import copy
from collections import defaultdict
import numpy as np
import time

import embodied

from . import generic
from . import selectors
from . import limiters


class Uniform(generic.Generic):
    def __init__(
        self,
        length,
        capacity=None,
        directory=None,
        online=False,
        chunks=1024,
        min_size=1,
        samples_per_insert=None,
        tolerance=1e4,
        seed=0,
    ):
        if samples_per_insert:
            limiter = limiters.SamplesPerInsert(samples_per_insert, tolerance, min_size)
        else:
            limiter = limiters.MinSize(min_size)
        assert not capacity or min_size <= capacity
        super().__init__(
            length=length,
            capacity=capacity,
            remover=selectors.Fifo(),
            sampler=selectors.Uniform(seed),
            limiter=limiter,
            directory=directory,
            online=online,
            chunks=chunks,
        )


class UniformRelabel(generic.Generic):
    def __init__(
        self,
        length,
        reward_model,
        add_mode,
        capacity=None,
        directory=None,
        online=False,
        chunks=1024,
        min_size=1,
        samples_per_insert=None,
        tolerance=1e4,
        seed=0,
    ):
        if samples_per_insert:
            limiter = limiters.SamplesPerInsert(samples_per_insert, tolerance, min_size)
        else:
            limiter = limiters.MinSize(min_size)
        assert not capacity or min_size <= capacity
        super().__init__(
            length=length,
            capacity=capacity,
            remover=selectors.Fifo(),
            sampler=selectors.Uniform(seed),
            limiter=limiter,
            directory=directory,
            online=online,
            chunks=chunks,
        )
        self.reward_model = reward_model
        self.add_mode = add_mode
        if self.add_mode == "chunk":
            self.added_seq_len = self.length - self.reward_model.seq_len_steps + 1
            self.prev_seq = defaultdict(tuple)
        elif self.add_mode == "episode":
            self.episode_streams = defaultdict(list)
            self.full_episodes = defaultdict(list)
        else:
            raise ValueError(f"Unknown add_mode: {self.add_mode}")

    def add(self, step, worker=0, load=False):
        if self.add_mode == "chunk":
            return self.add_chunk(step, worker, load)
        elif self.add_mode == "episode":
            return self.add_episode(step, worker, load)

    def add_chunk(self, step, worker=0, load=False):
        step = {k: v for k, v in step.items() if not k.startswith("log_")}
        step["id"] = np.asarray(embodied.uuid(step.get("id")))
        stream = self.streams[worker]
        stream.append(step)
        self.counters[worker] += 1
        # if 'log_immutable_density' not in step:
        #   step['log_immutable_density'] = 0
        #   self.saver and self.saver.add(step, worker)
        #   del step['log_immutable_density']
        # else:
        #   self.saver and self.saver.add(step, worker)
        if len(stream) < self.length or self.counters[worker] < self.stride:
            return

        # Batch reward computation.
        next_step_processed = self.reward_model.is_step_processed(
            stream[self.reward_model.seq_len_steps - 1]
        )
        is_last_idxs = [
            i for i, step in enumerate(stream) if step["is_last"] or step["is_terminal"]
        ]
        seq_has_last_step = (len(is_last_idxs) == 1) and (
            is_last_idxs[-1] == len(stream) - 1
        )
        first_step_is_first = stream[0]["is_first"]
        if len(is_last_idxs) > 0 and not seq_has_last_step:
            # Only handle continuous sequences for now.
            return
        elif not next_step_processed:
            seq = self.reward_model(tuple(stream))
        elif seq_has_last_step:
            split_idx = 0
            for i in range(len(stream)):
                if not self.reward_model.is_step_processed(stream[i]):
                    split_idx = i
                    break
            seq = self.reward_model(tuple(stream))[
                split_idx - self.reward_model.seq_len_steps + 1 :
            ]
        else:
            return

        # New first step in sequence.
        first_step = copy.deepcopy(seq[0])
        first_step["is_first"] = first_step_is_first
        seq = (first_step,) + seq[1:]

        # Add sequences to replay.
        combined = self.prev_seq[worker] + seq
        start_idx = 1 if len(self.prev_seq[worker]) > 0 else 0
        for i in range(start_idx, len(combined) - self.added_seq_len + 1):
            self.add_seq(combined[i : i + self.added_seq_len], load)
        self.prev_seq[worker] = copy.deepcopy(seq)
        self.counters[worker] = 0

    def add_episode(self, step, worker=0, load=False):
        step = {k: v for k, v in step.items() if not k.startswith("log_")}
        step["id"] = np.asarray(embodied.uuid(step.get("id")))
        episode = self.episode_streams[worker]
        episode.append(step)

        self.counters[worker] += 1
        if not (step["is_last"] or step["is_terminal"]):
            return
        self.counters[worker] = 0

        if not self.reward_model.is_seq_processed(episode):
            try:
                # Compute likelihoods for entire episode and save it.
                episode = self.reward_model(episode)
            except Exception as err:
                print(f"{type(err).__name__} was raised: {err}. Skipping sequence.")
                self.episode_streams[worker] = []
                return

        # for step in seq:
        #   self.saver and self.saver.add(step, worker)

        self.full_episodes[worker].extend(episode)
        end_idx = len(self.full_episodes[worker]) - self.length + 1
        self.episode_streams[worker] = []
        if end_idx <= 0:  # Not enough steps to save.
            return
        for i in range(end_idx):
            # Add `self.length` size chunks to the replay.
            self.add_seq(self.full_episodes[worker][i : i + self.length], load)
        self.full_episodes[worker] = self.full_episodes[worker][end_idx:]

    def add_seq(self, seq, load=False):
        key = embodied.uuid()
        if load:
            assert self.limiter.want_load()[0]
        else:
            dur = wait(self.limiter.want_insert, "Replay insert is waiting")
            self.metrics["inserts"] += 1
            self.metrics["insert_wait_dur"] += dur
            self.metrics["insert_wait_count"] += int(dur > 0)

        wait(self.limiter.want_insert, lambda: "Replay insert is waiting")
        self.table[key] = seq
        self.remover[key] = seq
        self.sampler[key] = seq
        while self.capacity and len(self) > self.capacity:
            self._remove(self.remover())


def wait(predicate, message, sleep=0.001, notify=1.0):
    start = time.time()
    notified = False
    while True:
        allowed, detail = predicate()
        duration = time.time() - start
        if allowed:
            return duration
        if not notified and duration >= notify:
            print(f"{message} ({detail})")
            notified = True
        time.sleep(sleep)
