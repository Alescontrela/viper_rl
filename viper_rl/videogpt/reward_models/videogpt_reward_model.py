from typing import Any, Union, Dict, Tuple, Optional, List
import os
import pickle
import functools
import numpy as np
import jax
import jax.numpy as jnp
import tqdm
import argparse

from ..models import load_videogpt, AE
from .. import sampler

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

class InvalidSequenceError(Exception):
    def __init__(self, message):            
        super().__init__(message)


class VideoGPTRewardModel:

    PRIVATE_LIKELIHOOD_KEY = 'log_immutable_density'
    PUBLIC_LIKELIHOOD_KEY = 'density'

    def __init__(self, task: str, vqgan_path: str, videogpt_path: str,
                 camera_key: str='image',
                 reward_scale: Optional[Union[Dict[str, Tuple], Tuple]]=None,
                 reward_model_device: int=0,
                 nll_reduce_sum: bool=True,
                 compute_joint: bool=True,
                 minibatch_size: int=64,
                 encoding_minibatch_size: int=64):
        """VideoGPT likelihood model for reward computation.

        Args:
            task: Task name, used for conditioning when class_map.pkl exists in videogpt_path.
            vqgan_path: Path to vqgan weights.
            videogpt_path: Path to videogpt weights.
            camera_key: Key for camera observation.
            reward_scale: Range to scale logits from [0, 1].
            reward_model_device: Device to run reward model on.
            nll_reduce_sum: Whether to reduce sum over nll.
            compute_joint: Whether to compute joint likelihood or use conditional.
            minibatch_size: Minibatch size for VideoGPT.
            encoding_minibatch_size: Minibatch size for VQGAN.
        """
        self.domain, self.task = task.split('_', 1)
        self.vqgan_path = vqgan_path
        self.videogpt_path = videogpt_path
        self.camera_key = camera_key
        self.reward_scale = reward_scale
        self.nll_reduce_sum = nll_reduce_sum
        self.compute_joint = compute_joint
        self.minibatch_size = minibatch_size
        self.encoding_minibatch_size = encoding_minibatch_size

        # Load VQGAN and VideoGPT weights.
        self.device = jax.devices()[reward_model_device]
        print(f'Reward model devices: {self.device}')
        self.ae = AE(path=vqgan_path, mode='jit')
        self.ae.ae_vars = jax.device_put(self.ae.ae_vars, self.device)
        self.model, variables, self.class_map = load_videogpt(videogpt_path, ae=self.ae, replicate=False)
        config = self.model.config
        self.sampler = sampler.VideoGPTSampler(self.model, mode='jit')
        self.variables = jax.device_put(variables, self.device)

        self.model_name = config.model
        self.n_skip = getattr(config, 'frame_skip', 1)
        self.class_cond = getattr(config, 'class_cond', False)
        self.seq_len = config.seq_len * self.n_skip
        self.seq_len_steps = self.seq_len

        # Load frame mask.
        if self.ae.mask_map is not None:
            self.mask = self.ae.mask_map[self.task]
            print(f'Loaded mask for task {self.task} from mask_map with keys: {self.ae.mask_map.keys()}')
            self.mask = jax.device_put(self.mask.astype(np.uint8), self.device)
        else:
            self.mask = None

        # Load task id.
        if self.class_cond and self.class_map is not None:
            self.task_id = None
            print(f'Available tasks: {list(self.class_map.keys())}')
            assert (self.task in self.class_map,
                    f'{self.task} not found in class map.')
            self.task_id = int(self.class_map[self.task])
            print(f'Loaded conditioning information for task {self.task}')
        elif self.class_cond:
            raise ValueError(
                f'No class_map for class_conditional model. '
                f'VideoGPT loaded class_map? {self.class_map is not None}')
        else:
            self.task_id = None

        print(
            f'finished loading {self.__class__.__name__}:'
            f'\n\tseq_len: {self.seq_len}'
            f'\n\tclass_cond: {self.class_cond}'
            f'\n\ttask: {self.task}'
            f'\n\tmodel: {self.model_name}'
            f'\n\tcamera_key: {self.camera_key}'
            f'\n\tseq_len_steps: {self.seq_len_steps}'
            f'\n\tmask? {self.mask is not None}'
            f'\n\ttask_id: {self.task_id}'
            f'\n\tn_skip? {self.n_skip}')

    def __call__(self, seq, **kwargs):
        return self.process_seq(self.compute_reward(seq, **kwargs), **kwargs)

    def rollout_video(self, init_frames, video_length, seed=0, open_loop_ctx=4, inputs_are_codes=False, decode=True, pbar=False):
        if inputs_are_codes:
            encodings = init_frames
        else:
            init_frames = self.process_images(init_frames)
            encodings = self.ae.encode(init_frames)
        rollout_length = min(video_length, self.seq_len // self.n_skip)
        batch = dict(
            encodings=encodings,
            label=self.expand_scalar(self.task_id or 0, encodings.shape[0], jnp.int32))
        if rollout_length > init_frames.shape[1]:
            encodings = jnp.pad(encodings, ((0, 0), (0, rollout_length - init_frames.shape[1]), (0, 0), (0, 0)))
            batch['encodings'] = encodings
            encodings = self.sampler(
                self.variables, batch=batch, log_tqdm=False,
                seed=seed, open_loop_ctx=init_frames.shape[1], decode=False)
            batch['encodings'] = encodings
        all_samples = [encodings]

        remaining_frames = video_length - encodings.shape[1]
        extra_sample_steps = max(remaining_frames // ((self.seq_len // self.n_skip) - open_loop_ctx), 0) 
        vid_range = tqdm.tqdm(range(extra_sample_steps)) if pbar else range(extra_sample_steps)
        for _ in vid_range:
            batch['encodings'] = jnp.roll(encodings, -((self.seq_len // self.n_skip) - open_loop_ctx), axis=1)
            encodings = self.sampler(
                self.variables, batch=batch, log_tqdm=False,
                seed=seed, open_loop_ctx=open_loop_ctx, decode=False)
            all_samples.append(encodings[:, open_loop_ctx:])
        all_samples = jnp.concatenate(all_samples, axis=1)
        if decode:
            return np.array(255 * (self.ae.decode(all_samples) * 0.5 + 0.5)).astype(np.uint8)
        else:
            return all_samples

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_likelihood(self, variables, embeddings, encodings, label):
        print(f'Tracing likelihood: Original embeddings shape: {embeddings.shape}, Encodings shape: {encodings.shape}')
        if self.n_skip > 1:
            encodings = encodings[:, self.n_skip - 1::self.n_skip]
            embeddings = embeddings[:, self.n_skip - 1::self.n_skip]
            print(f'\tAfter applying frame skip: Embeddings shape: {embeddings.shape}, Encodings shape: {encodings.shape}')
        likelihoods = self.model.apply(
            variables, embeddings, encodings, label=label, reduce_sum=self.nll_reduce_sum,
            method=self.model.log_prob)
        if self.compute_joint:
            ll = likelihoods.sum(-1)
        else:
            ll = likelihoods[:, -1]
        return ll

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_likelihood_for_initial_elements(self, variables, embeddings, encodings, label):
        print(f'Tracing init frame likelihood: Embeddings shape: {embeddings.shape}, Encodings shape: {encodings.shape}')
        if self.n_skip > 1:
            first_encodings = jnp.concatenate([encodings[:1, i::self.n_skip] for i in range(self.n_skip)], axis=0)
            first_embeddings = jnp.concatenate([embeddings[:1, i::self.n_skip] for i in range(self.n_skip)], axis=0)
            print(f'\tAfter applying frame skip: Embeddings shape: {first_embeddings.shape}, Encodings shape: {first_encodings.shape}')
        else:
            first_encodings, first_embeddings = encodings[:1], embeddings[:1]
        likelihoods = self.model.apply(
            variables, first_embeddings, first_encodings, label=label, reduce_sum=self.nll_reduce_sum,
            method=self.model.log_prob)
        if self.n_skip > 1:
            idxs = np.arange(len(likelihoods.shape))
            idxs[0] = 1
            idxs[1] = 0
            ll = likelihoods.transpose(idxs.tolist()).reshape((-1,) + likelihoods.shape[2:])[:-1]
        else:
            ll = likelihoods[0, :-1]
        if self.compute_joint:
            ll = jnp.cumsum(ll) / jnp.arange(1, len(ll) + 1)
        return ll
            
    def _reward_scaler(self, reward):
        if self.reward_scale:
            if isinstance(self.reward_scale, dict) and (self.task not in self.reward_scale):
                return reward
            rs = self.reward_scale[self.task] if isinstance(self.reward_scale, dict) else self.reward_scale
            reward = np.array(np.clip((reward - rs[0]) / (rs[1] - rs[0]), 0.0, 1.0))
            return reward
        else:
            return reward

    def compute_reward(self, seq: List[Dict[str, Any]]):
        """Use VGPT model to compute likelihoods for input sequence.
        Args:
            seq: Input sequence of states.
        Returns:
            seq: Input sequence with additional keys in the state dict.
        """
        if len(seq) < self.seq_len_steps:
            raise InvalidSequenceError(f'Input sequence must be at least {self.seq_len_steps} steps long. Seq len is {len(seq)}')
        label = self.task_id if self.class_cond else None

        # Where in sequence to start computing likelihoods. Don't perform redundant likelihood computations.
        start_idx = 0
        for i in range(self.seq_len_steps - 1, len(seq)):
            if not self.is_step_processed(seq[i]):
                start_idx = i
                break
        start_idx = int(max(start_idx - self.seq_len_steps + 1, 0))
        T = len(seq) - start_idx

        # Compute encodings and embeddings for image sequence.
        image_batch = jnp.stack([seq[i][self.camera_key] for i in range(start_idx, len(seq))])
        image_batch = self.process_images(image_batch)
        encodings = self.ae.encode(jnp.expand_dims(image_batch, axis=0))
        embeddings = self.ae.lookup(encodings)
        encodings, embeddings = encodings[0], embeddings[0]

        # Compute batch of encodings and embeddings for likelihood computation.
        idxs = list(range(T - self.seq_len + 1))
        batch_encodings = [encodings[idx:(idx + self.seq_len)] for idx in idxs]
        batch_embeddings = [embeddings[idx:(idx + self.seq_len)] for idx in idxs]
        batch_encodings = jax.device_put(jnp.stack(batch_encodings), self.device)
        batch_embeddings = jax.device_put(jnp.stack(batch_embeddings), self.device)

        rewards = []
        for i in range(0, len(idxs), self.minibatch_size):
            mb_encodings = batch_encodings[i:(i + self.minibatch_size)]
            mb_embeddings = batch_embeddings[i:(i + self.minibatch_size)]
            mb_label = self.expand_scalar(label, mb_encodings.shape[0], jnp.int32)
            rewards.append(sg(self._compute_likelihood(
                self.variables, mb_embeddings, mb_encodings, mb_label)))
        rewards = jnp.concatenate(rewards, axis=0)
        if len(rewards.shape) <= 1:
            rewards = self._reward_scaler(rewards)
        assert len(rewards) == (T - self.seq_len_steps + 1), f'{len(rewards)} != {T - self.seq_len_steps + 1}'
        for i, rew in enumerate(rewards):
            idx = start_idx + self.seq_len_steps - 1 + i
            assert not self.is_step_processed(seq[idx])
            seq[idx][VideoGPTRewardModel.PRIVATE_LIKELIHOOD_KEY] = rew

        if seq[0]['is_first']:
            first_encodings = batch_encodings[:1]
            first_embeddings = batch_embeddings[:1]
            first_label = self.expand_scalar(label, first_encodings.shape[0], jnp.int32)
            first_rewards = sg(self._compute_likelihood_for_initial_elements(
                self.variables, first_embeddings, first_encodings, first_label))
            if len(first_rewards.shape) <= 1:
                first_rewards = self._reward_scaler(first_rewards)
            assert len(first_rewards) == self.seq_len_steps - 1, f'{len(first_rewards)} != {self.seq_len_steps - 1}'
            for i, rew in enumerate(first_rewards):
                assert not self.is_step_processed(seq[i]), f'Step {i} already processed'
                seq[i][VideoGPTRewardModel.PRIVATE_LIKELIHOOD_KEY] = rew

        return seq

    def expand_scalar(self, scalar, size, dtype):
        if scalar is None: return None
        return jnp.array([scalar] * size, dtype=dtype)

    def is_step_processed(self, step):
        return VideoGPTRewardModel.PRIVATE_LIKELIHOOD_KEY in step.keys()

    def is_seq_processed(self, seq):
        for step in seq:
            if not self.is_step_processed(step):
                return False
        return True

    def process_images(self, image_batch):
        image_batch = jax.device_put(jnp.array(image_batch).astype(jnp.uint8), self.device)
        image_batch = image_batch * self.mask if self.mask is not None else image_batch
        return image_batch.astype(jnp.float32) / 127.5 - 1.0

    def process_seq(self, seq):
        for step in seq:
            if not self.is_step_processed(step):
                continue
            step[VideoGPTRewardModel.PUBLIC_LIKELIHOOD_KEY] = step[VideoGPTRewardModel.PRIVATE_LIKELIHOOD_KEY]
        return seq[self.seq_len_steps - 1:]

