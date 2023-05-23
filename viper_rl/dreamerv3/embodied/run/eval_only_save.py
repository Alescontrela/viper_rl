import io
import re
from datetime import datetime

import embodied
import numpy as np


def eval_only_save(agent, env, logger, args):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    step = logger.step
    metrics = embodied.Metrics()
    print("Observation space:", env.obs_space)
    print("Action space:", env.act_space)

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        logger.add({"length": length, "score": score}, prefix="episode")
        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        metrics.add(stats, prefix="stats")

    epsdir = embodied.Path(args.logdir) / "saved_episodes"
    epsdir.mkdirs()
    print("Saving episodes:", epsdir)

    def save(ep):
        time = datetime.now().strftime("%Y%m%dT%H%M%S")
        uuid = str(embodied.uuid())
        score = str(np.round(ep["reward"].sum(), 1)).replace("-", "m")
        length = len(ep["reward"])
        filename = epsdir / f"{time}-{uuid}-len{length}-rew{score}.npz"
        with io.BytesIO() as stream:
            np.savez_compressed(stream, **ep)
            stream.seek(0)
            filename.write(stream.read(), mode="wb")
        print("Saved episode:", filename)

    saver = embodied.Worker(save, "thread")

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_episode(lambda ep, worker: saver(ep))
    driver.on_step(lambda tran, _: step.increment())

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.from_checkpoint, keys=["agent"])

    print("Start evaluation loop.")
    policy = lambda *args: agent.policy(*args, mode="eval")
    while step < args.steps:
        driver(policy, steps=100)
        if should_log(step):
            logger.add(metrics.result())
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)
    logger.write()
