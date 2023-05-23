import os
import os.path as osp
import pathlib
import sys
import numpy as np
import time
import argparse
import yaml
import pickle
import random
import wandb
from datetime import datetime

import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax import jax_utils

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

from viper_rl.videogpt.models import AE, VideoGPT
from viper_rl.videogpt.sampler import VideoGPTSampler
from viper_rl.videogpt.data import load_dataset
from viper_rl.videogpt.train_utils import init_model_state_videogpt, get_first_device, ProgressMeter, \
    save_video_grid, add_border, save_video


def main():
    global model
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)

    config.ckpt = config.output_dir if osp.exists(config.output_dir) else None

    if is_master_process:
        wandb.init(project='viper_rl', config=config,
                   id=config.run_id, resume='allow', mode='online')
        wandb.run.name = config.run_id
        wandb.run.save()

    train_loader, class_map, _ = load_dataset(config, train=True, modality='video')
    test_loader, class_map_test, _ = load_dataset(config, train=False, modality='video')

    if config.class_cond:
        assert class_map == class_map_test, (class_map, class_map_test)
        pickle.dump(class_map, open(osp.join(config.output_dir, 'class_map.pkl'), 'wb'))

    ae = AE(config.ae_ckpt)

    batch = next(train_loader)
    batch = ae.prepare_batch(batch)
    batch = get_first_device(batch)

    model = VideoGPT(config, ae)
    sampler = VideoGPTSampler(model)
    state, schedule_fn = init_model_state_videogpt(init_rng, model, batch, config)

    if config.ckpt is not None:
        state = checkpoints.restore_checkpoint(osp.join(config.ckpt, 'checkpoints'), state)
        print(f'Restored from checkpoint {osp.join(config.ckpt)}, at itr {int(state.step)}')
    
    iteration = int(state.step)
    state = jax_utils.replicate(state)

    ckpt_dir = osp.join(config.output_dir, 'checkpoints')

    rng = jax.random.fold_in(rng, jax.process_index() + random.randint(0, 100000))
    rngs = jax.random.split(rng, jax.local_device_count())
    best_loss = float('inf')
    while iteration <= config.total_steps:
        iteration, state, rngs = train(iteration, ae, model, state, train_loader,
                                       schedule_fn, rngs)
        if iteration % config.test_interval == 0:
            val_loss, rngs = validate(iteration, ae, model, state, test_loader, rngs)
            is_best = val_loss < best_loss
            best_loss = min(best_loss, val_loss)
        if iteration % config.viz_interval == 0:
            visualize(sampler, ae, iteration, state, test_loader)
        if iteration % config.save_interval == 0 and is_master_process and is_best:
            state_ = jax_utils.unreplicate(state)
            save_path = checkpoints.save_checkpoint(ckpt_dir, state_, state_.step, keep=1, overwrite=True)
            print('Saved checkpoint to', save_path)
            del state_
        iteration += 1


def train_step(batch, state, rng):
    def loss_fn(params, batch, rng):
        rng_dropout, new_rng = jax.random.split(rng)
        variables = {'params': params}
        out = state.apply_fn(
            variables,
            **batch,
            training=True,
            rngs={'dropout': rng_dropout},
            method=model.loss
        )
        return out['loss'], (out, new_rng)
    aux, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch, rng)
    out, rng = aux[1]
    grads = jax.lax.pmean(grads, axis_name='device')
    new_state = state.apply_gradients(
        grads=grads,
    )

    if config.ema:
        decay = jnp.where(state.step == 0, 0.0, config.ema)
        ema_params = jax.tree_util.tree_map(
            lambda a, b: decay * a + (1.0 - decay) * b,
            state.ema_params, new_state.params
        )
        new_state = new_state.replace(ema_params=ema_params)
    return new_state, out, rng


def train(iteration, ae, model, state, train_loader, schedule_fn, rngs):
    progress = ProgressMeter(
        config.total_steps,
        ['time', 'data'] + model.metrics
    )

    p_train_step = jax.pmap(train_step, axis_name='device', donate_argnums=(0, 1, 2))

    end = time.time()
    while True:
        batch = next(train_loader)
        batch_size = batch[list(batch.keys())[0]].shape[1]
        progress.update(data=time.time() - end)

        batch = ae.prepare_batch(batch)
        state, return_dict, rngs = p_train_step(batch=batch, state=state, rng=rngs)

        metrics = {k: return_dict[k].mean() for k in model.metrics}
        metrics = {k: v.astype(jnp.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})

        if is_master_process:
            wandb.log({'train/lr': jax.device_get(schedule_fn(iteration))}, step=iteration)
            wandb.log(jax.device_get({**{f'train/{metric}': val
                        for metric, val in metrics.items()}
                    }), step=iteration)

        progress.update(time=time.time() - end)
        end = time.time()

        if iteration % config.log_interval == 0:
            progress.display(iteration)

        if iteration % config.viz_interval == 0 or \
        iteration % config.test_interval == 0 or \
        iteration % config.save_interval == 0 or \
        iteration >= config.total_steps:
            return iteration, state, rngs

        iteration += 1


def val_step(batch, state, rng):
    def loss_fn(params, batch, rng):
        rng_dropout, new_rng = jax.random.split(rng)
        variables = {'params': params}
        out = state.apply_fn(
            variables,
            **batch,
            training=False,
            rngs={'dropout': rng_dropout},
            method=model.loss
        )
        return out, new_rng
    out, rng = loss_fn(state.params, batch, rng)
    out = jax.lax.pmean(out, axis_name='device')
    return out, rng


def validate(iteration, ae, model, state, test_loader, rngs):
    progress = ProgressMeter(
        50,
        ['time', 'data'] + model.metrics,
        prefix='\tTest:'
    )

    p_val_step = jax.pmap(val_step, axis_name='device', donate_argnums=(0, 1, 2))

    end = time.time()
    for i in range(50):
        batch = next(test_loader)
        batch_size = batch[list(batch.keys())[0]].shape[1]
        progress.update(data=time.time() - end)

        batch = ae.prepare_batch(batch)
        return_dict, rngs = p_val_step(batch=batch, state=state, rng=rngs)

        metrics = {k: return_dict[k].mean() for k in model.metrics}
        metrics = jax.device_get({k: v.astype(jnp.float32) for k, v in metrics.items()})
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})
        progress.update(time=time.time() - end)
        end = time.time()

        if i % config.log_interval == 0:
            progress.display(i)

    progress.display(i)

    metrics = {metric: progress.meters[metric].avg
               for metric in model.metrics}

    if is_master_process:
        wandb.log({**{f'val/{metric}': val
                      for metric, val in metrics.items()}
                  }, step=iteration)
    return metrics['loss'], rngs


def visualize(sampler, ae, iteration, state, test_loader):
    batch = next(test_loader)
    video = batch['video']
    if len(video.shape) == 5: # NBTHW
        video = ae.decode(video)
    variables = {'params': state.ema_params if hasattr(state, 'ema_params') else state.params}
    samples = sampler(variables, batch).copy()
    samples = samples.reshape(-1, *samples.shape[-4:])
    real = jax.device_get(video)
    real = (real * 0.5 + 0.5).reshape(-1, *real.shape[-4:])
    add_border(samples[:, :config.open_loop_ctx], (0., 1., 0.))
    add_border(samples[:, config.open_loop_ctx:], (1., 0., 0.))

    videos = np.stack((samples, real), axis=1)
    videos = videos.reshape(-1, *videos.shape[2:])
    videos = (videos * 255).astype(np.uint8)

    videos = save_video_grid(videos)
    if is_master_process:
        videos = np.transpose(videos, (0, 3, 1, 2))
        wandb.log({'viz/sample': wandb.Video(videos, format='gif')}, step=iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    args.run_id = args.output_dir.split('/')[-1] + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    print(f'JAX process: {jax.process_index()} / {jax.process_count()}')
    print(f'JAX total devices: {jax.device_count()}')
    print(f'JAX local devices: {jax.local_device_count()}')

    config = yaml.safe_load(open(args.config, 'r'))
    if os.environ.get('DEBUG') == '1':
        config['viz_interval'] = 10
        config['log_interval'] = 1
        config['test_interval'] = 10
        args.output_dir = osp.join(osp.dirname(args.output_dir), f'DEBUG_{osp.basename(args.output_dir)}')
        args.run_id = f'DEBUG_{args.run_id}'

    print(f"Logging to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    args_d = vars(args)
    args_d.update(config)
    pickle.dump(args, open(osp.join(args.output_dir, 'args'), 'wb'))
    config = args

    is_master_process = jax.process_index() == 0

    main()
