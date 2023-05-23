import os
import os.path as osp
import pathlib
import sys
from unittest.mock import NonCallableMagicMock
import numpy as np
import time
import argparse
import yaml
import pickle
import wandb
import random
from datetime import datetime

import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax import jax_utils

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

from viper_rl.videogpt.loss_vqgan import VQPerceptualWithDiscriminator
from viper_rl.videogpt.data import load_dataset
from viper_rl.videogpt.train_utils import init_model_state_vqgan, ProgressMeter, save_image_grid, \
    get_first_device


def main():
    global model
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)

    config.ckpt = config.output_dir if osp.exists(config.output_dir) else None
    ckpt_dir = osp.join(config.output_dir, 'checkpoints')

    if is_master_process:
        wandb.init(project='viper_rl', config=config,
                   id=config.run_id, resume='allow', mode='online')
        wandb.run.name = config.run_id
        wandb.run.save()

    train_loader, _, mask_map = load_dataset(config, train=True, modality='image')
    test_loader, _, _ = load_dataset(config, train=False, modality='image')

    if mask_map is not None:
        pickle.dump(mask_map, open(osp.join(config.output_dir, 'mask_map.pkl'), 'wb'))
    
    batch = next(train_loader)
    batch = get_first_device(batch)

    model = VQPerceptualWithDiscriminator(config)

    state, lr_fn = init_model_state_vqgan(init_rng, model, batch, config)
    if config.ckpt is not None:
        state = checkpoints.restore_checkpoint(osp.join(config.ckpt, 'checkpoints'), state)
        print(f'Restored from checkpoint {osp.join(config.ckpt)}, at itr {int(state.step)}')

    iteration = int(state.step)
    state = jax_utils.replicate(state)

    # Randomize RNG so we get different rngs when restarting after preemptions
    # Otherwise we get the same sequence of noise
    rng = jax.random.fold_in(rng, jax.process_index() + random.randint(0, 100000))
    rngs = jax.random.split(rng, jax.local_device_count())
    while iteration <= config.total_steps:
        iteration, state, rngs = train(iteration, state, train_loader, lr_fn, rngs)
        if iteration % config.save_interval == 0 and is_master_process:
            state_ = jax_utils.unreplicate(state)
            save_path = checkpoints.save_checkpoint(ckpt_dir, state_, state_.step, keep=1, overwrite=True)
            print('Saved checkpoint to', save_path)
            del state_
        if iteration % config.viz_interval == 0:
            visualize(iteration, state, test_loader)
        iteration += 1

        
def train_step(batch, state, rng):
    # Generator update
    aux_G, grads_G = jax.value_and_grad(
        model.loss_G, has_aux=True
    )(state.vqgan_params, state.disc_params, state.disc_model_state,
      state.lpips_variables, batch, rng)
    aux_G, rng = aux_G[1]
    grads_G = jax.lax.pmean(grads_G, 'device')
    state = state.apply_vqgan_gradients(
        vqgan_grads=grads_G
    )

    # Discriminator update
    aux_D, grads_D = jax.value_and_grad(
        model.loss_D, has_aux=True
    )(state.disc_params, state.disc_model_state, state.vqgan_params,
      batch, rng)
    aux_D, disc_model_state, rng = aux_D[1]
    grads_D = jax.lax.pmean(grads_D, 'device')
    state = state.apply_disc_gradients(
        disc_grads=grads_D, disc_model_state=disc_model_state
    )

    aux = {**aux_G, **aux_D}
    return state, aux, rng


def train(iteration, state, train_loader, lr_fn, rngs):
    progress = ProgressMeter(
        config.total_steps,
        ['time', 'data'] + model.metrics
    )

    end = time.time()
    while True:
        batch = next(train_loader)
        batch_size = batch['image'].shape[1]
        progress.update(data=time.time() - end)

        state, metrics, rngs = jax.pmap(train_step, 'device')(batch, state, rngs)

        metrics = jax.device_get({k: metrics[k].mean() for k in model.metrics})
        metrics = {k: v.astype(np.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})

        if is_master_process:
            wandb.log({'train/lr': jax.device_get(lr_fn(iteration))}, step=iteration)
            wandb.log({**{f'train/{metric}': val
                        for metric, val in metrics.items()}
                    }, step=iteration)

        progress.update(time=time.time() - end)
        end = time.time()

        if iteration % config.log_interval == 0:
            progress.display(iteration)

        if iteration % config.save_interval == 0 or \
        iteration % config.viz_interval == 0 or \
        iteration >= config.total_steps:
            return iteration, state, rngs

        iteration += 1

        
def viz_step(batch, state):
    recon = model.vqgan.apply(
        {'params': state.vqgan_params},
        batch['image'],
        method=model.vqgan.reconstruct,
        rngs={'sample': jax.random.PRNGKey(0)}
    )
    recon = jnp.clip(recon, -1, 1)
    return recon


def visualize(iteration, state, test_loader):
    batch = next(test_loader)
    recon = jax.pmap(viz_step, 'device')(batch, state)
    
    batch = batch['image']
    batch = jax.device_get(batch).reshape(-1, *batch.shape[2:])
    recon = jax.device_get(recon).reshape(-1, *recon.shape[2:])
    viz = np.stack((recon, batch), axis=1)
    viz = viz.reshape(-1, *viz.shape[2:]) * 0.5 + 0.5

    viz = save_image_grid(viz)
    viz = wandb.Image(viz)

    if is_master_process:
        wandb.log({'eval/recon': viz}, step=iteration)


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
        config['save_interval'] = 10
        config['viz_interval'] = 10
        config['log_interval'] = 1
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
