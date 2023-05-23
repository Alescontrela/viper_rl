from typing import Any
from collections import OrderedDict
import numpy as np
import math
import tempfile
from PIL import Image
import ffmpeg
from moviepy.editor import ImageSequenceClip
import optax
import jax
import jax.numpy as jnp
import flax
from flax.training import train_state

from lpips_jax import LPIPSEvaluator


class TrainStateEMA(train_state.TrainState):
    ema_params: Any     


class TrainStateVQ(flax.struct.PyTreeNode):
    step: int
    tx: optax.GradientTransformation = flax.struct.field(pytree_node=False)

    vqgan_params: flax.core.FrozenDict[str, Any]
    vqgan_opt_state: optax.OptState

    disc_params: flax.core.FrozenDict[str, Any]
    disc_model_state: flax.core.FrozenDict[str, Any]
    disc_opt_state: optax.OptState

    lpips_variables: flax.core.FrozenDict[str, Any]

    def apply_vqgan_gradients(self, *, vqgan_grads, **kwargs):
        updates, new_vqgan_opt_state = self.tx.update(
            vqgan_grads, self.vqgan_opt_state, self.vqgan_params
        )
        new_vqgan_params = optax.apply_updates(self.vqgan_params, updates)
        return self.replace(
            step=self.step + 1,
            vqgan_params=new_vqgan_params,
            vqgan_opt_state=new_vqgan_opt_state,
            **kwargs,
        )

    def apply_disc_gradients(self, *, disc_grads, **kwargs):
        updates, new_disc_opt_state = self.tx.update(
            disc_grads, self.disc_opt_state, self.disc_params
        )
        new_disc_params = optax.apply_updates(self.disc_params, updates)
        return self.replace(
            step=self.step, # assume step is already updated in the other fn
            disc_params=new_disc_params,
            disc_opt_state=new_disc_opt_state,
            **kwargs,
        )

    @classmethod
    def create(
        cls, *, vqgan_params, disc_params, disc_model_state, 
        lpips_variables, tx
    ):
        vqgan_opt_state = tx.init(vqgan_params)
        disc_opt_state = tx.init(disc_params)
        return cls(
            step=0,
            tx=tx,
            vqgan_params=vqgan_params,
            vqgan_opt_state=vqgan_opt_state,
            disc_params=disc_params,
            disc_model_state=disc_model_state,
            disc_opt_state=disc_opt_state,
            lpips_variables=lpips_variables,
        )

def get_first_device(x):
    x = jax.tree_util.tree_map(lambda a: a[0], x)
    return jax.device_get(x)

    
def n2str(x):
    suffix = ''
    if x > 1e9:
        x /= 1e9
        suffix = 'B'
    elif x > 1e6:
        x /= 1e6
        suffix = 'M'
    elif x > 1e3:
        x /= 1e3
        suffix = 'K'
    return f'{x:.2f}{suffix}'


def print_model_size(params, name=''):
    model_params_size = jax.tree_util.tree_map(lambda x: x.size, params)
    total_params_size = sum(jax.tree_util.tree_flatten(model_params_size)[0])
    if name:
        print(f'{name} parameter count:', n2str(total_params_size))
    else:
        print('model parameter count:', n2str(total_params_size))


def get_optimizer(config):
    learning_rate_fn = optax.join_schedules([
        optax.linear_schedule(
            init_value=0.,
            end_value=config.lr,
            transition_steps=config.warmup_steps
        ),
        optax.constant_schedule(config.lr)
    ], [config.warmup_steps]) 
    tx = optax.adamw(learning_rate=learning_rate_fn)
    return tx, learning_rate_fn


def init_model_state_videogpt(rng, model, batch, config):
    variables = model.init(
        rngs={'params': rng, 'dropout': rng},
        **batch,
        training=True,
        method=model.loss
    ).unfreeze()
    params = variables.pop('params')
    assert len(variables) == 0
    print_model_size(params)

    tx, learning_rate_fn = get_optimizer(config)

    return TrainStateEMA.create(
        apply_fn=model.apply,
        params=params,
        ema_params=jax.tree_util.tree_map(jnp.array, params) if config.ema else None,
        tx=tx,
    ), learning_rate_fn

    
def init_model_state_vqgan(rng, model, batch, config):
    variables = model.vqgan.init(
        rngs={'params': rng, 'dropout': rng},
        image=batch['image'],
        deterministic=False
    ).unfreeze()
    vqgan_params = variables.pop('params')
    print_model_size(vqgan_params, name='vqgan')

    variables = model.disc.init(
        rngs={'params': rng},
        image=batch['image'],
        deterministic=False
    ).unfreeze()
    disc_params = variables.pop('params')
    disc_model_state = variables
    print_model_size(disc_params, name='disc')

    lpips_variables = LPIPSEvaluator(replicate=False, net='vgg16').params

    tx, learning_rate_fn = get_optimizer(config)

    return TrainStateVQ.create(
        vqgan_params=vqgan_params,
        disc_params=disc_params,
        disc_model_state=disc_model_state,
        lpips_variables=lpips_variables,
        tx=tx,
    ), learning_rate_fn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, total_iters, meter_names, prefix=""):
        self.iter_fmtstr = self._get_iter_fmtstr(total_iters)
        self.meters = OrderedDict({mn: AverageMeter(mn, ':6.3f')
                                   for mn in meter_names})
        self.prefix = prefix

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v, n=n)

    def display(self, iteration):
        entries = [self.prefix + self.iter_fmtstr.format(iteration)]
        entries += [str(meter) for meter in self.meters.values()]
        print('\t'.join(entries))

    def _get_iter_fmtstr(self, total_iters):
        num_digits = len(str(total_iters // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(total_iters) + ']'


def save_video_grid(video, fname=None, nrow=None, fps=10):
    b, t, h, w, c = video.shape

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
        if b % nrow != 0:
            nrow = 8
    ncol = math.ceil(b / nrow)
    padding = 1
    new_h = (padding + h) * ncol + padding
    new_h += new_h % 2
    new_w = (padding + w) * nrow + padding
    new_w += new_w % 2
    video_grid = np.zeros((t, new_h, new_w, c), dtype='uint8')
    for i in range(b):
        r = i // nrow
        c = i % nrow

        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]

    if fname is not None:
        clip = ImageSequenceClip(list(video_grid), fps=fps)
        clip.write_gif(fname, fps=fps)
        print('saved videos to', fname)
    
    return video_grid # THWC, uint8

    
def save_video(video, fname=None, fps=10):
    # video: TCHW, uint8
    T, H, W, C = video.shape
    if fname is None:
        fname = tempfile.NamedTemporaryFile().name + '.mp4'
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{W}x{H}')
        .output(fname, pix_fmt='yuv420p', vcodec='libx264', r=fps)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in video:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()
    print('Saved video to', fname)
    return fname


def add_border(video, color, width=0.025):
    # video: BTHWC in [0, 1]
    S = math.ceil(int(video.shape[3] * width))

    # top
    video[:, :, :S, :, 0] = color[0]
    video[:, :, :S, :, 1] = color[1]
    video[:, :, :S, :, 2] = color[2]

    # bottom
    video[:, :, -S:, :, 0] = color[0]
    video[:, :, -S:, :, 1] = color[1]
    video[:, :, -S:, :, 2] = color[2]

    # left
    video[:, :, :, :S, 0] = color[0]
    video[:, :, :, :S, 1] = color[1]
    video[:, :, :, :S, 2] = color[2]

    # right
    video[:, :, :, -S:, 0] = color[0]
    video[:, :, :, -S:, 1] = color[1]
    video[:, :, :, -S:, 2] = color[2]



def save_image_grid(images, fname=None, nrow=None):
    b, h, w, c = images.shape
    images = (images * 255).astype('uint8')

    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    image_grid = np.zeros(((padding + h) * ncol + padding,
                          (padding + w) * nrow + padding, c), dtype='uint8')
    for i in range(b):
        r = i // nrow
        c = i % nrow

        start_r = (padding + h) * r
        start_c = (padding + w) * c
        image_grid[start_r:start_r + h, start_c:start_c + w] = images[i]

    if fname is not None:
        image = Image.fromarray(image_grid)
        image.save(fname)
        print('saved image to', fname)

    return image_grid
