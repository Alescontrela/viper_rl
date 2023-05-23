from functools import cached_property, partial
import os.path as osp
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from flax import jax_utils
from flax.training import checkpoints

from .vqgan import VQGAN
from .videogpt import VideoGPT
from .stylegan_disc import StyleGANDisc
from .vqgan import VQGAN


def load_videogpt(path, replicate=True, ae=None):
    config = pickle.load(open(osp.join(path, 'args'), 'rb'))
    if ae is None:
        ae = AE(config.ae_ckpt, mode='pmap' if replicate else 'jit')
    model = VideoGPT(config, ae)

    class_file = osp.join(path, 'class_map.pkl')
    if osp.exists(class_file):
        class_map = pickle.load(open(class_file, 'rb'))
        class_map = {k: int(v) for k, v in class_map.items()}
    else:
        class_map = None

    state = checkpoints.restore_checkpoint(osp.join(path, 'checkpoints'), None)
    variables = {'params': state['ema_params'] or state['params']}
    if replicate:
        variables = jax_utils.replicate(variables)
    return model, variables, class_map

    
def load_vqgan(path, replicate=True):
    config = pickle.load(open(osp.join(path, 'args'), 'rb'))
    model = VQGAN(**config.ae)
    
    mask_file = osp.join(path, 'mask_map.pkl')
    if osp.exists(mask_file):
        mask_map = pickle.load(open(mask_file, 'rb'))
        mask_map = {k: v.astype(np.uint8) for k, v in mask_map.items()}
    else:
        mask_map = None
    
    state = checkpoints.restore_checkpoint(osp.join(path, 'checkpoints'), None)
    variables = {'params': state['vqgan_params']}
    if replicate:
        variables = jax_utils.replicate(variables)
    return model, variables, mask_map
    

class AE:
    def __init__(self, path, mode='pmap'):
        path = osp.expanduser(path)
        self.ae, self.ae_vars, self.mask_map = load_vqgan(path, replicate=mode == 'pmap')
        self.mode = mode

    def latent_shape(self, image_size):
        return self.ae.latent_shape(image_size)

    @property
    def channels(self):
        return self.ae.codebook_embed_dim
    
    @property
    def n_embed(self):
        return self.ae.n_embed

    def wrap_fn(self, fn):
        if self.mode == 'jit':
            fn = jax.jit(fn)
        elif self.mode == 'pmap':
            fn = jax.pmap(fn, 'device')
        else:
            raise ValueError(f'Unsupported mode: {self.mode}')
        return partial(fn, self.ae_vars)

    @cached_property
    def _encode(self):
        def fn(variables, video):
            T = video.shape[1]
            video = video.reshape(-1, *video.shape[2:])
            out = self.ae.apply(
                variables,
                video,
                deterministic=True,
                method=self.ae.encode
            )
            encodings = out['encodings']
            return encodings.reshape(-1, T, *encodings.shape[1:])
        return self.wrap_fn(fn)

    @cached_property
    def _decode(self):
        def fn(variables, encodings):
            T = encodings.shape[1]
            encodings = encodings.reshape(-1, *encodings.shape[2:])
            recon = self.ae.apply(
                variables,
                encodings,
                is_embed=False,
                deterministic=True,
                method=self.ae.decode
            )
            recon = jnp.clip(recon, -1, 1)
            return recon.reshape(-1, T, *recon.shape[1:])
        return self.wrap_fn(fn)

    @cached_property 
    def _lookup(self):
        def fn(variables, encodings):
            return self.ae.apply(
                variables,
                encodings,
                method=self.ae.codebook_lookup
            )
        return self.wrap_fn(fn)

    @cached_property
    def _lookup_jit(self):
        def fn(variables, encodings):
            return self.ae.apply(
                jax.tree_util.tree_map(lambda x: x[0], variables),
                encodings,
                method=self.ae.codebook_lookup
            )
        return partial(jax.jit(fn), self.ae_vars)

    @cached_property
    def _identity(self):
        return lambda x: x
    
    def encode(self, video):
        if self.mode == 'jit':
            is_pre_encoded = len(video.shape) == 4
        else:
            is_pre_encoded = len(video.shape) == 5
        if is_pre_encoded:
            return video
        encodings = self._encode(video)
        return encodings
    
    def decode(self, encodings):
        return self._decode(encodings)
    
    def lookup(self, encodings):
        return self._lookup(encodings)

    def lookup_jit(self, encodings):
        return self._lookup_jit(encodings)

    def prepare_batch(self, batch):
        if 'encodings' in batch:
            encodings = batch.pop('encodings')
        else:
            video = batch.pop('video')
            encodings = self.encode(video)

        if 'embeddings' in batch:
            embeddings = batch.pop('embeddings')
        else:
            embeddings = self.lookup(encodings)
        batch.update(embeddings=embeddings, encodings=encodings)
        return batch
