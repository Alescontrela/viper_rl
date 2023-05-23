from functools import cached_property
from tqdm import tqdm
import numpy as np
import jax


class VideoGPTSampler:
    def __init__(self, model, mode='pmap'):
        self.ae = model.ae
        self.model = model
        self.mode = mode
        self.config = model.config

    def wrap_fn(self, fn, mode):
        if mode == 'pmap':
            return jax.pmap(fn)
        elif mode == 'jit':
            return jax.jit(fn)
        else:
            raise NotImplementedError

    def split_rng(self, rng):
        if self.mode == 'pmap':
            rng = jax.random.split(rng, jax.local_device_count())
        elif self.mode == 'jit':
            pass # no action necessary
        else:
            raise NotImplementedError
        return rng

    @cached_property
    def _model_step(self):
        def _fn(variables, cache, embeddings, label, decode_step):
            if cache is not None:
                variables.update(cache)
            else:
                assert np.prod(embeddings.shape[1:-1]) > 1
            logits, cache = self.model.apply(
                variables, embeddings, label=label,
                training=False, decode_step=decode_step,
                mutable=['cache']
            )
            return logits, cache
        return self.wrap_fn(_fn, self.mode)

    @cached_property
    def _sample_step(self):
        def _fn(logits, rng):
            new_rng, rng = jax.random.split(rng)
            samples = jax.random.categorical(rng, logits, axis=-1)
            return samples, new_rng
        return self.wrap_fn(_fn, self.mode)

    def __call__(self, variables, batch, seed=0, log_tqdm=True, open_loop_ctx=None, decode=True):
        batch = {k: v.copy() for k, v in batch.items()}
        batch = jax.device_get(self.ae.prepare_batch(batch))
        encodings = batch.pop('encodings')
        label = batch.pop('label', None)

        rng = self.split_rng(jax.random.PRNGKey(seed))

        samples = np.zeros_like(encodings)
        latent_shape = samples.shape[-3:]
        ctx = open_loop_ctx or self.config.open_loop_ctx
        if self.mode == 'pmap':
            samples[:, :, :ctx] = encodings[:, :, :ctx]
        elif self.mode == 'jit':
            samples[:, :ctx] = encodings[:, :ctx]
        samples = samples.reshape(*samples.shape[:-3], -1)

        n_cond = np.prod(latent_shape[1:]) * ctx
        n_tokens = np.prod(latent_shape)
        itr = list(range(n_cond, n_tokens))
        if log_tqdm:
            itr = tqdm(itr)

        def get_index(idx):
            if self.mode == 'pmap':
                return np.full((jax.local_device_count(),), idx, dtype=np.int32)
            elif self.mode == 'jit':
                return idx
            raise Exception()
        _, cache = self._model_step(
            variables, None, self.ae.lookup(samples), label, get_index(0)
        )
        for i in itr:
            logits, cache = self._model_step(
                variables, cache, self.ae.lookup(samples[...,  i - 1, None]), label, get_index(i)
            )
            s, rng = self._sample_step(logits, rng)
            samples[..., i, None] = jax.device_get(s)

        samples = samples.reshape(*samples.shape[:-1], *latent_shape)
        if decode:
            samples = self.ae.decode(samples) * 0.5 + 0.5
        samples = jax.device_get(samples)
        return samples
