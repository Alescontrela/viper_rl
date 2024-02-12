from typing import Any
import numpy as np
import optax
import jax
import jax.numpy as jnp
import flax.linen as nn

from .transformer import Transformer


class VideoGPT(nn.Module):
    config: Any
    ae: Any

    def setup(self):
        self.shape = (self.config.seq_len, *self.ae.latent_shape(self.config.image_size))
        self.model = Transformer(
            **self.config.transformer,
            shape=self.shape,
            out_dim=self.ae.n_embed
        )

    @property
    def metrics(self):
        return ['loss']

    def __call__(self, embeddings, label=None, decode_step=None, training=False):
        if self.config.class_cond:
            assert label is not None, f"label is required for class conditioned model"

        L = np.prod(self.shape)
        mask = jnp.tril(jnp.ones((L, L), dtype=bool))
        if self.config.class_cond:
            label = jax.nn.one_hot(label, num_classes=self.config.n_classes)
        else:
            label = None

        return self.model(
            embeddings,
            mask=mask,
            label=label,
            decode_step=decode_step,
            deterministic=not training,
        )

    def log_prob(self, embeddings, encodings, label=None, text=None, text_mask=None, training=False, reduce_sum=True):
        logits = self(embeddings, label=label, text=text, text_mask=text_mask, training=training)
        labels = jax.nn.one_hot(encodings, self.ae.n_embed)
        nll = optax.softmax_cross_entropy(logits, labels)
        if self.config.class_cond:
            nll = nll.reshape(*nll.shape[:2], -1)
            nll = (nll.max(-1) * np.prod(encodings.shape[2:]) + nll.sum(-1)) / (2 * np.prod(encodings.shape[2:]))
        else:
            if reduce_sum:
                nll = nll.reshape(*nll.shape[:2], -1).sum(-1)
        return -nll

    def log_prob(self, embeddings, encodings, label=None, training=False, reduce_sum=True):
        logits = self(embeddings, label=label, training=training)
        labels = jax.nn.one_hot(encodings, self.ae.n_embed)
        nll = optax.softmax_cross_entropy(logits, labels)
        if reduce_sum:
            nll = nll.reshape(*nll.shape[:2], -1).sum(-1)
        return -nll

    def loss(self, embeddings, encodings, label=None, training=True):
        loss = -self.log_prob(
            embeddings, encodings, label, training=training
        ).mean() / np.prod(self.shape[1:])
        return dict(loss=loss)
