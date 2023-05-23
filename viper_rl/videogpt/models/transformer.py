from typing import Any, Tuple
import numpy as np
import flax.linen as nn
import flax.linen.initializers as nn_init
import jax
import jax.numpy as jnp


class Transformer(nn.Module):
    embed_dim: int
    mlp_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    attention_dropout: float
    shape: Tuple[int]
    out_dim: int
    
    @nn.compact
    def __call__(self, x, mask=None, deterministic=False, label=None, decode_step=None):
        old_shape = x.shape[1:-1]
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        x = nn.Dense(self.embed_dim)(x)
        if decode_step is None or x.shape[1] > 1:
            x = RightShift()(x)
        else:
            x_shift = RightShift()(x)
            x = jax.lax.cond(decode_step > 0, lambda: x, lambda: x_shift)

        position_bias = BroadcastPositionBiases(shape=self.shape)(x)

        if decode_step is not None and x.shape[1] == 1:
            position_bias = position_bias[decode_step]
        else:
            position_bias = position_bias[:x.shape[1]]
        x += position_bias

        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)

        for i in range(self.num_layers):
            x = TransformerLayer(
                embed_dim=self.embed_dim,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
            )(x, mask=mask, label=label, decode_step=decode_step, deterministic=deterministic)
        
        x = LayerNorm()(x, cond=label)
        x = nn.Dense(self.out_dim)(x)
        x = x.reshape(x.shape[0], *old_shape, x.shape[-1])
        return x 


class TransformerLayer(nn.Module):
    embed_dim: int
    mlp_dim: int
    num_heads: int
    dropout: float
    attention_dropout: float

    @nn.compact
    def __call__(self, x, mask=None, label=None, decode_step=None, deterministic=False):
        h = LayerNorm()(x, cond=label)
        h = MultiHeadAttention(
            num_heads=self.num_heads,
            head_dim=self.embed_dim // self.num_heads,
            dropout_rate=self.attention_dropout,
        )(h, mask=mask, decode_step=decode_step,
          deterministic=deterministic)
        h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
        x = x + h

        h = LayerNorm()(x, cond=label)
        h = nn.Sequential([
            nn.Dense(self.mlp_dim),
            gelu2,
            nn.Dropout(rate=self.dropout, deterministic=deterministic),
            nn.Dense(self.embed_dim)
        ])(h)
        h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
        x = x + h
        
        return x


class MultiHeadAttention(nn.Module):
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.

    @nn.compact
    def __call__(self, inputs, mask=None, decode_step=None, deterministic=False):
        qkv = nn.DenseGeneral(
            axis=-1, features=(self.num_heads, 3 * self.head_dim)
        )(inputs)
        query, key, value = jnp.split(qkv, 3, axis=-1)

        if decode_step is not None:
            cached_key = self.variable('cache', 'cached_key', lambda: key)
            cached_value = self.variable('cache', 'cached_value', lambda: value)

            is_slice = inputs.shape[1] == 1
            if is_slice:
                key = cached_key.value.at[:, decode_step].set(key[:, 0])
            else:
                key = cached_key.value.at[:].set(key)

            if is_slice:
                value = cached_value.value.at[:, decode_step].set(value[:, 0])
            else:
                value = cached_value.value.at[:].set(value)

            if mask is not None and is_slice:
                mask = mask[decode_step, None]
                mask = mask[:, :key.shape[1]]

            cached_key.value = key
            cached_value.value = value
        
        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.:
            dropout_rng = self.make_rng('dropout')
        
        x = nn.attention.dot_product_attention(
            query, key, value, mask=mask,
            dropout_rng=dropout_rng, dropout_rate=self.dropout_rate,
            deterministic=deterministic
        )
        out = nn.DenseGeneral(features=inputs.shape[-1], axis=(-2, -1))(x)
        return out

         
class LayerNorm(nn.Module):
    epsilon: float = 1e-6
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Any = nn_init.zeros
    scale_init: Any = nn_init.ones
    reduction_axes: Any = -1
    feature_axes: Any = -1

    @nn.compact
    def __call__(self, x, cond=None):
        features = x.shape[-1]
        x = jnp.asarray(x, jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
        var = jnp.maximum(0., mean2 - jax.lax.square(mean))

        y = x - mean
        mul = jax.lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            if cond is None:
                scale = self.param('scale', self.scale_init, (features,), jnp.float32)
                mul *= scale
            else:
                scale = nn.Dense(features, use_bias=False)(cond)
                scale = scale.reshape(scale.shape[0], *((1,) * (len(x.shape) - 2)), scale.shape[-1])
                mul *= 1 + scale
        y *= mul

        if self.use_bias:
            if cond is None:
                bias = self.param('bias', self.bias_init, (features,), jnp.float32)
            else:
                bias = nn.Dense(features, use_bias=False)(cond)
                bias = bias.reshape(bias.shape[0], *((1,) * (len(x.shape) - 2)), bias.shape[-1])
            y += bias
        return y


class BroadcastPositionBiases(nn.Module):
    shape: Tuple[int]

    @nn.compact
    def __call__(self, x):    
        shape = self.shape
        n_dim = len(shape)
        embed_dim = x.shape[-1]

        chunk_sizes = [embed_dim // n_dim + (i < (embed_dim % n_dim))
                       for i in range(n_dim)]
        assert sum(chunk_sizes) == embed_dim, f'sum({chunk_sizes}) = {sum(chunk_sizes)} != {embed_dim}'

        embs = [
            self.param(f'd_{i}', nn.initializers.normal(stddev=0.02),
                            (shape[i], chunk_sizes[i]), jnp.float32)
            for i in range(n_dim)
        ]

        out = []
        for i in range(n_dim):
            e = embs[i]
            e = jnp.reshape(e, (1,) + (1,) * i + (shape[i],) + (1,) * (n_dim - i - 1) + (-1,))
            e = jnp.broadcast_to(e, (1, *shape, e.shape[-1]))
            out.append(e)
        out = jnp.concatenate(out, axis=-1)
        out = jnp.reshape(out, (np.prod(shape), embed_dim))
        return out    


class RightShift(nn.Module):
    @nn.compact
    def __call__(self, x):
        sos = self.param('sos', nn.initializers.normal(stddev=0.02),
                         (x.shape[-1],), jnp.float32)
        sos = jnp.tile(sos[None, None], (x.shape[0], 1, 1))
        x = jnp.concatenate([sos, x[:, :-1]], axis=1)
        return x


def gelu2(x):
    return nn.sigmoid(1.702 * x) * x
