from typing import Any, Optional, Tuple
import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp


class VQGAN(nn.Module):
    ch: int
    ch_mult: int
    num_res_blocks: int
    attn_resolutions: Tuple[int]
    z_channels: int
    double_z: bool
    dropout: float
    n_embed: int
    embed_dim: int
    patch_size: Tuple[int]
    channels: Optional[int] = None
    dtype: Any = jnp.float32

    def setup(self):
        channels = self.channels or 3
        self.encoder = Encoder(ch=self.ch, ch_mult=self.ch_mult,
                               num_res_blocks=self.num_res_blocks,
                               attn_resolutions=self.attn_resolutions,
                               z_channels=self.z_channels,
                               double_z=self.double_z,
                               dropout=self.dropout,
                               downsample=self.patch_size)
        self.decoder = Decoder(ch=self.ch, ch_mult=self.ch_mult,
                               out_ch=channels, num_res_blocks=self.num_res_blocks,
                               attn_resolutions=self.attn_resolutions,
                               dropout=self.dropout,
                               upsample=self.patch_size)

        self.quantize = VectorQuantizer(n_e=self.n_embed, e_dim=self.embed_dim)

        ndims = len(self.patch_size)
        self.quant_conv = nn.Conv(self.embed_dim, [1] * ndims)
        self.post_quant_conv = nn.Conv(self.z_channels, [1] * ndims)

    @property
    def metrics(self):
        return ['vq_loss', 'ae_loss', 'perplexity']

    def latent_shape(self, image_size):
        return tuple([image_size // p for p in self.patch_size])

    def codebook_lookup(self, encodings):
        return self.quantize(None, encodings)    

    def reconstruct(self, image):
        vq_out = self.encode(image, deterministic=True)
        recon = self.decode(vq_out['encodings'], deterministic=True)
        return recon
    
    def encode(self, image, deterministic=True):
        h = self.encoder(image, deterministic=deterministic)
        h = self.quant_conv(h)
        vq_out = self.quantize(h)
        return vq_out
    
    def decode(self, encodings, is_embed=False, deterministic=True):
        encodings = encodings if is_embed else self.codebook_lookup(encodings)
        recon = self.decoder(self.post_quant_conv(encodings), deterministic)
        return recon
 
    def __call__(self, image, deterministic=True):
        vq_out = self.encode(image, deterministic=deterministic)
        recon = self.decode(vq_out['embeddings'], is_embed=True,
                            deterministic=deterministic)
        return {
            'recon': recon,
            'vq_loss': vq_out['vq_loss'],
            'perplexity': vq_out['perplexity']
        }


class VectorQuantizer(nn.Module):
    n_e: int
    e_dim: int
    beta: float = 0.25

    @nn.compact
    def __call__(self, z, encoding_indices=None):
        def quantize(encoding_indices):
            w = jax.device_put(embeddings)
            return w[(encoding_indices,)]
        embeddings = self.param(
            'embeddings',
            lambda rng, shape, dtype: jax.random.uniform(
                rng, shape, dtype, minval=-1.0 / self.n_e, maxval=1.0 / self.n_e
            ),
            [self.n_e, self.e_dim], jnp.float32
        )
        
        if encoding_indices is not None:
            return quantize(encoding_indices)

        z_flattened = z.reshape(-1, z.shape[-1])
        d = jnp.sum(z_flattened ** 2, axis=1, keepdims=True) + \
            jnp.sum(embeddings.T ** 2, axis=0, keepdims=True) - \
            2 * jnp.einsum('bd,nd->bn', z_flattened, embeddings)
        
        min_encoding_indices = jnp.argmin(d, axis=1)
        z_q = quantize(min_encoding_indices)
        z_q = jnp.reshape(z_q, z.shape)

        loss = self.beta * jnp.mean((jax.lax.stop_gradient(z_q) - z) ** 2) + \
               jnp.mean((z_q - jax.lax.stop_gradient(z)) ** 2)
        z_q = z + jax.lax.stop_gradient(z_q - z)

        encodings_one_hot = jax.nn.one_hot(min_encoding_indices, num_classes=self.n_e)
        assert len(encodings_one_hot.shape) == 2
        avg_probs = jnp.mean(encodings_one_hot, axis=0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))
        
        min_encoding_indices = jnp.reshape(min_encoding_indices, z.shape[:-1])
        print(f'Latents of shape {min_encoding_indices.shape[1:]}')

        return {
            'embeddings': z_q,
            'encodings': min_encoding_indices,
            'vq_loss': loss,
            'perplexity': perplexity
        }
        

class Encoder(nn.Module):
    ch: int
    ch_mult: Tuple
    num_res_blocks: int
    downsample: Tuple[int]
    attn_resolutions: Tuple
    z_channels: int
    double_z: bool = True
    dropout: float = 0.
    resample_with_conv: bool = True
 
    @nn.compact
    def __call__(self, x, deterministic=True):
        num_resolutions = len(self.ch_mult)
        downsample = self.downsample
        all_strides = []
        while not all([d == 1 for d in downsample]):
            strides = tuple([2 if d > 1 else 1 for d in downsample])
            downsample = tuple([max(1, d // 2) for d in downsample])
            all_strides.append(strides)
        assert len(all_strides) + 1 == num_resolutions
        ndims = len(x.shape[1:-1])

        cur_res = x.shape[2]
        h = nn.Conv(self.ch, [3] * ndims)(x)
        for i_level in range(num_resolutions):
            block_out = self.ch * self.ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                h = ResnetBlock(dropout=self.dropout,
                                out_channels=block_out,
                                deterministic=deterministic,
                               )(h)
                if cur_res in self.attn_resolutions:
                    h = AttnBlock()(h) 
            if i_level != num_resolutions - 1:
                h = Downsample(all_strides[i_level], self.resample_with_conv)(h)
                cur_res //= 2
        
        h = ResnetBlock(dropout=self.dropout, deterministic=deterministic)(h)
        h = AttnBlock()(h)
        h = ResnetBlock(dropout=self.dropout, deterministic=deterministic)(h)
        
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Conv(2 * self.z_channels if self.double_z else self.z_channels,
                    [3] * ndims)(h)
        return h


class Decoder(nn.Module):
    ch: int
    ch_mult: Tuple
    out_ch: int
    num_res_blocks: int
    upsample: Tuple[int]
    attn_resolutions: Tuple
    dropout: float = 0.
    resamp_with_conv: bool = True
    
    @nn.compact
    def __call__(self, z, deterministic=True):
        num_resolutions = len(self.ch_mult)
        upsample = self.upsample
        all_strides = []
        while not all([d == 1 for d in upsample]):
            strides = tuple([2 if d > 1 else 1 for d in upsample])
            upsample = tuple([max(1, d // 2) for d in upsample])
            all_strides.append(strides)
        assert len(all_strides) + 1 == num_resolutions
        ndims = len(z.shape[1:-1])

        block_in = self.ch * self.ch_mult[num_resolutions - 1]
        h = nn.Conv(block_in, [3] * ndims)(z)

        h = ResnetBlock(dropout=self.dropout, deterministic=deterministic)(h)
        h = AttnBlock()(h)
        h = ResnetBlock(dropout=self.dropout, deterministic=deterministic)(h)
        
        cur_res = z.shape[2]
        for i_level in reversed(range(num_resolutions)):
            block_out = self.ch * self.ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                h = ResnetBlock(dropout=self.dropout,
                                out_channels=block_out,
                                deterministic=deterministic,
                               )(h)
                if cur_res in self.attn_resolutions:
                    h = AttnBlock()(h)
            if i_level != 0:
                h = Upsample(all_strides[i_level - 1], self.resamp_with_conv)(h)
                cur_res *= 2
        
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Conv(self.out_ch, [3] * ndims)(h)
        return h


class Upsample(nn.Module):
    strides: Tuple[int]
    with_conv: bool

    @nn.compact
    def __call__(self, x):
        assert len(self.strides) == len(x.shape[1:-1])
        ndims = len(x.shape[1:-1])
        output_shape = (
            x.shape[0], 
            *[d * s for d, s in zip(x.shape[1:-1], self.strides)], 
            x.shape[-1]
        )
        x = jax.image.resize(x, output_shape, 
                             jax.image.ResizeMethod.NEAREST, antialias=False)
        if self.with_conv:
            x = nn.Conv(x.shape[-1], [3] * ndims)(x)
        return x

                
class Downsample(nn.Module):
    strides: Tuple[int]
    with_conv: bool

    @nn.compact
    def __call__(self, x):
        ndims = len(self.strides)
        if self.with_conv:
            x = nn.Conv(x.shape[-1], [3] * ndims, strides=self.strides)(x)
        else:
            x = nn.avg_pool(x, self.strides, strides=self.strides)
        return x


class ResnetBlock(nn.Module):
    dropout: float
    use_conv_shortcut: bool = False
    out_channels: Optional[int] = None
    deterministic: bool = False
    
    @nn.compact
    def __call__(self, x):
        out_channels = self.out_channels or x.shape[-1]
        ndims = len(x.shape[1:-1])
        
        h = x
        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Conv(out_channels, [3] * ndims)(h)

        h = nn.GroupNorm(num_groups=32)(h)
        h = nn.swish(h)
        h = nn.Dropout(self.dropout)(h, deterministic=self.deterministic)
        h = nn.Conv(out_channels, [3] * ndims)(h)

        if x.shape[-1] != out_channels:
            if self.use_conv_shortcut:
                x = nn.Conv(out_channels, [3] * ndims)(x)
            else:
                x = nn.Conv(out_channels, [1] * ndims)(x)
        return x + h


class AttnBlock(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        channels = x.shape[-1]
        ndims = len(x.shape[1:-1])

        h_ = x
        h_ = nn.GroupNorm(num_groups=32)(h_)
        q = nn.Conv(channels, [1] * ndims)(h_)
        k = nn.Conv(channels, [1] * ndims)(h_)
        v = nn.Conv(channels, [1] * ndims)(h_)

        B, *z_shape, C = q.shape
        z_tot = np.prod(z_shape)
        q = jnp.reshape(q, (B, z_tot, C))
        k = jnp.reshape(k, (B, z_tot, C))
        w_ = jnp.einsum('bqd,bkd->bqk', q, k)
        w_ = w_ * (int(C) ** (-0.5))
        w_ = jax.nn.softmax(w_, axis=2)

        v = jnp.reshape(v, (B, z_tot, C))
        h_ = jnp.einsum('bqk,bkd->bqd', w_, v)
        h_ = jnp.reshape(h_, (B, *z_shape, C))

        h_ = nn.Conv(channels, [1] * ndims)(h_)
        
        return x + h_
