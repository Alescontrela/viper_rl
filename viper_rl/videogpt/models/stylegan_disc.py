from typing import Tuple, Union, Callable, List
import functools as ft
from math import log2, sqrt
import numpy as np
import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp


ActivationFunction = Callable[[jnp.ndarray], jnp.ndarray]

 
def _apply_filter_2d(
    x: jnp.ndarray,
    filter_kernel: jnp.ndarray,
    padding: Tuple[int, int] = (0, 0),
) -> jnp.ndarray:
    """
    >>> x = jnp.zeros((1, 64, 64, 2))
    >>> kernel = jnp.zeros((4, 4))
    >>> _apply_filter_2d(x, kernel, padding=(2, 2)).shape
    (1, 65, 65, 2)
    """
    dimension_numbers = ("NHWC", "HWOI", "NHWC")
    filter_kernel = filter_kernel[:, :, None, None]
    x = x[..., None]
    vmap_over_axis = 3

    conv_func = ft.partial(
        jax.lax.conv_general_dilated,
        rhs=filter_kernel,
        window_strides=(1, 1),
        padding=[padding, padding],
        dimension_numbers=dimension_numbers,
    )
    y = jax.vmap(conv_func, in_axes=vmap_over_axis, out_axes=vmap_over_axis)(x)
    return jnp.squeeze(y, axis=vmap_over_axis + 1)


class ConvDownsample2D(nn.Module):
    """This is the `_simple_upfirdn_2d` part of
    https://github.com/NVlabs/stylegan2-ada/blob/main/dnnlib/tflib/ops/upfirdn_2d.py#L313

    >>> module = _init(
    ...     ConvDownsample2D,
    ...     output_channels=8,
    ...     kernel_shape=3,
    ...     resample_kernel=jnp.array([1, 3, 3, 1]),
    ...     downsample_factor=2)
    >>> x = jax.numpy.zeros((1, 64, 64, 4))
    >>> params = module.init(jax.random.PRNGKey(0), x)
    >>> y = module.apply(params, None, x)
    >>> tuple(y.shape)
    (1, 32, 32, 8)
    """

    output_channels: int
    kernel_shape: Union[int, Tuple[int, int]]
    resample_kernel: jnp.array
    downsample_factor: int = 1
    gain: float = 1.0
    dtype: jnp.dtype = jnp.float32

    def setup(
        self,
    ):
        if self.resample_kernel.ndim == 1:
            resample_kernel = self.resample_kernel[:, None] * self.resample_kernel[None, :]
        elif 0 <= self.resample_kernel.ndim > 2:
            raise ValueError(f"Resample kernel has invalid shape {self.resample_kernel.shape}")

        self.conv = nn.Conv(
            self.output_channels,
            kernel_size=(self.kernel_shape, self.kernel_shape),
            strides=(self.downsample_factor, self.downsample_factor),
            padding="VALID",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(),
        )
        self.resample_kernel_ = (jnp.array(resample_kernel) * self.gain / resample_kernel.sum()).astype(jnp.float32)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # pylint: disable=invalid-name
        kh, kw = self.resample_kernel_.shape
        ch, cw = self.kernel_shape, self.kernel_shape
        assert kh == kw
        assert ch == cw

        # See https://github.com/NVlabs/stylegan2-ada/blob/main/dnnlib/tflib/ops/upfirdn_2d.py#L362
        pad_0 = (kw - self.downsample_factor + cw) // 2
        pad_1 = (kw - self.downsample_factor + cw - 1) // 2
        y = _apply_filter_2d(
            x,
            self.resample_kernel_,
            padding=(pad_0, pad_1),
        )
        return self.conv(y)


def minibatch_stddev_layer(
    x: jnp.ndarray,
    group_size: int = None,
    num_new_features: int = 1,
    is_initializing: bool = False
) -> jnp.ndarray:
    """Minibatch standard deviation layer. Adds the standard deviation of
    subsets of size `group_size` taken over the batch dimension as features
    to x.

    Args:
        x ([type]): [description]
        group_size (int, optional): [description]. Defaults to None.
        num_new_features (int, optional): [description]. Defaults to 1.
        data_format (str, optional): [description]. Defaults to "channels_last".

    Returns:
        [type]: [description]

    >>> x = jnp.zeros((4, 23, 26, 3))
    >>> y = minibatch_stddev_layer(x, group_size=2, data_format=ChannelOrder.channels_last)
    >>> y.shape
    (4, 23, 26, 4)
    >>> x = jnp.zeros((4, 8, 23, 26))
    >>> y = minibatch_stddev_layer(x, num_new_features=4, data_format=ChannelOrder.channels_first)
    >>> y.shape
    (4, 12, 23, 26)

    FIXME Rewrite using allreduce ops like psum to allow non-batched definition
          of networks
    """
    # pylint: disable=invalid-name
    N, H, W, C = x.shape

    if is_initializing or group_size <= N: 
        group_size = min(group_size, N) if group_size is not None else N
        C_ = C // num_new_features

        y = jnp.reshape(x, (-1, group_size, H, W, num_new_features, C_))

        y_centered = y - jnp.mean(y, axis=1, keepdims=True)
        y_std = jnp.sqrt(jnp.mean(y_centered * y_centered, axis=1) + 1e-8)

        y_std = jnp.mean(y_std, axis=(1, 2, 4))
        y_std = y_std.reshape((-1, 1, 1, num_new_features))
        y_std = jnp.tile(y_std, (group_size, H, W, 1))
    else:
        assert group_size % N == 0, f"{group_size} % {N} != 0"
        index_group_size = group_size // N
        assert jax.device_count() % index_group_size == 0, f"{jax.device_count()} % {index_group_size} != 0"
        n_index_groups = jax.device_count() // index_group_size
        index_groups = [
            list(range(i * index_group_size, (i + 1) * index_group_size))
            for i in range(n_index_groups)
        ]

        C_ = C // num_new_features
        y = jnp.reshape(x, (-1, N, H, W, num_new_features, C_))
        
        y_mean = jnp.mean(y, axis=1, keepdims=True)
        y_mean = jax.lax.pmean(y_mean, 'device', axis_index_groups=index_groups)
        y_centered = y - y_mean

        y_std = jnp.mean(y_centered * y_centered, axis=1)
        y_std = jax.lax.pmean(y_std, 'device', axis_index_groups=index_groups)
        y_std = jnp.sqrt(y_std + 1e-8)
        y_std = jnp.mean(y_std, axis=(1, 2, 4))
        y_std = y_std.reshape((-1, 1, 1, num_new_features))
        y_std = jnp.tile(y_std, (N, H, W, 1))

    return jnp.concatenate((x, y_std), axis=3)


class DiscriminatorBlock(nn.Module):
    in_features: int
    out_features: int
    activation_function: ActivationFunction = jnn.leaky_relu
    resample_kernel: jnp.ndarray = jnp.array([1, 3, 3, 1])
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv_in = nn.Conv(
            self.in_features,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(),
        )
        self.downsample1 = ConvDownsample2D(
            self.out_features,
            kernel_shape=3,
            resample_kernel=self.resample_kernel,
            downsample_factor=2,
            dtype=self.dtype,
        )
        self.downsample2 = ConvDownsample2D(
            self.out_features,
            kernel_shape=1,
            resample_kernel=self.resample_kernel,
            downsample_factor=2,
            dtype=self.dtype,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        y = self.conv_in(x)
        y = self.activation_function(y)
        y = self.downsample1(y)
        y = self.activation_function(y)

        residual = self.downsample2(x)
        return (y + residual) / sqrt(2)


def _get_num_features(base_features: int, image_size: Tuple[int, int], max_hidden_feature_size: int) -> List[int]:
    """
    Gets number of features for the blocks. Each block includes a downsampling
    step by a factor of two and at the end, we want the resolution to be
    down to 4x4 (for square images)

    >>> features = _get_num_features(64, (512, 512), 1024)
    >>> 512 // 2**(len(features) - 1)
    4
    >>> features[3]
    512
    """
    for size in image_size:
        assert 2 ** int(log2(size)) == size, f"Image size must be a power of 2, got {image_size}"
    # determine the number of layers based on smaller side length
    shortest_side = min(*image_size)
    num_blocks = int(log2(shortest_side)) - 1
    num_features = (base_features * (2**i) for i in range(num_blocks))
    # we want to bring it down to 4x4 at the end of the last block
    return [min(n, max_hidden_feature_size) for n in num_features]


class StyleGANDisc(nn.Module):
    base_features: int
    max_hidden_feature_size: int
    mbstd_group_size: int
    mbstd_num_features: int
    gradient_penalty_weight: float
    dtype: jnp.dtype = jnp.float32

    @property
    def metrics(self):
        return ['g_loss', 'd_loss', 'd_grad_penalty']

    def compute_disc_logits(
        self, disc_params, disc_model_state, image
    ):
        return self.apply(
            {'params': disc_params, **disc_model_state}, image
        )

    def loss_G(self, disc_params, disc_model_state, fake):
        logits_fake = self.compute_disc_logits(
            disc_params, disc_model_state, fake
        )
        return {'g_loss': jnp.mean(nn.softplus(-logits_fake))}
    
    def loss_D(self, disc_params, disc_model_state, real, fake):
        logits_real = self.compute_disc_logits(
            disc_params, disc_model_state, real
        )
        logits_fake = self.compute_disc_logits(
            disc_params, disc_model_state, fake
        )
        d_loss = jnp.mean(
            nn.softplus(logits_fake) + nn.softplus(-logits_real)
        )

        # gradient penalty r1: https://github.com/NVlabs/stylegan2/blob/bf0fe0baba9fc7039eae0cac575c1778be1ce3e3/training/loss.py#L63-L67
        r1_grads = jax.grad(
            lambda x: jnp.mean(self.compute_disc_logits(
                disc_params, disc_model_state, x
            ))
        )(real)
        r1_grads = self.gradient_penalty_weight * jnp.mean(r1_grads ** 2)

        loss_D = d_loss + r1_grads
        return {
            'loss_D': loss_D,
            'd_loss': d_loss,
            'd_grad_penalty': r1_grads,
        }, {}

    @nn.compact
    def __call__(self, image: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = image
        size_t = x.shape[1:-1]
        num_features = _get_num_features(
            2 * self.base_features,
            size_t,
            self.max_hidden_feature_size
        )

        y = nn.Conv(self.base_features, [1, 1], padding='SAME')(x)
        y = jnn.leaky_relu(y)
        for n_in, n_out in zip(num_features[1:], num_features[:-1]):
            y = DiscriminatorBlock(
                n_in, n_out,
                activation_function=jnn.leaky_relu
            )(y)

        # final block running on 4x4 feature maps
        assert min(y.shape[1:3]) == 4

        y = minibatch_stddev_layer(
            y, group_size=self.mbstd_group_size,
            num_new_features=self.mbstd_num_features,
            is_initializing=self.is_initializing()
        )
        y = nn.Conv(num_features[-2], [3, 3], padding="VALID")(y)
        y = jnn.leaky_relu(y)
        y = jnp.reshape(y, (y.shape[0], -1))
        y = nn.Dense(num_features[-1])(y)
        y = jnn.leaky_relu(y)

        # Prediction head
        y = nn.Dense(1)(y)
        return y
