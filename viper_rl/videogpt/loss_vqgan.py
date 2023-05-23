import jax
import jax.numpy as jnp
from .models import VQGAN, StyleGANDisc
from lpips_jax import LPIPS


class VQPerceptualWithDiscriminator:
    def __init__(self, config):
        self.config = config
        self.vqgan = VQGAN(**config.ae)
        self.disc = StyleGANDisc(**config.disc)
        self.lpips = LPIPS(net_type='vgg16')

    @property
    def metrics(self):
        return ['loss_G', 'loss_D'] + self.vqgan.metrics + \
            self.disc.metrics + ['d_weight']

    def loss_recon(self, vqgan_params, lpips_variables, batch, rng):
        rng, old_rng = jax.random.split(rng)
        
        out = self.vqgan.apply(
            {'params': vqgan_params},
            batch['image'],
            deterministic=False,
            rngs={'dropout': old_rng}
        )

        rec_loss = jnp.abs(batch['image'] - out['recon'])
        if self.config.perceptual_weight > 0:
            p_loss = self.lpips.apply(
                lpips_variables, batch['image'], out['recon'],
            )
        else:
            p_loss = jnp.array([0.], dtype=jnp.float32)
        rec_loss = rec_loss + self.config.perceptual_weight * p_loss
        out['ae_loss'] = jnp.mean(rec_loss)
        return out, rng

    def loss_G(self, vqgan_params, disc_params, disc_model_state, 
               lpips_variables, batch, rng):
        vqgan_out, rng = self.loss_recon(
            vqgan_params, lpips_variables, batch, rng
        )

        g_out = self.disc.loss_G(
            disc_params, disc_model_state, vqgan_out['recon']
        )

        d_weight = jnp.array(self.config.disc_weight, dtype=jnp.float32)

        loss = vqgan_out['ae_loss'] + d_weight * g_out['g_loss'] + \
            self.config.codebook_weight * vqgan_out['vq_loss']

        return loss, ({
            'loss_G': loss,
            'd_weight': d_weight,
            **{k: vqgan_out[k] for k in self.vqgan.metrics},
            **g_out, 
        }, rng)


    def loss_D(self, disc_params, disc_model_state, vqgan_params, batch, rng):
        rng, old_rng = jax.random.split(rng)
        
        out = self.vqgan.apply(
            {'params': vqgan_params},
            batch['image'],
            deterministic=False,
            rngs={'dropout': old_rng}
        )

        d_out, disc_model_state = self.disc.loss_D(
            disc_params, disc_model_state, batch['image'], out['recon']
        )

        return d_out['loss_D'], (d_out, disc_model_state, rng)
