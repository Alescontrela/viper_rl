import embodied
import numpy as np
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj

from .. import CONFIG_PATH


@jaxagent.Wrapper
class Agent(nj.Module):
    configs = yaml.YAML(typ="safe").load(
        open(CONFIG_PATH / "dreamerv3/configs.yaml", "rb").read()
    )

    def __init__(self, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.wm = WorldModel(obs_space, act_space, config, name="wm")
        self.task_behavior = getattr(behaviors, config.task_behavior)(
            self.wm, self.act_space, self.config, name="task_behavior"
        )
        if config.expl_behavior == "None":
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(
                self.wm, self.act_space, self.config, name="expl_behavior"
            )

    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size),
        )

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)

    def policy(self, obs, state, mode="train"):
        self.config.jax.jit and print("Tracing policy function.")
        obs = self.preprocess(obs)
        (prev_latent, prev_action), task_state, expl_state = state
        embed = self.wm.encoder(obs)
        latent, _ = self.wm.rssm.obs_step(
            prev_latent, prev_action, embed, obs["is_first"]
        )
        self.expl_behavior.policy(latent, expl_state)
        task_outs, task_state = self.task_behavior.policy(latent, task_state)
        expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
        if mode == "eval":
            outs = task_outs
            outs["action"] = outs["action"].sample(seed=nj.rng())
            outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])
        elif mode == "explore":
            outs = expl_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
        elif mode == "train":
            outs = task_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
        state = ((latent, outs["action"]), task_state, expl_state)
        return outs, state

    def train(self, data, state, reference_data=None):
        self.config.jax.jit and print("Tracing train function.")
        metrics = {}
        data = self.preprocess(data)
        if reference_data is not None:
            reference_data = self.preprocess(reference_data)
        state, wm_outs, mets = self.wm.train(data, state, reference_data)
        metrics.update(mets)
        context = {**data, **wm_outs["post"]}
        start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
        _, mets = self.task_behavior.train(self.wm.imagine, start, context)
        metrics.update(mets)
        if self.config.expl_behavior != "None":
            _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        outs = {}
        return outs, state, metrics

    def report(self, data, reference_data=None):
        self.config.jax.jit and print("Tracing report function.")
        data = self.preprocess(data)
        if reference_data is not None:
            reference_data = self.preprocess(reference_data)
        report = {}
        report.update(self.wm.report(data, reference_data))
        mets = self.task_behavior.report(data)
        report.update({f"task_{k}": v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f"expl_{k}": v for k, v in mets.items()})
        return report

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_") or key in ("key",):
                continue
            if key.startswith("codes_") or key.startswith("embed_"):
                continue
            if len(value.shape) > 3 and value.dtype == jnp.uint8:
                value = jaxutils.cast_to_compute(value) / 255.0
            else:
                value = value.astype(jnp.float32)
            obs[key] = value
        obs["cont"] = 1.0 - obs["is_terminal"].astype(jnp.float32)
        return obs


class WorldModel(nj.Module):
    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith("log_")}
        self.encoder = nets.MultiEncoder(shapes, **config.encoder, name="enc")
        self.rssm = nets.RSSM(**config.rssm, name="rssm")

        self.heads = {
            "decoder": nets.MultiDecoder(shapes, **config.decoder, name="dec"),
            "reward": nets.MLP((), **config.reward_head, name="rew"),
            "cont": nets.MLP((), **config.cont_head, name="cont"),
        }

        if config.task_behavior == "Prior" or config.expl_behavior == "Prior":
            self.heads["density"] = nets.MLP((), **config.density_head, name="density")

        self.amp = (config.task_behavior == "MotionPrior") or (
            config.expl_behavior == "MotionPrior"
        )
        if self.amp:
            self.discriminator = nets.Discriminator(
                shapes, **config.discriminator, name="discriminator"
            )
            self.heads["discriminator_reward"] = nets.MLP(
                (), **config.discriminator_head, name="discriminator_head"
            )

        self.opt = jaxutils.Optimizer(name="model_opt", **config.model_opt)
        scales = self.config.loss_scales.copy()
        image, vector = scales.pop("image"), scales.pop("vector")
        scales.update({k: image for k in self.heads["decoder"].cnn_shapes})
        scales.update({k: vector for k in self.heads["decoder"].mlp_shapes})
        self.scales = scales

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action

    def train(self, data, state, reference_data=None):
        modules = [self.encoder, self.rssm, *self.heads.values()]
        if self.amp:
            modules.append(self.discriminator)
        mets, (state, outs, metrics) = self.opt(
            modules, self.loss, data, state, reference_data, has_aux=True
        )
        metrics.update(mets)
        return state, outs, metrics

    def loss(self, data, state, reference_data=None):
        embed = self.encoder(data)
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate(
            [prev_action[:, None], data["action"][:, :-1]], 1
        )
        post, prior = self.rssm.observe(
            embed, prev_actions, data["is_first"], prev_latent
        )
        dists = {}
        feats = {**post, "embed": embed}
        for name, head in self.heads.items():
            if name == "discriminator_reward":
                continue
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        losses = {}
        losses["dyn"] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses["rep"] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)

        additional_metrics = {}

        if self.amp:
            idxs = list(
                range(reference_data["is_first"].shape[1] - self.config.amp_window)
            )
            amp_data_stacks = {}
            amp_reference_data_stacks = {}
            for k, v in data.items():
                if k in self.discriminator.shapes:
                    amp_data_stacks[k] = jnp.stack(
                        [
                            v[:, idx : (idx + self.config.amp_window)].astype(
                                jnp.float32
                            )
                            for idx in idxs
                        ],
                        axis=1,
                    )
            for k, v in reference_data.items():
                if k in self.discriminator.shapes:
                    amp_reference_data_stacks[k] = jnp.stack(
                        [
                            v[:, idx : (idx + self.config.amp_window)].astype(
                                jnp.float32
                            )
                            for idx in idxs
                        ],
                        axis=1,
                    )
            amp_batch_dims = amp_data_stacks[list(amp_data_stacks.keys())[0]].shape[:2]
            amp_data_stacks = jax.tree_util.tree_map(
                lambda x: x.reshape((-1,) + x.shape[len(amp_batch_dims) :]),
                amp_data_stacks,
            )
            amp_reference_data_stacks = jax.tree_util.tree_map(
                lambda x: x.reshape((-1,) + x.shape[len(amp_batch_dims) :]),
                amp_reference_data_stacks,
            )
            policy_d = self.discriminator(sg(amp_data_stacks))
            reference_d, reference_d_grad = jax.vmap(
                jax.value_and_grad(self.discriminator, has_aux=False)
            )(sg(amp_reference_data_stacks))
            loss = lambda x, y: (x - y) ** 2
            reference_labels, policy_labels = jnp.ones_like(
                reference_d
            ), -1 * jnp.ones_like(policy_d)
            loss_reference = loss(reference_labels, reference_d)
            loss_policy = loss(policy_labels, policy_d)
            loss_disc = 0.5 * (loss_reference + loss_policy)

            # Gradient penalty.
            reference_d_grad = jax.tree_util.tree_map(
                lambda x: jnp.square(x), reference_d_grad
            )
            reference_d_grad = jax.tree_util.tree_map(
                lambda x: jnp.sum(x, axis=np.arange(1, len(x.shape))), reference_d_grad
            )
            reference_d_grad = jax.tree_util.tree_reduce(
                lambda x, y: x + y, reference_d_grad
            )
            loss_gp = 2.5 * reference_d_grad / self.config.amp_window
            losses["mp_amp"] = loss_disc
            losses["mp_amp_gp"] = loss_gp

            # Update discriminator head.
            policy_d = jax.tree_util.tree_map(
                lambda x: x.reshape(amp_batch_dims), policy_d
            )
            feats_new = {k: v[:, self.config.amp_window :] for k, v in feats.items()}
            discriminator_rewards = jnp.clip(
                1 - (1.0 / 4.0) * jnp.square(policy_d - 1), -2, 2
            )
            discriminator_reward_dist = self.heads["discriminator_reward"](
                feats_new
                if "discriminator_reward" in self.config.grad_heads
                else sg(feats_new)
            )
            losses["discriminator_reward"] = -discriminator_reward_dist.log_prob(
                discriminator_rewards.astype(jnp.float32)
            )

            additional_metrics.update(
                {
                    "mp_logits_reference_mean": reference_d.mean(),
                    "mp_logits_reference_std": reference_d.std(),
                    "mp_logits_policy_mean": policy_d.mean(),
                    "mp_logits_policy_std": policy_d.std(),
                    "mp_loss_reference_mean": loss_reference.mean(),
                    "mp_loss_reference_std": loss_reference.std(),
                    "mp_loss_policy_mean": loss_policy.mean(),
                    "mp_loss_policy_std": loss_policy.std(),
                }
            )

        for key, dist in dists.items():
            loss = -dist.log_prob(data[key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            losses[key] = loss

        scaled = {k: v.mean() * self.scales[k] for k, v in losses.items()}
        model_loss = sum(scaled.values())
        out = {"embed": embed, "post": post, "prior": prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data["action"][:, -1]
        state = last_latent, last_action
        metrics = self._metrics(data, dists, post, prior, losses, model_loss)
        metrics.update(additional_metrics)
        return model_loss, (state, out, metrics)

    def imagine(self, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start["action"] = policy(start)

        def step(prev, _):
            prev = prev.copy()
            state = self.rssm.img_step(prev, prev.pop("action"))
            return {**state, "action": policy(state)}

        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def report(self, data, reference_data=None):
        state = self.initial(len(data["is_first"]))
        report = {}
        report.update(self.loss(data, state, reference_data)[-1][-1])
        context, _ = self.rssm.observe(
            self.encoder(data)[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        start = {k: v[:, -1] for k, v in context.items()}
        recon = self.heads["decoder"](context)
        openl = self.heads["decoder"](self.rssm.imagine(data["action"][:6, 5:], start))

        for key in self.heads["decoder"].cnn_shapes.keys():
            truth = data[key][:6].astype(jnp.float32)
            model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
            error = (model - truth + 1) / 2
            video = jnp.concatenate([truth, model, error], 2)
            report[f"openl_{key}"] = jaxutils.video_grid(video)

        return report

    def _metrics(self, data, dists, post, prior, losses, model_loss):
        entropy = lambda feat: self.rssm.get_dist(feat).entropy()
        metrics = {}
        metrics.update(jaxutils.tensorstats(entropy(prior), "prior_ent"))
        metrics.update(jaxutils.tensorstats(entropy(post), "post_ent"))
        metrics.update({f"{k}_loss_mean": v.mean() for k, v in losses.items()})
        metrics.update({f"{k}_loss_std": v.std() for k, v in losses.items()})
        metrics["model_loss_mean"] = model_loss.mean()
        metrics["model_loss_std"] = model_loss.std()
        metrics["reward_max_data"] = jnp.abs(data["reward"]).max()
        metrics["reward_max_pred"] = jnp.abs(dists["reward"].mean()).max()
        if "reward" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["reward"], data["reward"], 0.1)
            metrics.update({f"reward_{k}": v for k, v in stats.items()})
        if "cont" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["cont"], data["cont"], 0.5)
            metrics.update({f"cont_{k}": v for k, v in stats.items()})
        return metrics


class ImagActorCritic(nj.Module):
    def __init__(self, critics, scales, act_space, config):
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.config = config
        disc = act_space.discrete
        self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
        self.actor = nets.MLP(
            name="actor",
            dims="deter",
            shape=act_space.shape,
            **config.actor,
            dist=config.actor_dist_disc if disc else config.actor_dist_cont,
        )
        self.retnorms = {
            k: jaxutils.Moments(**config.retnorm, name=f"retnorm_{k}") for k in critics
        }
        self.opt = jaxutils.Optimizer(name="actor_opt", **config.actor_opt)

    def initial(self, batch_size):
        return {}

    def policy(self, state, carry):
        return {"action": self.actor(state)}, carry

    def train(self, imagine, start, context):
        def loss(start):
            policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
            traj = imagine(policy, start, self.config.imag_horizon)
            loss, metrics = self.loss(traj)
            return loss, (traj, metrics)

        mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
        metrics.update(mets)
        for key, critic in self.critics.items():
            mets = critic.train(traj, self.actor)
            metrics.update({f"{key}_critic_{k}": v for k, v in mets.items()})
        return traj, metrics

    def loss(self, traj):
        metrics = {}
        advs = []
        total = sum(self.scales[k] for k in self.critics)
        for key, critic in self.critics.items():
            rew, ret, base = critic.score(traj, self.actor)
            offset, invscale = self.retnorms[key](ret)
            normed_ret = (ret - offset) / invscale
            normed_base = (base - offset) / invscale
            advs.append((normed_ret - normed_base) * self.scales[key] / total)
            metrics.update(jaxutils.tensorstats(rew, f"{key}_reward"))
            metrics.update(jaxutils.tensorstats(ret, f"{key}_return_raw"))
            metrics.update(jaxutils.tensorstats(normed_ret, f"{key}_return_normed"))
            metrics[f"{key}_return_rate"] = (jnp.abs(ret) >= 0.5).mean()
        adv = jnp.stack(advs).sum(0)
        policy = self.actor(sg(traj))
        logpi = policy.log_prob(sg(traj["action"]))[:-1]
        loss = {"backprop": -adv, "reinforce": -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent
        loss *= sg(traj["weight"])[:-1]
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))
        return loss.mean(), metrics

    def _metrics(self, traj, policy, logpi, ent, adv):
        metrics = {}
        ent = policy.entropy()[:-1]
        rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        rand = rand.mean(range(2, len(rand.shape)))
        act = traj["action"]
        act = jnp.argmax(act, -1) if self.act_space.discrete else act
        metrics.update(jaxutils.tensorstats(act, "action"))
        metrics.update(jaxutils.tensorstats(rand, "policy_randomness"))
        metrics.update(jaxutils.tensorstats(ent, "policy_entropy"))
        metrics.update(jaxutils.tensorstats(logpi, "policy_logprob"))
        metrics.update(jaxutils.tensorstats(adv, "adv"))
        metrics["imag_weight_dist"] = jaxutils.subsample(traj["weight"])
        return metrics


class VFunction(nj.Module):
    def __init__(self, rewfn, config):
        self.rewfn = rewfn
        self.config = config
        self.net = nets.MLP((), name="net", dims="deter", **self.config.critic)
        self.slow = nets.MLP((), name="slow", dims="deter", **self.config.critic)
        self.updater = jaxutils.SlowUpdater(
            self.net,
            self.slow,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update,
        )
        self.opt = jaxutils.Optimizer(name="critic_opt", **self.config.critic_opt)

    def train(self, traj, actor):
        target = sg(self.score(traj)[1])
        mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
        metrics.update(mets)
        self.updater()
        return metrics

    def loss(self, traj, target):
        metrics = {}
        traj = {k: v[:-1] for k, v in traj.items()}
        dist = self.net(traj)
        loss = -dist.log_prob(sg(target))
        if self.config.critic_slowreg == "logprob":
            reg = -dist.log_prob(sg(self.slow(traj).mean()))
        elif self.config.critic_slowreg == "xent":
            reg = -jnp.einsum(
                "...i,...i->...", sg(self.slow(traj).probs), jnp.log(dist.probs)
            )
        else:
            raise NotImplementedError(self.config.critic_slowreg)
        loss += self.config.loss_scales.slowreg * reg
        loss = (loss * sg(traj["weight"])).mean()
        loss *= self.config.loss_scales.critic
        metrics = jaxutils.tensorstats(dist.mean())
        return loss, metrics

    def score(self, traj, actor=None):
        rew = self.rewfn(traj)
        assert (
            len(rew) == len(traj["action"]) - 1
        ), "should provide rewards for all but last action"
        discount = 1 - 1 / self.config.horizon
        disc = traj["cont"][1:] * discount
        value = self.net(traj).mean()
        vals = [value[-1]]
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = jnp.stack(list(reversed(vals))[:-1])
        return rew, ret, value[:-1]
