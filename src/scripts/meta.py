"""Trains ppo.py's recurrent agent with a meta-learned update rule (a disco.py checkpoint) instead of PPO, to compare the two on one environment:
    python src/scripts/meta.py --meta_rule v1adych-team/meta-rl/meta_baseline:v2 --env_name Craftax-Classic-Pixels-v1 --track

Every update is one rollout and one adam step on the frozen meta-network's targets (LPG eq. 2). Unlike disco.py's agents,
this one has no critic in its loss: the meta-network never saw values, and here nothing else needs them.
"""


import time
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import optax
import tyro
from flax import nnx
from jax import numpy as jnp

from modules.checkpoint import fetch_checkpoint, load_checkpoint, save_checkpoint
from modules.envs import make
from modules.log import Logger
from modules.nn import ActorCriticRNN, MetaNetwork, num_params, take
from scripts.ppo import StepCarry, unroll

SKIP_INFO = ("discount", "terminated", "truncated", "final_observation")  # info entries that aren't per-episode scores


@dataclass(frozen=True)
class Config:
    meta_rule: str  # checkpoint directory or wandb artifact of a disco.py meta-network
    env_name: str = "Craftax-Classic-Pixels-v1"
    seed: int = 0
    total_timesteps: int = 1_000_000_000
    num_envs: int = 2048
    num_steps: int = 64  # rollout length, also the BPTT horizon
    # network, as ppo.py
    embed_dim: int = 32
    hidden_dim: int = 256
    patch_size: tuple[int, int] = (7, 7)
    # the meta-network's sizes, as in the disco.Config that trained it
    y_dim: int = 30
    meta_hidden: int = 256
    # the agent's update, disco.Spec
    lr: float = 1e-3
    kl_cost: float = 0.5
    gamma: float = 0.99
    # logging
    log_interval: int = 100  # in updates
    save_interval: int = 10_000  # in updates, 0 disables checkpoints
    checkpoint_dir: str = "checkpoints"
    track: bool = False
    wandb_project: str = "meta-rl"
    run_name: str | None = None

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // (self.num_envs * self.num_steps)


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    terminated: jax.Array
    done: jax.Array  # terminated or truncated: the next stored obs starts a new episode
    y_next: jax.Array  # the agent's y at the next state


def step(env, env_params, model, carry: StepCarry, rngs: nnx.Rngs) -> tuple[tuple[Transition, dict], StepCarry]:
    rnn_state, logits, _, _ = model(carry.rnn_state, carry.last_obs)
    action = jax.random.categorical(rngs.next_key(), logits)
    obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(rngs.next_key(), action.shape[0]), carry.env_state, action, env_params)
    # y of the true next state where the env reports it; craftax only reports done, so its y_next after a done is that of the next episode's first state
    _, _, y_next, _ = model(rnn_state, info.get("final_observation", obs))
    rnn_state = jnp.where(done[:, None], 0.0, rnn_state)
    ep_return, ep_length = carry.ep_return + reward, carry.ep_length + 1
    stats = {"episodes": done.sum(), "return": (ep_return * done).sum(), "length": (ep_length * done).sum()}
    stats |= {k.lower(): v.sum() for k, v in info.items() if k not in SKIP_INFO}
    transition = Transition(carry.last_obs, action, reward, info.get("terminated", done), done, jax.nn.softmax(y_next))
    carry = StepCarry(env_state=env_state, last_obs=obs, rnn_state=rnn_state, ep_return=ep_return * (1 - done), ep_length=ep_length * (1 - done))
    return (transition, stats), carry


def update(env, env_params, model, meta_network, optimizer: nnx.Optimizer, carry: StepCarry, cfg: Config, rngs: nnx.Rngs) -> tuple[StepCarry, dict, dict]:
    """One rollout, then one adam step towards the meta-network's targets."""
    init_rnn_state = carry.rnn_state
    rollout = nnx.scan(partial(step, env, env_params), in_axes=(None, nnx.Carry, 0), out_axes=(0, nnx.Carry), length=cfg.num_steps)
    (traj, stats), carry = rollout(model, carry, rngs.split(cfg.num_steps))

    # targets: computed outside the loss, so they are constants to the agent's gradient
    logits, y_logits, _ = unroll(model, init_rnn_state, traj.obs, traj.done)
    gamma = jnp.full(traj.reward.shape, cfg.gamma)
    pi_hat, y_hat = meta_network(traj.reward, traj.terminated, traj.done, gamma, jnp.exp(take(jax.nn.log_softmax(logits), traj.action)), jax.nn.softmax(y_logits), traj.y_next)

    def loss_fn(model):
        logits, y_logits, _ = unroll(model, init_rnn_state, traj.obs, traj.done)
        log_probs, log_y = jax.nn.log_softmax(logits), jax.nn.log_softmax(y_logits)
        kl = (jnp.exp(log_y) * (log_y - jax.nn.log_softmax(y_hat))).sum(-1).mean()
        loss = -(take(jax.nn.log_softmax(logits), traj.action) * pi_hat).mean() + cfg.kl_cost * kl
        entropy = -(jnp.exp(log_probs) * log_probs).sum(-1).mean()
        return loss, {"loss": loss, "kl": kl, "entropy": entropy, "pi_hat_rms": jnp.sqrt((pi_hat**2).mean()), "y_hat_rms": jnp.sqrt((y_hat**2).mean())}

    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return carry, metrics | {"grad_norm": optax.tree.norm(grads)}, jax.tree.map(jnp.sum, stats)


def train(cfg: Config):
    run_name = cfg.run_name or f"meta_{cfg.env_name}_s{cfg.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    logger = Logger(cfg, cfg.wandb_project, run_name, cfg.track)  # first, so that wandb records the console of a crash during setup
    env, env_params = make(cfg.env_name)

    rngs = nnx.Rngs(cfg.seed)
    model = ActorCriticRNN(env.observation_space(env_params).shape, env.num_actions, cfg.embed_dim, cfg.hidden_dim, cfg.patch_size, rngs, y_dim=cfg.y_dim)
    optimizer = nnx.Optimizer(model, optax.adam(cfg.lr), wrt=nnx.Param)
    meta_network = MetaNetwork(cfg.y_dim, cfg.meta_hidden, nnx.Rngs(0))
    path, _ = fetch_checkpoint(cfg.meta_rule)
    load_checkpoint(path, meta_network)

    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(rngs(), cfg.num_envs), env_params)
    carry = StepCarry(env_state=env_state, last_obs=obs, rnn_state=model.initialize_carry(cfg.num_envs), ep_return=jnp.zeros(cfg.num_envs), ep_length=jnp.zeros(cfg.num_envs))

    # the meta-network is frozen: it rides along in the state but no optimizer touches it
    graphdef, state = nnx.split((model, optimizer, rngs, meta_network))

    @partial(jax.jit, donate_argnums=(0, 1))
    def update_fn(state, carry):
        model, optimizer, rngs, meta_network = nnx.merge(graphdef, state)
        carry, metrics, stats = update(env, env_params, model, meta_network, optimizer, carry, cfg, rngs)
        return nnx.state((model, optimizer, rngs, meta_network)), carry, metrics, stats

    add = jax.jit(lambda a, b: jax.tree.map(jnp.add, a, b))
    steps_per_update = cfg.num_envs * cfg.num_steps
    print(f"{run_name}: {num_params(model):,} params, {cfg.num_updates:,} updates of {steps_per_update:,} steps")

    episode_stats, last_time, last_update = None, time.perf_counter(), 0
    for i in range(1, cfg.num_updates + 1):
        state, carry, metrics, stats = update_fn(state, carry)
        episode_stats = stats if episode_stats is None else add(episode_stats, stats)

        if i % cfg.log_interval == 0 or i == cfg.num_updates:
            metrics, episode_stats = jax.device_get((metrics, episode_stats))
            now = time.perf_counter()
            log = {"sps": (i - last_update) * steps_per_update / max(1e-6, now - last_time), **{f"train/{k}": v for k, v in metrics.items()}}
            n = episode_stats.pop("episodes")
            if n > 0:
                log |= {f"episode/{k}": v / n for k, v in episode_stats.items() if not k.startswith("achievements/")}
                log |= {k: v / n for k, v in episode_stats.items() if k.startswith("achievements/")}
            logger.log(i * steps_per_update, log)
            episode_stats, last_time, last_update = None, now, i

        if cfg.save_interval and (i % cfg.save_interval == 0 or i == cfg.num_updates):
            model, _, _, _ = nnx.merge(graphdef, state)
            path = save_checkpoint(f"{cfg.checkpoint_dir}/{run_name}", i * steps_per_update, model)
            logger.upload_checkpoint(path, i * steps_per_update)

    logger.close()
    model, _, _, _ = nnx.merge(graphdef, state)
    return model


if __name__ == "__main__":
    train(tyro.cli(Config))
