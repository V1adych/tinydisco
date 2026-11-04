"""Train a fresh Craftax agent with a frozen Disco rule and Disco's replay/inner learner.

    python src/scripts/rule_craftax.py --meta_rule v1adych-team/tinydisco/tinydisco21:step-3600 --track

Match d and meta_hidden to the rule checkpoint. Each seed starts a new agent; the rule never updates.
"""

import time
from dataclasses import dataclass
from functools import partial

import jax
import numpy as np
import tyro
from flax import nnx
from jax import numpy as jnp

from modules.checkpoint import fetch_checkpoint, load_checkpoint, save_checkpoint
from modules.envs import make
from modules.log import Logger
from modules.nn import MetaNetwork, num_params
from modules.utils import entropy
from scripts.disco import Config as DiscoConfig
from scripts.disco import Spec, act, init_buffer, init_envs, learn, make_nets, sample, store


@dataclass(frozen=True)
class Config:
    meta_rule: str  # checkpoint directory or wandb artifact of a disco.py meta-network
    env_name: str = "Craftax-Classic-Pixels-v1"
    seed: int = 0
    total_timesteps: int = 20_000_000
    num_envs: int = 32  # one agent, with this many parallel environments
    num_steps: int = 29
    batch: int = 256  # replay trajectories per update
    replay: float = 0.9
    buffer_size: int = 131_072  # transitions, rounded down to whole trajectories
    convs: tuple[tuple[int, int], ...] = ((7, 32),)  # Classic's 63x63 pixels -> 9x9 patches
    hidden: int = 256
    d: int = 64
    meta_hidden: int = 256
    lr: float = 3e-4
    gamma: float = 0.997
    lam: float = 0.95
    q_cost: float = 0.5
    max_abs_update: float = 1.0
    log_interval: int = 100
    save_interval: int = 10_000  # in collection/learning updates; 0 disables checkpoints
    checkpoint_dir: str = "checkpoints"
    track: bool = False
    wandb_project: str = "tinydisco"
    run_name: str | None = None

    def learner_config(self) -> DiscoConfig:
        return DiscoConfig(
            num_envs=self.num_envs, num_steps=self.num_steps, batch=self.batch, replay=self.replay,
            buffer_size=self.buffer_size, d=self.d, gamma=self.gamma, lam=self.lam,
            q_cost=self.q_cost, max_abs_update=self.max_abs_update,
        )


def update(nets, cfg, meta_graphdef, env, meta_params, params, opt_state, buffer, envs, key):
    """Collect and replay with a frozen rule; skip learning until a full batch is available."""
    key_act, key_sample = jax.random.split(key)
    rollout, stats, envs = act(nets, cfg, env, params, envs, key_act)
    buffer = store(nets, buffer, rollout)
    batch, ready = sample(nets, cfg, buffer, key_sample)

    def train(_):
        new_params, new_opt, logits, _, target_entropy, _ = learn(nets, cfg, meta_graphdef, meta_params, params, opt_state, batch)
        return new_params, new_opt, entropy(logits).mean(), target_entropy

    params, opt_state, policy_entropy, target_entropy = jax.lax.cond(
        ready, train, lambda _: (params, opt_state, jnp.float32(0), jnp.float32(0)), None
    )
    metrics = {"ready": ready, "entropy": policy_entropy, "target_entropy": target_entropy, "buffer_trajectories": buffer.size}
    return params, opt_state, buffer, envs, metrics, stats


def train(cfg: Config):
    assert cfg.num_steps > 0 and cfg.num_envs > 0 and cfg.batch > 0
    assert cfg.buffer_size // cfg.num_steps >= max(cfg.batch, cfg.num_envs), "buffer must hold a learning batch and a collection rollout"
    assert 0 <= cfg.replay <= 1 and cfg.log_interval > 0 and cfg.save_interval >= 0 and cfg.total_timesteps >= 0
    learner = cfg.learner_config()
    run_name = cfg.run_name or f"rule_{cfg.env_name}_s{cfg.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    logger = Logger(cfg, cfg.wandb_project, run_name, cfg.track)
    env, _ = make(cfg.env_name)
    nets = make_nets(learner, env, Spec(cfg.convs, cfg.hidden, cfg.lr, (1,)))  # no lifetime resets in evaluation
    meta_network = MetaNetwork(cfg.d, cfg.meta_hidden, nnx.Rngs(0))
    path, _ = fetch_checkpoint(cfg.meta_rule)
    load_checkpoint(path, meta_network)
    meta_graphdef, meta_params = nnx.split(meta_network)
    key, key_agent, key_envs = jax.random.split(jax.random.key(cfg.seed), 3)
    model = nets.new_agent(rngs=nnx.Rngs(key_agent))
    params = nnx.state(model)
    # JIT gives identical zero leaves separate allocations, so donation can reuse them safely.
    opt_state, buffer = jax.jit(lambda p: (nets.agent_tx.init(p), init_buffer(nets, learner)))(params)
    envs = jax.jit(partial(init_envs, nets, env, learner))(key_envs)
    update_fn = jax.jit(partial(update, nets, learner, meta_graphdef, env), donate_argnums=(1, 2, 3, 4))
    add = jax.jit(lambda a, b: jax.tree.map(jnp.add, a, b))
    steps_per_update = cfg.num_envs * cfg.num_steps
    num_updates = cfg.total_timesteps // steps_per_update
    print(f"{run_name}: {num_params(model):,} agent params, {num_updates:,} updates of {steps_per_update:,} steps")

    totals, last_time, last_update = None, time.perf_counter(), 0
    for i in range(1, num_updates + 1):
        params, opt_state, buffer, envs, metrics, stats = update_fn(
            meta_params, params, opt_state, buffer, envs, jax.random.fold_in(key, i)
        )
        totals = stats if totals is None else add(totals, stats)
        if i % cfg.log_interval == 0 or i == num_updates:
            metrics, stats = jax.device_get((metrics, totals))
            now = time.perf_counter()
            log = {"sps": (i - last_update) * steps_per_update / max(1e-6, now - last_time), **{f"train/{k}": v for k, v in metrics.items()}}
            n = stats["episodes"]
            log["episode/count"] = n
            if n > 0:
                log["episode/score"] = stats["score"] / n
                achievements = {k: v / n for k, v in stats.items() if k.startswith("achievements/")}
                log |= achievements
                if achievements:
                    # Craftax score is the geometric mean of achievement success percentages across episodes.
                    log["episode/score"] = np.expm1(np.log1p(list(achievements.values())).mean())
            logger.log(i * steps_per_update, log)
            totals, last_time, last_update = None, now, i
        if cfg.save_interval and (i % cfg.save_interval == 0 or i == num_updates):
            path = save_checkpoint(f"{cfg.checkpoint_dir}/{run_name}", i * steps_per_update, nnx.merge(nets.graphdef, params))
            logger.upload_checkpoint(path, i * steps_per_update)

    logger.close()
    return nnx.merge(nets.graphdef, params)


if __name__ == "__main__":
    train(tyro.cli(Config))
