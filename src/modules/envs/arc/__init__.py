"""No-click arcgpu games. Native state lives in the simulator, not in the JAX pytree.

Call prepare() before compiling resets/steps, with the total batch layout. Each
vmap adds to that batch; never rematerialize an environment call or branch/fork
its state. Hard resets draw fresh curricula from the engine's seeded generator.
"""

from pathlib import Path
from typing import NamedTuple

import arcgpu
import jax
from gymnax.environments import spaces
from jax import numpy as jnp

NUM_COLORS = 16
_METADATA = arcgpu.GAMES
ARC_GAMES = {name: meta for name, meta in _METADATA.items() if not meta["has_click"] and 6 not in meta["actions"]}
PROCEDURAL = tuple(name for name, meta in ARC_GAMES.items() if meta.get("procedural"))
OFFICIAL = tuple(name for name in ARC_GAMES if name not in PROCEDURAL)
TRAIN = tuple(Path(__file__).with_name("train.txt").read_text().split())
HOLDOUT = tuple(name for name in PROCEDURAL if name not in TRAIN)


class State(NamedTuple):
    steps: jax.Array
    score: jax.Array
    obs: jax.Array


class Arc:
    def __init__(self, game):
        if game not in ARC_GAMES:
            raise ValueError(f"{game!r} is not a supported no-click ARC game")
        self.game = game
        self.meta = ARC_GAMES[game]
        self.num_actions = len(self.meta["actions"])
        self.pool = None

    def observation_space(self, params=None):
        return spaces.Box(0, NUM_COLORS - 1, (64, 64, 1), jnp.uint8)

    def action_space(self, params=None):
        return spaces.Discrete(self.num_actions)

    def prepare(self, num_envs, agents=1, seed=0):
        if self.pool is not None:
            raise RuntimeError("Create a new Arc object for a new simulator batch")
        self.pool = arcgpu.Pool(self.game, num_envs, agents, seed)

    def reset_where(self, state, mask):
        if self.pool is None:
            raise RuntimeError("Call Arc.prepare(num_envs, agents, seed) before stepping")
        obs = self.pool.reset(mask, state.obs)
        return obs, State(jnp.where(mask, 0, state.steps), jnp.where(mask, 0, state.score), obs)

    def reset(self, key, params=None):
        state = State(jnp.int32(0), jnp.int32(0), jnp.zeros((64, 64, 1), jnp.uint8))
        # Keep the key's vmap axes even though the engine owns the seeded PRNG.
        return self.reset_where(state, jax.random.key_data(key)[0] >= jnp.uint32(0))

    def step(self, key, state, action, params=None):
        # Only declared keyboard actions are exposed; reset is controlled by the wrapper.
        rows = jnp.stack((action, jnp.zeros_like(action), jnp.zeros_like(action), jnp.zeros_like(action)), -1).astype(jnp.uint8)
        frame, _, flags, levels = self.pool.step(rows, state.obs)
        obs = frame.astype(jnp.uint8)[..., None]
        score = jnp.maximum(state.score, levels.astype(jnp.int32))
        reward = (score - state.score).astype(jnp.float32)
        steps = jnp.where(score > state.score, 0, state.steps + 1)
        budgets = jnp.asarray(self.meta["budget_steps"], jnp.int32)
        done = ((flags & 7) != 0) | (steps >= budgets[jnp.minimum(score, len(budgets) - 1)])
        new = State(steps, score, obs)
        next_obs, new = self.reset_where(new, done)
        return next_obs, new, reward, done, {"score": score.astype(jnp.float32), "final_observation": obs}
