"""ARC-AGI-3 games (JAX ports) behind make()'s interface: a game subclasses ArcGame with init_level, act and render."""

from typing import Any

import jax
from flax import struct
from gymnax.environments import spaces
from jax import numpy as jnp

SIZE, NUM_COLORS = 64, 16
MAX_STEPS = 1000  # an episode ends here if the game has not; the games themselves never do (lives are the only clock)


@struct.dataclass
class EnvState:
    game: Any
    time: jax.Array
    levels: jax.Array  # completed this episode


class ArcGame:
    """An episode is one playthrough: it ends when the game is won or lost, and the next one starts at level 0.
    shortcut: the toolkit's reset after a game over restarts only the level; upgrade if ARC-AGI-3's own scoring is needed."""

    num_actions = 4
    num_levels: int
    default_params = None

    def init_level(self, level: jax.Array):
        raise NotImplementedError

    def act(self, state, action: jax.Array) -> tuple[Any, jax.Array, jax.Array, jax.Array]:
        """The state after the action, and whether it finished the level, lost the game or won it."""
        raise NotImplementedError

    def render(self, state) -> jax.Array:
        """The (SIZE, SIZE) frame of color indices."""
        raise NotImplementedError

    def observation_space(self, params=None):
        return spaces.Box(0, NUM_COLORS - 1, (SIZE, SIZE, 1), jnp.uint8)

    def action_space(self, params=None):
        return spaces.Discrete(self.num_actions)

    def observe(self, state: EnvState) -> jax.Array:
        return self.render(state.game).astype(jnp.uint8)[..., None]

    def reset(self, key, params=None):
        state = EnvState(self.init_level(jnp.int32(0)), jnp.int32(0), jnp.int32(0))
        return self.observe(state), state

    def step(self, key, state: EnvState, action, params=None):
        game, advanced, lost, won = self.act(state.game, action)
        state = EnvState(game, state.time + 1, state.levels + advanced)
        obs = self.observe(state)
        terminated = lost | won
        truncated = (state.time >= MAX_STEPS) & ~terminated
        done = terminated | truncated
        info = {"terminated": terminated, "truncated": truncated, "final_observation": obs, "score": state.levels * done}
        reset_obs, reset_state = self.reset(key)
        obs, state = jax.tree.map(lambda r, x: jnp.where(done, r, x), (reset_obs, reset_state), (obs, state))
        return obs, state, advanced.astype(jnp.float32), done, info
