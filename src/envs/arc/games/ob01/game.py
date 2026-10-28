"""The community ARC-AGI-3 game ob01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Odd one out: three one-cell bodies on a 16x16 grid, each with its own pad; actions 1-4 move the active body (blocked by the
edge, walls and the other bodies), action 5 makes the next body active, and the level passes when every body stands on its
pad, 5 levels. There is no way to lose. A pad covers a wall on its cell, and the first sprite there is the pad, so a body
can step onto it. Draw order is the level's sprite order: bodies, pads, walls. Three dots in the frame's bottom-left
show which body is active. The arrays come from extract.py; a frame here is the last frame the original renders for an
action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("ob01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)])  # the original's actions 1-5 as (dx, dy); 5 switches bodies
BODIES, GOAL, WALL = DATA["colors"][:3], int(DATA["colors"][3]), int(DATA["colors"][4])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
UI_ROW, UI_COLS, UI_ACTIVE, UI_IDLE = SIZE - 2, np.array([2, 4, 6]), 11, 3  # the active-body dots


@struct.dataclass
class State:
    level: jax.Array
    bodies: jax.Array  # (3, 2) x, y
    active: jax.Array


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Ob01(ArcGame):
    num_actions = 5
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("starts", level), jnp.int32(0))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        goals = level_data("goals", state.level)
        body = state.bodies[state.active]
        target = body + jnp.asarray(MOVES)[action]
        inside = ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        wall = level_data("walls", state.level)[jnp.clip(target[1], 0, GRID - 1), jnp.clip(target[0], 0, GRID - 1)]
        # get_sprite_at returns the first sprite in list order: bodies, then pads, then walls
        moved = (action < 4) & inside & ~(state.bodies == target).all(-1).any() & ~(wall & ~(goals == target).all(-1).any())
        bodies = state.bodies.at[state.active].set(jnp.where(moved, target, body))
        advanced = moved & (bodies == goals).all()
        after = State(state.level, bodies, jnp.where(action == 4, (state.active + 1) % 3, state.active))
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.bodies[:, 1], state.bodies[:, 0]].set(jnp.asarray(BODIES))
        goals = level_data("goals", state.level)
        grid = grid.at[goals[:, 1], goals[:, 0]].set(GOAL)
        grid = jnp.where(level_data("walls", state.level), WALL, grid)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        return frame.at[UI_ROW, UI_COLS].set(jnp.where(jnp.arange(3) == state.active, UI_ACTIVE, UI_IDLE))
