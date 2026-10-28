"""The community ARC-AGI-3 game ul01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
The player (one cell) walks an 8x8 grid; stepping on the key picks it up, the door blocks until then and opening it (by
stepping into it with the key) passes the level, 5 levels. There is no way to lose. The camera scales the grid up to the
64x64 frame; a 4x4 square in the bottom-right corner lights up while the key is held. The arrays come from extract.py; a
frame here is the last frame the original renders for an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("ul01.npz")))
NUM_LEVELS = len(DATA["start"])
GRID = int(np.sqrt(DATA["cells"].max() + 1))  # the side of the padded grid the cell indices count in
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, KEY, DOOR = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
UI_SIZE, UI_HELD, UI_EMPTY = 4, 11, 5  # the key indicator in the frame's bottom-right corner


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    has_key: jax.Array
    opened: jax.Array


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Ul01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level), jnp.bool_(False), jnp.bool_(False))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        target = state.player + jnp.asarray(MOVES)[action]
        inside = ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        at_key = inside & ~state.has_key & (target == level_data("key", state.level)).all()
        at_door = inside & ~state.opened & (target == level_data("door", state.level)).all()
        blocked = ~inside | (at_door & ~state.has_key)
        advanced = at_door & state.has_key
        after = State(state.level, jnp.where(blocked, state.player, target), state.has_key | at_key, state.opened | advanced)
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32)
        # list order is draw order: player, key, door (they never share a cell)
        grid = grid.at[state.player[1], state.player[0]].set(PLAYER)
        key, door = level_data("key", state.level), level_data("door", state.level)
        grid = grid.at[key[1], key[0]].set(jnp.where(state.has_key, grid[key[1], key[0]], KEY))
        grid = grid.at[door[1], door[0]].set(jnp.where(state.opened, grid[door[1], door[0]], DOOR))
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        return frame.at[SIZE - UI_SIZE :, SIZE - UI_SIZE :].set(jnp.where(state.has_key, UI_HELD, UI_EMPTY))
