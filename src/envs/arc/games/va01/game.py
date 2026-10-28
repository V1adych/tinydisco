"""The community ARC-AGI-3 game va01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Coverage: the player (one cell) walks a small grid (4x4 to 8x8) around walls, leaving a trail on every cell it enters; the
level passes once every non-wall cell has been visited, 5 levels. There is no way to lose. The camera scales the grid up
to the 64x64 frame, drawing by layer: trail, walls, player; a 4x4 square in the bottom-right corner turns from green to
yellow when no cell is left unvisited. The arrays come from extract.py; a frame here is the last frame the original
renders for an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("va01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, WALL, TRAIL = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
UI_SIZE, UI_DONE, UI_PENDING = 4, 14, 11  # the indicator in the frame's bottom-right corner


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    visited: jax.Array  # (GRID, GRID)


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Va01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        start = level_data("start", level)
        return State(level, start, jnp.zeros((GRID, GRID), bool).at[start[1], start[0]].set(True))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        target = state.player + jnp.asarray(MOVES)[action]
        inside = ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        x, y = jnp.clip(target, 0, GRID - 1)
        moved = inside & ~level_data("walls", state.level)[y, x]
        player = jnp.where(moved, target, state.player)
        visited = state.visited.at[player[1], player[0]].set(True)
        advanced = moved & (visited.sum() == level_data("open", state.level))
        after = State(state.level, player, visited)
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        # layers: trail (0), walls (1), player (2)
        grid = jnp.where(state.visited, TRAIL, BACKGROUND)
        grid = jnp.where(level_data("walls", state.level), WALL, grid)
        grid = grid.at[state.player[1], state.player[0]].set(PLAYER)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        done = state.visited.sum() == level_data("open", state.level)
        return frame.at[SIZE - UI_SIZE :, SIZE - UI_SIZE :].set(jnp.where(done, UI_DONE, UI_PENDING))
