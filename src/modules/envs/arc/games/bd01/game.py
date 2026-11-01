"""The community ARC-AGI-3 game bd01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
No revisits: the player (one cell) walks a small grid (4x4 to 8x8) around walls to a target; every cell it leaves turns
red, and stepping onto a cell it has been on before loses the game, 5 levels. The camera scales the grid up to the 64x64
frame and draws in the level's sprite order (player, target, walls, red trail); a 4x4 square in the bottom-right corner
and a 3x3 one in the bottom-left turn red on the losing move. The arrays come from extract.py; a frame here is the last
frame the original renders for an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("bd01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, TARGET, WALL, VISITED = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
FAIL, RIGHT_OK, LEFT_OK = 8, 13, 11  # the corner squares: bottom-right 4x4, bottom-left 3x3 (from the frame's 4th last row)


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    visited: jax.Array  # (GRID, GRID) cells entered, the start included
    fail: jax.Array


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Bd01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        start = level_data("start", level)
        return State(level, start, jnp.zeros((GRID, GRID), bool).at[start[1], start[0]].set(True), jnp.bool_(False))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        target = state.player + jnp.asarray(MOVES)[action]
        inside = ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        x, y = jnp.clip(target, 0, GRID - 1)
        open_ = inside & ~level_data("walls", state.level)[y, x]
        lost = open_ & state.visited[y, x]
        moved = open_ & ~lost
        player = jnp.where(moved, target, state.player)
        after = State(state.level, player, state.visited.at[player[1], player[0]].set(True), lost)
        advanced = moved & (player == level_data("target", state.level)).all()
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, lost, won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: player, target, walls, then the trail as it was added
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        target = level_data("target", state.level)
        grid = grid.at[target[1], target[0]].set(TARGET)
        grid = jnp.where(level_data("walls", state.level), WALL, grid)
        trail = state.visited.at[state.player[1], state.player[0]].set(False)  # every visited cell but the current one was left
        grid = jnp.where(trail, VISITED, grid)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        frame = frame.at[SIZE - 4 :, SIZE - 4 :].set(jnp.where(state.fail, FAIL, RIGHT_OK))
        return frame.at[SIZE - 4 : SIZE - 1, :3].set(jnp.where(state.fail, FAIL, LEFT_OK))
