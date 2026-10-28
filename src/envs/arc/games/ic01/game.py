"""The community ARC-AGI-3 game ic01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Ice: on a small grid (8x8, one level 10x8) each move slides the player (one cell) in a straight line until the next cell
would be off the grid, a wall or a red hazard (it stops before them; hazards do not kill); the yellow target does not
stop a slide, so only a slide that ends on it passes the level, 5 levels. There is no way to lose. The camera scales the
grid up to the 64x64 frame and draws in the level's sprite order, so the target covers the player on it; a 4x4 square in
the bottom-right corner is light blue, yellow while the player stands on the target. The arrays come from extract.py; a
frame here is the last frame the original renders for an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("ic01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, TARGET, WALL, HAZARD = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
UI_SIZE, UI_ON_GOAL, UI_ICE = 4, 11, 10  # the square in the frame's bottom-right corner


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Ic01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        move = jnp.asarray(MOVES)[action]
        ray = state.player + jnp.arange(1, GRID + 1)[:, None] * move  # every cell ahead, nearest first
        inside = ((ray >= 0) & (ray < level_data("grid_size", state.level))).all(-1)
        x, y = jnp.clip(ray[:, 0], 0, GRID - 1), jnp.clip(ray[:, 1], 0, GRID - 1)
        free = inside & ~(level_data("walls", state.level) | level_data("hazards", state.level))[y, x]
        player = state.player + jnp.cumprod(free).sum() * move
        advanced = (player == level_data("target", state.level)).all()
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), State(state.level, player))
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: player, target, then walls or hazards (they never share a cell)
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        target = level_data("target", state.level)
        grid = grid.at[target[1], target[0]].set(TARGET)
        grid = jnp.where(level_data("walls", state.level), WALL, grid)
        grid = jnp.where(level_data("hazards", state.level), HAZARD, grid)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        on_goal = (state.player == target).all()
        return frame.at[SIZE - UI_SIZE :, SIZE - UI_SIZE :].set(jnp.where(on_goal, UI_ON_GOAL, UI_ICE))
