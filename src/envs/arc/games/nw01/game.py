"""The community ARC-AGI-3 game nw01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Arrow floor: the player (one cell) walks an 8x8 grid; a move that ends on a magenta arrow queues its direction, and the next
action moves that way instead of the chosen one (the queue clears even when that move is blocked); reaching the yellow
target passes the level, 5 levels. There is no way to lose. The camera scales the grid up to the 64x64 frame and draws in
the level's sprite order, so arrows and the target cover the player on them; a 3x3 orange patch near the bottom-left
corner shows while a forced move is queued. The arrays come from extract.py; a frame here is the last frame the original
renders for an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("nw01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, ARROW, TARGET, WALL = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
UI_SIZE, UI_QUEUED = 3, 12  # the patch at rows SIZE-4.., columns 0..


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    forced: jax.Array  # (2,) the queued direction
    queued: jax.Array


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Nw01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level), jnp.zeros(2, jnp.int32), jnp.bool_(False))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        target = state.player + jnp.where(state.queued, state.forced, jnp.asarray(MOVES)[action])
        inside = ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        wall = level_data("walls", state.level)[jnp.clip(target[1], 0, GRID - 1), jnp.clip(target[0], 0, GRID - 1)]
        moved = inside & ~wall
        player = jnp.where(moved, target, state.player)
        on = moved & (level_data("arrows", state.level) == player).all(-1)
        after = State(state.level, player, level_data("vectors", state.level)[jnp.argmax(on)], on.any())
        advanced = moved & (player == level_data("target", state.level)).all()
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: player, arrows, target, walls
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        arrows, target = level_data("arrows", state.level), level_data("target", state.level)
        grid = grid.at[arrows[:, 1], arrows[:, 0]].set(ARROW, mode="drop")
        grid = grid.at[target[1], target[0]].set(TARGET)
        grid = jnp.where(level_data("walls", state.level), WALL, grid)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        patch = frame[SIZE - 4 : SIZE - 4 + UI_SIZE, :UI_SIZE]
        return frame.at[SIZE - 4 : SIZE - 4 + UI_SIZE, :UI_SIZE].set(jnp.where(state.queued, UI_QUEUED, patch))
