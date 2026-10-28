"""The community ARC-AGI-3 game mb01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Magnet crates on a 10x10 grid: the player walks onto free cells (walls and crates block, the goal does not); after each
move every metal crate, in list order, slides one cell toward the player along the axis of the larger distance if that
cell is inside the grid and holds no player, crate or wall (a crate may slide onto the goal and then blocks it). Reaching
the goal passes the level, 5 levels with 1-2 crates, no way to lose. Drawn in the level's sprite order (player, crates,
goal, walls); the UI draws a tick per crate (counted at level start) in the frame's second-to-last row. The arrays come
from extract.py; a frame here is the last frame the original renders for an action (a level pass ends on the next
level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("mb01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, METAL, GOAL, WALL = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
UI_ROW, UI_COLOR, UI_MAX = SIZE - 2, 12, 12  # a tick per crate at columns 1..


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    crates: jax.Array  # (C, 2) x, y; padding slots sit at (GRID, GRID)


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Mb01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level), level_data("crates", level))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        size, walls, real = (level_data(k, state.level) for k in ("grid_size", "walls", "real"))
        inside = lambda v: ((v >= 0) & (v < size)).all()
        free = lambda v, crates: inside(v) & ~walls[jnp.clip(v[1], 0, GRID - 1), jnp.clip(v[0], 0, GRID - 1)] & ~(crates == v).all(-1).any()
        target = state.player + jnp.asarray(MOVES)[action]
        moved = free(target, state.crates)
        player = jnp.where(moved, target, state.player)
        crates = state.crates
        for i in range(crates.shape[0]):  # one after another: a crate sees the ones already pulled
            gap = player - crates[i]
            step = jnp.sign(gap) * jnp.where(jnp.abs(gap[0]) >= jnp.abs(gap[1]), jnp.array([1, 0]), jnp.array([0, 1]))
            pulled = crates[i] + step
            slide = moved & real[i] & free(pulled, crates) & (pulled != player).any()
            crates = crates.at[i].set(jnp.where(slide, pulled, crates[i]))
        advanced = moved & (player == level_data("goal", state.level)).all()
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = State(state.level, player, crates)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: player, crates, goal, walls
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        grid = grid.at[state.crates[:, 1], state.crates[:, 0]].set(METAL, mode="drop")
        goal = level_data("goal", state.level)
        grid = grid.at[goal[1], goal[0]].set(GOAL)
        grid = jnp.where(level_data("walls", state.level), WALL, grid)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        ticks = jnp.arange(UI_MAX)
        shown = ticks < level_data("real", state.level).sum()
        return frame.at[UI_ROW, 1 + ticks].set(jnp.where(shown, UI_COLOR, frame[UI_ROW, 1 + ticks]))
