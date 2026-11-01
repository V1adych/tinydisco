"""The community ARC-AGI-3 game gr01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Gravity: the player (one cell) walks an 8x8 grid of gray walls to a goal on the bottom row; after every move that is not
blocked it falls one cell if the cell below is free, 6 levels. After every action the original searches whether the goal
can still be reached under the same rules: the first time it cannot, a warning line shows, the second time in a row loses
the game. The camera scales the grid up to the 64x64 frame; a HUD shows the level (top row), a tick for the one target
(bottom row), the warning (row 2) and a bar (row 61) marking the final win or a loss. The arrays come from
extract.py; a frame here is the last frame the original renders for an action (a level pass ends on the next level's
first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("gr01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, GOAL, WALL = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
UI_DONE, UI_CURRENT, UI_AHEAD = 14, 11, 3  # level dots on the top row; UI_CURRENT is also the target tick's color
UI_WARN, UI_WON, UI_LOST = 12, 14, 8  # the dead-end warning on row 2, the end bar on row 61


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    warned: jax.Array  # the goal was unreachable after the last action (shown; another such action loses)
    ended: jax.Array  # 0 playing, 1 lost, 2 won: the end bar


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Gr01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level), jnp.bool_(False), jnp.int32(0))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        size, walls = level_data("grid_size", state.level), level_data("walls", state.level)
        free = lambda v: ((v >= 0) & (v < size)).all() & ~walls[jnp.clip(v[1], 0, GRID - 1), jnp.clip(v[0], 0, GRID - 1)]
        target = state.player + jnp.asarray(MOVES)[action]
        moved = jnp.where(free(target), target, state.player)
        below = moved + jnp.array([0, 1])
        player = jnp.where(free(target) & free(below), below, moved)
        advanced = (player == level_data("goal", state.level)).all()
        stuck = ~advanced & ~level_data("reachable", state.level)[player[1], player[0]]
        lost = stuck & state.warned
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = State(state.level, player, stuck, jnp.where(lost, 1, jnp.where(won, 2, 0)))
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, lost, won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: player, goal, walls
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        goal = level_data("goal", state.level)
        grid = jnp.where(level_data("walls", state.level), WALL, grid.at[goal[1], goal[0]].set(GOAL))
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        dots = jnp.arange(NUM_LEVELS)
        frame = frame.at[0, 1 + 2 * dots].set(jnp.where(dots < state.level, UI_DONE, jnp.where(dots == state.level, UI_CURRENT, UI_AHEAD)))
        frame = frame.at[SIZE - 1, 1].set(UI_CURRENT)  # the target is never removed, so one tick always
        frame = frame.at[2, :6].set(jnp.where(state.warned, UI_WARN, frame[2, :6]))
        return frame.at[SIZE - 3, :16].set(jnp.where(state.ended == 1, UI_LOST, jnp.where(state.ended == 2, UI_WON, frame[SIZE - 3, :16])))
