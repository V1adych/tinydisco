"""The community ARC-AGI-3 game wk01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Weak floor: the player (one cell) walks a 10x10 maze to a target; brown weak tiles turn into holes once it leaves them,
and stepping into a hole loses, as does running out of the level's move budget (12-25 moves) before the target, 5
levels. The camera scales the grid up to the 64x64 frame and draws in the level's sprite order (walls, weak tiles,
player, target, holes); the corner squares (4x4 bottom-right, 3x3 bottom-left) turn red on a loss and a row of ticks at
the top shows the moves left (as of the last move that did not pass the level). The arrays come from extract.py; a frame
here is the last frame the original renders for an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("wk01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, TARGET, WEAK, HOLE, WALL = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
FAIL, OK = 8, 13  # the corner squares: bottom-right 4x4, bottom-left 3x3 (from the frame's 4th last row)
TICKS, TICK_LEFT, TICK_SPENT = 14, 11, 3  # at most 14 budget ticks on row 1, every other column from column 1


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    holes: jax.Array  # (GRID, GRID) collapsed weak tiles
    steps_left: jax.Array
    shown_left: jax.Array  # the moves left the ticks show: not updated by the move that passes the level
    fail: jax.Array


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Wk01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        cap = level_data("step_cap", level)
        return State(level, level_data("start", level), jnp.zeros((GRID, GRID), bool), cap, cap, jnp.bool_(False))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        target = state.player + jnp.asarray(MOVES)[action]
        inside = ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        x, y = jnp.clip(target, 0, GRID - 1)
        open_ = inside & ~level_data("walls", state.level)[y, x]
        fell = open_ & state.holes[y, x]
        moved = open_ & ~fell
        here = state.player
        holes = state.holes.at[here[1], here[0]].set(state.holes[here[1], here[0]] | (moved & level_data("weak", state.level)[here[1], here[0]]))
        steps = state.steps_left - moved
        advanced = moved & (target == level_data("target", state.level)).all()
        broke = moved & ~advanced & (steps <= 0)
        shown = jnp.where(moved & ~advanced, steps, state.shown_left)
        after = State(state.level, jnp.where(moved, target, here), holes, steps, shown, fell | broke)
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, fell | broke, won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: walls, weak tiles, player, target, then holes as they collapse
        grid = jnp.where(level_data("walls", state.level), WALL, BACKGROUND)
        grid = jnp.where(level_data("weak", state.level) & ~state.holes, WEAK, grid)
        grid = grid.at[state.player[1], state.player[0]].set(PLAYER)
        target = level_data("target", state.level)
        grid = grid.at[target[1], target[0]].set(TARGET)
        grid = jnp.where(state.holes, HOLE, grid)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        frame = frame.at[SIZE - 4 :, SIZE - 4 :].set(jnp.where(state.fail, FAIL, OK))
        frame = frame.at[SIZE - 4 : SIZE - 1, :3].set(jnp.where(state.fail, FAIL, OK))
        i = jnp.arange(TICKS)
        ticks = jnp.where(i < jnp.minimum(level_data("step_cap", state.level), TICKS), jnp.where(i < state.shown_left, TICK_LEFT, TICK_SPENT), frame[1, 1 + 2 * i])
        return frame.at[1, 1 + 2 * i].set(ticks)
