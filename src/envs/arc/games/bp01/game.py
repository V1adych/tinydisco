"""The community ARC-AGI-3 game bp01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Battery mesh: the player (one cell) walks a 14x14 grid around walls; action 5 on a tower powers it (recolored), and
stepping onto the goal with every tower powered passes the level, 5 levels with 2-4 towers. There is no way to lose. The
camera scales the grid up to the 64x64 frame and draws in the level's sprite order, so towers and the goal cover the
player standing on them; a row of ticks in the bottom border counts the powered towers. The arrays come from extract.py;
a frame here is the last frame the original renders for an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("bp01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, TOWER, TOWER_ON, GOAL, WALL = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
UI_ON, UI_OFF = 14, 5  # the tick per tower in the frame's second-to-last row, from column 1


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    powered: jax.Array  # (T,) per tower slot


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Bp01(ArcGame):
    num_actions = 5
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level), jnp.zeros(DATA["towers"].shape[1], bool))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        size, walls, real = (level_data(k, state.level) for k in ("grid_size", "walls", "real"))
        power = action == 4
        powered = state.powered | (power & (level_data("towers", state.level) == state.player).all(-1))
        target = state.player + jnp.asarray(MOVES)[jnp.minimum(action, 3)]
        inside = ((target >= 0) & (target < size)).all()
        moved = ~power & inside & ~walls[jnp.clip(target[1], 0, GRID - 1), jnp.clip(target[0], 0, GRID - 1)]
        player = jnp.where(moved, target, state.player)
        advanced = ~power & (player == level_data("goal", state.level)).all() & (powered | ~real).all()
        after = State(state.level, player, powered)
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: player, towers (a powered one moves to the end, but no tower shares a
        # cell with the goal or a wall), goal, walls
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        towers = level_data("towers", state.level)
        grid = grid.at[towers[:, 1], towers[:, 0]].set(jnp.where(state.powered, TOWER_ON, TOWER), mode="drop")
        goal = level_data("goal", state.level)
        grid = jnp.where(level_data("walls", state.level), WALL, grid.at[goal[1], goal[0]].set(GOAL))
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        real = level_data("real", state.level)
        ticks = jnp.where(jnp.arange(len(real)) < state.powered.sum(), UI_ON, UI_OFF)
        return frame.at[SIZE - 2, 1 : 1 + len(real)].set(jnp.where(real, ticks, frame[SIZE - 2, 1 : 1 + len(real)]))
