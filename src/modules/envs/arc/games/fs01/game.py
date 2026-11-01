"""The community ARC-AGI-3 game fs01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Floor switches: the player (one cell) walks a small grid (8x8, one level 10x8); stepping on a yellow switch activates it
(recolored), and once every switch is active the gray door disappears; reaching the green target then passes the level,
5 levels. There is no way to lose. The camera scales the grid up to the 64x64 frame and draws in the level's sprite order,
so switches and the target cover the player standing on them; four 2x2 ticks in the bottom-right corner count the active
switches (green once all are). The arrays come from extract.py; a frame here is the last frame the original renders for an
action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("fs01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, SWITCH, DOOR, TARGET, WALL = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
ACTIVE = 10  # an activated switch's color
UI_ALL, UI_SOME, UI_OFF = 14, 11, 5  # the ticks' colors: every switch active, a tick of the active count, the rest


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    active: jax.Array  # (S,) per switch slot


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Fs01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level), jnp.zeros(DATA["switches"].shape[1], bool))

    def all_active(self, state: State) -> jax.Array:
        return (state.active | (level_data("switches", state.level)[:, 0] == GRID)).all()

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        target = state.player + jnp.asarray(MOVES)[action]
        inside = ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        wall = level_data("walls", state.level)[jnp.clip(target[1], 0, GRID - 1), jnp.clip(target[0], 0, GRID - 1)]
        door = ~self.all_active(state) & (target == level_data("door", state.level)).all()
        moved = inside & ~wall & ~door
        player = jnp.where(moved, target, state.player)
        after = State(state.level, player, state.active | (moved & (level_data("switches", state.level) == player).all(-1)))
        advanced = moved & (player == level_data("target", state.level)).all() & self.all_active(after)
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: player, switches, door, target, walls
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        switches = level_data("switches", state.level)
        grid = grid.at[switches[:, 1], switches[:, 0]].set(jnp.where(state.active, ACTIVE, SWITCH), mode="drop")
        door, target = level_data("door", state.level), level_data("target", state.level)
        grid = grid.at[door[1], door[0]].set(jnp.where(self.all_active(state), grid[door[1], door[0]], DOOR))
        grid = grid.at[target[1], target[0]].set(TARGET)
        grid = jnp.where(level_data("walls", state.level), WALL, grid)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        count = state.active.sum()
        ticks = jnp.where(jnp.arange(4) < count, jnp.where(self.all_active(state), UI_ALL, UI_SOME), UI_OFF)
        return frame.at[SIZE - 4 : SIZE - 2, SIZE - 8 :].set(jnp.repeat(ticks, 2)[None])
