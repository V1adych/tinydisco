"""The community ARC-AGI-3 game rv01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Rotating sparks: the player (one cell) collects every target on a small grid (8x8 to 16x16) while after each move the red
sparks all step one cell in a wind direction that turns N, E, S, W, one spark after the other in list order (a spark stops
at the edge, a wall or a cell whose first sprite is a spark), 5 levels. Walking into a spark is refused (nothing moves),
walking into a wall or the edge still lets the sparks move. The original loses when a spark is on the player, but checks
the first sprite on the player's cell, which is always the player, so the game cannot be lost; sparks pass over the
player and targets and stack where the first sprite is not a spark. The camera scales the grid up to the 64x64 frame and
draws in the level's sprite order (player, targets, sparks, walls); a HUD shows the level (top row), the wind (bottom) and
the targets left. The arrays come from extract.py; a frame here is the last frame the original renders for an action (a
level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("rv01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
WINDS = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])  # N, E, S, W
PLAYER, TARGET, SPARK, WALL = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
UI_DONE, UI_CURRENT, UI_AHEAD, UI_WIND_OFF = 14, 11, 3, 2  # HUD colors: level ticks, current level / wind / targets left


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    alive: jax.Array  # (T,) targets not yet collected; padding slots are never alive
    sparks: jax.Array  # (S, 2) x, y; padding slots sit at (GRID, GRID)
    wind: jax.Array  # spark moves so far; the next goes WINDS[wind % 4]


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Rv01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level), level_data("real", level), level_data("sparks", level), jnp.int32(0))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        size, walls, targets = (level_data(k, state.level) for k in ("grid_size", "walls", "targets"))
        inside = lambda v: ((v >= 0) & (v < size)).all()
        wall = lambda v: walls[jnp.clip(v[1], 0, GRID - 1), jnp.clip(v[0], 0, GRID - 1)]
        target = state.player + jnp.asarray(MOVES)[action]
        # the first sprite at the cell decides, in list order: player, targets, sparks, walls
        collect = inside(target) & state.alive & (targets == target).all(-1)
        refused = inside(target) & ~collect.any() & (state.sparks == target).all(-1).any()
        player = jnp.where(inside(target) & ~wall(target), target, state.player)
        alive = state.alive & ~collect
        sparks = state.sparks
        for i in range(sparks.shape[0]):
            there = sparks[i] + jnp.asarray(WINDS)[state.wind % 4]
            first_is_spark = (sparks == there).all(-1).any() & ~(player == there).all() & ~(alive & (targets == there).all(-1)).any()
            sparks = sparks.at[i].set(jnp.where(~inside(there) | wall(there) | first_is_spark, sparks[i], there))
        advanced = ~refused & ~alive.any()
        after = jax.tree.map(lambda a, b: jnp.where(refused, a, b), state, State(state.level, player, alive, sparks, state.wind + 1))
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        targets = jnp.where(state.alive[:, None], level_data("targets", state.level), GRID)
        grid = grid.at[targets[:, 1], targets[:, 0]].set(TARGET, mode="drop")
        grid = grid.at[state.sparks[:, 1], state.sparks[:, 0]].set(SPARK, mode="drop")
        grid = jnp.where(level_data("walls", state.level), WALL, grid)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        ticks = jnp.arange(NUM_LEVELS)
        frame = frame.at[0, 1 + 2 * ticks].set(jnp.where(ticks < state.level, UI_DONE, jnp.where(ticks == state.level, UI_CURRENT, UI_AHEAD)))
        frame = frame.at[SIZE - 2, 2:6].set(jnp.where(jnp.arange(4) == state.wind % 4, UI_CURRENT, UI_WIND_OFF))
        left = jnp.arange(1, 11) <= state.alive.sum()
        return frame.at[SIZE - 3, 1:11].set(jnp.where(left, UI_CURRENT, frame[SIZE - 3, 1:11]))
