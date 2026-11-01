"""The community ARC-AGI-3 game zq01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Zone timer: the player (one cell) walks a small grid (8x8, one level 10x8) to a target cell while a fixed set of red hazard
cells switches on and off every `period` actions (blocked moves count too); a hazard that is on blocks and is drawn, one
that is off is neither, 5 levels. There is no way to lose. The camera scales the grid up to the 64x64 frame and draws in
the level's sprite order, so a hazard switching on covers the player standing on it; a 3x3 square near the bottom-right
corner turns red while the hazards are on. The arrays come from extract.py; a frame here is the last frame the original
renders for an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("zq01.npz")))
NUM_LEVELS, GRID = DATA["hazards"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, TARGET, HAZARD = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
UI_ON, UI_OFF = 8, 14  # the 3x3 hazard indicator at rows 61-63, columns 60-62


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    ticks: jax.Array  # actions on this level; the hazards are on in every odd period


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


def hazards_on(state: State) -> jax.Array:
    return (state.ticks // level_data("period", state.level)) % 2 == 1


class Zq01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level), jnp.int32(0))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        ticked = state.replace(ticks=state.ticks + 1)
        target = state.player + jnp.asarray(MOVES)[action]
        inside = ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        hazard = level_data("hazards", state.level)[jnp.clip(target[1], 0, GRID - 1), jnp.clip(target[0], 0, GRID - 1)]
        player = jnp.where(inside & ~(hazard & hazards_on(ticked)), target, state.player)
        advanced = (player == level_data("target", state.level)).all()
        after = ticked.replace(player=player)
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: player, target, hazards (added last, drawn only while on)
        on = hazards_on(state)
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        target = level_data("target", state.level)
        grid = grid.at[target[1], target[0]].set(TARGET)
        grid = jnp.where(level_data("hazards", state.level) & on, HAZARD, grid)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        return frame.at[SIZE - 3 :, SIZE - 4 : SIZE - 1].set(jnp.where(on, UI_ON, UI_OFF))
