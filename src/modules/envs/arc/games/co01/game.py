"""The community ARC-AGI-3 game co01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Color gate: the player (one cell) walks a walled 10x10 grid to the goal, 5 levels. Stepping on a red or yellow pad sets the
active hue; only doors of the active hue (all drawn alike) can be walked through. There is no way to lose. The camera
scales the grid up to the 64x64 frame; a 3x3 square bottom-left shows the active hue, and when a pad opens a door a strip
bottom-right flashes for the next 8 rendered frames. The arrays come from extract.py; a frame here is the last frame the
original renders for an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("co01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
GOAL, PAD, DOOR = 0, 1, 2  # item kinds, as extract.py numbers them
PLAYER, WALL = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
START_HUE, FLASH_FRAMES, FLASH_EVEN, FLASH_ODD = 11, 8, 14, 0  # Co01UI: the flash counts down once per rendered frame


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    active: jax.Array  # the hue whose doors are open
    flash: jax.Array  # the door-open flash counter as this frame renders it


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Co01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level), jnp.int32(START_HUE), jnp.int32(0))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        items, kinds, hues = (level_data(k, state.level) for k in ("items", "kinds", "hues"))
        target = state.player + jnp.asarray(MOVES)[action]
        inside = ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        hit = (items == target).all(-1)
        first = jnp.argmax(hit)  # the first sprite there in list order (items come before walls)
        wall = ~hit.any() & level_data("walls", state.level)[jnp.clip(target[1], 0, GRID - 1), jnp.clip(target[0], 0, GRID - 1)]
        closed = hit.any() & (kinds[first] == DOOR) & (hues[first] != state.active)
        moved = inside & ~wall & ~closed
        pad = moved & hit.any() & (kinds[first] == PAD)
        active = jnp.where(pad, hues[first], state.active)
        opened = pad & ((kinds == DOOR) & (hues == active) & (hues != state.active)).any()
        player = jnp.where(moved, target, state.player)
        advanced = moved & ((kinds == GOAL) & (items == player).all(-1)).any()
        after = State(state.level, player, active, jnp.where(opened, FLASH_FRAMES, jnp.maximum(state.flash - 1, 0)))
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        # draw order: the items and walls in list order, the player (layer 1) on top
        items = level_data("items", state.level)
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[items[:, 1], items[:, 0]].set(level_data("item_colors", state.level), mode="drop")
        grid = jnp.where(level_data("walls", state.level), WALL, grid).at[state.player[1], state.player[0]].set(PLAYER)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        frame = frame.at[SIZE - 3 :, 1:4].set(state.active)
        strip = jnp.where(state.flash % 2 == 0, FLASH_EVEN, FLASH_ODD)
        return frame.at[SIZE - 1, SIZE - 4 :].set(jnp.where(state.flash > 0, strip, frame[SIZE - 1, SIZE - 4 :]))
