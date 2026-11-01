"""The community ARC-AGI-3 game tp01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Teleporters: the player (one cell) walks a small grid (8x8, one level 10x8) split by a wall; stepping onto either end of a
magenta portal pair warps it to the other end, and reaching the yellow target passes the level, 5 levels. There is no way
to lose. The camera scales the grid up to the 64x64 frame and draws in the level's sprite order, so portals and the target
cover the player standing on them; a fixed HUD shows a magenta block with white caps bottom-right, one tick per pair and a
difficulty-colored dot bottom-left. The arrays come from extract.py; a frame here is the last frame the original renders
for an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("tp01.npz")))
NUM_LEVELS, GRID = DATA["walls"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, PORTAL, TARGET, WALL = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
UI_BLOCK, UI_CAP, UI_TICK, UI_MAX_TICKS = 7, 1, 7, 6
UI_PALETTE = np.array([10, 11, 12, 14, 15])  # the difficulty dot's color, by difficulty 1-5


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Tp01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        target = state.player + jnp.asarray(MOVES)[action]
        inside = ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        wall = level_data("walls", state.level)[jnp.clip(target[1], 0, GRID - 1), jnp.clip(target[0], 0, GRID - 1)]
        moved = inside & ~wall
        player = jnp.where(moved, target, state.player)
        on = moved & (level_data("portals", state.level) == player).all(-1)  # a blocked move returns before the warp
        player = jnp.where(on.any(), level_data("partners", state.level)[jnp.argmax(on)], player)
        advanced = moved & (player == level_data("target", state.level)).all()
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), State(state.level, player))
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        # the level's sprite order is the draw order: player, portals, target, walls
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        portals, target = level_data("portals", state.level), level_data("target", state.level)
        grid = grid.at[portals[:, 1], portals[:, 0]].set(PORTAL, mode="drop")
        grid = grid.at[target[1], target[0]].set(TARGET)
        grid = jnp.where(level_data("walls", state.level), WALL, grid)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        frame = frame.at[SIZE - 3 : SIZE - 1, SIZE - 4 : SIZE - 2].set(UI_BLOCK)
        frame = frame.at[SIZE - 3, jnp.array([SIZE - 5, SIZE - 2])].set(UI_CAP)
        ticks = jnp.arange(UI_MAX_TICKS) < level_data("pairs", state.level)
        frame = frame.at[SIZE - 1, 1 : 1 + UI_MAX_TICKS].set(jnp.where(ticks, UI_TICK, frame[SIZE - 1, 1 : 1 + UI_MAX_TICKS]))
        dot = jnp.asarray(UI_PALETTE)[jnp.clip(level_data("difficulty", state.level) - 1, 0, len(UI_PALETTE) - 1)]
        return frame.at[SIZE - 2, 1].set(dot)
