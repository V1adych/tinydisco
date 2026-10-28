"""The community ARC-AGI-3 game ms01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Blind Sapper: the player (one cell) crosses a hidden minefield (8x8 to 16x16) to a goal; each ground tile it steps on is
recolored by its count of adjacent mines, stepping on a mine loses, 5 levels. The camera scales the grid up to the 64x64
frame and draws in the level's sprite order (player, goal, mines, ground tiles), so the tiles hide the player everywhere
but on its start cell and the mines always. The HUD marks the level (row 1) and counts revealed cells in ticks (row 62);
a mine plays an 8-frame burst whose last frame (red border, rings and cross) ends the action, centered as if the grid
were 16x16 whatever its size. The arrays come from extract.py; a frame here is the last frame the original renders for
an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("ms01.npz")))
NUM_LEVELS, GRID = DATA["mines"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PLAYER, GOAL, TILE = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
HUD_MARK, LEVEL_COLORS, TICK = 9, np.array([10, 11, 12, 14, 15]), 10  # frame[1, 2], frame[1, 3] by difficulty, row 62 ticks
BURST_RED, BURST_LIGHT = 8, 11
# the burst's last frame (animation step 7): a 5-pixel red border, rings at Chebyshev radius 2, 4, ... 10 alternating
# red / yellow, a red cross with 10-pixel arms and a yellow center, at the cell's center in a 16x16-grid camera (4x scale)
BORDER, RINGS, ARM, BURST_SCALE = 5, {2: BURST_RED, 4: BURST_LIGHT, 6: BURST_RED, 8: BURST_LIGHT, 10: BURST_RED}, 10, 4


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    revealed: jax.Array  # (GRID, GRID) cells stepped on (the start only once re-entered)
    dead: jax.Array


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Ms01(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level), jnp.zeros((GRID, GRID), bool), jnp.bool_(False))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        target = state.player + jnp.asarray(MOVES)[action]
        inside = ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        x, y = jnp.clip(target, 0, GRID - 1)
        lost = inside & level_data("mines", state.level)[y, x]
        player = jnp.where(inside, target, state.player)
        revealed = state.revealed.at[y, x].set(state.revealed[y, x] | (inside & ~lost))
        advanced = inside & ~lost & (target == level_data("goal", state.level)).all()
        after = State(state.level, player, revealed, lost)
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, lost, won

    def render(self, state: State) -> jax.Array:
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[state.player[1], state.player[0]].set(PLAYER)
        goal = level_data("goal", state.level)
        grid = grid.at[goal[1], goal[0]].set(GOAL)
        tiles = jnp.where(state.revealed, level_data("clues", state.level), TILE)
        grid = jnp.where(level_data("tiles", state.level), tiles, grid)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        frame = frame.at[1, 2].set(HUD_MARK).at[1, 3].set(jnp.asarray(LEVEL_COLORS)[(level_data("difficulty", state.level) - 1) % len(LEVEL_COLORS)])
        i = jnp.arange(SIZE)
        frame = frame.at[SIZE - 2].set(jnp.where((i >= 2) & (i < 2 + jnp.minimum(SIZE - 4, state.revealed.sum())), TICK, frame[SIZE - 2]))
        rows, cols = jnp.mgrid[:SIZE, :SIZE]
        cx, cy = state.player * BURST_SCALE + BURST_SCALE // 2
        dx, dy = cols - cx, rows - cy
        ring = jnp.maximum(jnp.abs(dx), jnp.abs(dy))
        burst = jnp.where(jnp.minimum(jnp.minimum(rows, cols), jnp.minimum(SIZE - 1 - rows, SIZE - 1 - cols)) < BORDER, BURST_RED, frame)
        for r, color in RINGS.items():
            burst = jnp.where(ring == r, color, burst)
        burst = jnp.where(((dx == 0) | (dy == 0)) & (ring <= ARM), BURST_RED, burst)
        burst = jnp.where(ring == 0, BURST_LIGHT, burst)
        return jnp.where(state.dead, burst, frame)
