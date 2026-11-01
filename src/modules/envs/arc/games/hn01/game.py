"""The community ARC-AGI-3 game hn01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Tower of Hanoi with three disks on a 10x12 grid: actions 1-3 select a peg (a held disk hovers over it), action 5 picks up
the selected peg's top disk or drops the held one there (refused onto a smaller disk), action 4 does nothing; the level
passes when all disks stand on the right peg, 5 identical levels, no way to lose. The UI draws over the scaled frame: a
dot per level in the top row (done, current, ahead), a red pixel for three actions after a refused drop (the countdown
also ticks on a level change and carries over), and a bar when the game is won. The arrays come from extract.py; a
frame here is the last frame the original renders for an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("hn01.npz")))
NUM_LEVELS = len(DATA["cells"])
GRID = int(DATA["grid_size"].max())  # the side of the padded grid the cell indices count in
PEG_X, BASE_Y, HAND_Y = DATA["peg_x"], int(DATA["base_y"]), 2  # a held disk hovers at HAND_Y over the selected peg
DISKS, PEG = DATA["colors"][:3], int(DATA["colors"][3])  # disk k (size k + 1)
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
HELD = 3  # the peg index of the disk in hand
DOT_DONE, DOT_HERE, DOT_AHEAD = 14, 11, 3  # the UI's level dots in row 0 at columns 1, 3, 5...
REJECT_PIXEL, REJECT_COLOR, REJECT_FRAMES = (SIZE - 2, 14), 8, 3
BAR_ROW, BAR_WIDTH, BAR_WIN = SIZE - 3, 16, 14


@struct.dataclass
class State:
    level: jax.Array
    place: jax.Array  # (3,) the peg of each disk, smallest first; HELD = in hand
    peg: jax.Array  # selected
    reject: jax.Array  # frames left showing the refused-drop pixel
    won: jax.Array


class Hn01(ArcGame):
    num_actions = 5
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, jnp.zeros(3, jnp.int32), jnp.int32(0), jnp.int32(0), jnp.bool_(False))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        held, on_peg = state.place == HELD, state.place == state.peg
        hand, top = jnp.argmax(held), jnp.argmax(on_peg)  # the top disk is the smallest one on the peg
        drop_ok = ~on_peg.any() | (top > hand)
        pick = (action == 4) & ~held.any() & on_peg.any()
        drop = (action == 4) & held.any() & drop_ok
        place = jnp.where(pick, state.place.at[top].set(HELD), state.place)
        place = jnp.where(drop, place.at[hand].set(state.peg), place)
        reject = jnp.where((action == 4) & held.any() & ~drop_ok, REJECT_FRAMES, jnp.maximum(state.reject - 1, 0))
        advanced = (place == 2).all()
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = State(state.level, place, jnp.where(action < 3, action, state.peg), reject, won)
        # the next level's on_set_level ticks the reject countdown once more
        fresh = self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)).replace(reject=jnp.maximum(reject - 1, 0))
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), fresh, after)
        return after, advanced, jnp.bool_(False), won

    def render(self, state: State) -> jax.Array:
        grid = jnp.full((GRID, GRID), BACKGROUND, jnp.int32).at[BASE_Y, PEG_X].set(PEG)
        below = ((state.place[None, :] == state.place[:, None]) & (jnp.arange(3)[None, :] > jnp.arange(3)[:, None])).sum(-1)
        held = state.place == HELD
        x = jnp.asarray(PEG_X)[jnp.where(held, state.peg, state.place)]
        y = jnp.where(held, HAND_Y, BASE_Y - 1 - below)
        grid = grid.at[y, x].set(jnp.asarray(DISKS))
        cells = jnp.asarray(DATA["cells"])[state.level]
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        dots = jnp.arange(NUM_LEVELS)
        frame = frame.at[0, 1 + 2 * dots].set(jnp.where(dots < state.level, DOT_DONE, jnp.where(dots == state.level, DOT_HERE, DOT_AHEAD)))
        frame = frame.at[REJECT_PIXEL].set(jnp.where(state.reject > 0, REJECT_COLOR, frame[REJECT_PIXEL]))
        return frame.at[BAR_ROW, :BAR_WIDTH].set(jnp.where(state.won, BAR_WIN, frame[BAR_ROW, :BAR_WIDTH]))
