"""The ARC-AGI-3 game ls20 (official games have ids, not names) in JAX, frame-exact to the original Python game.
The player moves 5 pixels per action through a maze carrying a symbol with a shape, a color and a rotation; floor tiles
cycle one of the three; a lock opens when stepped on while carrying the symbol it displays; all locks open = next level,
7 levels. Every move spends one unit of a budget that pickups refill; an empty budget costs one of 3 lives and restarts
the level; the last level is dark beyond a radius around the player. Nothing of this is told to the agent: it sees 64x64
pixels of 16 colors. The arrays come from extract.py; a frame here is the last frame the original renders for an
action (a life loss also renders a flash frame first, which an agent never sees)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("ls20.npz")))
NUM_LEVELS, NUM_SLOTS = DATA["planes"].shape[:2]
CELL = 5
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
ROLE = {name: i for i, name in enumerate(DATA["roles"])}
LIVES, BUDGET = 3, int(DATA["budget"][0])
assert (DATA["budget"] == BUDGET).all(), "the HUD assumes one budget for all levels"
FOG_RADIUS, FOG_COLOR = 20.0, 5
BACKGROUND = int(DATA["background"])
CARRIED_AT, CARRIED_SCALE = (55, 3), 2  # where the HUD shows the carried symbol (row, column)
CYCLE = np.array([len(DATA["shapes"]), len(DATA["palette"]), 4])  # shapes, colors, rotations
# the symbol pixels per shape and rotation (0 = ink, -1 = transparent), rotated as the original's sprite renderer does
SYMBOLS = np.stack([[np.rot90(shape, k=(4 - r) % 4) for r in range(4)] for shape in DATA["shapes"]])
PLAYER_SLOT = np.array([np.flatnonzero(roles == ROLE["player"])[0] for roles in DATA["role"]])
CARRIED_SLOT = np.array([np.flatnonzero(roles == ROLE["carried"])[0] for roles in DATA["role"]])
HIGHLIGHT_SLOT = np.array([np.flatnonzero(roles == ROLE["highlight"])[0] for roles in DATA["role"]])


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y of the 5x5 player sprite
    symbol: jax.Array  # (3,) shape, color and rotation indices of the carried symbol
    lives: jax.Array
    budget: jax.Array  # moves left; the move that takes it below zero costs a life
    solved: jax.Array  # (L,) per lock
    visible: jax.Array  # (S,) per sprite plane: taken pickups and opened locks vanish, hint boxes appear


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Ls20(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        solved = jnp.zeros(DATA["lock"].shape[1], bool)
        return State(level, level_data("start", level), level_data("start_symbol", level), jnp.int32(LIVES), jnp.int32(BUDGET), solved, level_data("visible", level))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        role, x, y = (level_data(k, state.level) for k in ("role", "x", "y"))
        lock, wants = level_data("lock", state.level), level_data("lock_wants", state.level)
        real = lock < NUM_SLOTS
        target = state.player + CELL * jnp.asarray(MOVES)[action]
        # the sprites whose top-left corner lies in the target cell decide what the move does
        window = (x >= target[0]) & (x < target[0] + CELL) & (y >= target[1]) & (y < target[1] + CELL)
        lock_here = real & ~state.solved & window.at[lock].get(mode="fill", fill_value=False)
        # a lock that wants another symbol refuses the whole action (the original flashes its border and returns early)
        refused = (lock_here & ~(wants == state.symbol).all(-1)).any()
        wall = (window & (role == ROLE["wall"])).any()
        picked = window & (role == ROLE["pickup"]) & state.visible
        hit = jnp.array([(window & (role == ROLE[t])).any() for t in ("shape_tile", "color_tile", "rotate_tile")])
        symbol = (state.symbol + hit) % jnp.asarray(CYCLE)
        match = real & (wants == symbol).all(-1)
        visible = state.visible & ~picked
        # a tile refreshes the hints: a box around each unsolved lock the symbol now fits, and the highlight if there is one
        hinted = visible.at[level_data("lock_hint", state.level)].set(match & ~state.solved, mode="drop")
        hinted = hinted.at[jnp.asarray(HIGHLIGHT_SLOT)[state.level]].set((match & ~state.solved).any())
        visible = jnp.where(hit.any(), hinted, visible)
        player = jnp.where(wall, state.player, target)
        at_lock = (player[0] == x.at[lock].get(mode="fill", fill_value=-1)) & (player[1] == y.at[lock].get(mode="fill", fill_value=-1))
        opened = real & ~state.solved & match & at_lock
        solved = state.solved | opened
        clears = opened & level_data("lock_box_clears", state.level)
        visible = visible.at[lock].min(~opened, mode="drop").at[level_data("lock_display", state.level)].min(~opened, mode="drop")
        visible = visible.at[level_data("lock_box", state.level)].min(~clears, mode="drop").at[level_data("lock_hint", state.level)].min(~clears, mode="drop")
        visible = visible.at[jnp.asarray(HIGHLIGHT_SLOT)[state.level]].min(~clears.any())
        advanced = (solved | ~real).all()
        refilled = picked.any()
        budget = jnp.where(refilled, BUDGET, jnp.where(advanced, state.budget, state.budget - 1))
        exhausted = ~advanced & ~refilled & (budget < 0)
        lives = state.lives - exhausted
        lost = exhausted & (lives == 0)
        after = State(state.level, player, symbol, lives, budget, solved, visible)
        # out of budget with lives left: the level starts over
        after = jax.tree.map(lambda a, b: jnp.where(exhausted & ~lost, a, b), self.init_level(state.level).replace(lives=lives), after)
        after = jax.tree.map(lambda a, b: jnp.where(refused, a, b), state, after)
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, lost, won

    def render(self, state: State) -> jax.Array:
        planes = level_data("planes", state.level)
        blank = jnp.full((SIZE, SIZE), -1, jnp.int8)
        px, py = state.player
        player = jax.lax.dynamic_update_slice(blank, jnp.asarray(DATA["player"]), (py, px))
        shape, color, rotation = state.symbol
        symbol = jnp.asarray(SYMBOLS)[shape, rotation]
        symbol = jnp.where(symbol == 0, jnp.asarray(DATA["palette"])[color], -1).astype(jnp.int8)
        symbol = jnp.repeat(jnp.repeat(symbol, CARRIED_SCALE, 0), CARRIED_SCALE, 1)
        carried = jax.lax.dynamic_update_slice(blank, symbol, CARRIED_AT)
        planes = planes.at[jnp.asarray(PLAYER_SLOT)[state.level]].set(player).at[jnp.asarray(CARRIED_SLOT)[state.level]].set(carried)
        # the topmost visible, non-transparent plane wins (planes are in draw order)
        covers = state.visible[:, None, None] & (planes >= 0)
        top = (jnp.arange(NUM_SLOTS)[:, None, None] * covers).max(0)
        frame = jnp.where(covers.any(0), jnp.take_along_axis(planes, top[None], 0)[0], BACKGROUND)
        rows, cols = jnp.mgrid[:SIZE, :SIZE]
        far = jnp.hypot(rows - (py + 1.5), cols - (px + 1.5)) > FOG_RADIUS
        fogged = jnp.where(carried >= 0, carried, jnp.where(far, FOG_COLOR, frame))
        frame = jnp.where(level_data("fog", state.level), fogged, frame)
        # the HUD: the budget as ticks lit from the right, then the lives
        lit = (BUDGET - 1 - jnp.arange(BUDGET)) < state.budget
        frame = frame.at[61:63, 13 : 13 + BUDGET].set(jnp.where(lit, 11, 3)[None])
        for k in range(LIVES):
            frame = frame.at[61:63, 56 + 3 * k : 58 + 3 * k].set(jnp.where(state.lives > k, 8, 3))
        return frame
