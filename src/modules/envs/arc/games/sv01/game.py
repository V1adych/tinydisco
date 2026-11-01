"""The community ARC-AGI-3 game sv01 (github.com/theredbluepill/arc-interactive) in JAX, frame-exact to the original Python game.
Survival: the player (one cell) walks an open grid (8x8 up to 24x24 over 5 levels); every action (moves 1-4, 5 waits)
costs 2 hunger and, outside the warm zones, 2 warmth (both start at 100); stepping onto food eats it (+20 hunger, capped
at 100). Either reaching 0 loses the game; surviving 60 actions passes the level. The camera scales the grid up to the
64x64 frame and draws by layer: warm zones, food, player. Bars at the top show hunger, warmth and the actions left, two
pixels the top-right the level. The arrays come from extract.py; a frame here is the last frame the original renders
for an action (a level pass ends on the next level's first frame)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("sv01.npz")))
NUM_LEVELS, GRID = DATA["warm"].shape[:2]
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)])  # the original's actions 1-5: up, down, left, right, wait
PLAYER, FOOD, WARM = (int(c) for c in DATA["colors"])
BACKGROUND, PADDING = int(DATA["background"]), int(DATA["padding"])
FULL, DECAY, MEAL, SURVIVE = 100, 2, 20, 60
BAR, BAR_HUNGER, BAR_WARMTH, BAR_LEFT, UI_MARK, UI_LEVEL = 20, 14, 12, 3, 9, 3  # the UI's bar length and colors


@struct.dataclass
class State:
    level: jax.Array
    player: jax.Array  # (2,) x, y on the grid
    eaten: jax.Array  # (F,) per food slot
    hunger: jax.Array
    warmth: jax.Array
    steps: jax.Array


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


class Sv01(ArcGame):
    num_actions = 5
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("start", level), jnp.zeros(DATA["foods"].shape[1], bool), jnp.int32(FULL), jnp.int32(FULL), jnp.int32(0))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        target = state.player + jnp.asarray(MOVES)[action]
        moved = (action < 4) & ((target >= 0) & (target < level_data("grid_size", state.level))).all()
        player = jnp.where(moved, target, state.player)
        # only a move eats: level 1's player starts on a food and keeps it while bumping the edge
        eat = moved & ~state.eaten & (level_data("foods", state.level) == player).all(-1)
        hunger = jnp.maximum(0, jnp.minimum(FULL, state.hunger + MEAL * eat.any()) - DECAY)
        warm = level_data("warm", state.level)[player[1], player[0]]
        warmth = jnp.where(warm, state.warmth, jnp.maximum(0, state.warmth - DECAY))
        after = State(state.level, player, state.eaten | eat, hunger, warmth, state.steps + 1)
        lost = (hunger <= 0) | (warmth <= 0)
        advanced = ~lost & (after.steps >= SURVIVE)
        won = advanced & (state.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, advanced, lost, won

    def render(self, state: State) -> jax.Array:
        # drawn by layer: warm zones, food, player
        grid = jnp.where(level_data("warm", state.level), WARM, BACKGROUND)
        foods = jnp.where(state.eaten[:, None], GRID, level_data("foods", state.level))
        grid = grid.at[foods[:, 1], foods[:, 0]].set(FOOD, mode="drop")
        grid = grid.at[state.player[1], state.player[0]].set(PLAYER)
        cells = level_data("cells", state.level)
        frame = jnp.where(cells >= 0, grid.reshape(-1)[cells], PADDING)
        bar = jnp.arange(BAR)
        for row, value, total, color in ((1, state.hunger, FULL, BAR_HUNGER), (2, state.warmth, FULL, BAR_WARMTH), (3, SURVIVE - state.steps, SURVIVE, BAR_LEFT)):
            frame = frame.at[row, 2 : 2 + BAR].set(jnp.where(bar < jnp.clip(value * BAR // total, 0, BAR), color, frame[row, 2 : 2 + BAR]))
        return frame.at[1, SIZE - 3].set(UI_MARK).at[1, SIZE - 4].set(UI_LEVEL + state.level)
