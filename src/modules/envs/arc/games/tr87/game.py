"""The ARC-AGI-3 game tr87 (official games have ids, not names) in JAX, frame-exact to the original Python game.
A rewriting puzzle: the top area shows rules (a left tile sequence, an arrow, a right tile sequence), the bottom area a
fixed source row of tiles above an editable answer row. A cursor (two brackets) selects an answer tile; two actions move
it left/right (wrapping), two cycle the selected tile through its 7 variants. The level is solved when the answer row is
the source row rewritten left to right by the first matching rule at each position; level flags change the rewriting
(double translation chains a rule's right side through a second rule, tree translation replaces each right tile by the
right side of the rule that starts with it) or make the rule sides the editable things instead of the answer row (alter
rules: the cursor walks over all sides and cycling turns every tile of the selected side). Every action spends one unit
of a budget shown as a bar on the bottom row; an empty budget loses the game. 6 levels. The arrays come from extract.py;
a frame here is the last frame the original renders for an action: a solving action plays an animation the agent never
sees and ends on the next level's first frame, or on the last level on the final animation frame (tiles painted black)."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import NUM_COLORS, SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("tr87.npz")))
NUM_LEVELS, NUM_TILES = DATA["x"].shape
NUM_RULES, SIDE = DATA["left"].shape[1:]  # rule slots and tiles per rule side; NUM_TILES / NUM_RULES index the padding
NUM_VARIANTS = DATA["glyphs"].shape[1]
FLAG = {name: i for i, name in enumerate(DATA["flags_names"])}
TILE, STEP = 5, 7  # a tile's size and the spacing of tiles in a row
BRACKET_GAP = 4  # rows between a tile and its cursor bracket
MARK = 9  # the black square the win animation leaves behind each rule tile of the last match
PAINT = 0  # the color the win animation paints matched tiles
BAR_LEFT, BAR_RIGHT = 1, 4  # the budget bar: the remaining fraction of the bottom row, and the rest
# the tile glyphs per family, variant and rotation (-1 = transparent), rotated as the original's sprite renderer does
GLYPHS = np.stack([[[np.rot90(glyph, k=(4 - r) % 4) for r in range(4)] for glyph in family] for family in DATA["glyphs"]])


@struct.dataclass
class State:
    level: jax.Array
    variant: jax.Array  # (T,) per tile slot
    cursor: jax.Array  # position in the answer row, or among all rule sides when the level's rules are editable
    budget: jax.Array
    won: jax.Array  # the last level was just solved: the frame shows the end of the win animation


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


def selection(level: jax.Array, cursor: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The (S,) tiles under the cursor (padded with NUM_TILES) and how many cursor positions the level has."""
    alter = level_data("flags", level)[FLAG["alter_rules"]]
    answer, left, right = (level_data(k, level) for k in ("answer", "left", "right"))
    sides = jnp.stack([left, right], 1).reshape(-1, SIDE)  # the cursor walks left side, right side, next rule...
    tiles = jnp.where(alter, sides[cursor], jnp.where(jnp.arange(SIDE) == 0, answer[cursor], NUM_TILES))
    count = jnp.where(alter, (sides < NUM_TILES).any(-1).sum(), (answer < NUM_TILES).sum())
    return tiles, count


def parse(level: jax.Array, variant: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The original's solved check, and what its win animation paints: whether the answer row is the source row rewritten
    left to right by the first usable rule matching at each position, the (T,) source and answer tiles that were matched,
    and the (T,) rule tiles of the last match."""
    flags = level_data("flags", level)
    double, tree = flags[FLAG["double_translation"]], flags[FLAG["tree_translation"]]
    name = jnp.append(level_data("family", level) * NUM_VARIANTS + variant, -1)  # what a tile is; -1 for the padding index
    source, answer, left, right = (level_data(k, level) for k in ("source", "answer", "left", "right"))
    left_name, right_name, source_name, answer_name = name[left], name[right], name[source], name[answer]
    left_len, right_len = (left < NUM_TILES).sum(-1), (right < NUM_TILES).sum(-1)
    real = left_len > 0
    offsets = jnp.arange(SIDE)
    # double translation: a rule's right side continues with the first rule whose left side it equals, compared over
    # the shorter of the two (the original zips them)
    overlap = jnp.minimum(right_len[:, None], left_len[None, :])
    chained = ((offsets >= overlap[..., None]) | (right_name[:, None] == left_name[None])).all(-1) & real[None]  # (R, R)
    chain = chained.argmax(-1)
    # tree translation: every tile of a rule's right side is replaced by the right side of the first rule starting with it
    looked = (right_name[:, :, None] == left_name[None, None, :, 0]) & real[None, None]  # (R, S, R)
    lookup, found = looked.argmax(-1), looked.any(-1)
    usable = real & jnp.where(tree, (found | (offsets >= right_len[:, None])).all(-1), ~double | chained.any(-1))
    # what the answer must show for each rule, as (R, S*S) names, and the partner rules it goes through (R, S)
    single = jnp.where(double, chain, jnp.arange(NUM_RULES))
    expect_single = jnp.pad(right_name[single], ((0, 0), (0, SIDE * SIDE - SIDE)), constant_values=-1)
    branch_len = jnp.where(offsets < right_len[:, None], right_len[lookup], 0)
    branch_at = jnp.cumsum(branch_len, -1) - branch_len
    expect_tree = jnp.full((NUM_RULES, SIDE * SIDE), -1)
    for j in range(SIDE):
        for k in range(SIDE):
            at = jnp.where(k < branch_len[:, j], branch_at[:, j] + k, SIDE * SIDE)
            expect_tree = expect_tree.at[jnp.arange(NUM_RULES), at].set(right_name[lookup[:, j], k], mode="drop")
    expect = jnp.where(tree, expect_tree, expect_single)
    expect_len = jnp.where(tree, branch_len.sum(-1), right_len[single])
    partners = jnp.where(tree, jnp.where(offsets < right_len[:, None], lookup, NUM_RULES), jnp.where(double & (offsets == 0), chain[:, None], NUM_RULES))
    left_ext, right_ext = (jnp.pad(a, ((0, 1), (0, 0)), constant_values=NUM_TILES) for a in (left, right))
    source_len = (source < NUM_TILES).sum()
    none = jnp.zeros(NUM_TILES, bool)

    def consume(_, carry):
        source_at, answer_at, failed, framed = carry
        ahead = source_name.at[source_at + offsets].get(mode="fill", fill_value=-1)
        matches = ((offsets >= left_len[:, None]) | (ahead == left_name)).all(-1) & usable
        rule = matches.argmax()
        have = answer_name.at[answer_at + jnp.arange(SIDE * SIDE)].get(mode="fill", fill_value=-1)
        ok = matches.any() & ((jnp.arange(SIDE * SIDE) >= expect_len[rule]) | (have == expect[rule])).all()
        active = ~failed & (source_at < source_len)
        used = jnp.concatenate([left[rule], right[rule], left_ext[partners[rule]].ravel(), right_ext[partners[rule]].ravel()])
        framed = jnp.where(active & ok, none.at[used].set(True, mode="drop"), framed)
        advance = active & ok
        return source_at + advance * left_len[rule], answer_at + advance * expect_len[rule], failed | (active & ~ok), framed

    _, answer_at, failed, framed = jax.lax.fori_loop(0, len(source), consume, (jnp.int32(0), jnp.int32(0), jnp.bool_(False), none))
    painted = none.at[source].set(True, mode="drop").at[jnp.where(jnp.arange(len(answer)) < answer_at, answer, NUM_TILES)].set(True, mode="drop")
    return ~failed, painted, framed


class Tr87(ArcGame):
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        return State(level, level_data("variant", level), jnp.int32(0), level_data("budget", level), jnp.bool_(False))

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        tiles, count = selection(state.level, state.cursor)
        moving = action >= 2  # the original's actions 1, 2 cycle the selection down, up; 3, 4 move the cursor left, right
        direction = jnp.where(action % 2 == 1, 1, -1)
        cursor = jnp.where(moving, (state.cursor + direction) % count, state.cursor)
        variant = jnp.where(moving, state.variant, state.variant.at[tiles].add(direction, mode="drop") % NUM_VARIANTS)
        solved = ~moving & parse(state.level, variant)[0]
        budget = state.budget - 1
        won = solved & (state.level == NUM_LEVELS - 1)
        lost = ~solved & (budget == 0)
        after = State(state.level, variant, cursor, budget, won)
        after = jax.tree.map(lambda a, b: jnp.where(solved & ~won, a, b), self.init_level(jnp.minimum(state.level + 1, NUM_LEVELS - 1)), after)
        return after, solved, lost, won

    def render(self, state: State) -> jax.Array:
        family, rotation, x, y = (level_data(k, state.level) for k in ("family", "rotation", "x", "y"))
        _, painted, framed = parse(state.level, state.variant)
        painted, framed = painted & state.won, framed & state.won
        frame = level_data("under", state.level).astype(jnp.int32)
        # the win animation's black squares lie between the two scenery layers (they overlap each other, hence min)
        rows, cols = (y - (MARK - TILE) // 2)[:, None, None] + jnp.arange(MARK)[None, :, None], (x - (MARK - TILE) // 2)[:, None, None] + jnp.arange(MARK)[None, None, :]
        frame = frame.at[rows, cols].min(jnp.broadcast_to(jnp.where(framed, PAINT, NUM_COLORS - 1)[:, None, None], rows.shape), mode="drop")
        over = level_data("over", state.level)
        frame = jnp.where(over >= 0, over, frame)
        glyph = jnp.asarray(GLYPHS)[family, state.variant, rotation]
        glyph = jnp.where(painted[:, None, None] & (glyph >= 0), PAINT, glyph)
        rows, cols = y[:, None, None] + jnp.arange(TILE)[None, :, None], x[:, None, None] + jnp.arange(TILE)[None, None, :]
        frame = frame.at[rows, cols].set(jnp.where(glyph >= 0, glyph, frame.at[rows, cols].get(mode="fill", fill_value=0)), mode="drop")
        # the cursor: a bracket above and below the selected tile(s), hidden once the puzzle is solved
        tiles, _ = selection(state.level, state.cursor)
        first, width = tiles[0], STEP * ((tiles < NUM_TILES).sum() - 1) + TILE
        rows, cols = jnp.mgrid[:SIZE, :SIZE]
        inside = (cols >= x[first]) & (cols < x[first] + width)
        edge = (cols == x[first]) | (cols == x[first] + width - 1)
        top, bottom = y[first] - BRACKET_GAP, y[first] + TILE - 1 + BRACKET_GAP
        bracket = ((rows == top) | (rows == bottom)) & inside | ((rows == top + 1) | (rows == bottom - 1)) & edge
        frame = jnp.where(bracket & ~state.won, 0, frame)
        # the HUD: the bottom row shows the remaining budget (rounded up to whole pixels)
        lit = jnp.arange(SIZE) < -(-SIZE * state.budget // level_data("budget", state.level))
        return frame.at[SIZE - 1].set(jnp.where(lit, BAR_LEFT, BAR_RIGHT))
