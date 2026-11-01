"""The ARC-AGI-3 game g50t (official games have ids, not names) in JAX, frame-exact to the original Python game.
The player is a square on a maze of 6-pixel cells; actions 1-4 move it one cell up, down, left, right, and reaching the
goal cell clears the level, 7 levels. Plates open doors while something stands on them (some doors toggle on each press
instead), or start a teleporter that swaps whatever stands on its two pads. Action 5 rewinds the player's moves back to
the start, and the run just undone becomes a ghost that replays it step by step alongside the next run, so a ghost can
hold a plate the player needs open; the icons top left count the runs, and the rewind after the last one clears the
ghosts. A patrol walks its own track (it turns at dead ends), and touching it or being under a closing door kills the
player. A line at the bottom shrinks every two actions; dying or the line running out loses the game. Rewinds undo the
patrols' and ghosts' moves too.
The original resolves an action over many sub-frames driven by an event queue (things move a pixel per frame, wires and
animations wait a few frames), and who gets crushed depends on where things are in that frame, so this port runs the
same queue frame by frame; the arrays come from extract.py and a frame here is the last frame the original renders."""

from pathlib import Path

import jax
import numpy as np
from flax import struct
from jax import numpy as jnp

from ..base import SIZE, ArcGame

DATA = dict(np.load(Path(__file__).with_name("g50t.npz")))
NUM_LEVELS = len(DATA["start"])
NB, NP, ND, NPL, NL, NPAD = (DATA[k].shape[1] for k in ("icon_pos", "patrol_start", "door_pos", "plate_door", "link_pads", "pad_pos"))
NA = NB + NP  # actors: the bodies (the player and its ghosts) first, then the patrols; bit a of an occupancy mask is actor a
# the entities the event queue holds, by id: bodies, patrols, doors, wires (one per plate), teleporter links, the marker
DOOR0, WIRE0 = NA, NA + ND
LINK0 = WIRE0 + NPL
MARKER = LINK0 + NL
BODY, PATROL, DOOR, WIRE, LINK, MARK = range(6)
KIND = np.array([BODY] * NB + [PATROL] * NP + [DOOR] * ND + [WIRE] * NPL + [LINK] * NL + [MARK])
CELL, CENTER = 6, 3  # moves are one cell; an actor's point for every test is its center pixel
PAD_BASE = 16  # trigger map values: plates 0.., pads PAD_BASE..
TIMER_WIDTH = int(DATA["timer_width"])
HISTORY = 2 * (TIMER_WIDTH + 1)  # the timer ends a level within this many actions, so no history grows longer
QUEUE = 64  # shortcut: event queue slots, far above what the levels reach (about 20); raise it if a new level chains more events
DEATH_FRAMES, LINK_WAIT, LINK_FRAMES, WIRE_WAIT, REWIND_WAIT = 9, 4, 14, 3, 3
MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])  # the original's actions 1-4: up, down, left, right, as (dx, dy)
PATROL_DIRS = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)])  # a patrol's heading index
KEY_STRIDE, COLORS = 1 << 16, 16
ACTOR_KEY = (int(DATA["actor_layer"]) + 16) * KEY_STRIDE  # draw key base of sprites the game adds later (new players, revived patrols)
FAR = -100


@struct.dataclass
class State:
    level: jax.Array
    pos: jax.Array  # (E, 2) x, y of each entity's sprite (bodies, patrols, doors, the marker)
    target: jax.Array  # (E, 2) where it is moving to
    delay: jax.Array  # (E,) frames it waits before acting
    instant: jax.Array  # (E,) its next move jumps to the target (rewinds move instantly)
    flag: jax.Array  # (E,) a door is open, a wire is on, a link is teleporting
    dying: jax.Array  # (E,) an actor's death animation runs
    anim: jax.Array  # (E,) frame of the death or teleport animation
    dead: jax.Array  # (E,)
    visible: jax.Array  # (E,) bodies and patrols shown (a dead or removed one is not)
    key: jax.Array  # (E,) draw key of bodies and patrols
    occupied: jax.Array  # (NPL + NPAD,) actor bitmask per plate, then per pad
    stuck: jax.Array  # (NPL,) held by a removed ghost: the original never takes it off the plate
    plate_on: jax.Array  # (NPL,)
    player: jax.Array  # the body slot that is the player
    ghosts: jax.Array  # (NB,) body slots of the ghosts, oldest first, -1 = none
    recording: jax.Array  # (NB, H, 2) per body slot, the moves a ghost replays
    recording_len: jax.Array  # (NB,)
    history: jax.Array  # (H, 2) the player's moves since the level (or the last rewind) started
    history_len: jax.Array
    snapshot: jax.Array  # (H, 2) the history when the rewind started: the next ghost's recording
    snapshot_len: jax.Array
    patrol_moves: jax.Array  # (NP, H, 2)
    patrol_len: jax.Array  # (NP,)
    patrol_dir: jax.Array  # (NP,)
    icon: jax.Array  # the current run's counter icon
    icon_pending: jax.Array  # the icons update once the marker arrives
    lit: jax.Array  # (NB,) per icon
    spent: jax.Array  # (NB,)
    rewinding: jax.Array
    more: jax.Array  # the rewind has more moves to undo (moves then go instantly)
    actions: jax.Array  # taken this level: the timer
    appended: jax.Array  # sprites the level's list has gained: the draw order of the next one
    queue: jax.Array  # (QUEUE,) entity ids, in the original's list order
    queue_len: jax.Array


def level_data(name: str, level: jax.Array) -> jax.Array:
    return jnp.asarray(DATA[name])[level]


def put(array: jax.Array, index, value, go=True) -> jax.Array:
    return array.at[index].set(jnp.where(go, value, array[index]))


def at(grid: jax.Array, point: jax.Array, fill) -> jax.Array:
    """grid[y, x] at point (x, y), fill outside the grid."""
    x, y = point[0], point[1]
    inside = (x >= 0) & (x < grid.shape[1]) & (y >= 0) & (y < grid.shape[0])
    return jnp.where(inside, grid[jnp.clip(y, 0, grid.shape[0] - 1), jnp.clip(x, 0, grid.shape[1] - 1)], fill)


def enqueue(s: State, e, go=True) -> State:
    return s.replace(queue=s.queue.at[s.queue_len].set(jnp.where(go, e, s.queue[jnp.minimum(s.queue_len, QUEUE - 1)]), mode="drop"), queue_len=s.queue_len + go)


def in_box(point: jax.Array, corner: jax.Array, size: int = 7) -> jax.Array:
    return ((point >= corner) & (point < corner + size)).all(-1)


def door_at(s: State, point: jax.Array) -> jax.Array:
    """Whether a door covers the point (doors are solid 7x7 squares)."""
    return in_box(point, s.pos[DOOR0:WIRE0]).any()


def kill(s: State, a, go) -> State:
    """The original's death: an actor shown and not yet dying starts its animation and vanishes; it is queued either way."""
    start = go & s.visible[a] & ~s.dying[a]
    s = s.replace(dying=put(s.dying, a, True, start), anim=put(s.anim, a, 0, start), visible=put(s.visible, a, False, start))
    return enqueue(s, a, go)


def swap(s: State, link, go) -> State:
    """A teleport: the two pads swap what stands on them, which jumps to its new pad."""
    pads = level_data("link_pads", s.level)[link]
    pad_pos = level_data("pad_pos", s.level)[pads]
    first, second = s.occupied[NPL + pads[0]], s.occupied[NPL + pads[1]]
    s = s.replace(occupied=put(put(s.occupied, NPL + pads[0], second, go), NPL + pads[1], first, go))
    for here, members in ((pad_pos[0], second), (pad_pos[1], first)):
        for a in range(NA):
            jump = go & ((members >> a) & 1 == 1)
            s = s.replace(pos=put(s.pos, a, here, jump), target=put(s.target, a, here, jump))
    return s


def door_signal(s: State, door, value, rewind, go) -> State:
    e = DOOR0 + jnp.maximum(door, 0)
    once = level_data("door_once", s.level)[jnp.maximum(door, 0)]
    go = go & (door >= 0) & ~(once & ~value)  # a toggling door ignores releases, and a press closes it when open
    value = value & ~(once & s.flag[e])
    go = go & (s.flag[e] != value)
    step = level_data("door_dir", s.level)[jnp.maximum(door, 0)] * jnp.where(value, CELL, -CELL)
    s = s.replace(flag=put(s.flag, e, value, go), target=put(s.target, e, s.pos[e] + step, go), instant=put(s.instant, e, rewind, go))
    s = s.replace(delay=put(s.delay, e, jnp.where(rewind & ~value, REWIND_WAIT, 0), go))
    return enqueue(s, e, go)


def link_signal(s: State, link, value, rewind, go) -> State:
    e = LINK0 + jnp.maximum(link, 0)
    go = go & (link >= 0)
    s = swap(s, jnp.maximum(link, 0), go & rewind & ~value)  # rewinding off a plate teleports at once
    start = go & ~rewind & value & ~s.flag[e]
    s = s.replace(flag=put(s.flag, e, True, start), delay=put(s.delay, e, LINK_WAIT, start), anim=put(s.anim, e, 0, start))
    return enqueue(s, e, start)


def press(s: State, a, value, rewind, go=True) -> State:
    """An actor steps on (value) or off whatever plate or pad its center is on; a plate passes changes on to its wire."""
    t = at(level_data("trigger", s.level), s.pos[a] + CENTER, -1).astype(jnp.int32)
    plate = go & (t >= 0) & (t < PAD_BASE)
    slot = jnp.where(t >= PAD_BASE, NPL + t - PAD_BASE, jnp.maximum(t, 0))
    bit = jnp.left_shift(1, a)
    occupied = jnp.where(value, s.occupied[slot] | bit, s.occupied[slot] & ~bit)
    s = s.replace(occupied=put(s.occupied, slot, occupied, go & (t >= 0)))
    p = jnp.minimum(slot, NPL - 1)
    on = s.plate_on[p]
    fire = plate & jnp.where((occupied != 0) | s.stuck[p], ~on, on)  # the plate is an OR of what stands on it
    s = s.replace(plate_on=put(s.plate_on, p, value, fire))
    # its wire: on a change, it signals its output and waits
    w = WIRE0 + p
    fire = fire & (s.flag[w] != value)
    s = s.replace(flag=put(s.flag, w, value, fire), delay=put(s.delay, w, jnp.where(rewind, jnp.where(value, 0, 2), WIRE_WAIT), fire))
    s = door_signal(s, level_data("plate_door", s.level)[p], value, rewind, fire)
    s = link_signal(s, level_data("plate_link", s.level)[p], value, rewind, fire)
    return enqueue(s, w, fire)


def step_actor(s: State, a, move, rewind, go=True) -> tuple[State, jax.Array]:
    """The original's move attempt: the cell must be floor and free of doors; the actor leaves its plate and is queued."""
    target = s.pos[a] + CELL * move
    s = s.replace(target=put(s.target, a, s.pos[a], go))  # the original's test parks the target on the position
    ok = go & at(level_data("floor", s.level), target + CENTER, False) & ~door_at(s, target + CENTER)
    s = press(s, a, False, rewind, ok)
    s = s.replace(target=put(s.target, a, target, ok))
    return enqueue(s, a, ok), ok


def move(s: State, step: jax.Array) -> State:
    s, ok = step_actor(s, s.player, step, s.more)
    s = s.replace(history=put(s.history, s.history_len, step, ok), history_len=s.history_len + ok)
    k = s.history_len - 1
    for rank in range(NB):  # the ghosts replay their recordings, each a frame later than the one before
        g = jnp.maximum(s.ghosts[rank], 0)
        go = ok & (s.ghosts[rank] >= 0) & (k < s.recording_len[g])
        s, _ = step_actor(s, g, s.recording[g, jnp.maximum(k, 0)], s.more, go)
        s = s.replace(delay=put(s.delay, g, rank + 2, go))
    for q in range(NP):
        a, go = NA - NP + q, ok & ~s.dead[NA - NP + q]
        for _ in range(4):  # ahead, left, right, back along its track; if blocked, turn around and look again
            path, heading = level_data("patrol_path", s.level)[q], s.patrol_dir[q]
            options = (heading + jnp.array([0, -1, 1, 2])) % 4
            fits = jnp.array([at(path, s.pos[a] + CELL * jnp.asarray(PATROL_DIRS)[o], False) for o in options])
            heading = jnp.where(fits.any(), options[fits.argmax()], heading)
            step = jnp.where(fits.any(), jnp.asarray(PATROL_DIRS)[heading], 0)
            s, moved = step_actor(s, a, step, s.more, go)
            s = s.replace(patrol_moves=s.patrol_moves.at[q].set(put(s.patrol_moves[q], s.patrol_len[q], step, moved)), patrol_len=s.patrol_len.at[q].add(moved))
            s = s.replace(patrol_dir=s.patrol_dir.at[q].set(jnp.where(go, jnp.where(moved, heading, (heading + 2) % 4), s.patrol_dir[q])))
            go = go & ~moved
    return s


def undo(s: State, rewind) -> State:
    """One rewind step: the last move of the player, and the matching ones of patrols and ghosts, reversed."""
    n = s.history_len - 1
    step = s.history[n]
    s = s.replace(history_len=n)
    for q in range(NP):
        a = NA - NP + q
        go = n < s.patrol_len[q]
        revive = go & ~s.visible[a]  # a dead patrol comes back, drawn above what the level had so far
        s = s.replace(visible=put(s.visible, a, True, revive), dead=put(s.dead, a, False, revive), key=put(s.key, a, ACTOR_KEY + s.appended, revive), appended=s.appended + revive)
        last = s.patrol_moves[q, jnp.maximum(s.patrol_len[q] - 1, 0)]
        s = s.replace(patrol_len=s.patrol_len.at[q].add(-go.astype(jnp.int32)))
        s, _ = step_actor(s, a, -last, rewind, go)
        s = s.replace(delay=put(s.delay, a, REWIND_WAIT, go), instant=put(s.instant, a, True, go))
    for rank in range(NB):
        g = jnp.maximum(s.ghosts[rank], 0)
        go = (s.ghosts[rank] >= 0) & (n < s.recording_len[g])
        s, _ = step_actor(s, g, -s.recording[g, n], rewind, go)
        s = s.replace(delay=put(s.delay, g, REWIND_WAIT, go), instant=put(s.instant, g, True, go))
    s, _ = step_actor(s, s.player, -step, rewind)
    return s.replace(delay=put(s.delay, s.player, REWIND_WAIT), instant=put(s.instant, s.player, True))


def rewind(s: State) -> State:
    go = s.history_len > 0
    s = s.replace(snapshot=s.history, snapshot_len=s.history_len, rewinding=s.rewinding | go)
    s = jax.lax.cond(go, lambda s: undo(s, s.more), lambda s: s, s)
    return s.replace(more=s.more | (go & (s.history_len > 0)))


def finish_rewind(s: State) -> State:
    """Back at the start: the run becomes a ghost and a fresh player appears, or after the last run the ghosts go."""
    start = level_data("start", s.level)
    icon = s.icon + 1
    s = s.replace(rewinding=False, pos=put(s.pos, s.player, start), target=put(s.target, s.player, start))
    wrap = icon == level_data("num_icons", s.level)
    is_ghost = (jnp.arange(NB)[:, None] == s.ghosts[None]).any(-1)
    ghost_bits = (is_ghost.astype(jnp.int32) << jnp.arange(NB)).sum()
    # removed ghosts stay on their plates in the original (nothing takes them off); on pads they no longer matter
    wrapped = s.replace(stuck=s.stuck | ((s.occupied[:NPL] & ghost_bits) != 0), occupied=s.occupied & ~ghost_bits, visible=s.visible.at[:NB].set(s.visible[:NB] & ~is_ghost), ghosts=jnp.full(NB, -1), icon=0)
    old = s.player
    used = is_ghost.at[old].set(True)
    new = used.argmin()
    fresh = dict(pos=put(s.pos, new, s.pos[old]), target=put(s.target, new, s.pos[old]), delay=put(s.delay, new, 0), instant=put(s.instant, new, False), dying=put(s.dying, new, False), anim=put(s.anim, new, 0))
    fresh |= dict(dead=put(s.dead, new, False), visible=put(s.visible, new, s.visible[old]), key=put(s.key, new, ACTOR_KEY + s.appended), appended=s.appended + 1)
    fresh |= dict(ghosts=s.ghosts.at[is_ghost.sum()].set(old), recording=s.recording.at[old].set(s.snapshot), recording_len=s.recording_len.at[old].set(s.snapshot_len), player=new, icon=icon)
    s = jax.tree.map(lambda a, b: jnp.where(wrap, a, b), wrapped, s.replace(**fresh))
    s = s.replace(target=put(s.target, MARKER, level_data("icon_pos", s.level)[s.icon]), icon_pending=True)
    return enqueue(s, MARKER)


def update_icons(s: State) -> State:
    r = s.icon
    lit = jnp.where(r == 0, False, s.lit.at[r - 1].set(False))
    spent = jnp.where(r == 0, False, s.spent.at[r - 1].set(True))
    return s.replace(lit=lit.at[r].set(True), spent=spent.at[r].set(False), icon_pending=False)


def settle(s: State) -> State:
    """The original's step once the queue has drained: icons, then everyone presses what it stands on, then the rewind."""

    def press_all(s: State) -> State:
        s = press(s, s.player, True, s.more)
        for q in range(NP):
            s = press(s, NA - NP + q, True, s.more)
        for rank in range(NB):
            s = press(s, jnp.maximum(s.ghosts[rank], 0), True, s.more, s.ghosts[rank] >= 0)
        go_on = (s.queue_len == 0) & s.rewinding
        undo_next = lambda s: (lambda s: s.replace(more=s.history_len > 0))(undo(s, True))
        return jax.lax.cond(go_on, lambda s: jax.lax.cond(s.more, undo_next, finish_rewind, s), lambda s: s, s)

    return jax.lax.cond(s.icon_pending, update_icons, press_all, s)


def tick(s: State, e) -> tuple[State, jax.Array]:
    """One queue entry's frame; False when it is done (the original's per-object step)."""
    kind = jnp.asarray(KIND)[e]
    waiting = s.delay[e] > 0
    actor = kind <= PATROL
    animating = actor & ~waiting & s.dying[e]
    linking = (kind == LINK) & ~waiting & s.flag[e]
    playing = linking & (s.anim[e] <= LINK_FRAMES)
    anim = s.anim[e] + (animating | playing)
    died = animating & (anim == DEATH_FRAMES)
    mover = ((kind <= DOOR) | (kind == MARK)) & ~waiting & ~animating
    pos, target = s.pos[e], s.target[e]
    jump = mover & s.instant[e] & ((pos != target).any() | (kind == DOOR))
    pos = jnp.where(jump, target, jnp.where(mover, pos + jnp.sign(target - pos), pos))
    still = (pos != target).any()
    s = s.replace(delay=s.delay.at[e].add(-waiting.astype(jnp.int32)), anim=s.anim.at[e].set(anim), dying=put(s.dying, e, False, died), dead=put(s.dead, e, True, died))
    s = s.replace(pos=s.pos.at[e].set(pos), instant=put(s.instant, e, False, jump))
    resting = mover & ~still
    # a patrol that comes to rest on the player kills it; so does a door that comes to rest closed, and it crushes patrols
    player = s.player
    touched = (kind == PATROL) & resting & at(jnp.asarray(DATA["player_art"]) >= 0, pos + CENTER - s.pos[player], False)
    crushing = (kind == DOOR) & resting & ~s.flag[e]
    s = kill(s, player, touched | (crushing & in_box(s.pos[player] + CENTER, pos)))
    for q in range(NP):
        s = kill(s, NA - NP + q, crushing & in_box(s.pos[NA - NP + q] + CENTER, pos))
    teleport = linking & ~playing
    s = swap(s, jnp.clip(e - LINK0, 0, NL - 1), teleport)
    s = s.replace(flag=put(s.flag, e, False, teleport))
    return s, waiting | animating | (mover & still) | playing


def frame(s: State) -> State:
    n = s.queue_len

    def entry(j, carry):
        s, keep = carry
        s, k = tick(s, s.queue[n - 1 - j])  # the original walks its list from the end
        return s, keep.at[n - 1 - j].set(k)

    s, keep = jax.lax.fori_loop(0, n, entry, (s, jnp.zeros(QUEUE, bool)))
    slots = jnp.arange(QUEUE)
    kept = jnp.where(slots < n, keep, slots < s.queue_len)  # what was queued during this frame stays behind the rest
    s = s.replace(queue=s.queue[jnp.argsort(~kept, stable=True)], queue_len=kept.sum())
    return jax.lax.cond(s.queue_len == 0, settle, lambda s: s, s)


class G50t(ArcGame):
    num_actions = 5
    num_levels = NUM_LEVELS

    def init_level(self, level: jax.Array) -> State:
        e = len(KIND)
        pos = jnp.full((e, 2), FAR).at[0].set(level_data("start", level)).at[NA - NP : NA].set(level_data("patrol_start", level))
        pos = pos.at[DOOR0:WIRE0].set(level_data("door_pos", level)).at[MARKER].set(level_data("marker_start", level))
        visible = jnp.zeros(e, bool).at[0].set(True).at[NA - NP : NA].set(level_data("patrol_start", level)[:, 0] != FAR)
        key = jnp.zeros(e, jnp.int32).at[0].set(level_data("player_key", level)).at[NA - NP : NA].set(level_data("patrol_key", level))
        no = jnp.zeros(e, bool)
        moves = jnp.zeros((HISTORY, 2), jnp.int32)
        return State(
            level, pos, pos, jnp.zeros(e, jnp.int32), no, no, no, jnp.zeros(e, jnp.int32), no, visible, key,
            jnp.zeros(NPL + NPAD, jnp.int32), jnp.zeros(NPL, bool), jnp.zeros(NPL, bool), jnp.int32(0), jnp.full(NB, -1),
            jnp.zeros((NB, HISTORY, 2), jnp.int32), jnp.zeros(NB, jnp.int32), moves, jnp.int32(0), moves, jnp.int32(0),
            jnp.zeros((NP, HISTORY, 2), jnp.int32), jnp.zeros(NP, jnp.int32), jnp.zeros(NP, jnp.int32),
            jnp.int32(0), jnp.bool_(False), jnp.zeros(NB, bool).at[0].set(True), jnp.zeros(NB, bool), jnp.bool_(False), jnp.bool_(False),
            jnp.int32(0), level_data("num_sprites", level), jnp.zeros(QUEUE, jnp.int32), jnp.int32(0),
        )

    def act(self, state: State, action: jax.Array) -> tuple[State, jax.Array, jax.Array, jax.Array]:
        s = jax.lax.cond(action < 4, lambda s: move(s, jnp.asarray(MOVES)[jnp.minimum(action, 3)]), rewind, state)
        s = s.replace(actions=s.actions + 1)
        s = jax.lax.while_loop(lambda s: s.queue_len > 0, frame, s)
        lost = (s.actions // 2 > TIMER_WIDTH) | s.dead[s.player]
        advanced = ~lost & (s.pos[s.player] == level_data("win", s.level)).all()
        won = advanced & (s.level == NUM_LEVELS - 1)
        after = jax.tree.map(lambda a, b: jnp.where(advanced & ~won, a, b), self.init_level(jnp.minimum(s.level + 1, NUM_LEVELS - 1)), s)
        return after, advanced, lost, won

    def render(self, state: State) -> jax.Array:
        s = state
        rows, cols = jnp.mgrid[:SIZE, :SIZE]

        def paint(art, corner, key, shown=True):
            """The sprite as draw-key-tagged colors over the screen (key * 16 + color, -1 where transparent)."""
            r, c = rows - corner[1], cols - corner[0]
            inside = (r >= 0) & (r < art.shape[0]) & (c >= 0) & (c < art.shape[1])
            color = art[jnp.clip(r, 0, art.shape[0] - 1), jnp.clip(c, 0, art.shape[1] - 1)]
            return jnp.where(inside & (color >= 0) & shown, key * COLORS + color, -1)

        def static(planes, keys, shown=True):
            return jnp.where((planes >= 0) & shown, keys * COLORS + planes, -1).max(0)

        layers = [static(level_data("planes", s.level), level_data("keys", s.level)[:, None, None])]
        shown = jnp.stack([s.lit, s.spent], -1)[..., None, None]
        layers.append(static(level_data("overlay_planes", s.level).reshape(-1, SIZE, SIZE), level_data("overlay_keys", s.level).reshape(-1, 1, 1), shown.reshape(-1, 1, 1)))
        is_ghost = (jnp.arange(NB)[:, None] == s.ghosts[None]).any(-1)
        for b in range(NB):
            art = jnp.where(is_ghost[b], jnp.asarray(DATA["ghost_art"]), jnp.asarray(DATA["player_art"]))
            layers.append(paint(art, s.pos[b], s.key[b], s.visible[b]))
        for q in range(NP):
            layers.append(paint(jnp.asarray(DATA["patrol_art"]), s.pos[NA - NP + q], s.key[NA - NP + q], s.visible[NA - NP + q]))
        for d in range(ND):
            layers.append(paint(level_data("door_art", s.level)[d], s.pos[DOOR0 + d], level_data("door_key", s.level)[d]))
        layers.append(paint(jnp.asarray(DATA["marker_art"]), s.pos[MARKER], level_data("marker_key", s.level)))
        timer = jnp.full((1, TIMER_WIDTH), DATA["timer_color"], jnp.int8)
        layers.append(paint(timer, jnp.array([-(s.actions // 2), SIZE - 1]), level_data("timer_key", s.level)))
        top = jnp.stack(layers).max(0)
        return jnp.where(top >= 0, top % COLORS, DATA["background"])
