"""Turns the original ARC-AGI-3 game g50t (originals/g50t.py, run with the arcengine package) into the arrays game.py
plays on:  python src/modules/envs/arc/games/g50t/extract.py  -> g50t.npz next to it.
Every sprite that never moves or changes is pre-rendered into a 64x64 plane (-1 = transparent) with its draw key (layer,
then position in the level's sprite list); the moving ones (player, ghosts, patrols, doors, the ghost-counter marker,
the timer line, the counter overlays) are drawn at run time from their art. The point tests the logic needs come out as
64x64 maps computed with the game's own lookups (which treat the -2 padding of sprites as solid): the walkable floor,
each patrol's path, and the plate or pad a point triggers. Plates, doors, pads and teleporter links come out as index
arrays, padded to the largest level."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).parents[2] / "originals/g50t.py"
OUT = Path(__file__).with_name("g50t.npz")
SIZE = 64
LAYER_OFFSET, KEY_STRIDE = 16, 1 << 16  # draw key = (layer + 16) * 2**16 + position in the level's sprite list
FAR = -100  # the position of a padded (absent) object, off screen and away from every point test
PAD_BASE = 16  # in the trigger map: plates are 0.., pads PAD_BASE.., nothing -1


def load(source: Path):
    spec = importlib.util.spec_from_file_location("g50t_original", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["g50t_original"] = module
    spec.loader.exec_module(module)
    return module


def plane(sprite) -> np.ndarray:
    out = np.full((SIZE, SIZE), -1, np.int8)
    pixels = sprite.render()
    h, w = pixels.shape
    y0, x0 = max(sprite.y, 0), max(sprite.x, 0)
    y1, x1 = min(sprite.y + h, SIZE), min(sprite.x + w, SIZE)
    region = pixels[y0 - sprite.y : y1 - sprite.y, x0 - sprite.x : x1 - sprite.x]
    out[y0:y1, x0:x1] = np.where(region >= 0, region, -1)  # the camera treats every negative value as transparent
    return out


def mask(hit) -> np.ndarray:
    """mask[y, x] = hit(x, y), the game's own point test over the whole screen."""
    return np.array([[hit(x, y) for x in range(SIZE)] for y in range(SIZE)])


def extract(game, module) -> dict:
    levels = []
    for i in range(len(module.levels)):
        game.set_level(i)
        world = game.vgwycxsxjz
        sprites = game.current_level.get_sprites()
        index = {id(s): k for k, s in enumerate(sprites)}
        key = lambda s: (s.layer + LAYER_OFFSET) * KEY_STRIDE + index[id(s)]
        doors, patrols, icons = world.uwxkstolmf, list(world.kgvnkyaimw), world.drofvwhbxb
        plates = [o for o in world.hamayflsib if isinstance(o, module.lqtxaumfed)]
        pads = [o for o in world.hamayflsib if isinstance(o, module.ulhhdeoyok)]
        assert world.hamayflsib == plates + pads, "plates come before pads in the trigger order"
        assert all(pl.nexhtmlmxh is not None and len(pl.nexhtmlmxh.ytztewxdin) == 1 for pl in plates), "a plate drives one output"
        assert len({id(pl.nexhtmlmxh) for pl in plates}) == len(plates), "a wire has one plate (wire state lives with its plate)"
        outputs = [pl.nexhtmlmxh.ytztewxdin[0] for pl in plates]
        links = [o for o in outputs if isinstance(o, module.crfcpstubm)]
        assert all(link.jvabfgsorb and link.xixykigflg for link in links), "every link joins two pads"
        assert all(p.baygqyisjz is not None for p in patrols), "every patrol has a path"
        assert all((d.xmdjbwrpmv.render() >= 0).all() and d.width == d.height == 7 for d in doors), "doors are solid 7x7 squares"
        moving = {id(world.dzxunlkwxt.xmdjbwrpmv), id(world.zmqoxwsfgh.xmdjbwrpmv), id(game.twyixucrqi)}
        moving |= {id(o.xmdjbwrpmv) for o in doors + patrols} | {id(s) for ic in icons for s in (ic.qhrqjofewa, ic.wypkgqewkd)}
        static = [s for s in sprites if id(s) not in moving and s.is_visible]

        def trigger(x, y):
            for k, o in enumerate(world.hamayflsib):
                if o.esidlbhbhw(x, y) != -1:
                    return k if k < len(plates) else PAD_BASE + k - len(plates)
            return -1

        levels.append(dict(
            planes=np.stack([plane(s) for s in static]),
            keys=np.array([key(s) for s in static], np.int32),
            floor=mask(lambda x, y: world.afbbgvkpip.esidlbhbhw(x, y) != -1),
            trigger=mask(trigger).astype(np.int8),
            start=np.array([world.dzxunlkwxt.x, world.dzxunlkwxt.y], np.int32),
            win=np.array([world.whftgckbcu.x + 1, world.whftgckbcu.y + 1], np.int32),  # the player position that clears the level
            num_sprites=np.int32(len(sprites)),
            player_key=np.int32(key(world.dzxunlkwxt.xmdjbwrpmv)),
            marker_key=np.int32(key(world.zmqoxwsfgh.xmdjbwrpmv)),
            marker_start=np.array([world.zmqoxwsfgh.x, world.zmqoxwsfgh.y], np.int32),
            timer_key=np.int32(key(game.twyixucrqi)),
            num_icons=np.int32(len(icons)),
            icon_pos=np.array([[ic.x, ic.y] for ic in icons], np.int32),
            overlay_planes=np.stack([[plane(ic.qhrqjofewa), plane(ic.wypkgqewkd)] for ic in icons]),  # lit, then spent
            overlay_keys=np.array([[key(ic.qhrqjofewa), key(ic.wypkgqewkd)] for ic in icons], np.int32),
            door_pos=np.array([[d.x, d.y] for d in doors], np.int32).reshape(-1, 2),
            door_dir=np.array([d.hluvhlvimq() for d in doors], np.int32).reshape(-1, 2),
            door_once=np.array([d.dpdubazedr for d in doors], bool),
            door_key=np.array([key(d.xmdjbwrpmv) for d in doors], np.int32),
            door_art=np.stack([d.xmdjbwrpmv.render() for d in doors]).astype(np.int8),
            plate_door=np.array([doors.index(o) if o in doors else -1 for o in outputs], np.int32),
            plate_link=np.array([links.index(o) if o in links else -1 for o in outputs], np.int32),
            pad_pos=np.array([[p.x, p.y] for p in pads], np.int32).reshape(-1, 2),
            link_pads=np.array([[pads.index(l.jvabfgsorb), pads.index(l.xixykigflg)] for l in links], np.int32).reshape(-1, 2),
            patrol_start=np.array([[p.x, p.y] for p in patrols], np.int32).reshape(-1, 2),
            patrol_key=np.array([key(p.xmdjbwrpmv) for p in patrols], np.int32),
            patrol_path=np.stack([mask(p.mvbhoupyur) for p in patrols]) if patrols else np.zeros((0, SIZE, SIZE), bool),
        ))
    out = {}
    for name in levels[0]:
        arrays = [lv[name] for lv in levels]
        if not arrays[0].ndim or name in ("floor", "trigger", "start", "win", "marker_start"):
            out[name] = np.stack(arrays)
            continue
        n = max(len(a) for a in arrays)
        padded = []
        for a in arrays:
            pad = np.zeros((n - len(a),) + a.shape[1:], a.dtype)
            # padding: transparent planes, far-away positions, and wiring indices -1 (no output, which the game drops)
            if name in ("planes", "door_art", "overlay_planes", "plate_door", "plate_link"):
                pad[...] = -1
            elif name.endswith("_pos") or name.endswith("_start"):
                pad[...] = FAR
            padded.append(np.concatenate([a, pad]))
        out[name] = np.stack(padded)
    player = game.vgwycxsxjz.dzxunlkwxt
    out["player_art"] = np.array(player.xmdjbwrpmv.pixels, np.int8)
    ghost = player.clone()
    ghost.bwuyldyhdx()  # the game's own recoloring of a player that becomes a ghost
    out["ghost_art"] = np.array(ghost.xmdjbwrpmv.pixels, np.int8)
    out["patrol_art"] = np.array(module.sprites["vtwcsmdoqp"].pixels, np.int8)
    out["marker_art"] = np.array(game.vgwycxsxjz.zmqoxwsfgh.xmdjbwrpmv.pixels, np.int8)
    timer = game.twyixucrqi.pixels
    assert timer.shape[0] == 1 and (timer == timer[0, 0]).all(), "the timer is a one-color line"
    out["timer_color"], out["timer_width"] = np.int32(timer[0, 0]), np.int32(timer.shape[1])
    out["actor_layer"] = np.int32(player.xmdjbwrpmv.layer)
    assert module.sprites["vtwcsmdoqp"].layer == player.xmdjbwrpmv.layer, "revived patrols and new players share a layer"
    out["background"] = np.int32(module.BACKGROUND_COLOR)
    return out


if __name__ == "__main__":
    module = load(SOURCE)
    data = extract(module.G50t(), module)
    np.savez_compressed(OUT, **data)
    print(f"{OUT}: {len(data['planes'])} levels, {data['planes'].shape[1]} static planes, {OUT.stat().st_size // 1024} KB")
