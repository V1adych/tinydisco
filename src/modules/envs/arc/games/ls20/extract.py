"""Turns the original ARC-AGI-3 game ls20 (originals/ls20.py, version cb3b57cc, run with the arcengine package) into the arrays
game.py plays on:  python src/modules/envs/arc/games/ls20/extract.py  -> ls20.npz next to it.
Every sprite of every level is pre-rendered at its position into a 64x64 plane (-1 = transparent), sorted by draw order
(layer, then list order); the game's own numbers (budget, fog, the symbol each lock wants, the starting symbol) and the
cross-references the logic needs (which plane is a lock, its display, its hint boxes) come out as index arrays, padded
to the largest level. The player and the carried-symbol display are the only sprites drawn at run time."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).parents[2] / "originals/ls20.py"
OUT = Path(__file__).with_name("ls20.npz")
SIZE = 64
# what a sprite is to the game, by its (obfuscated) tag; every plane gets one of these
ROLES = ("other", "wall", "lock", "display", "pickup", "shape_tile", "color_tile", "rotate_tile", "border", "highlight", "box", "lock_box", "player", "carried")
TAG_ROLE = {"jdd": "wall", "mae": "lock", "axa": "display", "iri": "pickup", "gsu": "shape_tile", "gic": "color_tile", "bgt": "rotate_tile",
            "nfq": "border", "fng": "highlight", "yar": "box", "qex": "lock_box", "caf": "player", "wex": "carried"}


def load(source: Path):
    spec = importlib.util.spec_from_file_location("ls20_original", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ls20_original"] = module
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


def role_of(sprite) -> str:
    for tag in sprite.tags:
        if tag in TAG_ROLE:
            return TAG_ROLE[tag]
    return "other"


def extract(game, module) -> dict:
    levels = []
    for i in range(len(module.levels)):
        game.set_level(i)
        level = game.current_level
        sprites = [s for s in level.get_sprites() if s is not game.egb]  # the flash sprite is never in a frame an agent sees
        order = sorted(range(len(sprites)), key=lambda k: (sprites[k].layer, k))
        sprites = [sprites[k] for k in order]
        index = {id(s): k for k, s in enumerate(sprites)}
        locks = list(game.qqv)  # the game's lock list, in its own order (its per-lock data follows it)
        at = lambda x, y, tag: level.get_sprite_at(x, y, tag)
        box = [at(s.x - 1, s.y - 1, "yar") for s in locks]
        levels.append(dict(
            planes=np.stack([plane(s) for s in sprites]),
            role=np.array([ROLES.index(role_of(s)) for s in sprites], np.int32),
            visible=np.array([s.is_visible for s in sprites]),
            x=np.array([s.x for s in sprites], np.int32),
            y=np.array([s.y for s in sprites], np.int32),
            lock=np.array([index[id(s)] for s in locks], np.int32),
            lock_display=np.array([index[id(s)] for s in game.pca], np.int32),
            lock_box=np.array([index[id(b)] for b in box], np.int32),
            lock_box_clears=np.array(["vdr" in b.tags for b in box]),
            lock_hint=np.array([index[id(at(s.x - 1, s.y - 1, "qex"))] for s in locks], np.int32),
            lock_wants=np.array(list(zip(game.gfy, game.vxy, game.cjl)), np.int32),  # shape, color, rotation indices
            budget=np.int32(level.get_data("vxy")),
            fog=np.bool_(level.get_data("kdy")),
            start_symbol=np.array([game.snw, game.tmx, game.tuv], np.int32),
            start=np.array([game.mgu.x, game.mgu.y], np.int32),
        ))
    num_planes = max(len(lv["planes"]) for lv in levels)
    num_locks = max(len(lv["lock"]) for lv in levels)
    out = {}
    for key in levels[0]:
        arrays = [lv[key] for lv in levels]
        if key in ("planes", "role", "visible", "x", "y"):
            n = num_planes
        elif key.startswith("lock"):
            n = num_locks
        else:
            out[key] = np.stack(arrays)
            continue
        # padding: transparent planes, and sprite indices one past the last slot (the game indexes them with drop/fill modes)
        pad_value = -1 if key == "planes" else num_planes if key in ("lock", "lock_display", "lock_box", "lock_hint") else 0
        out[key] = np.stack([np.concatenate([a, np.full((n - len(a),) + a.shape[1:], pad_value, a.dtype)]) for a in arrays])
    out["roles"] = np.array(ROLES)
    out["shapes"] = np.stack([np.array(s.pixels, np.int8) for s in game.hep])  # (6, 3, 3), 0 = ink, -1 = transparent
    out["palette"] = np.array(game.hul, np.int32)
    out["player"] = np.array(game.mgu.pixels, np.int8)
    out["background"] = np.int32(module.BACKGROUND_COLOR)
    return out


if __name__ == "__main__":
    module = load(SOURCE)
    data = extract(module.Ls20(), module)
    np.savez_compressed(OUT, **data)
    print(f"{OUT}: {len(data['planes'])} levels, {data['planes'].shape[1]} sprite slots, {OUT.stat().st_size // 1024} KB")
