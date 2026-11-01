"""Turns the original ARC-AGI-3 game tr87 (originals/tr87.py, run with the arcengine package) into the arrays game.py plays on:
python src/modules/envs/arc/games/tr87/extract.py  -> tr87.npz next to it.
Per level: the fixed scenery pre-rendered into two 64x64 planes (the layer -2 background and arrows with the camera's
background color baked in, and the layer -1 squares behind the tiles, -1 = transparent), every tile's family, starting
variant, position and rotation (the seeded random rotation is applied by the original's own level setup), and the
structure the logic needs as tile indices: the source row, the answer row and each rule's left and right side. The 21
tile glyphs are stored unrotated. Everything is padded to the largest level with tile index = tile count."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).parents[2] / "originals/tr87.py"
OUT = Path(__file__).with_name("tr87.npz")
SIZE = 64
TILE_TAG, FAMILIES, NUM_VARIANTS = "nxkictbbvzt", "ABC", 7  # tiles are named <tag><family letter><variant digit 1-7>
FLAGS = ("double_translation", "tree_translation", "alter_rules")


def load(source: Path):
    spec = importlib.util.spec_from_file_location("tr87_original", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tr87_original"] = module
    spec.loader.exec_module(module)
    return module


def draw(plane: np.ndarray, sprite) -> None:
    pixels = sprite.render()
    h, w = pixels.shape
    y0, x0 = max(sprite.y, 0), max(sprite.x, 0)
    y1, x1 = min(sprite.y + h, SIZE), min(sprite.x + w, SIZE)
    region = pixels[y0 - sprite.y : y1 - sprite.y, x0 - sprite.x : x1 - sprite.x]
    plane[y0:y1, x0:x1] = np.where(region >= 0, region, plane[y0:y1, x0:x1])  # the camera treats negative values as transparent


def extract(game, module) -> dict:
    levels = []
    for i in range(len(module.levels)):
        game.set_level(i)
        level = game.current_level
        tiles = sorted(level.get_sprites_by_tag(TILE_TAG), key=lambda s: (s.y, s.x))
        index = {id(s): k for k, s in enumerate(tiles)}
        scenery = [s for s in level.get_sprites() if TILE_TAG not in s.tags and s not in game.qvtymdcqear_parts]
        assert all(s.is_visible and s.layer in (-2, -1) for s in scenery)
        under, over = np.full((SIZE, SIZE), module.BACKGROUND_COLOR, np.int8), np.full((SIZE, SIZE), -1, np.int8)
        for s in scenery:  # list order within a layer is the draw order
            draw(under if s.layer == -2 else over, s)
        levels.append(dict(
            under=under,
            over=over,
            family=np.array([FAMILIES.index(s.name[-2]) for s in tiles], np.int32),
            variant=np.array([int(s.name[-1]) - 1 for s in tiles], np.int32),
            x=np.array([s.x for s in tiles], np.int32),
            y=np.array([s.y for s in tiles], np.int32),
            rotation=np.array([s.rotation // 90 for s in tiles], np.int32),
            source=np.array([index[id(s)] for s in game.zvojhrjxxm], np.int32),
            answer=np.array([index[id(s)] for s in game.ztgmtnnufb], np.int32),
            left=[[index[id(s)] for s in left] for left, _ in game.cifzvbcuwqe],
            right=[[index[id(s)] for s in right] for _, right in game.cifzvbcuwqe],
            flags=np.array([bool(level.get_data(f)) for f in FLAGS]),
            budget=np.int32(game.vfpimnmtnta),
        ))
    num_tiles = max(len(lv["x"]) for lv in levels)
    num_rules = max(len(lv["left"]) for lv in levels)
    side = max(len(s) for lv in levels for s in lv["left"] + lv["right"])
    out = {}
    for key in levels[0]:
        arrays = [lv[key] for lv in levels]
        if key in ("left", "right"):
            padded = np.full((len(levels), num_rules, side), num_tiles, np.int32)
            for l, sides in enumerate(arrays):
                for r, tiles in enumerate(sides):
                    padded[l, r, : len(tiles)] = tiles
            out[key] = padded
        elif key in ("source", "answer"):
            n = max(len(a) for a in arrays)
            out[key] = np.stack([np.concatenate([a, np.full(n - len(a), num_tiles, a.dtype)]) for a in arrays])
        elif key in ("family", "variant", "x", "y", "rotation"):
            pad = -2 * SIZE if key in ("x", "y") else 0  # padding tiles lie off screen
            out[key] = np.stack([np.concatenate([a, np.full(num_tiles - len(a), pad, a.dtype)]) for a in arrays])
        else:
            out[key] = np.stack(arrays)
    out["flags_names"] = np.array(FLAGS)
    out["glyphs"] = np.stack([[np.array(module.sprites[f"{TILE_TAG}{f}{v}"].pixels, np.int8) for v in range(1, NUM_VARIANTS + 1)] for f in FAMILIES])
    return out


if __name__ == "__main__":
    module = load(SOURCE)
    data = extract(module.Tr87(), module)
    np.savez_compressed(OUT, **data)
    print(f"{OUT}: {len(data['x'])} levels, {data['x'].shape[1]} tile slots, {data['left'].shape[1]}x{data['left'].shape[2]} rule slots, {OUT.stat().st_size // 1024} KB")
