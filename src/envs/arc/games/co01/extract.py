"""Turns the community ARC-AGI-3 game co01 (originals/co01.py, run with the arcengine package) into the arrays game.py plays
on:  python src/envs/arc/games/co01/extract.py  -> co01.npz next to it.
Per level (all 10x10): the player's start, the goal, pads and doors as items in sprite-list order (cell, kind, hue, color;
padded with cells at GRID, which every lookup drops), the wall mask, and the camera's grid -> frame mapping
(row * GRID + column, -1 on the letterbox)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).parents[2] / "originals/co01.py"
OUT = Path(__file__).with_name("co01.npz")
SIZE = 64
KINDS = ("goal", "recolor", "door")  # item kinds, by tag


def load(source: Path):
    spec = importlib.util.spec_from_file_location("co01_original", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["co01_original"] = module
    spec.loader.exec_module(module)
    return module


def pixel_cells(camera, grid: int) -> np.ndarray:
    """The grid cell (row * grid + column) each frame pixel shows, -1 = letterbox; grid is the padded grid's side."""
    scale, x0, y0 = camera._calculate_scale_and_offset()
    rows, cols = np.mgrid[:SIZE, :SIZE]
    gy, gx = (rows - y0) // scale, (cols - x0) // scale
    inside = (rows >= y0) & (cols >= x0) & (gy < camera.height) & (gx < camera.width)
    return np.where(inside, gy * grid + gx, -1).astype(np.int32)


def extract(game, module) -> dict:
    grid = max(max(level.grid_size) for level in module.levels)
    count = max(sum("wall" not in s.tags and "player" not in s.tags for s in level.get_sprites()) for level in module.levels)
    levels = []
    for i in range(len(module.levels)):
        game.set_level(i)
        level = game.current_level
        is_wall = ["wall" in s.tags for s in level.get_sprites() if "player" not in s.tags]
        assert is_wall == sorted(is_wall), "game.py draws (and looks up) walls after every other sprite"
        items = [s for s in level.get_sprites() if "wall" not in s.tags and "player" not in s.tags]
        pad = count - len(items)
        walls = np.zeros((grid, grid), bool)
        for w in level.get_sprites_by_tag("wall"):
            walls[w.y, w.x] = True
        levels.append(dict(
            start=np.array([game._player.x, game._player.y], np.int32),
            items=np.array([[s.x, s.y] for s in items] + [[grid, grid]] * pad, np.int32),
            kinds=np.array([next(k for k, tag in enumerate(KINDS) if tag in s.tags) for s in items] + [0] * pad, np.int32),
            hues=np.array([8 if "c8" in s.tags else 11 for s in items] + [0] * pad, np.int32),
            item_colors=np.array([int(s.pixels[0, 0]) for s in items] + [0] * pad, np.int32),
            walls=walls,
            cells=pixel_cells(game.camera, grid),
        ))
    out = {k: np.stack([lv[k] for lv in levels]) for k in levels[0]}
    out["grid_size"] = np.array([level.grid_size for level in module.levels], np.int32)
    out["colors"] = np.array([int(module.sprites[name].pixels[0, 0]) for name in ("player", "wall")], np.int32)
    out["background"], out["padding"] = np.int32(module.BACKGROUND_COLOR), np.int32(module.PADDING_COLOR)
    return out


if __name__ == "__main__":
    module = load(SOURCE)
    data = extract(module.Co01(), module)
    np.savez_compressed(OUT, **data)
    print(f"{OUT}: {len(data['start'])} levels, {OUT.stat().st_size} bytes")
