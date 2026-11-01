"""Turns the community ARC-AGI-3 game rs01 (originals/rs01.py, run with the arcengine package) into the arrays game.py plays
on:  python src/modules/envs/arc/games/rs01/extract.py  -> rs01.npz next to it.
Per level, on a grid padded to the largest level (GRID x GRID): the player's start, the targets' cells and colors in
sprite-list order (padded with GRID, which every lookup drops), the wall mask, the safe-color cycle (padded with -1), its
interval, the difficulty the HUD shows, and the camera's grid -> frame mapping (row * GRID + column, -1 on the letterbox)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).parents[2] / "originals/rs01.py"
OUT = Path(__file__).with_name("rs01.npz")
SIZE = 64


def load(source: Path):
    spec = importlib.util.spec_from_file_location("rs01_original", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rs01_original"] = module
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
    count = max(len(level.get_sprites_by_tag("target")) for level in module.levels)
    cycle = max(len(level.get_data("safe_colors")) for level in module.levels)
    levels = []
    for i in range(len(module.levels)):
        game.set_level(i)
        level = game.current_level
        targets = [[t.x, t.y] for t in game._targets]
        walls = np.zeros((grid, grid), bool)
        for w in level.get_sprites_by_tag("wall"):
            walls[w.y, w.x] = True
        safe = list(level.get_data("safe_colors"))
        levels.append(dict(
            grid_size=np.array(level.grid_size, np.int32),
            start=np.array([game._player.x, game._player.y], np.int32),
            targets=np.array(targets + [[grid, grid]] * (count - len(targets)), np.int32),
            target_colors=np.array([int(t.pixels[0, 0]) for t in game._targets] + [-1] * (count - len(targets)), np.int32),
            real=np.arange(count) < len(targets),
            walls=walls,
            safe_colors=np.array(safe + [-1] * (cycle - len(safe)), np.int32),
            num_safe=np.int32(len(safe)),
            cycle_interval=np.int32(level.get_data("cycle_interval")),
            difficulty=np.int32(level.get_data("difficulty")),
            cells=pixel_cells(game.camera, grid),
        ))
    out = {k: np.stack([lv[k] for lv in levels]) for k in levels[0]}
    out["colors"] = np.array([int(module.sprites[name].pixels[0, 0]) for name in ("player", "wall")], np.int32)
    out["background"], out["padding"] = np.int32(module.BACKGROUND_COLOR), np.int32(module.PADDING_COLOR)
    return out


if __name__ == "__main__":
    module = load(SOURCE)
    data = extract(module.Rs01(), module)
    np.savez_compressed(OUT, **data)
    print(f"{OUT}: {len(data['start'])} levels, {OUT.stat().st_size} bytes")
