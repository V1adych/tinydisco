"""Downloads the ARC-AGI-3 game sources into originals/<id>.py (gitignored; MIT, ARC Prize Foundation), as the official
toolkit does: python src/envs/arc/download.py [id ...]. With ARC_API_KEY set (free at https://three.arcprize.org) more
games are visible than with the anonymous key."""

import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://three.arcprize.org/api/games"
OUT = Path(__file__).with_name("originals")


def get(path: str, key: str = "") -> bytes:
    with urlopen(Request(f"{BASE}{path}", headers={"Accept": "application/json", "X-Api-Key": key}), timeout=30) as response:
        return response.read()


if __name__ == "__main__":
    key = os.environ.get("ARC_API_KEY") or json.loads(get("/anonkey"))["api_key"]
    games = {g["game_id"].split("-")[0]: g["game_id"] for g in json.loads(get("", key))}
    wanted = sys.argv[1:] or sorted(games)
    OUT.mkdir(exist_ok=True)
    for name in wanted:
        (OUT / f"{name}.py").write_bytes(get(f"/{games[name]}/source", key))
        print(name, games[name])
