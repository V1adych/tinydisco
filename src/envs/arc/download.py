"""Downloads ARC-AGI-3 game sources into originals/<id>.py (gitignored; MIT): the official games as the official toolkit does,
and the community games of github.com/theredbluepill/arc-interactive: python src/envs/arc/download.py [id ...]
With ARC_API_KEY set (free at https://three.arcprize.org) more official games are visible than with the anonymous key."""

import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

BASE = "https://three.arcprize.org/api/games"
COMMUNITY = "https://api.github.com/repos/theredbluepill/arc-interactive/contents/environment_files"
COMMUNITY_REF = "02152bc4806869ace6ef167468970ca84bac0c5a"
OUT = Path(__file__).with_name("originals")


def get(url: str, key: str = "") -> bytes:
    with urlopen(Request(url, headers={"Accept": "application/json", "X-Api-Key": key}), timeout=30) as response:
        return response.read()


def community(name: str) -> bytes:
    """The game's source at the pinned commit; its folder holds one version subfolder."""
    version = json.loads(get(f"{COMMUNITY}/{name}?ref={COMMUNITY_REF}"))[0]["name"]
    return get(f"https://raw.githubusercontent.com/theredbluepill/arc-interactive/{COMMUNITY_REF}/environment_files/{name}/{version}/{name}.py")


if __name__ == "__main__":
    key = os.environ.get("ARC_API_KEY") or json.loads(get(f"{BASE}/anonkey"))["api_key"]
    official = {g["game_id"].split("-")[0]: g["game_id"] for g in json.loads(get(BASE, key))}
    wanted = sys.argv[1:] or sorted(official)
    OUT.mkdir(exist_ok=True)
    for name in wanted:
        source = get(f"{BASE}/{official[name]}/source", key) if name in official else community(name)
        (OUT / f"{name}.py").write_bytes(source)
        print(name, official.get(name, "community"))
