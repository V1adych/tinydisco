"""Rents a GPU that runs the given command in the repo and deletes itself when it exits, e.g.
    python launch.py python src/disco.py --track
PROVIDER=vast (default) rents the cheapest acceptable Vast.ai offer; PROVIDER=runpod a RunPod pod (CLOUD=SECURE when community is out of stock).
GPU names the card in the provider's own naming, e.g. GPU="RTX 5090" or GPU="NVIDIA GeForce RTX 5090".
Needs WANDB_API_KEY, GITHUB_DEPLOY_KEY (base64) and the provider's VASTAI_API_KEY or RUNPOD_API_KEY in the environment."""

import json
import os
import sys
import urllib.request
from pathlib import Path

IMAGE = "nvidia/cuda:12.8.1-base-ubuntu24.04"  # tiny; start.sh installs everything else
DISK_GB = 30
# Vast: verified, reliable hosts with a fast link and enough cores for XLA's compile; hosts in China could not download our dependencies
VAST_OFFER_FILTER = {
    "num_gpus": {"eq": 1},
    "verified": {"eq": True},
    "rentable": {"eq": True},
    "reliability": {"gte": 0.99},
    "cuda_max_good": {"gte": 12.8},
    "inet_down": {"gte": 500},
    "disk_space": {"gte": 40},
    "cpu_cores_effective": {"gte": 8},
    "type": "on-demand",
    "order": [["dph_total", "asc"]],
    "limit": 8,
}


def call(url: str, key: str, body: dict, method: str = "POST") -> dict:
    request = urllib.request.Request(url, method=method, data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # the provider's explanation is in the body, which the exception alone doesn't show
        return json.loads(error.read())


def vast(script: str, env: dict):
    key = os.environ["VASTAI_API_KEY"]
    offers = call("https://console.vast.ai/api/v0/bundles/", key, VAST_OFFER_FILTER | {"gpu_name": {"eq": os.environ.get("GPU", "RTX 4090")}})["offers"]
    body = {"image": IMAGE, "disk": DISK_GB, "runtype": "ssh", "label": "meta-rl", "onstart": script, "env": env}
    for offer in offers:  # listings go stale within minutes, so the first offer may already be gone
        if "CN" in str(offer.get("geolocation")):
            continue
        result = call(f"https://console.vast.ai/api/v0/asks/{offer['id']}/", key, body, method="PUT")
        if result.get("success"):
            return f"instance {result['new_contract']}: {offer['gpu_name']} in {offer['geolocation']} at ${offer['dph_total']:.3f}/h"
        print(f"offer {offer['id']}: {result.get('msg')}")
    sys.exit("no offer accepted")


def runpod(script: str, env: dict):
    body = {
        "name": "meta-rl",
        "cloud": os.environ.get("CLOUD", "COMMUNITY"),
        "gpu": {"id": os.environ.get("GPU", "NVIDIA GeForce RTX 4090"), "count": 1, "minCudaVersion": "12.8"},
        "image": IMAGE,
        "entrypoint": ["/bin/bash", "-c"],
        "cmd": [script],
        "disk": DISK_GB,
        "env": env,
    }
    return call("https://api.runpod.io/v2/pods", os.environ["RUNPOD_API_KEY"], body)


env = {k: os.environ[k] for k in ("WANDB_API_KEY", "GITHUB_DEPLOY_KEY")} | {"RUN_COMMAND": " ".join(sys.argv[1:])}
print({"vast": vast, "runpod": runpod}[os.environ.get("PROVIDER", "vast")](Path("start.sh").read_text(), env))
