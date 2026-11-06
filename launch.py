"""Rents a GPU that runs the given command in the repo and deletes itself when it exits, e.g.
    python launch.py python src/scripts/disco.py --track
PROVIDER=vast (default) rents the cheapest acceptable Vast.ai offer; PROVIDER=runpod a RunPod pod (CLOUD=SECURE when community is out of stock).
GPU names the card in the provider's own naming, e.g. GPU="RTX 5090" or GPU="NVIDIA GeForce RTX 5090".
DISK_GB=30 and MIN_RAM_GIB=16 set disk space and minimum host RAM.
Needs WANDB_API_KEY, GITHUB_DEPLOY_KEY (base64) and the provider's VASTAI_API_KEY or RUNPOD_API_KEY in the environment. ARCGPU_DEPLOY_KEY (base64) is needed if the main key cannot also read V1adych/arcgpu."""

import json
import os
import sys
import urllib.request
from pathlib import Path

IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"  # nvcc builds arcgpu for the rented GPU
PROVIDER = os.environ.get("PROVIDER", "vast")
GPU = os.environ.get("GPU", "NVIDIA GeForce RTX 5090" if PROVIDER == "runpod" else "RTX 5090")
CLOUD = os.environ.get("CLOUD", "COMMUNITY")
DISK_GB = int(os.environ.get("DISK_GB", "30"))
MIN_RAM_GIB = int(os.environ.get("MIN_RAM_GIB", "64"))
VASTAI_API_KEY = os.environ.get("VASTAI_API_KEY")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
GITHUB_DEPLOY_KEY = os.environ.get("GITHUB_DEPLOY_KEY")
ARCGPU_DEPLOY_KEY = os.environ.get("ARCGPU_DEPLOY_KEY")  # optional separate read-only key for the private submodule
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
    "cpu_ram": {"gte": MIN_RAM_GIB * 1024},  # Vast reports MiB; filter the rental's share below too
    "duration": {"gte": 24 * 60 * 60},
    "type": "on-demand",
    "order": [["dph_total", "asc"]],
    "limit": 100,  # RAM shares are checked locally, so fetch more than the cheapest few
}


def call(url: str, key: str, body: dict, method: str = "POST") -> dict:
    request = urllib.request.Request(url, method=method, data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # the provider's explanation is in the body, which the exception alone doesn't show
        return json.loads(error.read())


def vast_ram_gib(offer: dict) -> float:
    """The rental's RAM share, proportional to its allocated CPU cores."""
    if not offer.get("cpu_cores"):
        return 0.0
    return (offer.get("cpu_ram") or 0) / 1024 * (offer.get("cpu_cores_effective") or 0) / offer["cpu_cores"]


def vast(script: str, env: dict):
    assert VASTAI_API_KEY, "VASTAI_API_KEY is required"
    offers = call("https://console.vast.ai/api/v0/bundles/", VASTAI_API_KEY, VAST_OFFER_FILTER | {"gpu_name": {"eq": GPU}})["offers"]
    body = {"image": IMAGE, "disk": DISK_GB, "runtype": "ssh", "label": "meta-rl", "onstart": script, "env": env}
    for offer in offers:  # listings go stale within minutes, so the first offer may already be gone
        if "CN" in str(offer.get("geolocation")) or vast_ram_gib(offer) < MIN_RAM_GIB:
            continue
        result = call(f"https://console.vast.ai/api/v0/asks/{offer['id']}/", VASTAI_API_KEY, body, method="PUT")
        if result.get("success"):
            return f"instance {result['new_contract']}: {offer['gpu_name']} in {offer['geolocation']} at ${offer['dph_total']:.3f}/h"
        print(f"offer {offer['id']}: {result.get('msg')}")
    sys.exit("no offer accepted")


def runpod(script: str, env: dict):
    assert RUNPOD_API_KEY, "RUNPOD_API_KEY is required"
    body = {
        "name": "meta-rl",
        "cloud": CLOUD,
        "gpu": {"id": GPU, "count": 1, "minCudaVersion": "12.8", "minRamPerGpu": MIN_RAM_GIB},
        "image": IMAGE,
        "entrypoint": ["/bin/bash", "-c"],
        "cmd": [script],
        "disk": DISK_GB,
        "env": env,
    }
    result = call("https://api.runpod.io/v2/pods", RUNPOD_API_KEY, body)
    if not result.get("id"):
        sys.exit("RunPod did not return a pod ID; check the provider dashboard for the launch status")
    return f"instance {result['id']}: {GPU} in {result.get('dataCenterId', 'unknown location')} at ${result['cost']:.3f}/h ({CLOUD})"


if __name__ == "__main__":  # importing this module must not rent anything
    assert WANDB_API_KEY and GITHUB_DEPLOY_KEY, "WANDB_API_KEY and GITHUB_DEPLOY_KEY are required"
    env = {"WANDB_API_KEY": WANDB_API_KEY, "GITHUB_DEPLOY_KEY": GITHUB_DEPLOY_KEY, "RUN_COMMAND": " ".join(sys.argv[1:])}
    if ARCGPU_DEPLOY_KEY:
        env["ARCGPU_DEPLOY_KEY"] = ARCGPU_DEPLOY_KEY
    print({"vast": vast, "runpod": runpod}[PROVIDER](Path("start.sh").read_text(), env))
