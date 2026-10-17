import json
import os
import sys
import urllib.request

body = {
    "name": "meta-rl",
    "templateId": "rmm9avyayn",
    "cloud": os.environ.get("CLOUD", "COMMUNITY"),
    "gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 1, "minCudaVersion": "12.8"},
    "startSsh": True,
    "env": {"TRAIN_ARGS": " ".join(sys.argv[1:])},
}
request = urllib.request.Request(
    "https://api.runpod.io/v2/pods",
    data=json.dumps(body).encode(),
    headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}", "Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(request) as response:
        print(response.read().decode())
except urllib.error.HTTPError as error:
    # RunPod's explanation is in the body, which the exception alone doesn't show
    sys.exit(error.read().decode())
