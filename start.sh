#!/usr/bin/env bash
# Startup script for a bare nvidia/cuda image on Vast.ai (sent as the onstart script) or RunPod (sent as the container command).
# launch.py passes this file's text and the env; everything is installed here, a custom image was slower to pull than this is to run.

# Delete the instance when this script ends for any reason, so a failed install or a crashed run doesn't bill forever;
# checkpoints and console output are in wandb. Each provider injects its own id and a key that may delete the instance.
destroy() {
    if [ -n "$RUNPOD_POD_ID" ]; then
        curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H "Content-Type: application/json" \
            -d "{\"query\": \"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}"
    else
        curl -s -X DELETE -H "Authorization: Bearer $CONTAINER_API_KEY" "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/"
    fi
}
trap destroy EXIT

apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates openssh-client > /dev/null
curl -LsSf https://astral.sh/uv/0.13.0/install.sh | sh > /dev/null
export PATH=/root/.local/bin:$PATH
mkdir -p ~/.ssh

# Repo access: a read-only deploy key, base64-encoded in GITHUB_DEPLOY_KEY
echo "$GITHUB_DEPLOY_KEY" | base64 -d > ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
# trust github.com's host key, since nobody is there to answer ssh's "are you sure?" prompt
ssh-keyscan github.com >> ~/.ssh/known_hosts 2> /dev/null
git clone git@github.com:V1adych/meta-rl.git /root/meta-rl
cd /root/meta-rl
uv sync --frozen
# jaxtari's sprite pack; the owner confirmed owning the Atari 2600 ROMs
JAXTARI_CONFIRM_OWNERSHIP=1 uv run install-sprites
env >> /etc/environment  # so that an ssh session sees the same variables as this script

# Without RUN_COMMAND the instance is a plain dev box that stays up
if [ -z "$RUN_COMMAND" ]; then
    trap - EXIT
    sleep infinity
fi

# e.g. python launch.py python src/disco.py --track
uv run $RUN_COMMAND
