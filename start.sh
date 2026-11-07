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
# sshd refuses the provider's authorized_keys if it, ~/.ssh or ~ is not root's alone, as on some Vast hosts
chown root:root ~ ~/.ssh ~/.ssh/authorized_keys 2> /dev/null
chmod go-w ~ && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys 2> /dev/null

# Repo access: a read-only deploy key, base64-encoded in TINYDISCO_DEPLOY_KEY
echo "$TINYDISCO_DEPLOY_KEY" | base64 -d > ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
# trust github.com's host key, since nobody is there to answer ssh's "are you sure?" prompt
ssh-keyscan github.com >> ~/.ssh/known_hosts 2> /dev/null
git clone git@github.com:V1adych/tinydisco.git /root/tinydisco
cd /root/tinydisco
if [ -n "$ARCGPU_DEPLOY_KEY" ]; then
    echo "$ARCGPU_DEPLOY_KEY" | base64 -d > ~/.ssh/arcgpu
    chmod 600 ~/.ssh/arcgpu
    cat >> ~/.ssh/config <<'SSH'
Host github-arcgpu
    HostName github.com
    User git
    IdentityFile ~/.ssh/arcgpu
    IdentitiesOnly yes
SSH
    git config submodule.arcgpu.url git@github-arcgpu:V1adych/arcgpu.git
else
    git config submodule.arcgpu.url git@github.com:V1adych/arcgpu.git
fi
git submodule update --init src/modules/envs/arc/arcgpu || exit 1
uv sync --frozen --extra gpu --config-settings-package arcgpu:backend=cuda || exit 1
# Prefer host drivers over CUDA compatibility stubs; JAX supplies its own toolkit libraries.
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu
# jaxtari's sprite pack; the owner confirmed owning the Atari 2600 ROMs
JAXTARI_CONFIRM_OWNERSHIP=1 uv run --no-sync install-sprites
env >> /etc/environment  # so that an ssh session sees the same variables as this script

# Without RUN_COMMAND the instance is a plain dev box that stays up
if [ -z "$RUN_COMMAND" ]; then
    trap - EXIT
    sleep infinity
fi

# prints reach the provider's log as they happen, and every compile reports how long it took
export PYTHONUNBUFFERED=1
export JAX_LOG_COMPILES=1
export JAX_COMPILATION_CACHE_DIR=~/jax_cache
# arcgpu allocates CUDA memory outside JAX's pool.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# e.g. python launch.py python src/scripts/disco.py --track
uv run --no-sync $RUN_COMMAND
