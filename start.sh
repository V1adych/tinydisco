#!/usr/bin/env bash
# SSH in with the account keys RunPod passes in PUBLIC_KEY
mkdir -p ~/.ssh
echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
ssh-keygen -A
service ssh start

# Repo access: a read-only deploy key, stored base64-encoded in a RunPod secret
echo "$GITHUB_DEPLOY_KEY" | base64 -d > ~/.ssh/id_ed25519
chmod 600 ~/.ssh/id_ed25519
# trust github.com's host key, since nobody is there to answer ssh's "are you sure?" prompt
ssh-keyscan github.com >> ~/.ssh/known_hosts

# /workspace survives a pod stop, so a restarted pod pulls instead of cloning again
if [ -d /workspace/meta-rl ]; then
    git -C /workspace/meta-rl pull --ff-only
else
    git clone git@github.com:V1adych/meta-rl.git /workspace/meta-rl
fi
cd /workspace/meta-rl

# in case dependencies changed since the image was built; a no-op otherwise
uv sync --frozen

# Without TRAIN_ARGS the pod is a plain dev box
if [ -z "$TRAIN_ARGS" ]; then
    sleep infinity
fi

python src/ppo.py --checkpoint_dir /workspace/checkpoints $TRAIN_ARGS

# Delete the pod so nothing bills, even if training crashed: checkpoints and console output are already in wandb.
# RUNPOD_API_KEY is the pod's own key from RunPod; only the GraphQL API accepts it for this
curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -d "{\"query\": \"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}"
sleep infinity
