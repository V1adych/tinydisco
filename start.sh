#!/usr/bin/env bash
# SSH in with the account keys RunPod passes in PUBLIC_KEY
mkdir -p ~/.ssh
echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
ssh-keygen -A
service ssh start

# Without TRAIN_ARGS the pod is a plain dev box
if [ -z "$TRAIN_ARGS" ]; then
    sleep infinity
fi

python ppo.py --checkpoint_dir /workspace/checkpoints $TRAIN_ARGS

# stop the pod so the GPU stops billing, even if training crashed
runpodctl pod stop "$RUNPOD_POD_ID"
sleep infinity
