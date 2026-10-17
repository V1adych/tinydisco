#!/usr/bin/env bash
# Without TRAIN_ARGS the pod is a plain dev box
if [ -z "$TRAIN_ARGS" ]; then
    exit 0
fi

cd /app
python ppo.py --checkpoint_dir /workspace/checkpoints $TRAIN_ARGS

# stop the pod so the GPU stops billing, even if training crashed
runpodctl pod stop "$RUNPOD_POD_ID"
