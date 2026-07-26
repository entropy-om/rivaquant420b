#!/bin/bash
# Run locally, not on the pod: ships the repo, installs deps, starts data
# prep + training detached in tmux so it survives SSH disconnect.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${RIVAQUANT_STAGE:-162m}"
INFO=$(cat remote/.pod_info.json)
IP=$(python3 -c "import json,sys; print(json.load(sys.stdin)['ip'])" <<< "$INFO")
PORT=$(python3 -c "import json,sys; print(json.load(sys.stdin)['port'])" <<< "$INFO")

echo "shipping repo to root@$IP:$PORT (stage=$STAGE) ..."
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
  --exclude .git --exclude train/data \
  ./ "root@$IP:/workspace/rivaquant420b/"

ssh -o StrictHostKeyChecking=no -p "$PORT" "root@$IP" bash -s <<REMOTE
set -e
cd /workspace/rivaquant420b
pip install -q -r requirements.txt
mkdir -p /workspace/rivaquant420b-out
tmux new-session -d -s rivaquant420b "
  cd /workspace/rivaquant420b &&
  python3 train/data.py 2>&1 | tee -a /workspace/rivaquant420b-out/train.log &&
  RIVAQUANT_STAGE=$STAGE PYTHONPATH=/workspace/rivaquant420b python3 train/train.py 2>&1 | tee -a /workspace/rivaquant420b-out/train.log
"
echo "launched in tmux session 'rivaquant420b' (stage=$STAGE)"
REMOTE
