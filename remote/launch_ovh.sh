#!/bin/bash
# Run locally, not on the box: ships the repo, installs deps, starts data
# prep + training detached in tmux so it survives SSH disconnect.
#
# Deliberately no CUDA-available-or-abort check here (unlike launch.sh,
# the RunPod path): OVH compute-optimized has no GPU at all, by design.
# CPU is the explicit, informed choice for the "quantal" stage, not a
# silent fallback — verified once (torch import + version) so a genuinely
# broken box still fails loudly, and echoed clearly either way.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="quantal"
KEY="$HOME/.ssh/id_pldev_ci"
INFO=$(cat remote/.pod_info_ovh.json)
IP=$(python3 -c "import json,sys; print(json.load(sys.stdin)['ip'])" <<< "$INFO")

echo "shipping repo to root@$IP (stage=$STAGE, OVH compute-optimized, CPU) ..."
rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  --exclude .git --exclude train/data \
  ./ "root@$IP:/workspace/rivaquant420b/"

ssh -i "$KEY" -o StrictHostKeyChecking=no "root@$IP" bash -s <<REMOTE
set -e
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qq -y python3-pip tmux
mkdir -p /workspace
cd /workspace/rivaquant420b
# requirements.txt has no torch pin (RunPod's base images ship it
# pre-installed; this bare Ubuntu box doesn't). CPU wheel index — much
# smaller download than a CUDA build, and correct for hardware with no GPU.
pip install -q --index-url https://download.pytorch.org/whl/cpu torch
pip install -q -r requirements.txt
python3 -c "import torch; print('torch', torch.__version__, '- CPU-only run, intentional, no CUDA check')"
mkdir -p /workspace/rivaquant420b-out
tmux new-session -d -s rivaquant420b_quantal "
  cd /workspace/rivaquant420b &&
  python3 train/data.py 2>&1 | tee -a /workspace/rivaquant420b-out/train.log &&
  RIVAQUANT_STAGE=$STAGE RIVAQUANT_OUT=/workspace/rivaquant420b-out PYTHONPATH=/workspace/rivaquant420b python3 train/train.py 2>&1 | tee -a /workspace/rivaquant420b-out/train.log
"
echo "launched in tmux session 'rivaquant420b_quantal'"
REMOTE
