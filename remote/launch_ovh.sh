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

echo "shipping repo to ubuntu@$IP (stage=$STAGE, OVH compute-optimized, CPU) ..."
rsync -az -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  --exclude .git --exclude train/data \
  ./ "ubuntu@$IP:/home/ubuntu/rivaquant420b/"

ssh -i "$KEY" -o StrictHostKeyChecking=no "ubuntu@$IP" bash -s <<REMOTE
set -e
sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -qq -y python3-pip python3-venv tmux
cd /home/ubuntu/rivaquant420b
python3 -m venv .venv
source .venv/bin/activate
# requirements.txt has no torch pin (RunPod's base images ship it
# pre-installed; this bare Ubuntu box doesn't). CPU wheel index — much
# smaller download than a CUDA build, and correct for hardware with no GPU.
pip install -q --index-url https://download.pytorch.org/whl/cpu torch
pip install -q -r requirements.txt
python3 -c "import torch; print('torch', torch.__version__, '- CPU-only run, intentional, no CUDA check')"
mkdir -p /home/ubuntu/rivaquant420b-out
# tmux spawns a fresh shell that does NOT inherit this script's venv
# activation — use the venv's own python binary directly instead of
# relying on \`source\` running again inside the tmux pane.
# BATCH_SIZE=2/GRAD_ACCUM=16 (effective batch still 32, same as the
# default 8/4): the default 8 micro-batch OOM-killed the first attempt at
# ~30GB RSS on this box's 32GB total (dmesg confirmed it by hand — a
# ~1B-param model's fp32 weights+grad+Adam-states alone is ~16GB before
# any activation memory, and BitLinear's STE keeps extra copies around
# for the backward pass on top of that). GRA9's own quota caps this
# account at 44GB total in this region, so a bigger box isn't available
# here — smaller micro-batch is the actual fix, not more RAM.
tmux new-session -d -s rivaquant420b_quantal "
  cd /home/ubuntu/rivaquant420b &&
  /home/ubuntu/rivaquant420b/.venv/bin/python3 train/data.py 2>&1 | tee -a /home/ubuntu/rivaquant420b-out/train.log &&
  RIVAQUANT_STAGE=$STAGE RIVAQUANT_OUT=/home/ubuntu/rivaquant420b-out PYTHONPATH=/home/ubuntu/rivaquant420b \\
  RIVAQUANT_BATCH_SIZE=2 RIVAQUANT_GRAD_ACCUM_STEPS=16 \\
  /home/ubuntu/rivaquant420b/.venv/bin/python3 train/train.py 2>&1 | tee -a /home/ubuntu/rivaquant420b-out/train.log
"
echo "launched in tmux session 'rivaquant420b_quantal'"
REMOTE
