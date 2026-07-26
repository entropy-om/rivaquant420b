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
# Community-cloud hosts have different drivers (seen both 560.35 and
# 580.126 across two provisions) — the bundled cu118 torch works fine on
# an older driver but hit "CUDA unknown error" on the newer one, even
# though nvidia-smi saw the GPU fine either way. Don't force a reinstall
# unconditionally (a cu128 build needs driver >=570, which would BREAK
# the 560-driver host that already works) — check first, only fix what's
# actually broken on this specific box.
if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "stock torch already sees the GPU, no reinstall needed"
else
  echo "stock torch can't see the GPU on this host, reinstalling against cu124"
  pip install -q --index-url https://download.pytorch.org/whl/cu124 torch
fi
python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "FATAL: torch.cuda.is_available() is still False — not launching on CPU silently."; exit 1; }
echo "GPU confirmed: \$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
mkdir -p /workspace/rivaquant420b-out
# data.py has no logging of its own (just print()/progress bars), so tee
# is its only persistence. train.py's own log() already writes every line
# to train.log directly -- piping its stdout through tee -a to the SAME
# path would double-write every line. Let train.py's stdout go to the
# tmux pane only (still visible via tmux attach / capture-pane).
tmux new-session -d -s rivaquant420b "
  cd /workspace/rivaquant420b &&
  python3 train/data.py 2>&1 | tee -a /workspace/rivaquant420b-out/train.log &&
  RIVAQUANT_STAGE=$STAGE PYTHONPATH=/workspace/rivaquant420b python3 train/train.py
"
echo "launched in tmux session 'rivaquant420b' (stage=$STAGE)"
REMOTE
