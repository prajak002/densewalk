#!/usr/bin/env bash
# GEM-X (video -> SOMA 77-joint pose) in its own venv. Same steps as GEM-X's README
# Quick Start + scripts/install_env.sh, except torch is the cu128 build of the pinned
# 2.10.0: the cu126 build in their README has no kernels for the RTX 5090 (sm_120).
# soma-retargeter (G1 retargeting, SSH-only submodule) is not needed and not installed.
set -euo pipefail
cd /workspace/GEM-X
disk() { local free; free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  echo "[$(date +%T)] disk free ${free} GB -- $1"; [ "$free" -ge 20 ] || { echo "STOP: under 20 GB"; exit 3; }; }
disk "venv + torch 2.10.0 cu128"
[ -d /workspace/gvenv ] || uv venv /workspace/gvenv --python 3.12
source /workspace/gvenv/bin/activate
uv pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
disk "soma (third_party, with LFS assets)"
(cd third_party/soma && git lfs pull)
uv pip install -e third_party/soma
disk "gem + SAM-3D-Body deps + detectron2 (install_env.sh)"
bash scripts/install_env.sh
python -c "import torch, gem, soma; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
disk "done"
