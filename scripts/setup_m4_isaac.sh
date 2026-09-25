#!/usr/bin/env bash
# M4 setup: Isaac Sim 6.0.1 + Isaac Lab on the vast box, in its OWN venv.
#
# Isolated from /venv/main on purpose: that env holds the working gsplat/torch
# build for the splat pipeline, and Isaac Sim pins its own torch. One broken
# dependency resolution should not cost us both.
#
# Version choice is not free: outputs/g1_policy/model_1499.pt was trained under
# Isaac Sim 6.0.1 + Isaac Lab 24.2.4, and its observation layout is tied to that
# env config. A different version silently changes the observation vector and
# the policy walks like it is drunk rather than erroring.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export OMNI_KIT_ACCEPT_EULA=YES

echo "=== disk before (stop if <20G) ==="; df -h / | tail -1

cd /workspace
uv venv env_isaac --python 3.12
source env_isaac/bin/activate
export UV_HTTP_TIMEOUT=1200

echo "=== installing isaacsim 6.0.1.0 (large: expect ~15-20GB) ==="
uv pip install "isaacsim[all,extscache]==6.0.1.0"

echo "=== disk after isaacsim ==="; df -h / | tail -1

echo "=== verifying isaac sim launches headless ==="
python - <<'PY'
from isaacsim.simulation_app import SimulationApp
app = SimulationApp({"headless": True})
print("ISAAC_SIM_OK")
app.close()
PY

echo "=== isaac lab ==="
apt-get install -y -qq cmake build-essential
[ -d IsaacLab ] || git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh -i
cd /workspace

python -c 'import isaaclab; print("ISAACLAB_OK", getattr(isaaclab, "__version__", "?"))'

echo "=== confirm the G1 task ids and config path actually present ==="
python - <<'PY'
import isaaclab_tasks, pkgutil, pathlib
root = pathlib.Path(isaaclab_tasks.__file__).parent
hits = [str(p.relative_to(root)) for p in root.rglob("*") if "g1" in p.name.lower()]
print("g1 paths:", hits[:10])
import gymnasium as gym
print("G1 tasks:", [k for k in gym.registry if "G1" in k][:10])
PY

echo "=== disk after ==="; df -h / | tail -1
echo "SETUP_M4_DONE"
