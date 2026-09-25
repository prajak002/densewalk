#!/bin/bash
# Phase 2 M3 setup: wait for Isaac Sim install, verify it, then install Isaac Lab.
cd /root/densewalk

while pgrep -f 'uv pip install' >/dev/null 2>&1; do sleep 20; done
echo "ISAACSIM_INSTALL_FINISHED $(date +%H:%M)"
echo "install errors: $(grep -c 'error:' isaacsim_install.log)"

source env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
export UV_NO_CACHE=1
export UV_LINK_MODE=hardlink

echo '=== verifying isaac sim launches headless ==='
timeout 420 python -c 'from isaacsim.simulation_app import SimulationApp; a=SimulationApp({"headless":True}); print("ISAAC_SIM_OK"); a.close()' 2>&1 | tail -4

echo "=== cloning isaac lab $(date +%H:%M) ==="
if [ ! -d IsaacLab ]; then
  git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git 2>&1 | tail -2
fi

apt-get install -y cmake build-essential >/dev/null 2>&1

cd IsaacLab
echo "=== isaaclab.sh -i $(date +%H:%M) ==="
./isaaclab.sh -i > ../isaaclab_install.log 2>&1
echo "isaaclab install rc=$?"
tail -5 ../isaaclab_install.log

cd /root/densewalk
python -c 'import isaaclab; print("ISAACLAB_OK", getattr(isaaclab, "__version__", "?"))' 2>&1 | tail -3

echo '=== hunting for pretrained G1 policy ==='
find / -path /proc -prune -o \( -iname 'policy.onnx' -o -iname '*.onnx' \) -print 2>/dev/null | head -20
echo '--- g1 task configs ---'
find / -path /proc -prune -o -ipath '*locomotion/velocity/config/g1*' -print 2>/dev/null | head -20
echo '--- rsl_rl logs ---'
find / -path /proc -prune -o -ipath '*logs/rsl_rl*' -print 2>/dev/null | head -20

echo "SETUP_M3_DONE $(date +%H:%M)"
