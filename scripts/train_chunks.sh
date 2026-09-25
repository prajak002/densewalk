#!/bin/bash
# M1 — static background, 4 spatial chunks over the 83m AV2 street.
# Config reproduces the run that reached 24.20 dB mean (24.45/25.39/24.33/22.61):
#   12000 iters, 2.2M cap, 900k lidar init, stride 1, downscale 2.
cd /root/densewalk
source .venv/bin/activate
export PYTHONPATH=src
SEQ="04994d08-156c-3018-9717-ba0e29be8153"

for i in 0 1 2 3; do
  LO=$(python3 -c "print($i*0.25)")
  HI=$(python3 -c "print(($i+1)*0.25)")
  echo "=== CHUNK $i [$LO,$HI] $(date +%H:%M) ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/train_background.py \
    --seq-dir data/av2/$SEQ \
    --mask-dir data/av2_masks_moving \
    --iters 12000 \
    --downscale 2 \
    --stride 1 \
    --init-points 900000 \
    --lidar-every 1 \
    --cap-max 2200000 \
    --opacity-reg 0.005 \
    --ckpt-every 3000 \
    --seg-start $LO --seg-end $HI \
    --out outputs/v2_chunk_$i.pt 2>&1 | \
    grep -E 'HELD-OUT|segment|noise scaler|Error|saved|lidar restricted'
  echo "CHUNK $i DONE $(date +%H:%M)"
done
echo "ALL CHUNKS DONE $(date +%H:%M)"
