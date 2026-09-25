#!/usr/bin/env bash
# Retrain the Varanasi static splat (the v2 checkpoint was lost with a reclaimed box).
# Same recipe as v2: 30k iters, MCMC cap 500k, YOLO11x-seg masks, the ORIGINAL COLMAP
# poses (outputs/varanasi/sparse, copied from the Mac) so world.json still applies.
# Launch: setsid nohup bash /workspace/vsplat/job.sh > /workspace/vsplat/job.log 2>&1 < /dev/null &
set -euo pipefail
source /venv/main/bin/activate
W=/workspace/vsplat; cd "$W"
disk() { local free; free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  echo "[$(date +%T)] disk free ${free} GB -- $1"; [ "$free" -ge 20 ] || { echo "STOP: under 20 GB"; exit 3; }; }

disk "1/4 frames"
[ -f images/.ok ] || { python scripts/varanasi_frames.py --video video/varanasi1.mp4 --out-dir images && touch images/.ok; }
disk "2/4 masks (yolo11x-seg)"
[ -f masks/.ok ] || { python scripts/varanasi_masks.py --images images/images --out-dir masks --imgsz 1280 && touch masks/.ok; }
disk "3/4 COLMAP bin -> txt"
python - <<'PY'
import pycolmap
r = pycolmap.Reconstruction("sparse")
r.write_text("sparse")
names = sorted(im.name for im in r.images.values())
print("cameras", len(r.cameras), "images", len(r.images), "points", len(r.points3D), "first/last", names[0], names[-1])
PY
ls images/images | head -2; ls masks/masks_train | head -2
mkdir -p out
disk "4/4 train splat (30k iters)"
python scripts/varanasi_train_splat.py --model-dir sparse --images images/images --masks masks/masks_train \
  --out out/varanasi_bg_v3.pt --iters 30000 --cap-max 500000
echo "[$(date +%T)] JOB DONE"
