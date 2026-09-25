#!/usr/bin/env bash
# M1-Varanasi step 3b: re-run mapping only, with the forward-motion defaults relaxed.
#
# The first attempt registered 2/361 and said "No good initial image pair found".
# The database showed why: 4425 of 6258 verified pairs are PLANAR_OR_PANORAMIC,
# because a forward walk produces little parallax between nearby frames. COLMAP's
# two initialisation guards are tuned for orbit-style capture and both reject
# exactly this motion:
#   init_max_forward_motion 0.95 -> 1.0   accept pairs whose motion is forward
#   init_min_tri_angle      16   -> 4     1m of walking vs a 20m facade is ~3-4deg
# filter_min_tri_angle is also lowered so low-parallax points survive instead of
# being filtered out of an already sparse model.
#
# Relaxing the guards alone was not enough: COLMAP then seeded on images #193
# and #185, only 8 frames (~0.3m) apart, and a model that shallow cannot register
# a third image -- it stopped at 2 again. So the seed is now forced to a pair the
# database says has genuine parallax: config=3 (not PLANAR_OR_PANORAMIC) and 64
# frames apart, ~2.1s of walking, 361 inliers.
#
# Features and matches are unchanged and reused; this re-runs mapping only.
set -euo pipefail

ROOT=${1:-/workspace/varanasi/sfm}
THREADS=${THREADS:-24}
SPARSE=${SPARSE_DIR:-$ROOT/sparse_retry2}

echo "=== disk ==="; df -h / | tail -1
rm -rf "$SPARSE"; mkdir -p "$SPARSE"

colmap mapper \
  --database_path "$ROOT/database.db" \
  --image_path "$ROOT/images" \
  --output_path "$SPARSE" \
  --Mapper.num_threads "$THREADS" \
  --Mapper.init_image_id1 ${INIT1:-125} \
  --Mapper.init_image_id2 ${INIT2:-187} \
  --Mapper.init_max_forward_motion 1.0 \
  --Mapper.init_min_tri_angle 4 \
  --Mapper.init_num_trials 500 \
  --Mapper.filter_min_tri_angle 0.5 \
  --Mapper.multiple_models 0 \
  --Mapper.ba_refine_principal_point 1

echo "=== result ==="
for m in "$SPARSE"/*/; do
  echo "--- model $m ---"
  colmap model_analyzer --path "$m" 2>&1 | sed 's/^/    /'
done
echo "RETRY_DONE"
