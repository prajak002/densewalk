#!/usr/bin/env bash
# M1-Varanasi step 3: camera poses by SfM.
#
# AV2 handed us calibrated poses; here every pose must be solved for. Three
# choices below are forced by this clip rather than by taste:
#   * sequential matcher   - it is video, consecutive frames overlap; exhaustive
#                            matching of 361 frames buys nothing and costs 65x.
#   * single_camera        - one phone, fixed lens, so intrinsics are shared and
#                            solving them once is far better conditioned.
#   * mask_path            - drops crowd features BEFORE triangulation, so moving
#                            people never enter the static model.
# COLMAP 3.9.1 from apt is built without CUDA, so SIFT runs on CPU, and the vast
# container is quota'd to ~21 effective cores despite nproc=128 (32 procs beat
# 128 by 3.5x in bench). Hence the thread cap.
set -euo pipefail

ROOT=${1:-/workspace/varanasi/sfm}
THREADS=${THREADS:-24}
DB=$ROOT/database.db
IMAGES=$ROOT/images
MASKS=$ROOT/masks_colmap
SPARSE=$ROOT/sparse

echo "=== disk ==="; df -h / | tail -1
rm -rf "$DB" "$SPARSE"; mkdir -p "$SPARSE"

echo "=== [1/3] feature extraction ==="
colmap feature_extractor \
  --database_path "$DB" \
  --image_path "$IMAGES" \
  --ImageReader.mask_path "$MASKS" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --SiftExtraction.use_gpu 0 \
  --SiftExtraction.num_threads "$THREADS" \
  --SiftExtraction.estimate_affine_shape 1 \
  --SiftExtraction.domain_size_pooling 1

echo "=== [2/3] sequential matching ==="
colmap sequential_matcher \
  --database_path "$DB" \
  --SiftMatching.use_gpu 0 \
  --SiftMatching.num_threads "$THREADS" \
  --SequentialMatching.overlap 15 \
  --SequentialMatching.quadratic_overlap 1

echo "=== [3/3] mapping ==="
colmap mapper \
  --database_path "$DB" \
  --image_path "$IMAGES" \
  --output_path "$SPARSE" \
  --Mapper.num_threads "$THREADS" \
  --Mapper.ba_refine_principal_point 1

echo "=== result ==="
for m in "$SPARSE"/*/; do
  echo "--- model $m ---"
  colmap model_analyzer --path "$m" 2>&1 | sed 's/^/    /'
done
echo "COLMAP_DONE"
