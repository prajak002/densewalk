#!/usr/bin/env bash
# DenseWalk Δt milestone -- GPU box job: trimmed Cosmos3-Nano -> DCA latents -> verification.
# Launch (token via env only, never written to disk):
#   HF_TOKEN=... setsid nohup bash /workspace/dw/box_cache_job.sh > /workspace/dw/job.log 2>&1 < /dev/null &
set -euo pipefail
source /venv/main/bin/activate
W=/workspace/dw; D=/workspace/dw_data; M=/workspace/cosmos3_nano_und
RUN=dca_results/cosmos3-nano_v2/run_20260820_000205
mkdir -p "$W" "$D"; cd "$W"

disk() {  # CLAUDE.md: check disk before every step, stop under 20 GB free
  local free; free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  echo "[$(date +%T)] disk free ${free} GB -- $1"
  if [ "$free" -lt 20 ]; then echo "STOP: under 20 GB free"; exit 3; fi
}

disk "step 1: trim base model (shard-by-shard, 31.5 GB streamed, ~17.5 GB kept)"
[ -f "$M/.trim_ok" ] || { python trim_cosmos3.py --out "$M" && touch "$M/.trim_ok"; }

disk "step 2: labels + keyframes + V2_full adapters/heads + recorded predictions"
if [ ! -f "$D/.data_ok" ]; then python fast_fetch.py; touch "$D/.data_ok"; fi
false && python - <<'EOF'
import os, time
from huggingface_hub import snapshot_download
RUN = "dca_results/cosmos3-nano_v2/run_20260820_000205"
pats = ["json_openvla/*", "keyframes/**",
        f"{RUN}/V2_full/dca_full/adapter_best/*", f"{RUN}/V2_full/dca_full/heads_best.pt",
        f"{RUN}/V2_full/lora_only/adapter_best/*", f"{RUN}/V2_full/lora_only/heads_best.pt",
        f"{RUN}/predictions_V2_full_dca_full.json", f"{RUN}/predictions_V2_full_lora_only.json"]
delay = 15
for attempt in range(1, 9):          # notebook's robust_snapshot: resumes, backs off on 429
    try:
        snapshot_download("s-alam/dense_walk_mini", repo_type="dataset", local_dir="/workspace/dw_data",
                          allow_patterns=pats, max_workers=4)
        print("snapshot complete"); break
    except Exception as e:
        print(f"retry {attempt}: {str(e)[:150]}"); time.sleep(delay); delay = min(2 * delay, 300)
else:
    raise SystemExit("download failed after 8 attempts")
n = sum(len(f) for _, _, f in os.walk("/workspace/dw_data/keyframes"))
print("keyframe images:", n)
EOF

for RUNG in dca_full lora_only; do
  disk "step 3: cache latents [$RUNG] (2 backbone forwards/frame)"
  [ -f "$W/latents_$RUNG.npz" ] || python cache_dca_latents.py \
      --run-dir "$D/$RUN/V2_full" --rung "$RUNG" --ckpt best --base-model "$M" \
      --json "$D/json_openvla/*.json" --data-root "$D" --out "$W/latents_$RUNG.npz"
  disk "step 4: verify [$RUNG] against the checkpoint's recorded test predictions"
  python verify_latents.py --latents "$W/latents_$RUNG.npz" \
      --pred "$D/$RUN/predictions_V2_full_$RUNG.json" | tee "$W/verify_$RUNG.txt"
done
disk "JOB DONE"
