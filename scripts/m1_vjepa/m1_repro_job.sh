#!/usr/bin/env bash
# M1: reproduce latent2rgb Stage D (pilot) + Stage H (decoder-floor audit) on the 5090.
# Uses the ORIGINAL decoder.pt / floor.json (not retrained) so any difference is the
# machine/device, not a new decoder. Writes to repro/ so the Mac reference files stay untouched.
# Launch: setsid nohup bash /workspace/l2r/m1_repro_job.sh > /workspace/l2r/repro/job.log 2>&1 < /dev/null &
set -euo pipefail
source /venv/main/bin/activate
cd /workspace/l2r
mkdir -p repro
disk() { local free; free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  echo "[$(date +%T)] disk free ${free} GB -- $1"; [ "$free" -ge 20 ] || { echo "STOP: under 20 GB"; exit 3; }; }

disk "stage D pilot"
python scripts/pilot.py --out-csv repro/stage_d_results.csv
disk "stage H decoder-floor audit"
python scripts/decoder_floor_audit.py --out-csv repro/decoder_floor_audit.csv --out-json repro/decoder_floor_audit.json
disk "compare against Mac reference"
python compare_repro.py | tee repro/COMPARE.txt
echo "[$(date +%T)] JOB DONE"
