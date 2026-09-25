#!/bin/bash
# Parallel range-request download for large files whose server advertises
# accept-ranges: bytes (confirmed for the WildTrack zip via curl -I).
# Each part resumes by appending from wherever it left off, instead of
# restarting from zero on retry -- a slow-but-steady server connection
# (measured ~250KB/s/part here) needs that to ever finish.
set -euo pipefail

URL="$1"
OUT="$2"
PARTS="${3:-16}"

SIZE=$(curl -sIL "$URL" | grep -i '^content-length:' | tail -1 | tr -d '\r' | awk '{print $2}')
if [ -z "$SIZE" ]; then
  echo "could not determine content-length, aborting" >&2
  exit 1
fi
echo "total size: $SIZE bytes, $PARTS parts"

CHUNK=$(( (SIZE + PARTS - 1) / PARTS ))
PARTFILES=()
PIDS=()
for i in $(seq 0 $((PARTS - 1))); do
  START=$(( i * CHUNK ))
  END=$(( START + CHUNK - 1 ))
  if [ "$END" -ge "$SIZE" ]; then END=$((SIZE - 1)); fi
  PART="${OUT}.part${i}"
  PARTFILES+=("$PART")
  (
    EXPECTED=$(( END - START + 1 ))
    for attempt in $(seq 1 30); do
      EXISTING=$(stat -f%z "$PART" 2>/dev/null || stat -c%s "$PART" 2>/dev/null || echo 0)
      if [ "$EXISTING" -ge "$EXPECTED" ]; then
        break
      fi
      RESUME_START=$(( START + EXISTING ))
      TMP="${PART}.chunk"
      rm -f "$TMP"
      if curl -sL --fail --connect-timeout 20 --max-time 1800 --speed-limit 1024 --speed-time 60 \
        --range "${RESUME_START}-${END}" -o "$TMP" "$URL"; then
        cat "$TMP" >> "$PART"
        rm -f "$TMP"
      else
        echo "part $i attempt $attempt failed at offset $RESUME_START, retrying..." >&2
        sleep 5
      fi
    done
    ACTUAL=$(stat -f%z "$PART" 2>/dev/null || stat -c%s "$PART" 2>/dev/null || echo 0)
    if [ "$ACTUAL" != "$EXPECTED" ]; then
      echo "part $i incomplete after retries: expected $EXPECTED got $ACTUAL" >&2
      exit 1
    fi
  ) &
  PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=1
done
if [ "$FAIL" -ne 0 ]; then
  echo "one or more parts failed" >&2
  exit 1
fi

cat "${PARTFILES[@]}" > "$OUT"
rm -f "${PARTFILES[@]}"

ACTUAL=$(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT")
if [ "$ACTUAL" != "$SIZE" ]; then
  echo "size mismatch: expected $SIZE, got $ACTUAL" >&2
  exit 1
fi
echo "PARALLEL_DOWNLOAD_DONE: $OUT ($ACTUAL bytes)"
