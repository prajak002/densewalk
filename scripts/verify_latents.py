"""Check that re-computed DCA outputs match the checkpoint's OWN recorded test predictions
(predictions_V2_full_<rung>.json, written by the notebook's evaluate() at training time).

Matching direction, obstacle distance and gate on the same keyframes proves the whole chain
-- trimmed base weights, LoRA adapter, heads, processor, prompt -- is the one that was trained.
Frames are matched by keyframe basename (e.g. 000285_kf0000.jpg), which is unique.

  python verify_latents.py --latents latents_dca_full.npz --pred predictions_V2_full_dca_full.json
"""
import argparse
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--latents", required=True)
p.add_argument("--pred", required=True)
a = p.parse_args()

z = np.load(a.latents)
row = {os.path.basename(str(k)): i for i, k in enumerate(z["images"])}
P = json.load(open(a.pred))
hit = [(r, row[os.path.basename(r["image"])]) for r in P if os.path.basename(r["image"]) in row]
print(f"recorded test frames {len(P)} | matched in cache {len(hit)}")
assert hit, "no overlap -- check image keys"

cs_rec = np.array([[r["cos_pred"], r["sin_pred"]] for r, _ in hit])
cs_new = z["dca_dir"][[i for _, i in hit]]
ang = np.degrees(np.arccos(np.clip((cs_rec * cs_new).sum(1), -1, 1)))
od_rec = np.array([r["obs_dist_pred_m"] for r, _ in hit])
od_new = z["dca_obs_dist_m"][[i for _, i in hit]]
g_rec = np.array([r["gate_pred"] for r, _ in hit])
g_new = z["dca_gate"][[i for _, i in hit]]

def rep(name, d, tol):
    ok = np.median(np.abs(d)) < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:28s} median |Δ| {np.median(np.abs(d)):.4f}  "
          f"p95 {np.percentile(np.abs(d), 95):.4f}  max {np.abs(d).max():.4f}  (tol {tol})")
    return ok

ok = all([rep("direction (deg)", ang, 1.0),
          rep("obstacle distance (m)", od_new - od_rec, 0.05),
          rep("gate", g_new - g_rec, 0.01)])
# bf16 forward passes are not bit-deterministic across GPUs/kernels: expect small, not zero
print("VERIFY:", "PASS -- latents come from the trained model" if ok else "FAIL -- do not train on these latents")
raise SystemExit(0 if ok else 1)
