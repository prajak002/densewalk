"""M1 check: reproduced numbers vs the reference results from the Mac run (tolerance 5% relative)."""
import json
import pandas as pd

TOL = 0.05
ref = pd.read_csv("stage_d_results.csv")
new = pd.read_csv("repro/stage_d_results.csv")
m = ref.merge(new, on=["clip_id", "t", "k"], suffixes=("_ref", "_new"))
print(f"stage D rows: ref {len(ref)} | repro {len(new)} | matched {len(m)}")
ok = len(m) == len(ref)
for col in ["latent_drift", "pixel_error", "excess_over_floor"]:
    rel = ((m[col + "_new"] - m[col + "_ref"]).abs() / m[col + "_ref"].abs().clip(lower=1e-6))
    print(f"  {col:18s} median rel diff {rel.median():.4f}  max {rel.max():.4f}")
for col in ["latent_drift", "pixel_error"]:
    a, b = m.groupby("k")[col + "_ref"].mean(), m.groupby("k")[col + "_new"].mean()
    r = ((b - a).abs() / a).max()
    print(f"  per-k mean {col:12s} max rel diff {r:.4f}  {'PASS' if r <= TOL else 'FAIL'}")
    ok &= r <= TOL

R = json.load(open("outputs/decoder_floor_audit.json"))
N = json.load(open("repro/decoder_floor_audit.json"))
keys = [("fraction_of_chance", "latent_level"), ("floor_local", "mean"), ("pixel_error_mean",),
        ("baselines", "copy_last_context_frame"), ("baselines", "context_mean_frame"),
        ("chance_reference", "latent_l2"), ("divergence_ratio", "raw", "ratio")]
print("stage H headline numbers:")
for k in keys:
    a, b = R, N
    for p in k:
        a, b = a[p], b[p]
    r = abs(b - a) / abs(a)
    good = r <= TOL
    ok &= good
    print(f"  {'.'.join(k):40s} ref {a:10.4f}  repro {b:10.4f}  rel {r:.4f}  {'PASS' if good else 'FAIL'}")
print("M1 REPRODUCTION:", "PASS" if ok else "FAIL")
