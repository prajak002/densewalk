"""Second cleaning pass on an exported metric 3DGS .ply: drop degenerate needle gaussians.

MCMC leaves many gaussians with one axis collapsed to ~0 and another stretched;
seen off the training path they render as long shards. Dropped when the largest
axis is longer than --needle-min-m AND anisotropy (max/min axis) exceeds --max-aniso.
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--max-aniso", type=float, default=25.0)
ap.add_argument("--needle-min-m", type=float, default=0.08)
ap.add_argument("--no-needle", action="store_true")
ap.add_argument("--facade-box", action="store_true",
                help="keep only building fronts: beside the corridor (|y| in [ymin,ymax]), below zmax, "
                     "opaque (>= min-opacity) and compact (largest axis <= max-axis-m)")
ap.add_argument("--ymin", type=float, default=2.6); ap.add_argument("--ymax", type=float, default=14.0)
ap.add_argument("--zmax", type=float, default=11.0); ap.add_argument("--xmin", type=float, default=-22.0)
ap.add_argument("--xmax", type=float, default=40.0)
ap.add_argument("--min-opacity", type=float, default=0.25); ap.add_argument("--max-axis-m", type=float, default=0.45)
a = ap.parse_args()
b = open(a.inp, "rb").read(); h = b.index(b"end_header\n") + 11
hdr = b[:h].decode(); n = int([l for l in hdr.splitlines() if l.startswith("element vertex")][0].split()[-1])
P = np.frombuffer(b[h:], dtype=np.float32).reshape(n, 17)
sc = np.exp(P[:, 10:13].astype(np.float64))
needle = (sc.max(1) > a.needle_min_m) & (sc.max(1) / np.maximum(sc.min(1), 1e-9) > a.max_aniso)
if a.no_needle:
    needle[:] = False
if a.facade_box:
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    op = 1 / (1 + np.exp(-P[:, 9].astype(np.float64)))
    inside = ((np.abs(y) >= a.ymin) & (np.abs(y) <= a.ymax) & (z > -0.1) & (z < a.zmax)
              & (x > a.xmin) & (x < a.xmax) & (op >= a.min_opacity) & (sc.max(1) <= a.max_axis_m))
    needle |= ~inside
Q = P[~needle]
with open(a.out, "wb") as f:
    f.write(hdr.replace(f"element vertex {n}", f"element vertex {len(Q)}").encode()); f.write(Q.tobytes())
rep = {"input": n, "removed": int(needle.sum()), "kept": int(len(Q)), "args": vars(a)}
open(a.out + ".json", "w").write(json.dumps(rep, indent=2)); print(rep)
