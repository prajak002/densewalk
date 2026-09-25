"""Per-horizon metrics with clip-level bootstrap CIs (frames inside a clip are correlated;
the DCA post-mortem measured an effective n of ~41 on 91 stop frames -- never report
frame-level CIs)."""
from __future__ import annotations

import numpy as np

from densewalk.horizon.head import LAPLACE_K68, LAPLACE_K95


def fde(pred_disp: np.ndarray, true_disp: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred_disp - true_disp, axis=-1)


def clip_bootstrap(values: np.ndarray, clip: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """Mean and 95% CI of `values`, resampling whole clips."""
    uc = np.unique(clip)
    if len(uc) == 0:
        return float("nan"), float("nan"), float("nan")
    sums = np.array([values[clip == c].sum() for c in uc])
    cnts = np.array([(clip == c).sum() for c in uc])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uc), size=(n_boot, len(uc)))
    boots = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    return float(sums.sum() / cnts.sum()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def horizon_table(rows: dict, preds: dict[str, dict], horizons, n_boot: int = 1000) -> dict:
    """rows: tuple arrays (dt, disp, clip, event, clr_t1, clr_valid_t1).
    preds: method -> {"disp": [N,2], optional "b_xy": [N,2], "clr": [N], "b_c": [N]}.
    Returns method -> horizon -> metrics, plus paired Δ vs "B1_cv" and "B2_state"."""
    out = {}
    fdes = {m: fde(p["disp"], rows["disp"]) for m, p in preds.items()}
    for m, p in preds.items():
        out[m] = {}
        for H in horizons:
            sel = np.isclose(rows["dt"], H)
            if not sel.any():
                continue
            e, cl = fdes[m][sel], rows["clip"][sel]
            mean, lo, hi = clip_bootstrap(e, cl, n_boot)
            r = {"n": int(sel.sum()), "n_clips": int(len(np.unique(cl))),
                 "fde": mean, "fde_lo": lo, "fde_hi": hi}
            ev = rows["event"][sel] > 0.5
            r["fde_event"] = float(e[ev].mean()) if ev.any() else float("nan")
            r["fde_steady"] = float(e[~ev].mean()) if (~ev).any() else float("nan")
            r["n_event"] = int(ev.sum())
            if "B1_cv" in fdes:
                ref = fdes["B1_cv"][sel]
                # skill is undefined where CV is (near-)exact; report NaN, not -1e6
                r["skill_vs_cv"] = float(1 - e.mean() / ref.mean()) if ref.mean() > 0.01 else float("nan")
                if ev.any() and ref[ev].mean() > 0.01:
                    r["skill_vs_cv_event"] = float(1 - e[ev].mean() / ref[ev].mean())
            for base in ("B1_cv", "B2_state"):
                if base in fdes and base != m:
                    d, dlo, dhi = clip_bootstrap(e - fdes[base][sel], cl, n_boot)
                    r[f"delta_vs_{base}"] = (d, dlo, dhi)       # negative = better than base
            if "b_xy" in p:
                z = np.abs(p["disp"][sel] - rows["disp"][sel]) / p["b_xy"][sel]
                r["cov68"] = float((z < LAPLACE_K68).mean())   # nominal 0.68 per axis
                r["cov95"] = float((z < LAPLACE_K95).mean())   # nominal 0.95 per axis
                r["mean_b"] = float(p["b_xy"][sel].mean())
            if "clr" in p:
                mv = (rows["clr_valid_t1"][sel] > 0.5)
                if mv.any():
                    r["clr_mae"] = float(np.abs(p["clr"][sel][mv] - rows["clr_t1"][sel][mv]).mean())
            out[m][H] = r
        # ADE@H: mean of per-horizon FDE over queried horizons <= H
        hs = sorted(out[m])
        for H in hs:
            out[m][H]["ade"] = float(np.mean([out[m][h]["fde"] for h in hs if h <= H]))
    return out


def format_table(tab: dict, horizons) -> str:
    lines = []
    head = f"{'method':22s}" + "".join(f"{f'FDE@{H}s':>22s}" for H in horizons)
    lines.append(head)
    for m, per in tab.items():
        cells = []
        for H in horizons:
            r = per.get(H)
            cells.append(f"{r['fde']:.3f} [{r['fde_lo']:.3f},{r['fde_hi']:.3f}]" if r else "-")
        lines.append(f"{m:22s}" + "".join(f"{c:>22s}" for c in cells))
    return "\n".join(lines)
