"""Δt horizon experiment driver.

  audit      python scripts/horizon_run.py audit --json 'dw_data/json_openvla*/*.json'
  run        python scripts/horizon_run.py run --json '...' --latents h_cache.npz --out outputs/horizon/real
  synthetic  python scripts/horizon_run.py synthetic --out outputs/horizon/synthetic

`run` rebuilds the DCA notebook's exact clip split, builds (clip, frame, Δt) tuples, trains
Ours / F / B2 / B3 on cached latents, evaluates B0/B1 and the two falsification controls, and
writes results.json + horizon_curve.png + a printed verdict.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

from densewalk.horizon import dataset as ds
from densewalk.horizon import train as tr
from densewalk.horizon.metrics import format_table


def load_latents(path, key="h"):
    z = np.load(path, allow_pickle=False)
    images = [str(x) for x in z["images"]]
    return z[key], {im: i for i, im in enumerate(images)}


def plot(res, path, horizons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = res["table"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, style in [("B0_zero", ":"), ("B1_cv", "--"), ("B2_state", "-."), ("F_concat", "-"),
                     ("Ours_film", "-"), ("B3_per_horizon", "none"),
                     ("CTRL_shuffled_dt", ":"), ("CTRL_shuffled_h", ":")]:
        if m not in T:
            continue
        hs = [h for h in horizons if h in T[m]]
        y = [T[m][h]["fde"] for h in hs]
        lo = [T[m][h]["fde_lo"] for h in hs]; hi = [T[m][h]["fde_hi"] for h in hs]
        kw = dict(marker="o" if m != "B3_per_horizon" else "s", lw=2.5 if m == "Ours_film" else 1.3)
        ln, = ax[0].plot(hs, y, linestyle=style if style != "none" else "", label=m, **kw)
        ax[0].fill_between(hs, lo, hi, alpha=0.12)
        if m not in ("B1_cv", "B0_zero") and not m.startswith("CTRL"):
            ax[1].plot(hs, [T[m][h].get("skill_vs_cv", np.nan) for h in hs], marker="o", label=m,
                       color=ln.get_color(), lw=kw["lw"])
    for H in ds.HORIZONS_INTERP:
        for a in ax:
            a.axvline(H, color="0.85", lw=0.8, zorder=0)
    ax[0].set(xlabel="Δt (s)", ylabel="FDE (m)", title="FDE vs horizon (95% clip-bootstrap CI)")
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_ylim(-1.0, 1.0)
    ax[1].set(xlabel="Δt (s)", ylabel="skill vs constant velocity", title="1 − FDE/FDE_CV  (grey = unseen Δt)")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run(json_glob, latents, out, cfg, seeds, max_epochs, log=print, latent_key="h"):
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    clips = ds.load_all(json_glob, cfg)
    H, img2row = load_latents(latents, latent_key)
    tr_c, va_c, te_c = ds.notebook_split(clips)
    log(f"clips {len(clips)} -> train {len(tr_c)} / val {len(va_c)} / test {len(te_c)} | latents {H.shape}")
    rows = {}
    for name, cs, hz in [("train", tr_c, ds.HORIZONS_TRAIN), ("val", va_c, ds.HORIZONS_TRAIN),
                         ("test", te_c, ds.HORIZONS_TRAIN + ds.HORIZONS_INTERP)]:
        r = ds.build_tuples(cs, hz, cfg)
        rows[name] = ds.attach_hrow(r, cs, img2row)
        cnt = {round(float(h), 3): int(np.isclose(r["dt"], h).sum()) for h in np.unique(r["dt"])}
        log(f"  {name:5s} tuples {len(r['dt']):6d} | per Δt {cnt}")
    horizons = sorted({round(float(x), 3) for x in rows["test"]["dt"]})
    res = tr.run_experiment(rows["train"], rows["val"], rows["test"], H, horizons,
                            seeds=seeds, max_epochs=max_epochs, log=log)
    res["verdict"] = tr.verdict(res)
    res["meta"] = {"json_glob": json_glob, "latents": latents, "n_clips": len(clips),
                   "split_clip_ids": {"train": [c.clip_id for c in tr_c], "val": [c.clip_id for c in va_c],
                                      "test": [c.clip_id for c in te_c]},
                   "minutes": (time.time() - t0) / 60}
    tr.save(res, os.path.join(out, "results.json"))
    plot(res, os.path.join(out, "horizon_curve.png"), horizons)
    log("\n" + format_table(res["table"], horizons))
    T = res["table"]["Ours_film"]
    log("\nB3 = one head per trained Δt: if it matches Ours, conditioning's only gain is answering unseen Δt")
    log("Ours skill vs CV : " + "  ".join(f"{h}s:{T[h]['skill_vs_cv']:+.3f}" for h in horizons))
    log("Ours coverage68  : " + "  ".join(f"{h}s:{T[h].get('cov68', float('nan')):.2f}" for h in horizons))
    log("Ours clr MAE     : " + "  ".join(f"{h}s:{T[h].get('clr_mae', float('nan')):.2f}" for h in horizons))
    log("\nVERDICT (Section 7 checks):")
    for k, v in res["verdict"].items():
        log(f"  [{'PASS' if v else 'FAIL'}] {k}")
    log(f"\nwrote {out}/results.json, horizon_curve.png  ({res['meta']['minutes']:.1f} min)")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit"); a.add_argument("--json", required=True); a.add_argument("--n", type=int, default=2)
    r = sub.add_parser("run")
    r.add_argument("--json", required=True); r.add_argument("--latents", required=True)
    r.add_argument("--out", required=True)
    s = sub.add_parser("synthetic"); s.add_argument("--out", required=True)
    for p in (r, s):
        p.add_argument("--seeds", type=int, default=3); p.add_argument("--max-epochs", type=int, default=80)
    r.add_argument("--latent-key", default="h", choices=["h", "z0", "za"])
    r.add_argument("--time-key"); r.add_argument("--yaw-key")
    r.add_argument("--yaw-unit", default="deg", choices=["deg", "rad"])
    r.add_argument("--yaw-sign", type=float, default=-1.0)
    r.add_argument("--assume-zero-yaw", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "audit":
        for p in sorted(glob.glob(args.json))[: args.n]:
            print(p); ds.audit(p)
        return
    seeds = tuple(range(args.seeds))
    if args.cmd == "run":
        cfg = ds.SchemaConfig(time_key=args.time_key, yaw_key=args.yaw_key, yaw_unit=args.yaw_unit,
                              yaw_sign=args.yaw_sign, assume_zero_yaw=args.assume_zero_yaw)
        run(args.json, args.latents, args.out, cfg, seeds, args.max_epochs, latent_key=args.latent_key)
        return
    from densewalk.horizon.synthetic import write_dataset
    for tag, signal in [("signal", 1.0), ("null", 0.0)]:
        root = os.path.join(args.out, tag)
        jg, lat = write_dataset(root, signal=signal, seed=0)
        print(f"\n{'=' * 30} SYNTHETIC [{tag}] signal={signal} {'=' * 30}")
        run(jg, lat, root, ds.SchemaConfig(), seeds, args.max_epochs)


if __name__ == "__main__":
    sys.exit(main())
