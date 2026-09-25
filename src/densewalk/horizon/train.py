"""Train the horizon head and every baseline on cached latents, then run the horizon
experiment with its falsification controls.

Everything here reads cached h(t) -- the Cosmos3-Nano backbone is never touched, so a full
run (all variants x seeds) is minutes on CPU.
"""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import torch

from densewalk.horizon.head import HorizonHead, cv_displacement, horizon_loss
from densewalk.horizon.metrics import horizon_table

VARIANTS = {
    "Ours_film":   dict(cond="film", use_h=True),
    "F_concat":    dict(cond="concat", use_h=True),
    "B2_state":    dict(cond="film", use_h=False),
}


def _batch(rows, H, idx, dev, hrow=None, dt=None):
    hr = rows["hrow"][idx] if hrow is None else hrow[idx]
    return (torch.as_tensor(H[hr], device=dev).float() if H is not None else None,
            torch.as_tensor(rows["s"][idx], device=dev),
            torch.as_tensor(rows["dt"][idx] if dt is None else dt[idx], device=dev),
            {"disp": torch.as_tensor(rows["disp"][idx], device=dev),
             "clr_t1": torch.as_tensor(rows["clr_t1"][idx], device=dev),
             "clr_valid_t1": torch.as_tensor(rows["clr_valid_t1"][idx], device=dev)})


def train_head(tr, va, H, variant: dict, seed=0, d=256, lr=1e-3, wd=1e-4, bs=512,
               max_epochs=80, patience=10, beta=0.0, dev="cpu", log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    model = HorizonHead(d_h=H.shape[1], d=d, **variant).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # balance horizons: every Δt contributes equally per epoch regardless of how many
    # anchors survive the end-of-clip cut (long horizons lose the clip tail)
    dts, inv = np.unique(tr["dt"], return_inverse=True)
    w = 1.0 / np.bincount(inv)[inv]; w /= w.sum()
    n = len(tr["dt"])
    best, best_state, bad = math.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        order = np.random.choice(n, size=n, replace=True, p=w)
        for k in range(0, n, bs):
            h, s, dt, y = _batch(tr, H, order[k:k + bs], dev)
            loss, _ = horizon_loss(model(h, s, dt), y, beta=beta)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl = eval_loss(model, va, H, dev)
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    log(f"    trained {variant} seed{seed}: {ep + 1} epochs, best val NLL {best:.4f}")
    return model


@torch.no_grad()
def eval_loss(model, rows, H, dev, bs=4096):
    model.eval(); tot, n = 0.0, len(rows["dt"])
    for k in range(0, n, bs):
        idx = np.arange(k, min(n, k + bs))
        h, s, dt, y = _batch(rows, H, idx, dev)
        tot += float(horizon_loss(model(h, s, dt), y)[0]) * len(idx)
    return tot / n


@torch.no_grad()
def predict_rows(model, rows, H, dev, hrow=None, dt=None, bs=4096) -> dict:
    model.eval(); outs = []
    n = len(rows["dt"])
    for k in range(0, n, bs):
        idx = np.arange(k, min(n, k + bs))
        h, s, dtt, _ = _batch(rows, H, idx, dev, hrow=hrow, dt=dt)
        outs.append({k2: v.cpu().numpy() for k2, v in model(h, s, dtt).items()})
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def _shuffle_h_across_clips(rows, rng):
    """Give every row the latent of a frame from a DIFFERENT clip."""
    hrow = rows["hrow"].copy()
    clips = np.unique(rows["clip"])
    perm = np.roll(rng.permutation(clips), 1)            # derangement of clips
    for c, c2 in zip(clips, perm):
        src = np.unique(rows["hrow"][rows["clip"] == c2])
        m = rows["clip"] == c
        hrow[m] = rng.choice(src, size=int(m.sum()))
    return hrow


def run_experiment(tr, va, te, H, horizons_eval, seeds=(0, 1, 2), per_horizon_b3=True,
                   dev="cpu", log=print, **kw) -> dict:
    """Returns {"table": seed-0 horizon table incl. controls, "seeds": per-seed FDE}."""
    preds = {
        "B0_zero": {"disp": np.zeros_like(te["disp"])},
        "B1_cv": {"disp": cv_displacement(torch.as_tensor(te["s"]),
                                          torch.as_tensor(te["dt"])).numpy()},
    }
    seed_fde = {}
    rng = np.random.default_rng(0)
    for name, var in VARIANTS.items():
        for sd in seeds:
            m = train_head(tr, va, H, var, seed=sd, dev=dev, log=log, **kw)
            p = predict_rows(m, te, H, dev)
            seed_fde.setdefault(name, []).append(
                {float(Hh): float(np.linalg.norm(p["disp"] - te["disp"], axis=-1)[np.isclose(te["dt"], Hh)].mean())
                 for Hh in horizons_eval if np.isclose(te["dt"], Hh).any()})
            if sd == seeds[0]:
                preds[name] = p
                if name == "Ours_film":
                    # CONTROL 1: wrong Δt at test time -> must hurt, else Δt is ignored
                    preds["CTRL_shuffled_dt"] = predict_rows(m, te, H, dev, dt=rng.permutation(te["dt"]))
                    # CONTROL 2: latent from another clip -> gain over B2 must vanish
                    preds["CTRL_shuffled_h"] = predict_rows(m, te, H, dev, hrow=_shuffle_h_across_clips(te, rng))
    if per_horizon_b3:
        # B3: one Δt-agnostic head per TRAINED horizon (cannot answer unseen horizons)
        disp = np.full_like(te["disp"], np.nan)
        for Hh in np.unique(tr["dt"]):
            sel = lambda r: {k: v[np.isclose(r["dt"], Hh)] for k, v in r.items()}
            m = train_head(sel(tr), sel(va), H, dict(cond="none", use_h=True), seed=seeds[0], dev=dev, log=log, **kw)
            ts = np.isclose(te["dt"], Hh)
            if ts.any():
                disp[ts] = predict_rows(m, sel(te), H, dev)["disp"]
        ok = ~np.isnan(disp[:, 0])
        preds["B3_per_horizon"] = {"disp": np.where(ok[:, None], disp, 1e6)}   # unseen Δt -> not answerable
    tab = horizon_table(te, preds, horizons_eval)
    if "B3_per_horizon" in tab:
        tab["B3_per_horizon"] = {h: r for h, r in tab["B3_per_horizon"].items() if r["fde"] < 1e5}
    return {"table": tab, "seeds": seed_fde}


def verdict(res: dict, long_h=(1.0, 2.0, 3.0)) -> dict:
    """The Section-7 falsification checks, evaluated mechanically."""
    T = res["table"]; ours = T["Ours_film"]
    hs = sorted(ours)
    chk = {}
    short = hs[0]
    # at the shortest horizon CV is ~exact: Ours must stay within 2 cm of it (absolute, since
    # a ratio against a ~0 CV error is meaningless)
    chk["near_cv_at_shortest"] = ours[short]["fde"] - T["B1_cv"][short]["fde"] < 0.02
    sk = [ours[h]["skill_vs_cv"] for h in hs if h in long_h]
    chk["skill_rises_with_horizon"] = bool(len(sk) > 1 and sk[-1] > sk[0])
    lh = [h for h in hs if h in long_h]
    chk["beats_state_only_CI"] = all(ours[h]["delta_vs_B2_state"][2] < 0 for h in lh)
    chk["shuffled_dt_hurts"] = all(T["CTRL_shuffled_dt"][h]["fde"] > 1.2 * ours[h]["fde"] for h in lh)
    chk["shuffled_h_loses_gain"] = all(
        T["CTRL_shuffled_h"][h]["fde"] >= T["B2_state"][h]["fde"] * 0.97 for h in lh)
    ev = [ours[h].get("skill_vs_cv_event", float("nan")) - ours[h]["skill_vs_cv"] for h in lh]
    chk["gain_concentrated_on_events"] = bool(np.nanmean(ev) > 0)
    chk["ALL"] = all(chk.values())
    return chk


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def save(res: dict, path: str):
    json.dump(to_jsonable(res), open(path, "w"), indent=2)
