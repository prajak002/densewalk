"""Δt-horizon training tuples from DenseWalk clip JSONs (json_openvla/*.json).

One row per (clip, anchor frame, Δt):
    s      = [v, cos a, sin a, clr, clr_valid]    ego state at t (odometry/depth at deploy)
    disp   = (dx, dy) ego position at t+Δt, in the ego frame at t (x fwd, y left), metres
    clr_t1 = nearest-obstacle distance at t+Δt (masked by clr_valid_t1)
    event  = 1 if the 6-class mode changes anywhere in (t, t+Δt]

Fields the DCA notebook already reads (verified in its load_clip): frames[].action
{velocity_mps, direction_deg, label}, frames[].navigability.nearest_obstacle_m,
frames[].flags, frames[].image. The TIMESTAMP and YAW-RATE fields have NOT been
audited yet -- the candidate key lists below are guesses and `audit()` exists to
settle them. A missing field raises SchemaError; it is never silently defaulted.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from densewalk import frames as fr

# Keyframes are 6 Hz (median gap 0.167 s, audited over 250 clips), so 0.1 s is NOT labelable:
# a 0.1 s target would be an interpolation of a zero-order-hold integration, i.e. constant
# velocity by construction. The shortest real horizon is one keyframe.
HORIZONS_TRAIN = (1 / 6, 0.5, 1.0, 2.0, 3.0)
HORIZONS_INTERP = (0.33, 0.75, 1.5, 2.5)   # evaluation only: never trained on

# same mapping as the DCA notebook's MODE6_FROM_LABEL
MODE6_FROM_LABEL = {
    "standing_still": 0,
    "walking_slow": 1, "walking_normal": 1, "walking_fast": 1,
    "turning_left": 2, "turning_right": 3,
    "stepping_left": 4, "stepping_right": 5,
}
STOP_LABELS = {"standing_still"}
TURN_STEP = {"turning_left", "turning_right", "stepping_left", "stepping_right"}

TIME_KEYS = ("time_sec", "timestamp", "timestamp_s", "t", "time", "time_s", "ts", "t_sec")
FRAME_IDX_KEYS = ("frame_idx", "frame_index", "frame", "index", "src_frame")
YAW_KEYS = ("yaw_rate", "yaw_rate_dps", "yaw_rate_deg_s", "yaw_rate_rad_s", "omega")


class SchemaError(KeyError):
    pass


@dataclass
class SchemaConfig:
    time_key: str | None = None        # None -> search TIME_KEYS, then FRAME_IDX_KEYS + fps
    yaw_key: str | None = None         # None -> search YAW_KEYS in frame and frame["action"]
    yaw_unit: str = "deg"              # "deg" or "rad" per second; *_rad_s keys force rad
    # DenseWalk yaw_rate_dps is LEFT-NEGATIVE (audited 2026-09-25 over 250 clips: turning_left
    # is negative in 1256/1256 frames, turning_right positive in 1470/1470) -> flip to CCW+
    yaw_sign: float = -1.0
    dir_sign: float = fr.DENSEWALK_DIR_SIGN
    assume_zero_yaw: bool = False      # ONLY for a first look if no yaw exists; biases turns
    max_gap_s: float = 0.5
    close_m: float = 2.0


@dataclass
class Clip:
    clip_id: str
    instruction: str
    t: np.ndarray
    v: np.ndarray
    angle: np.ndarray          # motion angle rel. body forward, rad, CCW+
    yaw_rate: np.ndarray       # rad/s, CCW+
    clr: np.ndarray
    clr_valid: np.ndarray
    mode6: np.ndarray
    labels: list[str]
    images: list[str]
    bad: np.ndarray            # solve_failed flags (kept in the time axis, never used as anchors)
    poses: np.ndarray = field(default=None)


def _find(d: dict, keys, where: str):
    for k in keys:
        for src in (d, d.get("action") or {}):
            if k in src and isinstance(src[k], (int, float)):
                return k, float(src[k])
    return None, None


def audit(path: str) -> dict:
    """Print and return the schema of one clip JSON: top-level keys, per-frame keys,
    and which timestamp / yaw candidates resolve. Run this FIRST on real data."""
    d = json.load(open(path))
    f0 = d["frames"][0]
    rep = {
        "top_keys": sorted(d.keys()),
        "frame_keys": sorted(f0.keys()),
        "action_keys": sorted((f0.get("action") or {}).keys()),
        "navigability_keys": sorted((f0.get("navigability") or {}).keys()),
        "time_key": _find(f0, TIME_KEYS, "")[0],
        "frame_idx_key": _find(f0, FRAME_IDX_KEYS, "")[0],
        "fps": d.get("fps") or d.get("frame_rate"),
        "yaw_key": _find(f0, YAW_KEYS, "")[0],
        "n_frames": len(d["frames"]),
    }
    ts = _clip_times(d, SchemaConfig(), strict=False)
    if ts is not None and len(ts) > 1:
        gaps = np.diff(ts)
        rep["frame_gap_s"] = {"median": float(np.median(gaps)), "min": float(gaps.min()),
                              "max": float(gaps.max())}
        rep["clip_duration_s"] = float(ts[-1] - ts[0])
        rep["dt_0.1_labelable"] = bool(np.median(gaps) <= 0.1 + 1e-6)
    for k, v in rep.items():
        print(f"  {k:20s} {v}")
    return rep


def _clip_times(d: dict, cfg: SchemaConfig, strict: bool = True):
    fs = d["frames"]
    keys = (cfg.time_key,) if cfg.time_key else TIME_KEYS
    k, _ = _find(fs[0], keys, "time")
    if k:
        return np.array([_find(f, (k,), "time")[1] for f in fs])
    fps = d.get("fps") or d.get("frame_rate")
    k, _ = _find(fs[0], FRAME_IDX_KEYS, "frame")
    if k and fps:
        return np.array([_find(f, (k,), "frame")[1] for f in fs]) / float(fps)
    if strict:
        raise SchemaError(
            f"no timestamp in clip JSON. Tried frame keys {TIME_KEYS} and "
            f"{FRAME_IDX_KEYS}+top-level fps. Frame keys present: {sorted(fs[0].keys())}. "
            "Set SchemaConfig.time_key.")
    return None


def load_clip(path: str, cfg: SchemaConfig) -> Clip:
    d = json.load(open(path))
    fs = d["frames"]
    t = _clip_times(d, cfg)
    if cfg.assume_zero_yaw:
        yaw = np.zeros(len(fs))
    else:
        keys = (cfg.yaw_key,) if cfg.yaw_key else YAW_KEYS
        yk, _ = _find(fs[0], keys, "yaw")
        if yk is None:
            raise SchemaError(
                f"no yaw rate in clip JSON (tried {keys} in frame and frame['action']). "
                "Set SchemaConfig.yaw_key, or assume_zero_yaw=True for a biased first look.")
        yaw = np.array([_find(f, (yk,), "yaw")[1] for f in fs])
        if cfg.yaw_unit == "deg" and not yk.endswith("rad_s"):
            yaw = np.radians(yaw)
        yaw = yaw * cfg.yaw_sign
    instr = d["instruction"]["text"] if isinstance(d["instruction"], dict) else d["instruction"]
    clr, clr_valid = [], []
    for f in fs:
        nod = (f.get("navigability") or {}).get("nearest_obstacle_m")
        ok = isinstance(nod, (int, float)) and nod >= 0
        clr.append(float(nod) if ok else 0.0); clr_valid.append(float(ok))
    order = np.argsort(t, kind="stable")
    pick = lambda xs: [xs[i] for i in order]
    c = Clip(
        clip_id=str(d.get("video", d.get("video_id", os.path.basename(path)))),
        instruction=instr,
        t=t[order],
        v=np.array([float(f["action"]["velocity_mps"]) for f in pick(fs)]),
        angle=fr.label_dir_to_angle([float(f["action"]["direction_deg"]) for f in pick(fs)],
                                    cfg.dir_sign),
        yaw_rate=yaw[order],
        clr=np.array(clr)[order], clr_valid=np.array(clr_valid)[order],
        mode6=np.array([MODE6_FROM_LABEL[f["action"]["label"]] for f in pick(fs)]),
        labels=[f["action"]["label"] for f in pick(fs)],
        images=[f["image"] for f in pick(fs)],
        bad=np.array(["solve_failed" in (f.get("flags") or []) for f in pick(fs)]),
    )
    # solve_failed motion is a fabricated zero placeholder: zero-order-hold the previous
    # good sample through it for integration, and never cross it when building targets
    v, a, w = c.v.copy(), c.angle.copy(), c.yaw_rate.copy()
    for k in range(1, len(v)):
        if c.bad[k]:
            v[k], a[k], w[k] = v[k - 1], a[k - 1], w[k - 1]
    c.poses = fr.integrate_planar_odometry(c.t, v, a, w)
    return c


def state_vector(c: Clip, i: int) -> np.ndarray:
    return np.array([c.v[i], math.cos(c.angle[i]), math.sin(c.angle[i]),
                     c.clr[i], c.clr_valid[i]], dtype=np.float32)


def build_tuples(clips: list[Clip], horizons=HORIZONS_TRAIN, cfg: SchemaConfig | None = None):
    """-> dict of numpy arrays, one row per valid (clip, anchor, Δt)."""
    cfg = cfg or SchemaConfig()
    rows = defaultdict(list)
    for ci, c in enumerate(clips):
        T = c.t
        for i in range(len(T)):
            if c.bad[i]:
                continue
            for dt in horizons:
                t1 = T[i] + dt
                if t1 > T[-1] + 1e-9:
                    continue                                  # leaves the clip: no target
                j1 = int(np.searchsorted(T, t1 - 1e-9))
                j0 = max(j1 - 1, 0)
                if abs(T[j1] - t1) < 1e-9:
                    j0, alpha = j1, 0.0
                else:
                    alpha = (t1 - T[j0]) / (T[j1] - T[j0])
                if T[j1] - T[j0] > cfg.max_gap_s:
                    continue
                if c.bad[i + 1:j1 + 1].any():
                    continue                                  # never integrate across a failed solve
                p1 = fr.se2_interp(c.poses[j0], c.poses[j1], alpha)
                disp = fr.se2_relative_xy(c.poses[i], p1)
                if c.clr_valid[j0] and c.clr_valid[j1]:
                    clr1, cv1 = (1 - alpha) * c.clr[j0] + alpha * c.clr[j1], 1.0
                else:
                    clr1, cv1 = 0.0, 0.0
                rows["clip"].append(ci)
                rows["frame"].append(i)
                rows["dt"].append(dt)
                rows["s"].append(state_vector(c, i))
                rows["disp"].append(disp.astype(np.float32))
                rows["clr_t1"].append(clr1)
                rows["clr_valid_t1"].append(cv1)
                rows["event"].append(float((c.mode6[i + 1:j1 + 1] != c.mode6[i]).any()))
    out = {k: np.asarray(v, dtype=np.float32) for k, v in rows.items()}
    for k in ("clip", "frame"):
        out[k] = out[k].astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# Split: a faithful copy of the DCA notebook's stratified 3-way split (cell 8), so the
# horizon head is trained on exactly the clips the V2_full LoRA was trained on and
# tested on clips it never saw. Clip ORDER matters: sorted json paths, as the notebook.
# ---------------------------------------------------------------------------
def notebook_split(clips: list[Clip], test_frac=0.20, val_frac=0.10, seed=42):
    def stratum(c):
        labs = {l for l, b in zip(c.labels, c.bad) if not b}   # notebook drops solve_failed first
        return (bool(labs & STOP_LABELS), bool(labs & TURN_STEP))
    by = defaultdict(list)
    for c in clips:
        by[stratum(c)].append(c)
    rng = random.Random(seed)
    tr, va, te = [], [], []
    for _, g in sorted(by.items()):
        g = g[:]; rng.shuffle(g)
        n = len(g)
        n_te = max(1, round(n * test_frac)) if n > 2 else (1 if n > 1 else 0)
        n_va = max(1, round(n * val_frac)) if (val_frac > 0 and n - n_te > 1) else 0
        te += g[:n_te]; va += g[n_te:n_te + n_va]; tr += g[n_te + n_va:]
    return tr, va, te


def load_all(json_dir_glob: str, cfg: SchemaConfig) -> list[Clip]:
    paths = sorted(glob.glob(json_dir_glob))
    if not paths:
        raise FileNotFoundError(f"no clip JSONs match {json_dir_glob}")
    clips = [load_clip(p, cfg) for p in paths]
    return [c for c in clips if len(c.t) > 0]


def attach_hrow(rows: dict, clips: list[Clip], image_to_row: dict[str, int]) -> dict:
    """Add rows["hrow"]: index into the latent cache for each anchor frame.
    The cache is keyed by image path, exactly like the notebook's Z0_CACHE."""
    hr = np.empty(len(rows["clip"]), dtype=np.int64)
    for n, (ci, fi) in enumerate(zip(rows["clip"], rows["frame"])):
        key = clips[ci].images[fi]
        if key not in image_to_row:
            key = os.path.basename(os.path.dirname(key)) + "/" + os.path.basename(key)
        hr[n] = image_to_row[key]
    rows["hrow"] = hr
    return rows
