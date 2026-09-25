"""Synthetic DenseWalk-format clips with a KNOWN answer, to validate the horizon pipeline
before real data/weights are available.

Each clip: a walker at 10 Hz who stops, turns and sidesteps at random. The fake "latent"
h(t) encodes, with noise, what happens in the next 3 s (upcoming event type + time until it).
    signal > 0 -> the image contains future information: Ours must beat state-only B2
    signal = 0 -> h is pure noise: Ours must NOT beat B2 (no false discovery)
This is a pipeline test, not evidence about the real model.
"""
from __future__ import annotations

import json
import os

import numpy as np

EVENTS = ("walk", "stand", "turn_left", "turn_right", "step_left", "step_right")
LABEL = {"walk": None, "stand": "standing_still", "turn_left": "turning_left",
         "turn_right": "turning_right", "step_left": "stepping_left", "step_right": "stepping_right"}


def _walk_label(v):
    return "walking_slow" if v < 0.5 else ("walking_normal" if v < 1.0 else "walking_fast")


def make_clip(rng, n=150, hz=10.0):
    """-> (frames list in notebook JSON schema, per-frame event id array)."""
    ev = np.zeros(n, dtype=int)
    k = int(rng.integers(10, 30))
    while k < n:
        e = int(rng.integers(1, len(EVENTS)))
        dur = int(rng.integers(8, 20))
        ev[k:k + dur] = e
        k += dur + int(rng.integers(15, 40))
    v0 = rng.uniform(0.8, 1.5)
    clr = np.clip(4 + np.cumsum(rng.normal(0, 0.15, n)), 0.3, 9.0)
    frames = []
    for i in range(n):
        name = EVENTS[ev[i]]
        v, d, yaw = v0, 0.0, 0.0
        if name == "stand":
            v = 0.0
            clr[i] = min(clr[i], 1.2 + 0.2 * rng.random())    # people stop because it's crowded
        elif name == "turn_left":
            yaw = 35.0
        elif name == "turn_right":
            yaw = -35.0
        elif name == "step_left":
            d = -70.0                                        # dir_deg: negative = LEFT
        elif name == "step_right":
            d = 70.0
        lab = LABEL[name] or _walk_label(v)
        frames.append({
            "image": f"kf/{i:05d}.jpg", "timestamp": i / hz,
            "action": {"velocity_mps": float(v + rng.normal(0, 0.03)) if v > 0 else 0.0,
                       "direction_deg": float(d + rng.normal(0, 2.0)), "label": lab,
                       "mode": "stand" if name == "stand" else "walk",
                       "yaw_rate": float(yaw + rng.normal(0, 1.0))},   # deg/s, CCW(left)+
            "navigability": {"nearest_obstacle_m": float(clr[i])},
            "flags": [],
        })
    return frames, ev


def future_code(ev, i, hz=10.0, horizon_s=3.0):
    """What an oracle 'image' would reveal at frame i: one-hot next event + time to it."""
    code = np.zeros(len(EVENTS) + 1)
    j_end = min(len(ev), i + int(horizon_s * hz) + 1)
    for j in range(i, j_end):
        if ev[j] != ev[i] or j == i and ev[i] != 0:
            code[ev[j]] = 1.0
            code[-1] = 1.0 - (j - i) / (horizon_s * hz)
            break
    else:
        code[0] = 1.0
    return code


def write_dataset(root, n_clips=60, signal=1.0, d_h=64, noise=1.0, seed=0):
    """Writes root/json/*.json and root/latents.npz (images, h). Returns (json_glob, latents_path)."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "json"), exist_ok=True)
    W = rng.normal(0, 1.0, (len(EVENTS) + 1, d_h))
    images, H = [], []
    for c in range(n_clips):
        frames, ev = make_clip(rng)
        vid = f"syn{c:03d}"
        for i, f in enumerate(frames):
            f["image"] = f"keyframes/{vid}/{i:05d}.jpg"
            images.append(f["image"])
            H.append(signal * future_code(ev, i) @ W + noise * rng.normal(0, 1.0, d_h))
        json.dump({"video": vid, "instruction": "walk through the crowd", "frames": frames},
                  open(os.path.join(root, "json", f"{vid}.json"), "w"))
    lat = os.path.join(root, "latents.npz")
    np.savez(lat, images=np.array(images), h=np.asarray(H, dtype=np.float16))
    return os.path.join(root, "json", "*.json"), lat
