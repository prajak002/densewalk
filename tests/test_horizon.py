import json
import math

import numpy as np
import pytest

from densewalk import frames as fr
from densewalk.horizon import dataset as ds

torch = pytest.importorskip("torch")
from densewalk.horizon.head import HorizonHead, cv_displacement, horizon_loss  # noqa: E402


def test_straight_walk_integrates_to_v_times_t():
    t = np.arange(0, 3.01, 0.1)
    P = fr.integrate_planar_odometry(t, np.full(len(t), 1.2), np.zeros(len(t)), np.zeros(len(t)))
    assert np.allclose(fr.se2_relative_xy(P[0], P[-1]), [1.2 * 3.0, 0.0], atol=1e-9)


def test_dir_sign_negative_is_left():
    # DenseWalk: stepping_left has negative direction_deg -> ego +y (left)
    a = fr.label_dir_to_angle(-90.0)
    t = np.array([0.0, 1.0])
    P = fr.integrate_planar_odometry(t, np.array([1.0, 1.0]), np.array([a, a]), np.zeros(2))
    assert np.allclose(fr.se2_relative_xy(P[0], P[1]), [0.0, 1.0], atol=1e-9)


def test_relative_xy_is_in_anchor_ego_frame():
    # after turning 90 deg left, walking "forward" is world +y; seen from the anchor frame
    P_ref = np.array([0.0, 0.0, math.pi / 2])
    assert np.allclose(fr.se2_relative_xy(P_ref, np.array([0.0, 2.0, 0.0])), [2.0, 0.0], atol=1e-9)


def _write_clip(tmp_path, n=31, hz=10.0, bad=(), gap_at=None, v=1.0):
    frames = []
    t = 0.0
    for i in range(n):
        if gap_at is not None and i == gap_at:
            t += 1.0
        frames.append({"image": f"keyframes/c/{i:05d}.jpg", "timestamp": t,
                       "action": {"velocity_mps": v, "direction_deg": 0.0, "label": "walking_normal",
                                  "yaw_rate": 0.0},
                       "navigability": {"nearest_obstacle_m": 3.0},
                       "flags": ["solve_failed"] if i in bad else []})
        t += 1.0 / hz
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"video": "c", "instruction": "go", "frames": frames}))
    return str(p)


def test_targets_and_end_of_clip(tmp_path):
    c = ds.load_clip(_write_clip(tmp_path), ds.SchemaConfig())
    r = ds.build_tuples([c], (0.5, 3.0))
    first = (r["frame"] == 0) & np.isclose(r["dt"], 3.0)
    assert np.allclose(r["disp"][first], [[3.0, 0.0]], atol=1e-5)
    # 3.0 s horizon only exists for the anchor at t=0 in a 3.0 s clip
    assert (np.isclose(r["dt"], 3.0)).sum() == 1
    assert (np.isclose(r["dt"], 0.5)).sum() == 26


def test_never_crosses_solve_failed_or_gap(tmp_path):
    c = ds.load_clip(_write_clip(tmp_path, bad=(10,)), ds.SchemaConfig())
    r = ds.build_tuples([c], (0.5,))
    assert 10 not in r["frame"]
    assert not any(f < 10 <= f + 5 for f in r["frame"])
    c2 = ds.load_clip(_write_clip(tmp_path, gap_at=15), ds.SchemaConfig(max_gap_s=0.5))
    r2 = ds.build_tuples([c2], (0.5,))
    assert 14 not in r2["frame"]           # its target falls inside the 1 s hole


def test_missing_timestamp_raises(tmp_path):
    p = _write_clip(tmp_path)
    d = json.load(open(p))
    for f in d["frames"]:
        del f["timestamp"]
    json.dump(d, open(p, "w"))
    with pytest.raises(ds.SchemaError):
        ds.load_clip(p, ds.SchemaConfig())


def test_head_is_constant_velocity_at_init():
    torch.manual_seed(0)
    m = HorizonHead(d_h=16, d=32)
    s = torch.tensor([[1.3, 0.8, 0.6, 2.0, 1.0]] * 4)
    dt = torch.tensor([0.1, 0.5, 1.0, 3.0])
    out = m(torch.randn(4, 16), s, dt)
    assert torch.allclose(out["disp"], cv_displacement(s, dt), atol=1e-6)
    assert torch.all(out["b_xy"][1:] > out["b_xy"][:-1])      # uncertainty grows with Δt


def test_zero_horizon_is_zero_displacement():
    m = HorizonHead(d_h=16, d=32)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_()
    out = m(torch.randn(3, 16), torch.randn(3, 5), torch.zeros(3))
    assert torch.allclose(out["disp"], torch.zeros(3, 2))


def test_loss_is_finite_and_differentiable():
    m = HorizonHead(d_h=16, d=32)
    out = m(torch.randn(8, 16), torch.randn(8, 5).abs(), torch.rand(8) * 3)
    y = {"disp": torch.randn(8, 2), "clr_t1": torch.rand(8) * 5, "clr_valid_t1": torch.ones(8)}
    loss, _ = horizon_loss(out, y)
    loss.backward()
    assert math.isfinite(float(loss))


def test_split_is_deterministic_and_disjoint(tmp_path):
    c = ds.load_clip(_write_clip(tmp_path), ds.SchemaConfig())
    import copy
    clips = []
    for k in range(20):
        cc = copy.deepcopy(c); cc.clip_id = f"c{k}"; clips.append(cc)
    a = [[x.clip_id for x in s] for s in ds.notebook_split(clips)]
    b = [[x.clip_id for x in s] for s in ds.notebook_split(clips)]
    assert a == b
    assert not (set(a[0]) & set(a[2])) and not (set(a[0]) & set(a[1]))
