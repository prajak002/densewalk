"""Analytical static-visibility-fraction computation.

For datasets with lidar + 3D cuboids (JRDB), visibility of a background
region is computed from lidar-ray occlusion. WildTrack has no lidar, but
does have calibrated multi-view cameras and annotated ground-plane person
positions -- so the equivalent analytical signal here is multi-view
geometric occlusion: for each camera and each time frame, does the line
segment from the camera center to a background point pass through any
person's body (approximated as a vertical cylinder)? This is derived
purely from calibration + annotated positions, not estimated or learned.

All points and person positions are in the project's canonical world
frame (metres, Z-up, floor at z=0) -- see frames.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_PERSON_RADIUS_M = 0.25
DEFAULT_PERSON_HEIGHT_M = 1.8


@dataclass(frozen=True)
class Person:
    id: int
    x: float
    y: float
    radius: float = DEFAULT_PERSON_RADIUS_M
    height: float = DEFAULT_PERSON_HEIGHT_M


@dataclass(frozen=True)
class FrameOccupants:
    """The set of people present at one time frame (ground positions),
    used to test occlusion of background points at that frame."""

    people: list[Person] = field(default_factory=list)


def segment_intersects_cylinder(
    cam: np.ndarray, point: np.ndarray, person: Person, eps: float = 1e-12
) -> bool:
    """True if the 3D segment cam->point passes through `person`'s body,
    modelled as a vertical cylinder of given radius/height rooted at
    (person.x, person.y, z=0)."""
    cam = np.asarray(cam, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    d = point - cam
    a = d[0] ** 2 + d[1] ** 2

    if a < eps:
        # segment has no horizontal extent: constant xy, either always
        # inside the cylinder's circle or never.
        inside = (cam[0] - person.x) ** 2 + (cam[1] - person.y) ** 2 <= person.radius**2
        if not inside:
            return False
        s_lo, s_hi = 0.0, 1.0
    else:
        b = 2.0 * (d[0] * (cam[0] - person.x) + d[1] * (cam[1] - person.y))
        c = (cam[0] - person.x) ** 2 + (cam[1] - person.y) ** 2 - person.radius**2
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return False
        sqrt_disc = np.sqrt(disc)
        s1 = (-b - sqrt_disc) / (2.0 * a)
        s2 = (-b + sqrt_disc) / (2.0 * a)
        s_lo, s_hi = max(0.0, min(s1, s2)), min(1.0, max(s1, s2))
        if s_lo > s_hi:
            return False

    z_lo_s = cam[2] + s_lo * d[2]
    z_hi_s = cam[2] + s_hi * d[2]
    z_min, z_max = min(z_lo_s, z_hi_s), max(z_lo_s, z_hi_s)
    return bool(z_min <= person.height and z_max >= 0.0)


def is_point_visible_from_camera(
    point: np.ndarray, cam: np.ndarray, people: list[Person]
) -> bool:
    """A background point is visible from a camera if no person's body
    occludes the line of sight."""
    return not any(segment_intersects_cylinder(cam, point, p) for p in people)


def is_point_visible_any_camera(
    point: np.ndarray, cams: list[np.ndarray], people: list[Person]
) -> bool:
    """Visible at this time frame if at least one camera has a clear
    line of sight -- reconstruction only needs one usable observation."""
    return any(is_point_visible_from_camera(point, cam, people) for cam in cams)


def static_visibility_fraction(
    points: np.ndarray,
    cams: list[np.ndarray],
    frames: list[FrameOccupants],
) -> np.ndarray:
    """For each of the (N,3) background `points`, the fraction of `frames`
    in which it is visible from at least one camera in `cams`."""
    if len(frames) == 0:
        raise ValueError("static_visibility_fraction requires at least one frame")
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    n = points.shape[0]
    visible_count = np.zeros(n, dtype=np.float64)
    for frame in frames:
        for i in range(n):
            if is_point_visible_any_camera(points[i], cams, frame.people):
                visible_count[i] += 1.0
    return visible_count / len(frames)


def static_visibility_fraction_view_level_batch(
    points: np.ndarray,
    cams: list[np.ndarray],
    frames: list[FrameOccupants],
) -> np.ndarray:
    """Fraction of (camera, frame) observation pairs in which each point
    is visible from that specific camera -- NOT collapsed across cameras
    with an OR first. With C cameras and F frames this gives up to C*F
    distinct values per point, vs. at most F+1 for the frame-level metric
    above (visible from >=1 camera). Use this one for stratification --
    it is 'share of views unoccluded', a different and finer-grained
    quantity than 'share of frames unoccluded'."""
    if len(frames) == 0:
        raise ValueError("static_visibility_fraction_view_level_batch requires >=1 frame")
    if len(cams) == 0:
        raise ValueError("static_visibility_fraction_view_level_batch requires >=1 camera")
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    points_xy = points[:, :2]
    points_z = points[:, 2]
    n = points.shape[0]
    visible_count = np.zeros(n, dtype=np.float64)
    total_pairs = len(cams) * len(frames)

    for frame in frames:
        if len(frame.people) == 0:
            visible_count += float(len(cams))
            continue
        px = np.array([p.x for p in frame.people], dtype=np.float64)
        py = np.array([p.y for p in frame.people], dtype=np.float64)
        radius = np.array([p.radius for p in frame.people], dtype=np.float64)
        height = np.array([p.height for p in frame.people], dtype=np.float64)
        for cam in cams:
            occluded = _occluded_mask_one_camera(points_xy, points_z, np.asarray(cam), px, py, radius, height)
            visible_count += (~occluded).astype(np.float64)

    return visible_count / total_pairs


def _occluded_mask_one_camera(
    points_xy: np.ndarray,
    points_z: np.ndarray,
    cam: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    radius: np.ndarray,
    height: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """Vectorized twin of segment_intersects_cylinder + is_point_visible_
    from_camera's aggregation, for one camera against all (N) points and
    all (M) people at once. Returns an (N,) occluded-by-any-person mask."""
    n = points_xy.shape[0]
    m = px.shape[0]
    if m == 0:
        return np.zeros(n, dtype=bool)

    dx = points_xy[:, 0] - cam[0]  # (N,)
    dy = points_xy[:, 1] - cam[1]
    dz = points_z - cam[2]
    a = dx**2 + dy**2  # (N,)

    ex = cam[0] - px  # (M,)
    ey = cam[1] - py
    c = ex**2 + ey**2 - radius**2  # (M,)
    b = 2.0 * (dx[:, None] * ex[None, :] + dy[:, None] * ey[None, :])  # (N,M)

    a_safe = np.where(a < eps, 1.0, a)  # avoid /0; a<eps points handled below
    disc = b**2 - 4.0 * a_safe[:, None] * c[None, :]
    valid = disc >= 0.0
    sqrt_disc = np.sqrt(np.clip(disc, 0.0, None))
    s1 = (-b - sqrt_disc) / (2.0 * a_safe[:, None])
    s2 = (-b + sqrt_disc) / (2.0 * a_safe[:, None])
    s_lo = np.maximum(0.0, np.minimum(s1, s2))
    s_hi = np.minimum(1.0, np.maximum(s1, s2))
    valid &= s_lo <= s_hi

    z_lo = cam[2] + s_lo * dz[:, None]
    z_hi = cam[2] + s_hi * dz[:, None]
    z_min = np.minimum(z_lo, z_hi)
    z_max = np.maximum(z_lo, z_hi)
    hits = valid & (z_min <= height[None, :]) & (z_max >= 0.0)

    # degenerate a~0 case: point directly above/below the camera in xy
    degenerate = a < eps
    if np.any(degenerate):
        inside_circle = ex[None, :] ** 2 + ey[None, :] ** 2 <= radius[None, :] ** 2
        z_deg_lo = np.minimum(cam[2], cam[2] + dz[:, None])
        z_deg_hi = np.maximum(cam[2], cam[2] + dz[:, None])
        deg_hits = inside_circle & (z_deg_lo <= height[None, :]) & (z_deg_hi >= 0.0)
        hits[degenerate] = deg_hits[degenerate]

    return hits.any(axis=1)


def static_visibility_fraction_batch(
    points: np.ndarray,
    cams: list[np.ndarray],
    frames: list[FrameOccupants],
) -> np.ndarray:
    """Vectorized equivalent of static_visibility_fraction. Loops over
    frames and cameras (both typically small: tens-hundreds and single
    digits respectively) but evaluates all points against all people in
    a frame with numpy broadcasting, instead of Python-level loops over
    points and people. Use this for real datasets; the scalar functions
    above remain the tested reference semantics."""
    if len(frames) == 0:
        raise ValueError("static_visibility_fraction_batch requires at least one frame")
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    points_xy = points[:, :2]
    points_z = points[:, 2]
    n = points.shape[0]
    visible_count = np.zeros(n, dtype=np.float64)

    for frame in frames:
        if len(frame.people) == 0:
            visible_count += 1.0
            continue
        px = np.array([p.x for p in frame.people], dtype=np.float64)
        py = np.array([p.y for p in frame.people], dtype=np.float64)
        radius = np.array([p.radius for p in frame.people], dtype=np.float64)
        height = np.array([p.height for p in frame.people], dtype=np.float64)

        occluded_all_cams = np.ones(n, dtype=bool)
        for cam in cams:
            occluded = _occluded_mask_one_camera(points_xy, points_z, np.asarray(cam), px, py, radius, height)
            occluded_all_cams &= occluded
        visible_count += (~occluded_all_cams).astype(np.float64)

    return visible_count / len(frames)
