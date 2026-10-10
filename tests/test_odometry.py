"""Metric distances: floor-plane visual odometry, fusing it with the steps, and measured distances in the review."""

import json

import cv2
import numpy as np
import pytest
from test_navigation import BUILD, lshape

from nav.mapping.build import finish_map
from nav.mapping.motion import RATE_HZ, MotionTrack
from nav.mapping.odometry import FloorSegment, floor_motion, fuse, intrinsics, to_floor
from nav.mapping.review import MeasuredDistance, Review, apply_review, from_template, template

W, H = 540, 960
FOV = 36.0
HEIGHT = 1.3


def _floor_texture(seed: int = 0) -> np.ndarray:
    """20 m x 20 m of floor at 1 cm per pixel, with blobs for the tracker."""
    rng = np.random.default_rng(seed)
    tex = np.full((2000, 2000), 120, np.uint8)
    for _ in range(6000):
        x, y, r = int(rng.integers(0, 2000)), int(rng.integers(0, 2000)), int(rng.integers(2, 8))
        cv2.circle(tex, (x, y), r, int(rng.integers(0, 255)), -1)
    return tex


def _render(tex: np.ndarray, up: np.ndarray, offset_m: tuple[float, float], yaw_deg: float = 0.0) -> np.ndarray:
    """What a camera at HEIGHT sees of the floor, moved by `offset_m` and turned by `yaw_deg` on the spot."""
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    px = np.c_[uu.ravel(), vv.ravel()].astype(float)
    q, hit = to_floor(px, up, np.linalg.inv(intrinsics(W, H, FOV)), HEIGHT)
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    world = q @ np.array([[c, -s], [s, c]]).T + np.array(offset_m)
    map_x = np.full(H * W, -1.0, np.float32)
    map_y = np.full(H * W, -1.0, np.float32)
    map_x[hit], map_y[hit] = (world[:, 0] + 10) * 100, world[:, 1] * 100
    return cv2.remap(tex, map_x.reshape(H, W), map_y.reshape(H, W), cv2.INTER_LINEAR)


def test_floor_motion_measures_walking_and_ignores_turning_on_the_spot():
    tex = _floor_texture()
    k = intrinsics(W, H, FOV)
    pitch = np.radians(15)  # phone tilted 15 degrees down, like the Itnig walk
    up = np.array([0.0, -np.cos(pitch), -np.sin(pitch)])
    start = _render(tex, up, (0.0, 0.0))
    d, scale, inliers = floor_motion([start, _render(tex, up, (0.0, 0.30))], up, up, k, HEIGHT)
    assert inliers >= 15 and scale == pytest.approx(1.0, abs=0.05)
    assert d == pytest.approx(0.30, abs=0.03)
    d, _, _ = floor_motion([start, _render(tex, up, (0.0, 0.0), yaw_deg=4.0)], up, up, k, HEIGHT)
    assert d < 0.05


def _walk_track(seconds: float = 20.0) -> MotionTrack:
    """Stand 0-2 s, walk east 2-18 s with a step every 0.5 s (0.7 m each by the recorder's default), stand."""
    t = np.arange(0, seconds * 1000 + 1, 1000 / RATE_HZ)
    step_t = np.arange(2000, 18000, 500.0)
    steps = np.isin(np.round(t), step_t)
    walked = np.cumsum(steps) * 0.7
    n = len(t)
    return MotionTrack(t, walked, np.full(n, 90.0), walked.copy(), np.zeros(n), np.zeros(n),
                       (t >= 2000) & (t < 18000), step_t, True, "chrome", up=np.tile([0, 1.0, 0], (n, 1)),
                       step_length_m=0.7)


def test_fuse_uses_the_video_rejects_jitter_and_fills_gaps_with_rescaled_steps():
    track = _walk_track()
    segs = []
    for t0 in np.arange(0, 20000, 500.0):
        walking = 2000 <= t0 < 18000
        gap = 8000 <= t0 < 9000  # the floor can't be seen for a second
        segs.append(FloorSegment(t0, t0 + 500, 0.3 if walking else 0.03, 0.0 if gap else 1.0, 0 if gap else 200))
    f = fuse(track, segs, step_length_m=0.7)
    assert f.step_length_m == pytest.approx(0.3, abs=0.01)  # video metres / steps
    # 30 walking half-seconds from the video (0.3 m each) + 2 from the steps at the re-estimated 0.3 m; no jitter
    assert f.track.walked_m[-1] == pytest.approx(32 * 0.3, abs=0.05)
    assert f.track.distance_source == "video" and f.video_share == pytest.approx(30 / 32, abs=0.01)
    assert f.track.x_m[-1] == pytest.approx(f.track.walked_m[-1], abs=0.05) and abs(f.track.y_m[-1]) < 1e-6
    assert fuse(track, None, 0.7).track.walked_m[-1] == track.walked_m[-1]  # no odometry: the steps


def test_measured_distances_rescale_the_published_map():
    draft, cset, tracks = lshape()
    raw = finish_map(draft, cset, BUILD, "builds/b", tracks)
    e1 = next(e for e in raw.edges if e.id == "e1")
    assert e1.distance_m == pytest.approx(10)  # n1 -> n2 walked 10 m east
    review = Review(distances=[MeasuredDistance(between=("n1", "n2"), meters=7.0, sigma_m=0.3, note="tape"),
                               MeasuredDistance(between=("n1", "n9"), meters=3.0)])
    m = apply_review(raw, review)
    e1m = next(e for e in m.edges if e.id == "e1")
    assert e1m.distance_m == pytest.approx(7.0)
    assert m.frame.distance_scale == pytest.approx(0.7)
    n2 = next(n for n in m.nodes if n.id == "n2")
    assert n2.position.x_m == pytest.approx(next(n for n in raw.nodes if n.id == "n2").position.x_m * 0.7, abs=0.1)
    check = m.frame.measured_distances[0]
    assert check.map_m == pytest.approx(7.0) and abs(check.residual_m) < 0.01
    assert m.frame.measured_distances[1].map_m is None  # n9 isn't in the map: reported, not used
    assert any("n1 - n9" in w for w in m.warnings)
    assert e1.distance_m == pytest.approx(10)  # the build's map is untouched

    t = json.loads(json.dumps(template(raw, review), default=str))
    assert from_template(t).distances == review.distances
