"""The 3D model put in the map frame (nav/mapping/geometry.py place_in_map), on a synthetic model: cameras of a walk
whose dead reckoning is known, seen in a model frame that is rotated, scaled and shifted."""

import json
import math

import numpy as np
import pytest

from nav.mapping import geometry


class Track:
    """MotionTrack.at for a walk east 10 m (0-10 s) then north 10 m, 1 m/s, compass headings."""

    def __init__(self, drift_after_s: float | None = None, drift_m: float = 0.0):
        self.drift_after_s, self.drift_m = drift_after_s, drift_m

    def xy(self, t_s: float) -> tuple[float, float]:
        x, y = (t_s, 0.0) if t_s <= 10 else (10.0, t_s - 10)
        return x, y

    def at(self, t_ms: float) -> dict:
        t = t_ms / 1000
        x, y = self.xy(t)
        if self.drift_after_s is not None and t > self.drift_after_s:
            x += self.drift_m
        return {"walked_m": t, "heading_deg": 90.0 if t <= 10 else 0.0, "x_m": x, "y_m": y, "pitch_deg": 0.0,
                "moving": True}


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def cam_from_map(heading_deg: float) -> np.ndarray:
    """An upright camera facing `heading_deg` (clockwise from +y): rows are its right, down and forward axes."""
    h = math.radians(heading_deg)
    f = np.array([math.sin(h), math.cos(h), 0.0])
    d = np.array([0.0, 0.0, -1.0])
    return np.stack([np.cross(d, f), d, f])


def synthetic_job(tmp_path, flung: dict[int, tuple[float, float]] | None = None):
    """A job folder and raw.npz: one walk sampled every 0.5 s; the model frame = map frame rotated 70° about an
    oblique axis, scaled by 0.5 and shifted. `flung`: frame index -> metres added to its model position (a wrong
    join)."""
    work = tmp_path / "work"
    work.mkdir()
    times = np.arange(0, 20.01, 0.5)
    frames = [geometry.GeoFrame(name=f"r1/{round(t * 1000):07d}.jpg", rec_id="r1", t_ms=float(t * 1000),
                                sharpness=10.0, brightness=100.0) for t in times]
    (work / "job.json").write_text(json.dumps({"version": 1, "place_model": "m",
                                               "frames": [f.__dict__ for f in frames],
                                               "walks": {"r1": {"fov_deg": 50, "width": 576, "height": 1024}}}))
    axis = np.array([1.0, 2.0, 3.0]) / math.sqrt(14)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = math.radians(70)
    Q = np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k  # model <- map rotation
    s_m, o = 0.5, np.array([3.0, -2.0, 7.0])
    track = Track()
    n = len(frames)
    R, t = np.zeros((n, 3, 3)), np.zeros((n, 3))
    for i, f in enumerate(frames):
        x, y = track.xy(f.t_ms / 1000)
        c_map = np.array([x, y, 1.4])
        if flung and i in flung:
            c_map = c_map + np.array([*flung[i], 0.0])
        Rm = cam_from_map(track.at(f.t_ms)["heading_deg"]) @ Q.T  # cam <- model
        c_model = s_m * Q @ c_map + o
        R[i], t[i] = Rm, -Rm @ c_model
    pts_map = np.array([[5.0, 5.0, 2.0], [12.0, 15.0, 1.0], [8.0, 25.0, 0.5]])
    points = (s_m * pts_map @ Q.T) + o
    obs_frame = np.array([0, 1, 2, 3, 4, 5], np.int32)
    obs_kp = np.zeros(6, np.int32)
    obs_point = np.array([0, 0, 1, 1, 2, 2], np.int32)
    kps = np.tile(np.array([[10.0, 20.0]], np.float32), (n, 1))
    np.savez(work / "raw.npz", registered=np.ones(n, bool), R=R, t=t, focal=np.full(n, 800.0), points=points,
             obs_frame=obs_frame, obs_kp=obs_kp, obs_point=obs_point, kps=kps, kp_offsets=np.arange(n + 1),
             place=np.eye(n, 4, dtype=np.float32))
    return work, frames, pts_map


def test_the_model_lands_where_the_walk_is_levelled_scaled_and_turned(tmp_path):
    work, frames, pts_map = synthetic_job(tmp_path)
    geo = geometry.place_in_map(work, {"r1": Track()}, "r1")
    assert geo.posed.all() and geo.info["scale"] == pytest.approx(2.0, rel=1e-3)
    track = Track()
    for i, f in enumerate(frames):
        x, y = track.xy(f.t_ms / 1000)
        assert geo.xyz[i, 0] == pytest.approx(x, abs=1e-3) and geo.xyz[i, 1] == pytest.approx(y, abs=1e-3)
        assert abs((geo.heading_deg[i] - track.at(f.t_ms)["heading_deg"] + 180) % 360 - 180) < 0.01
    # the 3D points seen by posed frames, in the map frame (z relative to the cameras' median height)
    seen = sorted(set(geo.obs_point.tolist()))
    assert len(seen) == 3 and geo.points[:, :2] == pytest.approx(pts_map[:, :2], abs=1e-3)
    assert geo.obs_xy.shape == (6, 2) and geo.obs_frame.tolist() == [0, 1, 2, 3, 4, 5]
    assert geo.position("r1", 10_200) == pytest.approx((10.0, 0.0), abs=1e-3)


def test_a_frame_flung_away_and_a_stretch_glued_wrong_are_not_posed(tmp_path):
    flung = {5: (12.0, 0.0)} | {i: (0.0, 15.0) for i in range(30, 41)}  # one frame, then the last 5 s
    work, frames, _ = synthetic_job(tmp_path, flung)
    geo = geometry.place_in_map(work, {"r1": Track()}, "r1")
    assert not geo.posed[5] and not geo.posed[30:].any() and geo.posed[:5].all() and geo.posed[6:30].all()
    # they keep a position: dead reckoning moved into the map frame
    assert geo.xyz[5, :2] == pytest.approx((2.5, 0.0), abs=0.05)
    assert geo.xyz[35, :2] == pytest.approx((10.0, 7.5), abs=0.05)
    assert geo.info["walks"]["r1"]["posed"] == len(frames) - 12
    assert geo.obs_frame.tolist() == [0, 1, 2, 3]  # frame 5's point now has one trusted view (frame 4's): dropped


def test_dead_reckoning_drift_is_not_mistaken_for_a_wrong_join():
    t = np.arange(0, 120_001, 500.0)
    off = np.stack([t / 1000 * 0.04, np.zeros_like(t)], axis=1)  # 4.8 m drift over two minutes, smoothly
    assert geometry._check_walk(t, off).all()
    off[160:] += [6.0, 0.0]  # a sudden 6 m jump: the last 40 s glued in the wrong place
    keep = geometry._check_walk(t, off)
    assert keep[:155].all() and not keep[165:].any()  # the walk's longest stretch is the reference


def test_too_little_posed_is_no_geometry(tmp_path):
    work, frames, _ = synthetic_job(tmp_path)
    raw = dict(np.load(work / "raw.npz"))
    raw["registered"][5:] = False
    np.savez(work / "raw.npz", **raw)
    assert geometry.place_in_map(work, {"r1": Track()}, "r1") is None


def test_write_and_load_round_trip(tmp_path):
    work, frames, _ = synthetic_job(tmp_path)
    geo = geometry.place_in_map(work, {"r1": Track()}, "r1")
    geometry.write(geo, tmp_path)
    back = geometry.load(tmp_path)
    assert [f.name for f in back.frames] == [f.name for f in frames] and back.posed.all()
    assert back.xyz[:, :2] == pytest.approx(geo.xyz[:, :2], abs=1e-3) and back.info == geo.info
    assert back.points == pytest.approx(geo.points, abs=1e-3) and back.place.shape == geo.place.shape


def test_pairs_are_the_walk_ahead_and_the_most_similar_elsewhere(monkeypatch):
    monkeypatch.setattr(geometry, "RETRIEVAL_K", 1)
    frames = [geometry.GeoFrame(f"a/{i}.jpg", "a", i * 500.0, 1, 1) for i in range(10)]
    frames += [geometry.GeoFrame(f"b/{i}.jpg", "b", i * 500.0, 1, 1) for i in range(10)]
    place = np.zeros((20, 20), np.float32)
    for i in range(10):  # b's frame i looks like a's frame i
        place[i, i] = place[10 + i, i] = 1.0
    pairs = set(geometry.make_pairs(frames, place))
    assert (0, 5) in pairs and (0, 6) not in pairs  # 2.5 s ahead in the same walk, not 3 s
    assert all((i, 10 + i) in pairs for i in range(10))  # each one's look-alike in the other walk


def test_an_unreachable_host_fails_the_build_other_failures_fall_back(tmp_path, monkeypatch):
    from nav import config
    from nav.infer.errors import InferenceUnavailable
    from nav.mapping import inference

    frames = [geometry.GeoFrame(f"r1/{i}.jpg", "r1", i * 500.0, 1, 1) for i in range(20)]
    monkeypatch.setattr(geometry, "prepare_job", lambda work, *a: frames)
    monkeypatch.setattr(config, "INFER_URL", "http://host")

    class Host:
        def __init__(self, error):
            self.error = error

        def geometry(self, work, job_id):
            raise self.error

    monkeypatch.setattr(inference, "remote", lambda: Host(InferenceUnavailable("queue busy")))
    with pytest.raises(InferenceUnavailable):
        geometry.build_geometry([], tmp_path, tmp_path / "b1", "m", {}, None, [])
    monkeypatch.setattr(inference, "remote", lambda: Host(RuntimeError("no model")))
    warnings = []
    assert geometry.build_geometry([], tmp_path, tmp_path / "b2", "m", {}, None, warnings) is None
    assert warnings and "dead reckoning" in warnings[0]
