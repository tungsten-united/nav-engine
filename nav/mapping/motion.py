"""Motion from the phone's IMU: steps, walked distance, heading, camera pitch and a dead-reckoned path.

- Gravity is the low-passed accelerationIncludingGravity (it points up, in the device frame).
- Steps are peaks of the band-passed |acceleration| while the phone is moving; distance = steps x step length. When the
  video shows the floor, odometry.py replaces that distance with a measured one (prepare.load_walk).
- Heading: the gyro's rotation about gravity is integrated (smooth, no magnetic disturbance, but drifts) and its
  offset to the compass (the camera-axis heading the phone computed from deviceorientationabsolute) is taken as
  the circular mean over the whole walk. Indoors the compass is disturbed, so this is only roughly north-aligned.
- Pitch is the camera axis elevation (device -z against gravity): 0 = level, -90 = looking at the floor.
- Without sensors (an imported video), `video_motion` makes a track from the video's own yaw (calibrate.py):
  heading relative to the start, and walked metres guessed from time at VIDEO_SPEED_M_S.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.signal import butter, find_peaks, sosfiltfilt

RATE_HZ = 50.0
STEP_FLOOR = 0.15  # m/s^2, minimum peak prominence
STEP_FRAC = 0.8  # x rolling std of the band-passed signal
BOUT_GAP_S = 1.5  # steps further apart than this start a new bout
BOUT_MIN = 4  # steps
VIDEO_SPEED_M_S = 1.0  # assumed walking speed when only the video is known


@dataclass
class MotionTrack:
    t_ms: np.ndarray  # uniform RATE_HZ grid on the recording clock
    walked_m: np.ndarray
    heading_deg: np.ndarray  # nan when unknown; 0 = north, clockwise
    x_m: np.ndarray  # dead-reckoned, east
    y_m: np.ndarray  # north
    pitch_deg: np.ndarray
    moving: np.ndarray  # bool
    step_t_ms: np.ndarray
    has_compass: bool
    gyro_layout: str  # which rotationRate axis layout matched (see load_motion)
    source: str = "imu"  # imu | video (no sensors: heading from the video, distance from time) | none
    up: np.ndarray | None = None  # [N, 3] unit up vector in device axes (from gravity); None without an IMU
    distance_source: str = "steps"  # steps | video (floor odometry, odometry.py) | time (video_motion)
    step_length_m: float | None = None  # the step length used (re-estimated from the video when it could be)
    video_share: float = 0.0  # fraction of walked_m measured by the floor odometry

    @property
    def heading_kind(self) -> str:
        return {"imu": "compass" if self.has_compass else "gyro", "video": "video"}.get(self.source, "none")

    def index(self, t_ms: float) -> int:
        return int(np.clip(np.searchsorted(self.t_ms, t_ms), 0, len(self.t_ms) - 1))

    def at(self, t_ms: float) -> dict:
        i = self.index(t_ms)
        h = self.heading_deg[i]
        return {"walked_m": round(float(self.walked_m[i]), 2),
                "heading_deg": None if np.isnan(h) else round(float(h), 1) % 360,
                "x_m": round(float(self.x_m[i]), 2), "y_m": round(float(self.y_m[i]), 2),
                "pitch_deg": None if np.isnan(self.pitch_deg[i]) else round(float(self.pitch_deg[i]), 1),
                "moving": bool(self.moving[i])}

    def mean_heading(self, t0_ms: float, t1_ms: float) -> float | None:
        h = self.heading_deg[self.index(t0_ms):self.index(t1_ms) + 1]
        h = h[~np.isnan(h)]
        return None if not len(h) else float(np.degrees(np.angle(np.exp(1j * np.radians(h)).mean())) % 360)


def angle_diff(a: float, b: float) -> float:
    """Signed b - a in degrees, in [-180, 180): + means clockwise (a right turn when going from a to b)."""
    return float((b - a + 180) % 360 - 180)


def _col(table: dict, name: str) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in table.get(name, [])], dtype=float)


def _lowpass(x: np.ndarray, hz: float) -> np.ndarray:
    return sosfiltfilt(butter(2, hz, "low", fs=RATE_HZ, output="sos"), x, axis=0) if len(x) > 30 else x


def _interp(t_grid: np.ndarray, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(x)
    return np.interp(t_grid, t[ok], x[ok]) if ok.sum() >= 2 else np.full_like(t_grid, np.nan)


def _detrend_angle(d: np.ndarray) -> np.ndarray:
    """Differences between two heading tracks around their median offset, in degrees."""
    return (d - np.median(d) + 180) % 360 - 180


def _bouts(peaks: np.ndarray) -> list[np.ndarray]:
    if not len(peaks):
        return []
    runs = np.split(peaks, np.where(np.diff(peaks) > BOUT_GAP_S * RATE_HZ)[0] + 1)
    return [r for r in runs if len(r) >= BOUT_MIN]


def load_motion(rec_dir: Path, step_length_m: float, duration_ms: float | None = None,
                user_agent: str = "") -> MotionTrack:
    imu = pq.read_table(rec_dir / "imu.parquet").to_pydict() if (rec_dir / "imu.parquet").is_file() else {}
    ori = pq.read_table(rec_dir / "orientation.parquet").to_pydict() if (rec_dir / "orientation.parquet").is_file() else {}
    t_imu = _col(imu, "t_ms")
    end = max(duration_ms or 0, t_imu[-1] if len(t_imu) else 0, 1000.0)
    t = np.arange(0, end + 1, 1000 / RATE_HZ)
    n = len(t)
    if len(t_imu) < 2 * RATE_HZ:  # no motion data: stand still, heading unknown
        z = np.zeros(n)
        nan = np.full(n, np.nan)
        return MotionTrack(t, z, nan, z, z.copy(), nan, np.zeros(n, bool), np.array([]), False, "none", "none")

    acc = np.stack([_interp(t, t_imu, _col(imu, k)) for k in ("ax", "ay", "az")], axis=1)
    gyro = np.nan_to_num(np.stack([_interp(t, t_imu, _col(imu, k)) for k in ("gx", "gy", "gz")], axis=1))
    grav = _lowpass(acc, 0.3)
    up = grav / np.linalg.norm(grav, axis=1, keepdims=True).clip(1e-6)
    pitch = np.degrees(np.arcsin(np.clip(-up[:, 2], -1, 1)))

    # Steps: peaks of band-passed vertical acceleration above an adaptive threshold (handheld walking can be very
    # smooth: 0.1-0.4 m/s^2), kept only in bouts of a few regular steps so hand jitter while standing doesn't count.
    vert = (acc * up).sum(axis=1)
    band = sosfiltfilt(butter(2, [0.8, 3.0], "band", fs=RATE_HZ, output="sos"), vert - vert.mean())
    win = int(3 * RATE_HZ)
    sd = np.sqrt(np.convolve(band ** 2, np.ones(win) / win, mode="same"))
    peaks, props = find_peaks(band, distance=int(0.4 * RATE_HZ), prominence=0)
    peaks = peaks[props["prominences"] > np.maximum(STEP_FLOOR, STEP_FRAC * sd[peaks])]
    peaks = np.array([p for bout in _bouts(peaks) for p in bout], dtype=int)
    steps = np.zeros(n)
    steps[peaks] = 1
    walked = np.cumsum(steps) * step_length_m
    moving = np.zeros(n, bool)
    for bout in _bouts(peaks):
        moving[bout[0]:bout[-1] + 1] = True

    # Heading: integrate yaw rate (rotation about up; + is counter-clockwise from above), anchor to the compass.
    # The recorder stores gx/gy/gz = rotationRate beta/gamma/alpha, which the spec puts around x/y/z, but Chrome
    # (Android) reports alpha/beta/gamma around x/y/z. Pick the layout whose turns match the compass, else the UA.
    layouts = {"spec": gyro, "chrome": gyro[:, [2, 0, 1]]}
    compass = np.full(n, np.nan)
    has_compass = False
    if ori:
        t_ori = _col(ori, "t_ms")
        hd = _col(ori, "heading_deg")
        if np.isnan(hd).all():
            hd = _col(ori, "compass_deg")
        ok = ~np.isnan(hd)
        if ok.sum() >= 10:
            # interpolate unwrapped, then wrap back
            unwrapped = np.degrees(np.unwrap(np.radians(hd[ok])))
            compass = np.interp(t, t_ori[ok], unwrapped)
            has_compass = True
    integrated = {k: -np.cumsum((g * up).sum(axis=1)) / RATE_HZ for k, g in layouts.items()}
    if has_compass:
        layout = min(integrated, key=lambda k: np.mean(np.abs(_detrend_angle(integrated[k] - compass))))
    else:
        layout = "chrome" if "Chrome" in user_agent and "CriOS" not in user_agent else "spec"
    gyro_heading = integrated[layout]
    compass = compass % 360
    level = np.abs(pitch) < 50
    use = level & ~np.isnan(compass)
    if use.sum() > RATE_HZ:
        offset = np.degrees(np.angle(np.exp(1j * np.radians(compass[use] - gyro_heading[use])).mean()))
        heading = (gyro_heading + offset) % 360
    else:
        heading = gyro_heading % 360  # relative to the starting direction

    # Dead reckoning: each step moves step_length along the heading.
    hr = np.radians(heading)
    x = np.cumsum(steps * step_length_m * np.sin(hr))
    y = np.cumsum(steps * step_length_m * np.cos(hr))
    return MotionTrack(t, walked, heading, x, y, pitch, moving, t[peaks], has_compass, layout, up=up,
                       step_length_m=step_length_m)


def video_motion(t_video_ms: np.ndarray, yaw_units: np.ndarray, fov_deg: float, duration_ms: float | None = None,
                 speed_m_s: float = VIDEO_SPEED_M_S) -> MotionTrack:
    """A track for a recording without sensors: heading = the video's yaw x field of view (0 = the starting
    direction, clockwise), walking all the time at `speed_m_s`, dead-reckoned along that heading."""
    end = max(duration_ms or 0, float(t_video_ms[-1]) if len(t_video_ms) else 0, 1000.0)
    t = np.arange(0, end + 1, 1000 / RATE_HZ)
    n = len(t)
    if len(t_video_ms) >= 2:
        heading = np.interp(t, t_video_ms, yaw_units * fov_deg) % 360
    else:
        heading = np.zeros(n)
    step = speed_m_s / RATE_HZ
    walked = np.arange(n) * step
    hr = np.radians(heading)
    x, y = np.cumsum(step * np.sin(hr)), np.cumsum(step * np.cos(hr))
    return MotionTrack(t, walked, heading, x, y, np.full(n, np.nan), np.ones(n, bool), np.array([]), False, "none",
                       "video", distance_source="time")
