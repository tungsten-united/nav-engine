"""Recording format (schema_version 1): what the phone recorder captures during a walkthrough.

Every timestamp `t_ms` is milliseconds since the recording started, on the phone's monotonic clock
(performance.now() - t0). `Manifest.t0_epoch_ms` anchors that clock to UTC.

On disk (one folder per recording; paths are relative so the folder maps 1:1 to object storage):

    recordings/<rec_id>/
      manifest.json          Manifest
      video.mp4              full rear-camera video, starts at the `video_start` event
      voice/000.webm ...     push-to-talk clips (VoiceClip in the manifest)
      imu.parquet            ImuSample rows, native rate (~60 Hz)
      orientation.parquet    OrientationSample rows, native rate
      events.jsonl           Event rows (start/stop, voice, marks, live steps, ...)
      raw/                   uploads as received (video chunks, sample batches); video chunks are removed
                             once video.mp4 is built
"""

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1
REC_ID_RE = re.compile(r"^\d{8}T\d{6}Z_[0-9a-f]{4}$")  # 20261009T194012Z_a3f9

Holding = Literal["handheld_portrait", "handheld_landscape", "chest_mount", "lanyard", "other"]
Tag = Literal["door", "junction", "stairs", "lift", "destination"]
Status = Literal["recording", "finalizing", "complete", "incomplete", "failed"]
EventType = Literal[
    "start",  # recording started (t_ms = 0)
    "video_start",  # MediaRecorder started: video time 0 is this t_ms
    "video_chunk",  # a video chunk was produced; data: {seq, bytes}
    "stop",  # user pressed stop
    "voice_start",  # push-to-talk pressed; data: {clip}
    "voice_end",  # push-to-talk released; data: {clip}
    "mark",  # tag button; data: {tag}
    "step",  # live step detection (recomputed offline); data: {cadence, speed_mps}
    "orientation_change",  # screen rotated; data: {angle}
    "hidden",  # page went to background (sensors/camera may pause)
    "visible",
    "error",  # data: {message}
]


class DeviceInfo(BaseModel):
    user_agent: str
    platform: str | None = None
    screen_w: int | None = None
    screen_h: int | None = None
    pixel_ratio: float | None = None


class CameraInfo(BaseModel):
    width: int  # as delivered by the camera track (portrait phones often report 1080x1920)
    height: int
    fps: float | None = None
    mime: str  # MediaRecorder mime type, e.g. video/webm;codecs=vp9 or video/mp4;codecs=avc1
    facing: str | None = None
    label: str | None = None


class RecordingCreate(BaseModel):
    title: str = ""
    place: str = ""  # building / floor, e.g. "Office, 3rd floor"
    notes: str = ""
    holding: Holding = "handheld_portrait"
    step_length_m: float = Field(0.7, ge=0.3, le=1.2)
    t0_epoch_ms: int  # Date.now() at t_ms = 0
    device: DeviceInfo
    camera: CameraInfo


class RecordingUpdate(BaseModel):
    title: str | None = None
    place: str | None = None
    notes: str | None = None
    holding: Holding | None = None
    step_length_m: float | None = Field(None, ge=0.3, le=1.2)


class VideoInfo(BaseModel):
    file: str = "video.mp4"
    source_mime: str
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    reencoded: bool = False  # True when the source codec could not be copied into mp4


class VoiceClip(BaseModel):
    clip: int
    file: str  # relative, e.g. voice/000.webm
    mime: str
    t_start_ms: float
    t_end_ms: float
    bytes: int


class Counts(BaseModel):
    video_chunks: int = 0
    sample_batches: int = 0
    voice_clips: int = 0
    imu: int = 0
    orientation: int = 0
    events: int = 0
    marks: int = 0


class FinishRequest(BaseModel):
    """What the phone believes it sent, so the server can report gaps."""

    duration_ms: float
    video_chunks: int = Field(ge=0)
    sample_batches: int = Field(ge=0)
    voice_clips: int = Field(ge=0)


class Manifest(RecordingCreate):
    schema_version: int = SCHEMA_VERSION
    rec_id: str
    created_at_utc: datetime
    created_by: str | None = None  # Cloudflare Access identity (email or service token) when present
    status: Status = "recording"
    duration_ms: float | None = None
    video: VideoInfo | None = None
    voice: list[VoiceClip] = []
    counts: Counts = Counts()
    finish: FinishRequest | None = None  # what the phone reported at stop
    gaps: list[str] = []  # missing uploads found at finalize, e.g. "video chunk 12"
    error: str | None = None


class ImuSample(BaseModel):
    t_ms: float
    ax: float | None = None  # acceleration including gravity, device frame, m/s^2
    ay: float | None = None
    az: float | None = None
    lax: float | None = None  # acceleration without gravity (not on every device)
    lay: float | None = None
    laz: float | None = None
    gx: float | None = None  # rotation rate, deg/s (alpha, beta, gamma axes)
    gy: float | None = None
    gz: float | None = None
    interval_ms: float | None = None


class OrientationSample(BaseModel):
    t_ms: float
    alpha: float | None = None  # W3C DeviceOrientation, degrees
    beta: float | None = None
    gamma: float | None = None
    absolute: bool = False  # alpha is relative to magnetic north
    compass_deg: float | None = None  # iOS webkitCompassHeading
    compass_acc: float | None = None  # iOS webkitCompassAccuracy, degrees
    heading_deg: float | None = None  # live camera-axis heading computed on the phone (display only)
    screen_angle: int | None = None


class Event(BaseModel):
    t_ms: float
    type: EventType
    data: dict[str, Any] = {}


class SampleBatch(BaseModel):
    seq: int = Field(ge=0)  # 0, 1, 2 ... per recording; re-sending a seq overwrites it
    imu: list[ImuSample] = []
    orientation: list[OrientationSample] = []
    events: list[Event] = []


class RecordingSummary(BaseModel):
    rec_id: str
    title: str
    place: str
    created_at_utc: datetime
    status: Status
    duration_ms: float | None
    counts: Counts


class FileInfo(BaseModel):
    path: str
    bytes: int
