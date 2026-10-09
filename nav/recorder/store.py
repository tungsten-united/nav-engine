"""Recording storage on the local filesystem (layout in nav/schemas/recording.py).

Uploads are idempotent: each video chunk, sample batch and voice clip is one file named by its sequence number,
so a retried upload just overwrites itself. `finalize` turns the raw uploads into the clean files
(video.mp4, *.parquet, events.jsonl) and can be rerun at any time.
"""

import json
import secrets
import shutil
import subprocess
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nav.schemas.recording import (
    REC_ID_RE,
    Counts,
    FileInfo,
    FinishRequest,
    ImuSample,
    Manifest,
    OrientationSample,
    PlaceSummary,
    RecordingCreate,
    RecordingSummary,
    RecordingUpdate,
    SampleBatch,
    VideoInfo,
    VoiceClip,
)


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


def _ext(mime: str) -> str:
    base = mime.split(";")[0].strip().lower()
    return {"video/webm": "webm", "audio/webm": "webm", "video/mp4": "mp4", "audio/mp4": "m4a",
            "audio/ogg": "ogg", "audio/mpeg": "mp3", "audio/wav": "wav", "video/x-matroska": "mkv"}.get(base, "bin")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _arrow_schema(model) -> pa.Schema:
    types = {float: pa.float64(), bool: pa.bool_(), int: pa.int64()}
    fields = []
    for name, f in model.model_fields.items():
        base = next(t for t in types if t is f.annotation or t in getattr(f.annotation, "__args__", ()))
        fields.append(pa.field(name, types[base], nullable=name != "t_ms"))
    return pa.schema(fields)


IMU_SCHEMA = _arrow_schema(ImuSample)
ORIENTATION_SCHEMA = _arrow_schema(OrientationSample)


def _missing(present: set[int], expected: int) -> list[int]:
    return [i for i in range(max(expected, max(present, default=-1) + 1)) if i not in present]


class RecordingStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    # --- manifest ---------------------------------------------------------------------------------------------
    def dir(self, rec_id: str) -> Path:
        if not REC_ID_RE.match(rec_id):
            raise NotFound(rec_id)
        d = self.root / rec_id
        if not (d / "manifest.json").is_file():
            raise NotFound(rec_id)
        return d

    def get(self, rec_id: str) -> Manifest:
        return Manifest.model_validate_json((self.dir(rec_id) / "manifest.json").read_text())

    def _save(self, m: Manifest) -> None:
        _write_atomic(self.root / m.rec_id / "manifest.json", m.model_dump_json(indent=2).encode())

    def create(self, req: RecordingCreate, created_by: str | None = None) -> Manifest:
        now = datetime.now(timezone.utc)
        while True:
            rec_id = f"{now:%Y%m%dT%H%M%SZ}_{secrets.token_hex(2)}"
            if not (self.root / rec_id).exists():
                break
        (self.root / rec_id).mkdir(parents=True)
        m = Manifest(**req.model_dump(), rec_id=rec_id, created_at_utc=now, created_by=created_by)
        self._save(m)
        return m

    def update(self, rec_id: str, upd: RecordingUpdate) -> Manifest:
        with self._locks[rec_id]:
            m = self.get(rec_id)
            m = m.model_copy(update=upd.model_dump(exclude_none=True))
            self._save(m)
            return m

    def summaries(self, place: str | None = None) -> list[RecordingSummary]:
        out = []
        for d in sorted(self.root.iterdir(), reverse=True):
            if REC_ID_RE.match(d.name) and (d / "manifest.json").is_file():
                m = self.get(d.name)
                if place is None or m.place.casefold() == " ".join(place.split()).casefold():
                    out.append(RecordingSummary(**m.model_dump(include=set(RecordingSummary.model_fields))))
        return out

    def places(self) -> list[PlaceSummary]:
        """One entry per place (case-insensitive), newest first, spelled as in its latest recording."""
        groups: dict[str, list[RecordingSummary]] = defaultdict(list)
        for s in self.summaries():
            groups[s.place.casefold()].append(s)
        return [PlaceSummary(place=g[0].place, recordings=len(g), duration_ms=sum(s.duration_ms or 0 for s in g),
                             last_recorded_utc=g[0].created_at_utc) for g in groups.values()]

    def delete(self, rec_id: str) -> None:
        shutil.rmtree(self.dir(rec_id))
        self._locks.pop(rec_id, None)

    # --- uploads ----------------------------------------------------------------------------------------------
    def _writable(self, rec_id: str) -> Path:
        d = self.dir(rec_id)
        if self.get(rec_id).status == "complete":
            raise Conflict(f"{rec_id} is complete; delete it or upload to a new recording")
        return d

    def put_video_chunk(self, rec_id: str, seq: int, data: bytes) -> None:
        _write_atomic(self._writable(rec_id) / "raw" / "video" / f"{seq:06d}.bin", data)

    def put_samples(self, rec_id: str, batch: SampleBatch) -> None:
        path = self._writable(rec_id) / "raw" / "samples" / f"{batch.seq:06d}.json"
        _write_atomic(path, batch.model_dump_json(exclude_defaults=True).encode())

    def put_voice(self, rec_id: str, clip: int, data: bytes, mime: str, t_start_ms: float, t_end_ms: float) -> VoiceClip:
        d = self._writable(rec_id)
        vc = VoiceClip(clip=clip, file=f"voice/{clip:03d}.{_ext(mime)}", mime=mime,
                       t_start_ms=t_start_ms, t_end_ms=t_end_ms, bytes=len(data))
        _write_atomic(d / vc.file, data)
        with self._locks[rec_id]:
            m = self.get(rec_id)
            m.voice = sorted([v for v in m.voice if v.clip != clip] + [vc], key=lambda v: v.clip)
            self._save(m)
        return vc

    def finish(self, rec_id: str, req: FinishRequest) -> Manifest:
        with self._locks[rec_id]:
            m = self.get(rec_id)
            m.finish, m.duration_ms, m.status = req, req.duration_ms, "finalizing"
            self._save(m)
            return m

    # --- finalize ---------------------------------------------------------------------------------------------
    def finalize(self, rec_id: str) -> Manifest:
        d = self.dir(rec_id)
        with self._locks[rec_id]:
            m = self.get(rec_id)
            m.status, m.error = "finalizing", None
            self._save(m)
        gaps: list[str] = []
        counts = Counts(voice_clips=len(m.voice))
        exp = m.finish or FinishRequest(duration_ms=0, video_chunks=0, sample_batches=0, voice_clips=0)
        try:
            video = self._finalize_video(d, m, exp.video_chunks, gaps, counts)
            self._finalize_samples(d, exp.sample_batches, gaps, counts)
            gaps += [f"voice clip {i}" for i in _missing({v.clip for v in m.voice}, exp.voice_clips)]
            error, status = None, ("incomplete" if gaps or m.finish is None else "complete")
            if m.finish is None:
                gaps.append("recording was not stopped (no finish request)")
        except Exception as e:  # keep raw uploads; the error goes into the manifest
            video, error, status = m.video, f"{type(e).__name__}: {e}", "failed"
        with self._locks[rec_id]:
            m = self.get(rec_id)  # voice clips may have arrived meanwhile
            m.video, m.counts, m.gaps, m.status, m.error = video, counts, gaps, status, error
            m.counts.voice_clips = len(m.voice)
            self._save(m)
            return m

    def _finalize_video(self, d: Path, m: Manifest, expected: int, gaps: list[str], counts: Counts) -> VideoInfo | None:
        chunks = sorted((d / "raw" / "video").glob("*.bin"))
        if not chunks:
            if m.video and (d / m.video.file).is_file():
                counts.video_chunks = m.counts.video_chunks
                return m.video  # already built and chunks cleaned up
            if expected:
                gaps.append(f"all {expected} video chunks")
            return None
        present = {int(p.stem) for p in chunks}
        gaps += [f"video chunk {i}" for i in _missing(present, expected)]
        counts.video_chunks = len(chunks)
        # MediaRecorder timeslice chunks are pieces of one stream: concatenated they form the original file.
        source = d / "raw" / f"source.{_ext(m.camera.mime)}"
        with source.open("wb") as out:
            for p in chunks:
                out.write(p.read_bytes())
        target = d / "video.mp4"
        reencoded = False
        base = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-fflags", "+genpts", "-i", str(source), "-map", "0:v:0"]
        tail = ["-movflags", "+faststart", str(target)]
        copy = subprocess.run(base + ["-c", "copy"] + tail, capture_output=True, text=True)
        if copy.returncode != 0:  # e.g. VP8 can't go into mp4
            reencoded = True
            enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
            subprocess.run(base + enc + tail, capture_output=True, text=True, check=True)
        info = _probe(target)
        source.unlink()
        for p in chunks:
            p.unlink()
        return VideoInfo(source_mime=m.camera.mime, reencoded=reencoded, **info)

    def _finalize_samples(self, d: Path, expected: int, gaps: list[str], counts: Counts) -> None:
        files = sorted((d / "raw" / "samples").glob("*.json"))
        batches = [SampleBatch.model_validate_json(p.read_text()) for p in files]
        gaps += [f"sample batch {i}" for i in _missing({b.seq for b in batches}, expected)]
        imu = sorted((s for b in batches for s in b.imu), key=lambda s: s.t_ms)
        ori = sorted((s for b in batches for s in b.orientation), key=lambda s: s.t_ms)
        events = sorted((e for b in batches for e in b.events), key=lambda e: e.t_ms)
        pq.write_table(pa.Table.from_pylist([s.model_dump() for s in imu], schema=IMU_SCHEMA), d / "imu.parquet")
        pq.write_table(pa.Table.from_pylist([s.model_dump() for s in ori], schema=ORIENTATION_SCHEMA),
                       d / "orientation.parquet")
        _write_atomic(d / "events.jsonl", "".join(e.model_dump_json() + "\n" for e in events).encode())
        counts.sample_batches, counts.imu, counts.orientation, counts.events = len(batches), len(imu), len(ori), len(events)
        counts.marks = sum(e.type == "mark" for e in events)

    # --- files ------------------------------------------------------------------------------------------------
    def files(self, rec_id: str) -> list[FileInfo]:
        d = self.dir(rec_id)
        return [FileInfo(path=p.relative_to(d).as_posix(), bytes=p.stat().st_size)
                for p in sorted(d.rglob("*")) if p.is_file() and not p.name.startswith(".")]

    def series(self, rec_id: str, stream: str, max_points: int | None = None) -> dict[str, list]:
        """imu / orientation as columns ({"t_ms": [...], "ax": [...]}), thinned to at most max_points rows."""
        if stream not in ("imu", "orientation"):
            raise NotFound(stream)
        path = self.dir(rec_id) / f"{stream}.parquet"
        if not path.is_file():
            raise NotFound(f"{rec_id}/{stream}")
        t = pq.read_table(path)
        if max_points and t.num_rows > max_points:
            t = t.take(list(range(0, t.num_rows, -(-t.num_rows // max_points))))
        return t.to_pydict()

    def file(self, rec_id: str, rel: str) -> Path:
        d = self.dir(rec_id).resolve()
        p = (d / rel).resolve()
        if not p.is_relative_to(d) or not p.is_file():
            raise NotFound(rel)
        return p


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    j = json.loads(out.stdout)
    s = (j.get("streams") or [{}])[0]
    dur = j.get("format", {}).get("duration")
    return {"codec": s.get("codec_name"), "width": s.get("width"), "height": s.get("height"),
            "duration_s": float(dur) if dur else None}

