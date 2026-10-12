"""Inference host: the local models of the map pipeline and the Claude Code map job, over HTTP (`nav infer serve`).

This machine only does inference here: no recordings or maps are stored, only scratch folders for map jobs
(NAV_INFER_JOBS_DIR, safe to delete). It is bound to 127.0.0.1 and published through the Cloudflare Tunnel; Cloudflare
Access is the gate (the workers send the service token), so nav-infer has no auth of its own.

It shares one GPU with whatever else runs on this computer, so it protects itself:
- one model call at a time (a single GPU slot), at most NAV_INFER_QUEUE requests waiting; beyond that 503 with
  Retry-After, which the client honours; live localization frames (nav-api) jump the queue of map-build calls;
- hard caps per request (images, pairs, bytes), so every request finishes well inside Cloudflare's 100 s;
- one Claude Code job at a time, at most NAV_INFER_JOB_QUEUE waiting; jobs are submitted and polled, never held open.

    GET  /infer/v1/health
    POST /infer/v1/embed       images[] (JPEG), model          -> {model, n, dim, data: base64 float32}
    POST /infer/v1/place       images[] (JPEG), model, priority? -> same, MegaLoc (priority=live: ahead of batches)
    POST /infer/v1/match       images[] (JPEG), pairs: [[i,j]] -> {inliers: [...]}
    POST /infer/v1/verify      queries[] (JPEG), refs: [key], ref_images[]? (named <index>.jpg) -> {inliers: [[...]]}
                               live (nav-api): the camera frames against visual-memory frames, whose features stay
                               cached by key; 409 {missing: [key]} asks for the images of the ones not cached
    POST /infer/v1/language    audio, model                    -> {probs: {code: p}}, Whisper's language guess
    POST /infer/v1/transcribe  audio, model, language?         -> {text}
    POST /infer/v1/jobs/map-draft   job_id, place, language, model, effort, budget_usd, stage (places | routes),
                                    candidates (places) or routes (routes), previous?, frames[]
    POST /infer/v1/jobs/geometry/{job_id}/frames   frames[] named <rec_id>__<t>.jpg (in chunks, before the job)
    POST /infer/v1/jobs/geometry    job_id, job (job.json): the 3D model of the frames (nav/mapping/geometry.py); its
                                    GPU stage is a job on the machine's shared GPU queue (gpu-run), the mapping runs here
    GET  /infer/v1/jobs/{job_id}                -> {status: queued|running|done|failed, error, files, ...}
    GET  /infer/v1/jobs/{job_id}/files/{name}   -> claude.json, claude.log, prompt.md, schema.json, command.txt;
                                                   geometry: raw.npz, summary.json, gpu-run.log
"""

import asyncio
import base64
import json
import logging
import re
import sys
import tempfile
import threading
import time
import queue as queue_mod
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from nav import config

log = logging.getLogger(__name__)

MAX_IMAGES = 32  # per embed / place request
MAX_PAIRS = 64  # per match request ...
MAX_VERIFY_QUERIES = 4  # a burst
MAX_VERIFY_REFS = 16
MAX_MATCH_IMAGES = 48  # ... over at most this many distinct images
MAX_BODY = 16 * 1024 * 1024  # model requests
MAX_JOB_BODY = 96 * 1024 * 1024  # map jobs (candidates + up to MAX_JOB_FRAMES frames); Cloudflare allows 100 MB
MAX_JOB_FRAMES = 100  # the Messages API limit too, for when the job moves there
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
JOB_FILES = ("claude.json", "claude.log", "prompt.md", "schema.json", "command.txt")
GEOMETRY_FILES = ("raw.npz", "summary.json", "gpu-run.log")
GEOMETRY_FRAME_RE = re.compile(r"^([A-Za-z0-9-]+(?:_[A-Za-z0-9-]+)*)__([0-9]{1,9}\.jpg)$")
MAX_GEOMETRY_FRAMES = 8000


class Backend(Protocol):
    """What the server runs; LocalBackend in production, stubs in tests."""

    models: dict[str, str]  # {"embed": name, "place": name, "transcribe": name}

    def embed(self, images: list[bytes], model: str) -> np.ndarray: ...
    def place(self, images: list[bytes], model: str) -> np.ndarray: ...
    def match(self, images: list[bytes], pairs: list[tuple[int, int]]) -> list[int]: ...
    def verify(self, queries: list[bytes], refs: list[tuple[str, bytes | None]]) -> list[list[int]]: ...
    def detect_language(self, path: Path, model: str) -> dict[str, float]: ...
    def transcribe(self, path: Path, model: str, language: str | None = None) -> str: ...
    def map_draft(self, work_dir: Path, place: str, language: str, model: str, effort: str, budget_usd: float,
                  stage: str = "places") -> None: ...
    def geometry(self, work_dir: Path) -> None: ...


class LocalBackend:
    def __init__(self):
        from nav.mapping.models import FeatureCache

        self.models = {"embed": config.EMBED_MODEL, "place": config.PLACE_MODEL, "transcribe": config.TRANSCRIBE_MODEL}
        self.features = FeatureCache(config.INFER_FEATURE_CACHE)

    def embed(self, images, model):
        from nav.mapping.inference import load_rgb
        from nav.mapping.models import embed_images

        return embed_images([load_rgb(b) for b in images], model)

    def place(self, images, model):
        from nav.mapping.inference import load_rgb
        from nav.mapping.models import embed_places

        return embed_places([load_rgb(b) for b in images], model)

    def match(self, images, pairs):
        from nav.mapping.inference import local_pair_inliers

        return local_pair_inliers(images, pairs)

    def verify(self, queries, refs):
        from nav.mapping.inference import load_rgb
        from nav.mapping.models import verify

        return verify([load_rgb(b) for b in queries],
                      [(k, None if b is None else (lambda b=b: load_rgb(b))) for k, b in refs], self.features)

    def detect_language(self, path, model):
        from nav.mapping.models import detect_language

        return detect_language(path, model)

    def transcribe(self, path, model, language=None):
        from nav.mapping.models import transcribe

        return transcribe(path, model, language)

    def map_draft(self, work_dir, place, language, model, effort, budget_usd, stage="places"):
        from nav.mapping.llm import run_claude_code

        run_claude_code(work_dir, place, language, model, budget_usd, effort, stage=stage)  # work_dir/job/<stage>/

    def geometry(self, work_dir):
        from nav.mapping.geometry import run_job

        run_job(work_dir, queue=True)  # a resident service: its batch GPU work waits its turn on the queue


class Busy(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after


class GpuSlot:
    """One model call at a time; at most `queue` more waiting, the rest are turned away (503).

    Live calls (a walking user's camera frame, from nav-api) go ahead of every waiting batch call (map builds) and have
    their own `live_queue` places, so they only ever wait for the call that is running and for other live calls."""

    def __init__(self, queue: int, live_queue: int = 2):
        self.queue, self.live_queue = queue, live_queue
        self.pending = 0  # batch calls running + waiting
        self.live_pending = 0  # live calls running + waiting
        self._busy = False
        self._waiting: dict[bool, deque[asyncio.Future]] = {True: deque(), False: deque()}

    async def run(self, fn: Callable, *args, live: bool = False):
        if live and self.live_pending >= 1 + self.live_queue:
            raise Busy(retry_after=1)
        if not live and self.pending >= 1 + self.queue:
            raise Busy(retry_after=min(30, 2 * self.pending))
        if live:
            self.live_pending += 1
        else:
            self.pending += 1
        try:
            await self._acquire(live)
            try:
                return await run_in_threadpool(fn, *args)
            finally:
                self._release()
        finally:
            if live:
                self.live_pending -= 1
            else:
                self.pending -= 1

    async def _acquire(self, live: bool) -> None:
        if not self._busy and not any(self._waiting.values()):
            self._busy = True
            return
        fut = asyncio.get_running_loop().create_future()
        self._waiting[live].append(fut)
        try:
            await fut  # _release hands the slot over by resolving it (_busy stays True)
        except asyncio.CancelledError:
            if fut in self._waiting[live]:
                self._waiting[live].remove(fut)
            elif fut.done() and not fut.cancelled():
                self._release()  # it was handed over just as this call was cancelled: pass it on
            raise

    def _release(self) -> None:
        for waiting in (self._waiting[True], self._waiting[False]):
            while waiting:
                fut = waiting.popleft()
                if not fut.done():
                    fut.set_result(None)
                    return
        self._busy = False


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


class Jobs:
    """Map jobs in root/<job_id>/ (status.json + work/). One runs at a time, in submission order."""

    def __init__(self, root: Path, backend: Backend, queue: int):
        self.root, self.backend, self.queue = root, backend, queue
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._q: queue_mod.Queue[str] = queue_mod.Queue()
        self.pending = 0
        for status in self.root.glob("*/status.json"):  # a restart killed whatever was queued or running
            s = json.loads(status.read_text())
            if s["status"] in ("queued", "running"):
                s.update(status="failed", error="interrupted: the inference host restarted; submit again")
                _write_atomic(status, json.dumps(s).encode())
        threading.Thread(target=self._worker, name="map-jobs", daemon=True).start()

    def _dir(self, job_id: str) -> Path:
        if not JOB_ID_RE.match(job_id):
            raise HTTPException(422, "bad job_id")
        return self.root / job_id

    def _files_dir(self, job_id: str, kind: str, stage: str = "places") -> Path:
        return self._dir(job_id) / "work" / ("" if kind == "geometry" else f"job/{stage}")

    def status(self, job_id: str) -> dict:
        path = self._dir(job_id) / "status.json"
        if not path.is_file():
            raise HTTPException(404, f"no job {job_id}")
        s = json.loads(path.read_text())
        p = s.get("params", {})
        kind = p.get("kind", "map-draft")
        names = GEOMETRY_FILES if kind == "geometry" else JOB_FILES
        s["files"] = [n for n in names if (self._files_dir(job_id, kind, p.get("stage", "places")) / n).is_file()]
        return s

    def _set(self, jid: str, /, **fields) -> None:
        path = self._dir(jid) / "status.json"
        s = json.loads(path.read_text()) if path.is_file() else {}
        s.update(fields)
        _write_atomic(path, json.dumps(s).encode())

    def upload(self, job_id: str, files: dict[str, bytes]) -> int:
        """Frames of a geometry job, before it is submitted (idempotent per name). -> frames there so far."""
        d = self._dir(job_id)
        if (d / "status.json").is_file():
            raise HTTPException(409, f"job {job_id} already submitted")
        images = d / "work" / "images"
        for name, data in files.items():
            (images / name).parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(images / name, data)
        return sum(1 for _ in images.glob("*/*.jpg"))

    def submit(self, job_id: str, params: dict, files: dict[str, bytes]) -> dict:
        d = self._dir(job_id)
        with self._lock:
            if (d / "status.json").is_file():  # idempotent: a retried submit gets the job it already made
                return self.status(job_id)
            if self.pending >= 1 + self.queue:
                raise Busy(retry_after=60)
            work = d / "work"
            (work / "frames").mkdir(parents=True, exist_ok=True)
            for name, data in files.items():
                (work / name).write_bytes(data)
            self._set(job_id, job_id=job_id, status="queued", params=params, error=None,
                      submitted_at_utc=datetime.now(timezone.utc).isoformat())
            self.pending += 1
            self._q.put(job_id)
        return self.status(job_id)

    def file(self, job_id: str, name: str) -> Path:
        p = self.status(job_id).get("params", {})
        kind = p.get("kind", "map-draft")
        path = self._files_dir(job_id, kind, p.get("stage", "places")) / name
        if name not in (GEOMETRY_FILES if kind == "geometry" else JOB_FILES) or not path.is_file():
            raise HTTPException(404, name)
        return path

    def _worker(self) -> None:
        while True:
            job_id = self._q.get()
            p = json.loads((self._dir(job_id) / "status.json").read_text())["params"]
            self._set(job_id, status="running", started_at_utc=datetime.now(timezone.utc).isoformat())
            t0 = time.monotonic()
            try:
                if p.get("kind") == "geometry":
                    self.backend.geometry(self._dir(job_id) / "work")
                else:
                    self.backend.map_draft(self._dir(job_id) / "work", p["place"], p["language"], p["model"],
                                           p["effort"], p["budget_usd"], p.get("stage", "places"))
                result = {"status": "done"}
            except Exception as e:  # the job files (claude.json, claude.log) say more; the client reads them
                log.warning("map job %s failed: %s", job_id, e)
                result = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
            finally:
                with self._lock:
                    self.pending -= 1
            self._set(job_id, duration_s=round(time.monotonic() - t0, 1), **result)


def create_infer_app(backend: Backend | None = None, jobs_dir: Path | None = None, queue: int | None = None,
                     job_queue: int | None = None) -> FastAPI:
    backend = backend or LocalBackend()
    gpu = GpuSlot(config.INFER_QUEUE if queue is None else queue)
    jobs = Jobs(jobs_dir or config.INFER_JOBS_DIR, backend,
                config.INFER_JOB_QUEUE if job_queue is None else job_queue)
    app = FastAPI(title="nav-engine inference host")
    app.state.gpu, app.state.jobs = gpu, jobs

    @app.exception_handler(Busy)
    async def _busy(_: Request, e: Busy):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": "busy, retry later"}, status_code=503,
                            headers={"Retry-After": str(e.retry_after)})

    @app.middleware("http")
    async def body_limit(request: Request, call_next):
        # Checked before the multipart body is read: oversized requests never reach the models.
        if request.method == "POST":
            limit = MAX_JOB_BODY if "/jobs/" in request.url.path else MAX_BODY
            length = request.headers.get("content-length")
            if length is None or int(length) > limit:
                from fastapi.responses import JSONResponse

                return JSONResponse({"detail": f"body must have a content-length of at most {limit} bytes"},
                                    status_code=411 if length is None else 413)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    def check_model(kind: str, model: str) -> None:
        if model != backend.models[kind]:
            raise HTTPException(409, f"{kind} model here is {backend.models[kind]!r}, not {model!r}")

    async def read_images(images: list[UploadFile], limit: int) -> list[bytes]:
        if not images or len(images) > limit:
            raise HTTPException(422, f"send 1..{limit} images")
        return [await f.read() for f in images]

    def vectors(model: str, v: np.ndarray) -> dict:
        v = np.ascontiguousarray(v, dtype="<f4")
        return {"model": model, "n": int(v.shape[0]), "dim": int(v.shape[1]) if v.ndim == 2 else 0,
                "data": base64.b64encode(v.tobytes()).decode()}

    api = "/infer/v1"

    @app.get(f"{api}/health")
    def health() -> dict:
        torch = sys.modules.get("torch")  # only reported once a model has loaded it
        gpu_name = torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else None
        return {"ok": True, "models": backend.models, "gpu": gpu_name, "gpu_pending": gpu.pending, "gpu_live_pending": gpu.live_pending,
                "gpu_queue": gpu.queue, "jobs_pending": jobs.pending, "jobs_queue": jobs.queue}

    @app.post(f"{api}/embed")
    async def embed(model: str = Form(...), images: list[UploadFile] = File(...)) -> dict:
        check_model("embed", model)
        data = await read_images(images, MAX_IMAGES)
        return vectors(model, await gpu.run(backend.embed, data, model))

    @app.post(f"{api}/place")
    async def place(model: str = Form(...), images: list[UploadFile] = File(...), priority: str = Form("batch")) -> dict:
        """priority=live: a camera frame of someone walking (nav-api), served before waiting map-build calls."""
        check_model("place", model)
        data = await read_images(images, MAX_IMAGES)
        return vectors(model, await gpu.run(backend.place, data, model, live=priority == "live"))

    @app.post(f"{api}/match")
    async def match(pairs: str = Form(...), images: list[UploadFile] = File(...)) -> dict:
        data = await read_images(images, MAX_MATCH_IMAGES)
        try:
            pp = [(int(i), int(j)) for i, j in json.loads(pairs)]
        except (ValueError, TypeError):
            raise HTTPException(422, "pairs must be a JSON list of [i, j]")
        if not pp or len(pp) > MAX_PAIRS or any(not (0 <= k < len(data)) for p in pp for k in p):
            raise HTTPException(422, f"send 1..{MAX_PAIRS} pairs of indexes into images")
        return {"inliers": [int(k) for k in await gpu.run(backend.match, data, pp)]}

    @app.post(f"{api}/verify")
    async def verify(queries: list[UploadFile] = File(...), refs: str = Form(...),
                     ref_images: list[UploadFile] | None = File(None)):
        """Live (nav-api, ahead of batch calls): LightGlue inliers of each camera frame with each memory frame."""
        from fastapi.responses import JSONResponse

        from nav.infer.errors import MissingFeatures

        q = await read_images(queries, MAX_VERIFY_QUERIES)
        try:
            keys = [str(k) for k in json.loads(refs)]
        except (ValueError, TypeError):
            raise HTTPException(422, "refs must be a JSON list of keys")
        if not keys or len(keys) > MAX_VERIFY_REFS:
            raise HTTPException(422, f"send 1..{MAX_VERIFY_REFS} refs")
        given: dict[str, bytes] = {}
        for f in ref_images or []:
            try:
                given[keys[int(Path(f.filename or "").stem)]] = await f.read()
            except (ValueError, IndexError):
                raise HTTPException(422, f"ref image {f.filename!r}: name it <index into refs>.jpg")
        try:
            got = await gpu.run(backend.verify, q, [(k, given.get(k)) for k in keys], live=True)
        except MissingFeatures as e:
            return JSONResponse({"missing": e.keys}, status_code=409)
        return {"inliers": [[int(k) for k in row] for row in got]}

    async def on_audio(audio: UploadFile, fn: Callable, *args):
        suffix = Path(audio.filename or "clip.webm").suffix[:8] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(await audio.read())
            tmp.flush()
            try:
                return await gpu.run(fn, Path(tmp.name), *args)
            except Busy:
                raise
            except Exception as e:  # undecodable clip or unknown language: the caller's input, not a host failure
                raise HTTPException(422, f"could not transcribe: {type(e).__name__}: {e}")

    @app.post(f"{api}/language")
    async def language(model: str = Form(...), audio: UploadFile = File(...)) -> dict:
        check_model("transcribe", model)
        return {"probs": await on_audio(audio, backend.detect_language, model)}

    @app.post(f"{api}/transcribe")
    async def transcribe(model: str = Form(...), audio: UploadFile = File(...),
                         language: str | None = Form(None)) -> dict:
        check_model("transcribe", model)
        return {"text": await on_audio(audio, backend.transcribe, model, language or None)}

    @app.post(f"{api}/jobs/map-draft", status_code=202)
    async def map_draft(job_id: str = Form(...), place: str = Form(...), language: str = Form(...),
                        model: str = Form(...), effort: str = Form(...), budget_usd: float = Form(...),
                        stage: str = Form("places"), candidates: UploadFile | None = File(None),
                        routes: UploadFile | None = File(None), previous: UploadFile | None = File(None),
                        frames: list[UploadFile] = File(...)) -> dict:
        if len(frames) > MAX_JOB_FRAMES:
            raise HTTPException(422, f"at most {MAX_JOB_FRAMES} frames")
        if stage not in ("places", "routes") or (candidates if stage == "places" else routes) is None:
            raise HTTPException(422, "stage places needs candidates (candidates.json), stage routes needs routes")
        files = {"candidates.json": await candidates.read()} if stage == "places" else \
            {"routes.json": await routes.read()}
        if previous is not None:
            files["previous.json"] = await previous.read()
        for f in frames:
            name = Path(f.filename or "").name
            if not re.fullmatch(r"[A-Za-z0-9_-]+\.jpg", name):
                raise HTTPException(422, f"bad frame name {f.filename!r}")
            files[f"frames/{name}"] = await f.read()
        params = {"place": place, "language": language, "model": model, "effort": effort, "budget_usd": budget_usd,
                  "stage": stage}
        return await run_in_threadpool(jobs.submit, job_id, params, files)

    @app.post(f"{api}/jobs/geometry/{{job_id}}/frames")
    async def geometry_frames(job_id: str, frames: list[UploadFile] = File(...)) -> dict:
        files = {}
        for f in frames:
            m = GEOMETRY_FRAME_RE.match(Path(f.filename or "").name)
            if not m:
                raise HTTPException(422, f"bad frame name {f.filename!r}: <rec_id>__<t>.jpg")
            files[f"{m[1]}/{m[2]}"] = await f.read()
        return {"frames": await run_in_threadpool(jobs.upload, job_id, files)}

    @app.post(f"{api}/jobs/geometry", status_code=202)
    async def geometry(job_id: str = Form(...), job: UploadFile = File(...)) -> dict:
        data = await job.read()
        try:
            frames = json.loads(data)["frames"]
        except (ValueError, KeyError, TypeError):
            raise HTTPException(422, "job must be job.json (nav/mapping/geometry.py)")
        if not frames or len(frames) > MAX_GEOMETRY_FRAMES:
            raise HTTPException(422, f"send 1..{MAX_GEOMETRY_FRAMES} frames")
        images = jobs._dir(job_id) / "work" / "images"
        if not (jobs._dir(job_id) / "status.json").is_file():
            missing = [f["name"] for f in frames if not (images / f["name"]).is_file()]
            if missing:
                raise HTTPException(409, f"{len(missing)} frames not uploaded, e.g. {missing[0]}")
        return await run_in_threadpool(jobs.submit, job_id, {"kind": "geometry"}, {"job.json": data})

    @app.get(f"{api}/jobs/{{job_id}}")
    def job_status(job_id: str) -> dict:
        return jobs.status(job_id)

    @app.get(f"{api}/jobs/{{job_id}}/files/{{name}}")
    def job_file(job_id: str, name: str) -> FileResponse:
        return FileResponse(jobs.file(job_id, name))

    return app
