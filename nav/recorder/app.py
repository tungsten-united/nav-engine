"""Recorder server: the recordings and maps API (/api/v1, OpenAPI docs at /docs) and the debugging frontend
(frontend/: phone app at /, /dash/, /viz/).

Two deployments, same code:
- helium: bound to localhost behind the Cloudflare Tunnel; Cloudflare Access is the gate (browser login for people,
  the "helium-clients" service token for workers), so the app has no auth of its own (NAV_API_TOKEN unset).
- Cloud Run: public, so NAV_API_TOKEN gates /api/* (Bearer header, or ?token= where a <video>/<img> can't send
  headers), NAV_CORS_ORIGINS lets other frontends call it, and NAV_ALLOW_DELETE=0 keeps the shared data.
"""

import hashlib
import hmac
import json
import logging
import mimetypes
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nav import config
from nav.mapping.store import MapStore
from nav.recorder.store import Conflict, NotFound, RecordingStore
from nav.schemas.map import MapSummary, PlaceMap
from nav.schemas.recording import (
    SCHEMA_VERSION,
    FileInfo,
    FinishRequest,
    Manifest,
    PlaceSummary,
    RecordingCreate,
    RecordingSummary,
    RecordingUpdate,
    SampleBatch,
    VoiceClip,
)

log = logging.getLogger(__name__)
_DEFAULT = object()
# Cloud Run cuts an HTTP/1 response with a Content-Length over 32 MiB (the browser sees a 500): larger files are
# served in ranges of at most this much, or streamed without a length.
MAX_RESPONSE = 8 * 1024 * 1024
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def _read(path: Path, start: int, end: int):
    with path.open("rb") as f:
        f.seek(start)
        left = end - start + 1
        while left > 0 and (chunk := f.read(min(1 << 20, left))):
            left -= len(chunk)
            yield chunk


def send_file(request: Request, path: Path) -> Response:
    """FileResponse, except for files over MAX_RESPONSE: a Range request gets at most MAX_RESPONSE bytes back (an
    open-ended range may be answered short; video players then ask for the next range), and a plain GET is
    streamed without a Content-Length."""
    size = path.stat().st_size
    if size <= MAX_RESPONSE:
        return FileResponse(path)
    media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    m = RANGE_RE.fullmatch((request.headers.get("range") or "").strip())
    if m and (m[1] or m[2]):
        start, end = (int(m[1]), int(m[2]) if m[2] else size - 1) if m[1] else (max(0, size - int(m[2])), size - 1)
        if start >= size or start > end:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        end = min(end, size - 1, start + MAX_RESPONSE - 1)
        return StreamingResponse(_read(path, start, end), status_code=206, media_type=media,
                                 headers={"Content-Range": f"bytes {start}-{end}/{size}",
                                          "Content-Length": str(end - start + 1), "Accept-Ranges": "bytes"})
    return StreamingResponse(_read(path, 0, size - 1), media_type=media, headers={"Accept-Ranges": "bytes"})


class BuildRequest(BaseModel):
    place: str
    language: str | None = None  # default NAV_MAP_LANGUAGE
    max_candidates: int = 60
    llm: bool = True  # false: stop after selecting candidates (no LLM cost)
    transcribe: bool = True


def create_app(store: RecordingStore | None = None, maps: MapStore | None = None, *, web_dir=_DEFAULT,
               api_token=_DEFAULT, cors_origins=_DEFAULT, cors_origin_regex=_DEFAULT, allow_delete=_DEFAULT,
               cache_dir: Path | None = None) -> FastAPI:
    """Settings default to nav/config.py (environment); tests pass them explicitly."""
    store = store or RecordingStore(config.RECORDINGS_DIR)
    maps = maps or MapStore(config.MAPS_DIR)
    web_dir = config.WEB_DIR if web_dir is _DEFAULT else web_dir
    api_token = config.API_TOKEN if api_token is _DEFAULT else api_token
    cors_origins = config.CORS_ORIGINS if cors_origins is _DEFAULT else cors_origins
    cors_origin_regex = config.CORS_ORIGIN_REGEX if cors_origin_regex is _DEFAULT else cors_origin_regex
    allow_delete = config.ALLOW_DELETE if allow_delete is _DEFAULT else allow_delete
    cache_dir = cache_dir or config.CACHE_DIR
    app = FastAPI(title="nav-engine recorder", version=f"schema-{SCHEMA_VERSION}")
    app.state.store = store

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, e: NotFound):
        return JSONResponse({"detail": f"not found: {e}"}, status_code=404)

    @app.exception_handler(Conflict)
    async def _conflict(_: Request, e: Conflict):
        return JSONResponse({"detail": str(e)}, status_code=409)

    @app.middleware("http")
    async def no_stale_assets(request: Request, call_next):
        # Cloudflare caches .js/.css at the edge by default: a new page with an old app.js breaks the app.
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
        return response

    if api_token:
        @app.middleware("http")
        async def require_token(request: Request, call_next):
            path = request.url.path
            if path.startswith("/api/") and path != "/api/v1/health" and request.method != "OPTIONS":
                # Either place: behind Cloud Run IAM the Authorization header carries Google's identity token.
                auth = request.headers.get("authorization", "")
                given = [auth[7:] if auth.lower().startswith("bearer ") else "", request.query_params.get("token", "")]
                if not any(hmac.compare_digest(g.encode(), api_token.encode()) for g in given):
                    return JSONResponse({"detail": "missing or wrong API token"}, status_code=401,
                                        headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"})
            return await call_next(request)

    if cors_origins or cors_origin_regex:  # added last: outermost, so 401s and preflights get CORS headers too
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_origin_regex=cors_origin_regex,
                           allow_methods=["*"], allow_headers=["Authorization", "Content-Type", "X-Nav-User"],
                           max_age=600)

    async def body(request: Request) -> bytes:
        if int(request.headers.get("content-length") or 0) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(413, "upload too large")
        data = await request.body()
        if not data:
            raise HTTPException(400, "empty body")
        return data

    def identity(request: Request) -> str | None:
        h = request.headers
        if email := h.get("cf-access-authenticated-user-email"):
            return email
        if h.get("cf-access-jwt-assertion"):
            return "service-token"
        if user := h.get("x-nav-user", "").strip()[:100]:
            return user
        return "token" if api_token else None

    api = "/api/v1"

    @app.get(f"{api}/health")
    def health() -> dict:
        return {"ok": True, "schema_version": SCHEMA_VERSION}

    @app.post(f"{api}/recordings", status_code=201)
    def create(req: RecordingCreate, request: Request) -> Manifest:
        return store.create(req, created_by=identity(request))

    @app.get(f"{api}/recordings")
    def list_recordings(place: str | None = None) -> list[RecordingSummary]:
        return store.summaries(place)

    @app.get(f"{api}/places")
    def places() -> list[PlaceSummary]:
        """Places with recordings; each place becomes one map."""
        return store.places()

    @app.get(f"{api}/recordings/{{rec_id}}")
    def get(rec_id: str) -> Manifest:
        return store.get(rec_id)

    @app.patch(f"{api}/recordings/{{rec_id}}")
    def update(rec_id: str, upd: RecordingUpdate) -> Manifest:
        return store.update(rec_id, upd)

    @app.delete(f"{api}/recordings/{{rec_id}}", status_code=204)
    def delete(rec_id: str) -> Response:
        if not allow_delete:
            raise HTTPException(403, "deleting is disabled here (NAV_ALLOW_DELETE=0): this is the team's shared data")
        store.delete(rec_id)
        return Response(status_code=204)

    @app.put(f"{api}/recordings/{{rec_id}}/video/{{seq}}", status_code=204)
    async def put_video(rec_id: str, seq: int, request: Request) -> Response:
        if seq < 0:
            raise HTTPException(422, "seq must be >= 0")
        store.put_video_chunk(rec_id, seq, await body(request))
        return Response(status_code=204)

    @app.post(f"{api}/recordings/{{rec_id}}/samples", status_code=204)
    def put_samples(rec_id: str, batch: SampleBatch) -> Response:
        store.put_samples(rec_id, batch)
        return Response(status_code=204)

    @app.put(f"{api}/recordings/{{rec_id}}/voice/{{clip}}")
    async def put_voice(rec_id: str, clip: int, request: Request,
                        t_start_ms: float = Query(...), t_end_ms: float = Query(...)) -> VoiceClip:
        if clip < 0 or t_end_ms < t_start_ms:
            raise HTTPException(422, "need clip >= 0 and t_end_ms >= t_start_ms")
        mime = request.headers.get("content-type", "application/octet-stream")
        return store.put_voice(rec_id, clip, await body(request), mime, t_start_ms, t_end_ms)

    @app.post(f"{api}/recordings/{{rec_id}}/finish", status_code=202)
    def finish(rec_id: str, req: FinishRequest, tasks: BackgroundTasks) -> Manifest:
        m = store.finish(rec_id, req)
        tasks.add_task(store.finalize, rec_id)
        return m

    @app.post(f"{api}/recordings/{{rec_id}}/finalize")
    def finalize(rec_id: str, tasks: BackgroundTasks, wait: bool = False) -> Manifest:
        """Rebuild video.mp4 / parquet / events.jsonl from the raw uploads (e.g. after late uploads)."""
        if wait:
            return store.finalize(rec_id)
        store.get(rec_id)
        tasks.add_task(store.finalize, rec_id)
        return store.get(rec_id)

    @app.get(f"{api}/recordings/{{rec_id}}/files")
    def files(rec_id: str) -> list[FileInfo]:
        return store.files(rec_id)

    @app.get(f"{api}/recordings/{{rec_id}}/files/{{path:path}}")
    def file(rec_id: str, path: str, request: Request) -> Response:
        return send_file(request, store.file(rec_id, path))

    @app.get(f"{api}/recordings/{{rec_id}}/series/{{stream}}")
    def series(rec_id: str, stream: str, max_points: int = Query(20000, ge=100)) -> dict[str, list]:
        """imu or orientation samples as columns, for plotting (the parquet files hold everything)."""
        return store.series(rec_id, stream, max_points)

    @app.get(f"{api}/maps")
    def list_maps() -> list[MapSummary]:
        """Built maps, newest first (one per place; build with `nav map build <place>`)."""
        return maps.summaries()

    @app.get(f"{api}/maps/{{map_id}}")
    def get_map(map_id: str) -> PlaceMap:
        return maps.get(map_id)

    @app.get(f"{api}/maps/{{map_id}}/builds")
    def map_builds(map_id: str) -> list[str]:
        return maps.builds(map_id)

    @app.get(f"{api}/maps/{{map_id}}/files/{{path:path}}")
    def map_file(map_id: str, path: str, request: Request) -> Response:
        """Frames, candidates.json, the Claude Code job files ... of any build (paths as in map.json)."""
        return send_file(request, maps.file(map_id, path))

    # --- map builds -------------------------------------------------------------------------------------------
    building: dict = {}  # the build in progress: {map_id, build_id}
    build_lock = threading.Lock()

    def write_status(build_dir: Path, **fields) -> None:
        path = build_dir / "status.json"
        s = json.loads(path.read_text()) if path.is_file() else {}
        s.update(fields)
        tmp = build_dir / ".status.json.tmp"
        tmp.write_text(json.dumps(s, indent=1))
        tmp.replace(path)

    def run_build(map_id: str, build_id: str, req: BuildRequest) -> None:
        from nav.mapping.build import build_map
        from nav.mapping.select import SelectParams

        build_dir = maps.root / map_id / "builds" / build_id
        try:
            res = build_map(req.place, recordings_root=store.root, maps_root=maps.root, cache_root=cache_dir,
                            params=SelectParams(max_candidates=req.max_candidates), embed_model=config.EMBED_MODEL,
                            place_model=config.PLACE_MODEL,
                            transcribe_model=config.TRANSCRIBE_MODEL if req.transcribe else None,
                            llm_model=config.MAP_LLM_MODEL, language=req.language or config.MAP_LANGUAGE,
                            budget_usd=config.MAP_LLM_BUDGET_USD, run_llm=req.llm, effort=config.MAP_LLM_EFFORT,
                            build_id=build_id)
            result = {"status": "done", "published": res.map is not None,
                      "candidates": len(res.candidates.candidates)}
        except Exception as e:  # the status says why; the build folder keeps whatever was made
            log.exception("map build %s/%s failed", map_id, build_id)
            result = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        finally:
            with build_lock:
                building.clear()
        write_status(build_dir, finished_at_utc=datetime.now(timezone.utc).isoformat(), **result)

    @app.post(f"{api}/maps/build", status_code=202)
    def build(req: BuildRequest) -> dict:
        """Build (or rebuild) the map of a place from all its finished recordings, in the background. One build at a
        time; poll `GET /maps/{map_id}/builds/{build_id}/status`. With `llm: false` it stops after selecting
        candidates (no LLM cost)."""
        from nav.mapping.build import recordings_for
        from nav.schemas.map import slugify

        rec_dirs = recordings_for(store.root, req.place)
        if not rec_dirs:
            raise HTTPException(404, f"no finished recordings with video for place {req.place!r}")
        map_id = slugify(Manifest.model_validate_json((rec_dirs[-1] / "manifest.json").read_text()).place)
        with build_lock:
            if building:
                raise HTTPException(409, f"a build is running: {building['map_id']}/{building['build_id']}")
            while True:  # build ids are UTC seconds: wait for a free one
                build_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
                build_dir = maps.root / map_id / "builds" / build_id
                if not build_dir.exists():
                    break
                time.sleep(0.2)
            build_dir.mkdir(parents=True)
            building.update(map_id=map_id, build_id=build_id)
        write_status(build_dir, status="running", map_id=map_id, build_id=build_id, request=req.model_dump(),
                     recordings=len(rec_dirs), started_at_utc=datetime.now(timezone.utc).isoformat())
        threading.Thread(target=run_build, args=(map_id, build_id, req), name=f"build-{map_id}", daemon=True).start()
        return {"map_id": map_id, "build_id": build_id, "status": "running",
                "status_url": f"{api}/maps/{map_id}/builds/{build_id}/status"}

    @app.get(f"{api}/maps/{{map_id}}/builds/{{build_id}}/status")
    def build_status(map_id: str, build_id: str) -> dict:
        """running | done | failed (with error), for builds started through the API."""
        if not build_id.replace("T", "").replace("Z", "").isdigit():
            raise NotFound(build_id)
        return json.loads(maps.file(map_id, f"builds/{build_id}/status.json").read_text())

    def page(folder: Path, assets: tuple[str, ...]) -> HTMLResponse:
        # Cache-bust assets by content hash: Cloudflare forces a 4 h browser TTL on .js/.css, the page itself isn't cached.
        html = (folder / "index.html").read_text()
        for name in assets:
            digest = hashlib.sha256((folder / name).read_bytes()).hexdigest()[:10]
            html = html.replace(f'"{name}"', f'"{name}?v={digest}"')
        return HTMLResponse(html)

    if web_dir:
        @app.get("/", include_in_schema=False)
        @app.get("/index.html", include_in_schema=False)
        def index() -> HTMLResponse:
            return page(web_dir, ("config.js", "app.js", "style.css"))

        @app.get("/dash", include_in_schema=False)
        def dash_redirect() -> RedirectResponse:
            return RedirectResponse("/dash/")

        @app.get("/dash/", include_in_schema=False)
        @app.get("/dash/index.html", include_in_schema=False)
        def dash() -> HTMLResponse:
            """Recordings dashboard: browse places and walks, play video with tags/voice/sensors on one timeline."""
            return page(web_dir / "dash", ("../config.js", "dash.js", "dash.css"))

        @app.get("/viz", include_in_schema=False)
        def viz_redirect() -> RedirectResponse:
            return RedirectResponse("/viz/")

        @app.get("/viz/", include_in_schema=False)
        @app.get("/viz/index.html", include_in_schema=False)
        def viz() -> HTMLResponse:
            """Map viewer: the graph of each place, node and edge details with their frames, directions between nodes."""
            return page(web_dir / "viz", ("../config.js", "viz.js", "viz.css"))

        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app

