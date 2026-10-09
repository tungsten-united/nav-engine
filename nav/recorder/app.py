"""Recorder server: the phone web app (/) and the recordings API (/api/v1, OpenAPI docs at /docs).

Bound to localhost and published through the Cloudflare Tunnel; Cloudflare Access is the gate (browser login
for people, the "helium-clients" service token for other workers), so the app itself has no auth.
"""

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from nav import config
from nav.recorder.store import Conflict, NotFound, RecordingStore
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

WEB_DIR = Path(__file__).parent / "web"


def create_app(store: RecordingStore | None = None) -> FastAPI:
    store = store or RecordingStore(config.RECORDINGS_DIR)
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
        return "service-token" if h.get("cf-access-jwt-assertion") else None

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
    def file(rec_id: str, path: str) -> FileResponse:
        return FileResponse(store.file(rec_id, path))

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app

