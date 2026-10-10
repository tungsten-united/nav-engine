"""map-api: recordings, map builds, review and map files (/api/v1, OpenAPI docs at /docs), and the debugging
frontend (frontend/: phone recorder at /, /dash/, /viz/, and /loc/ + /nav/, which exercise nav-api).

Two deployments, same code:
- Cloud Run (service map-api, deploy/cloudrun.sh): public, so NAV_API_TOKEN gates /api/* (Bearer header, or ?token=
  where a <video>/<img> can't send headers), NAV_CORS_ORIGINS lets other frontends call it, and NAV_ALLOW_DELETE=0
  keeps the shared data. This is where coworkers open the debugging frontend.
- helium (helium-nav.oscarmartinez.dev): bound to localhost behind the Cloudflare Tunnel; Cloudflare Access is the gate
  (browser login for people, the "helium-clients" service token for workers), so no token of its own. Its pages call
  the Cloud Run services (frontend/config.js).

The live side (maps, routes, localization, for Orient's orchestrator) is nav-api (nav/nav_api/app.py).
"""

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nav import config, web
from nav.mapping.store import MapStore
from nav.recorder.store import NotFound, RecordingStore
from nav.mapping.review import Review, load_review, save_review
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
# The debugging pages: folder -> (assets to cache-bust, what it is). "" is the phone recorder at /.
PAGES = {
    "": (("config.js", "app.js", "style.css"), "Phone recorder"),
    "dash": (("../config.js", "dash.js", "dash.css"), "Recordings dashboard: places and walks, video with tags, "
             "voice and sensors on one timeline"),
    "viz": (("../config.js", "viz.js", "viz.css"), "Map viewer: the graph of each place, details, directions"),
    "loc": (("../config.js", "loc-core.js", "loc.js", "loc.css"),
            "Localization debugger (nav-api): camera, recording or image -> ranked nodes"),
    "nav": (("../config.js", "../loc/loc-core.js", "arrival.js", "nav.js", "../loc/loc.css"),
            "Navigation debugger (nav-api): routes, spoken instructions, live checkpoints, auto-confirmation"),
}


class BuildRequest(BaseModel):
    place: str
    language: str | None = None  # default NAV_MAP_LANGUAGE
    max_candidates: int = 60
    llm: bool = True  # false: stop after selecting candidates (no LLM cost)
    transcribe: bool = True


def create_app(store: RecordingStore | None = None, maps: MapStore | None = None, *, web_dir=_DEFAULT,
               api_token=_DEFAULT, cors_origins=_DEFAULT, cors_origin_regex=_DEFAULT, allow_delete=_DEFAULT,
               cache_dir: Path | None = None, auto_build=_DEFAULT, build_map_fn=None) -> FastAPI:
    """Settings default to nav/config.py (environment); tests pass them explicitly. `build_map_fn` stands in for
    nav.mapping.build.build_map in tests."""
    store = store or RecordingStore(config.RECORDINGS_DIR)
    maps = maps or MapStore(config.MAPS_DIR)
    web_dir = config.WEB_DIR if web_dir is _DEFAULT else web_dir
    api_token = config.API_TOKEN if api_token is _DEFAULT else api_token
    cors_origins = config.CORS_ORIGINS if cors_origins is _DEFAULT else cors_origins
    cors_origin_regex = config.CORS_ORIGIN_REGEX if cors_origin_regex is _DEFAULT else cors_origin_regex
    allow_delete = config.ALLOW_DELETE if allow_delete is _DEFAULT else allow_delete
    auto_build = config.AUTO_BUILD if auto_build is _DEFAULT else auto_build
    cache_dir = cache_dir or config.CACHE_DIR
    app = FastAPI(title="nav-engine map-api", version=f"schema-{SCHEMA_VERSION}")
    app.state.store = store

    web.setup(app, api_token=api_token, cors_origins=cors_origins, cors_origin_regex=cors_origin_regex)

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
        return {"ok": True, "service": "map-api", "schema_version": SCHEMA_VERSION}

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
        tasks.add_task(finalize_and_build, rec_id)
        return m

    @app.post(f"{api}/recordings/{{rec_id}}/finalize")
    def finalize(rec_id: str, tasks: BackgroundTasks, wait: bool = False) -> Manifest:
        """Rebuild video.mp4 / parquet / events.jsonl from the raw uploads (e.g. after late uploads). A usable walk
        then rebuilds its place's map (see finalize_and_build)."""
        if wait:
            return finalize_and_build(rec_id)
        store.get(rec_id)
        tasks.add_task(finalize_and_build, rec_id)
        return store.get(rec_id)

    @app.get(f"{api}/recordings/{{rec_id}}/files")
    def files(rec_id: str) -> list[FileInfo]:
        return store.files(rec_id)

    @app.get(f"{api}/recordings/{{rec_id}}/files/{{path:path}}")
    def file(rec_id: str, path: str, request: Request) -> Response:
        return web.send_file(request, store.file(rec_id, path))

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

    @app.get(f"{api}/schema/map")
    def map_schema() -> dict:
        """JSON Schema of a published map (`GET /maps/{map_id}`), for nav-api's callers."""
        return PlaceMap.model_json_schema(mode="serialization")

    @app.get(f"{api}/maps/{{map_id}}/review")
    def get_review(map_id: str) -> Review:
        """The human review of a place's map (nav/mapping/review.py): approvals, corrections, hidden items."""
        maps.get(map_id)
        return load_review(maps.dir(map_id))

    @app.put(f"{api}/maps/{{map_id}}/review")
    def put_review(map_id: str, review: Review, request: Request) -> PlaceMap:
        """Replace the review and republish the map with it (no LLM call). Only verified edges route real users.
        Returns the published map."""
        from nav.mapping.build import republish

        maps.get(map_id)
        save_review(maps.dir(map_id), review, by=identity(request))
        return republish(maps.dir(map_id))

    @app.get(f"{api}/maps/{{map_id}}/builds")
    def map_builds(map_id: str) -> list[str]:
        return maps.builds(map_id)

    @app.get(f"{api}/maps/{{map_id}}/files/{{path:path}}")
    def map_file(map_id: str, path: str, request: Request) -> Response:
        """Frames, candidates.json, the Claude Code job files ... of any build (paths as in map.json)."""
        return web.send_file(request, maps.file(map_id, path))

    # --- map builds -------------------------------------------------------------------------------------------
    building: dict = {}  # the build in progress: {map_id, build_id}
    pending: dict[str, tuple[BuildRequest, list[str]]] = {}  # map_id -> (auto build to run next, walks it is for)
    build_lock = threading.Lock()

    def write_status(build_dir: Path, name: str = "status.json", **fields) -> None:
        path = build_dir / name
        s = json.loads(path.read_text()) if path.is_file() else {}
        s.update(fields)
        tmp = build_dir / f".{name}.tmp"
        tmp.write_text(json.dumps(s, indent=1))
        tmp.replace(path)

    def run_build(map_id: str, build_id: str, req: BuildRequest) -> None:
        from nav.mapping.select import SelectParams

        if build_map_fn is None:
            from nav.mapping.build import build_map
        else:
            build_map = build_map_fn
        build_dir = maps.root / map_id / "builds" / build_id
        try:
            res = build_map(req.place, recordings_root=store.root, maps_root=maps.root, cache_root=cache_dir,
                            params=SelectParams(max_candidates=req.max_candidates), embed_model=config.EMBED_MODEL,
                            place_model=config.PLACE_MODEL,
                            transcribe_model=config.TRANSCRIBE_MODEL if req.transcribe else None,
                            transcribe_language=config.TRANSCRIBE_LANGUAGE,
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
        start_pending()

    def start_build(req: BuildRequest, trigger: str) -> dict:
        """Start a build of the place in the background: 404 without usable walks, 409 while one is running."""
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
            pending.pop(map_id, None)  # this build takes every walk finalized so far
        write_status(build_dir, status="running", map_id=map_id, build_id=build_id, request=req.model_dump(),
                     trigger=trigger, recordings=len(rec_dirs), started_at_utc=datetime.now(timezone.utc).isoformat())
        threading.Thread(target=run_build, args=(map_id, build_id, req), name=f"build-{map_id}", daemon=True).start()
        log.info("map build %s/%s started (%s)", map_id, build_id, trigger)
        return {"map_id": map_id, "build_id": build_id, "status": "running",
                "status_url": f"{api}/maps/{map_id}/builds/{build_id}/status"}

    # Every walk of a place feeds its one map: when a walk finalizes (in either direction, or into new areas), the
    # place's map is rebuilt from all its walks with the published map as the previous one, so ids, the review and
    # what still holds carry over; a failed build leaves the published map as it was. One build at a time: walks
    # that finalize meanwhile queue one more build of their place, which takes all of them.
    def request_build(place: str, rec_id: str) -> None:
        from nav.schemas.map import slugify

        try:
            start_build(BuildRequest(place=place), f"auto: {rec_id} finalized")
            return
        except HTTPException as e:
            if e.status_code != 409:
                log.warning("no map build after %s: %s", rec_id, e.detail)
                return
        with build_lock:
            if building:
                req, recs = pending.get(slugify(place), (BuildRequest(place=place), []))
                pending[slugify(place)] = (req, [*recs, rec_id])
                log.info("map build of %s queued after the running one (%s finalized)", place, rec_id)
                return
        request_build(place, rec_id)  # it ended in between

    def start_pending() -> None:
        with build_lock:
            if building or not pending:
                return
            map_id, (req, recs) = next(iter(pending.items()))
        try:
            start_build(req, f"auto: {', '.join(recs)} finalized")
        except HTTPException as e:
            if e.status_code == 409:  # another build got in first: this one waits for it
                return
            with build_lock:
                pending.pop(map_id, None)
            log.warning("queued map build of %s dropped: %s", map_id, e.detail)

    def finalize_and_build(rec_id: str) -> Manifest:
        from nav.mapping.build import USABLE_STATUS

        m = store.finalize(rec_id)
        if auto_build and m.status in USABLE_STATUS and m.video:
            try:
                request_build(m.place, rec_id)
            except Exception:  # the walk is saved either way; a build can be started by hand
                log.exception("no map build after %s", rec_id)
        return m

    @app.post(f"{api}/maps/build", status_code=202)
    def build(req: BuildRequest) -> dict:
        """Build (or rebuild) the map of a place from all its finished recordings, in the background. One build at a
        time; poll `GET /maps/{map_id}/builds/{build_id}/status`. With `llm: false` it stops after selecting
        candidates (no LLM cost). Walks that finalize start one by themselves (NAV_AUTO_BUILD, on by default)."""
        return start_build(req, "api")

    def run_refinish(map_id: str, build_dir: Path) -> None:
        from nav.mapping.build import refinish_build

        try:
            res = refinish_build(build_dir, store.root, cache_dir)
            result = {"status": "done", "references": len(res.map.references),
                      "embedding": res.map.embedding.model if res.map.embedding else None}
        except Exception as e:
            log.exception("refinish %s/%s failed", map_id, build_dir.name)
            result = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        finally:
            with build_lock:
                building.clear()
        write_status(build_dir, name="refinish.json", finished_at_utc=datetime.now(timezone.utc).isoformat(), **result)
        start_pending()

    @app.post(f"{api}/maps/{{map_id}}/refinish", status_code=202)
    def refinish(map_id: str) -> dict:
        """Redo the finish of the place's latest LLM build from its saved answer (no LLM cost): reference images and
        their embeddings, headings, status, relations, quality check; then publish it with the review. In the
        background; poll `GET /maps/{map_id}/builds/{build_id}/refinish`."""
        from nav.mapping.build import latest_llm_build

        maps.get(map_id)
        try:
            build_dir = latest_llm_build(maps.dir(map_id))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        with build_lock:
            if building:
                raise HTTPException(409, f"a build is running: {building['map_id']}/{building['build_id']}")
            building.update(map_id=map_id, build_id=build_dir.name)
        write_status(build_dir, name="refinish.json", status="running",
                     started_at_utc=datetime.now(timezone.utc).isoformat(), finished_at_utc=None, error=None)
        threading.Thread(target=run_refinish, args=(map_id, build_dir), name=f"refinish-{map_id}", daemon=True).start()
        return {"map_id": map_id, "build_id": build_dir.name, "status": "running",
                "status_url": f"{api}/maps/{map_id}/builds/{build_dir.name}/refinish"}

    @app.get(f"{api}/maps/{{map_id}}/builds/{{build_id}}/refinish")
    def refinish_status(map_id: str, build_id: str) -> dict:
        if not build_id.replace("T", "").replace("Z", "").isdigit():
            raise NotFound(build_id)
        return json.loads(maps.file(map_id, f"builds/{build_id}/refinish.json").read_text())

    @app.get(f"{api}/maps/{{map_id}}/builds/{{build_id}}/status")
    def build_status(map_id: str, build_id: str) -> dict:
        """running | done | failed (with error), for builds started through the API."""
        if not build_id.replace("T", "").replace("Z", "").isdigit():
            raise NotFound(build_id)
        return json.loads(maps.file(map_id, f"builds/{build_id}/status.json").read_text())

    if web_dir:
        for folder, (assets, doc) in PAGES.items():
            def show(folder=folder, assets=assets) -> HTMLResponse:
                return web.page(web_dir / folder, assets)

            show.__doc__ = doc
            prefix = f"/{folder}/" if folder else "/"
            app.get(prefix, include_in_schema=False)(show)
            app.get(f"{prefix}index.html", include_in_schema=False)(show)
            if folder:
                app.get(f"/{folder}", include_in_schema=False)(lambda prefix=prefix: RedirectResponse(prefix))

        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app

