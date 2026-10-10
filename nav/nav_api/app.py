"""nav-api: the live side, for Orient's orchestrator (tungsten-united/orient-orchestrator), which calls it server to
server; the phone never does. Read-only on the data: the published maps of every place (the team bucket, mounted
read-only on Cloud Run), routes over them, and localization of live camera frames against their reference images.
The models run on the inference host (helium nav-infer, NAV_INFER_URL), like map-api's.

    GET  /api/v1/health[?deep=1]              deep: also the maps and the inference host
    GET  /api/v1/maps                         MapSummary[]
    GET  /api/v1/maps/{map_id}                PlaceMap: the published map (latest build + review)
    GET  /api/v1/schema/map                   its JSON Schema
    GET  /api/v1/maps/{map_id}/files/{path}   the published build's reference images, frames and embeddings
    POST /api/v1/maps/{map_id}/route          RouteRequest -> Route
    POST /api/v1/maps/{map_id}/localize       images[] (1-4 JPEGs), previous?, expected?, heading_deg?, top_k?,
                                              motion[]? (the phone's Motion JSON per image), previous_step_count?
                                              -> Localization

Same pattern as map-api (nav/web.py): NAV_API_TOKEN on /api/* (its own token), CORS, no frontend. Recordings, map
builds, reviews and build internals stay on map-api (nav/map_api/app.py), which also serves the debugging pages
that exercise this API (/loc/, /nav/); with NAV_FRONTEND_URL set, page paths here redirect there.
"""

import logging
import time

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from nav import config, web
from nav.infer.errors import InferenceError, InferenceUnavailable
from nav.nav_api.maps import MapCache
from nav.navigation.localize import localize
from nav.navigation.route import plan_route
from nav.schemas.map import MAP_SCHEMA_VERSION, MapSummary, PlaceMap
from nav.schemas.navigation import LiveMotion, Localization, Route, RouteHop, RouteRequest, Thresholds

log = logging.getLogger(__name__)
_DEFAULT = object()
MAX_FRAMES = 4  # a short burst
MAX_FRAME_BYTES = 4 * 1024 * 1024
PAGES = ("", "dash", "viz", "loc", "nav")  # the debugging frontend, on map-api


def live_place(images: list[bytes], model: str):
    from nav.mapping.inference import place_live

    return place_live(images, model, config.LIVE_INFER_DEADLINE_S)


def infer_health() -> dict:
    from nav.mapping.inference import remote

    if not (client := remote()):
        return {"ok": True, "where": "local models"}
    try:
        return {"ok": True, "where": client.base_url, **client.health(deadline_s=5)}
    except InferenceError as e:
        return {"ok": False, "where": client.base_url, "error": str(e)}


def create_nav_app(maps: MapCache | None = None, *, place_fn=None, infer_health_fn=None, api_token=_DEFAULT,
                   cors_origins=_DEFAULT, cors_origin_regex=_DEFAULT, frontend_url=_DEFAULT,
                   thresholds: Thresholds | None = None) -> FastAPI:
    """Settings default to nav/config.py (environment); tests pass them explicitly. `place_fn(images, model)` gives
    the place embeddings of JPEG frames (default: the inference host, live priority, short deadline)."""
    maps = maps or MapCache(config.MAPS_DIR, config.MAP_RELOAD_S)
    place_fn = place_fn or live_place
    infer_health_fn = infer_health_fn or infer_health
    api_token = config.API_TOKEN if api_token is _DEFAULT else api_token
    cors_origins = config.CORS_ORIGINS if cors_origins is _DEFAULT else cors_origins
    cors_origin_regex = config.CORS_ORIGIN_REGEX if cors_origin_regex is _DEFAULT else cors_origin_regex
    frontend_url = config.FRONTEND_URL if frontend_url is _DEFAULT else frontend_url
    thresholds = thresholds or config.loc_thresholds()
    app = FastAPI(title="nav-engine nav-api", version=f"map-schema-{MAP_SCHEMA_VERSION}",
                  description="The live API for Orient's orchestrator: published maps, routes, localization.")
    app.state.maps = maps
    web.setup(app, api_token=api_token, cors_origins=cors_origins, cors_origin_regex=cors_origin_regex)

    @app.exception_handler(StarletteHTTPException)
    async def _no_route(request: Request, e: StarletteHTTPException):
        if e.status_code == 404 and e.detail == "Not Found" and request.url.path.startswith("/api/"):
            where = f" ({frontend_url})" if frontend_url else ""
            return JSONResponse({"detail": "not on nav-api, the live API: recordings, map builds, reviews and build "
                                           f"files are on map-api{where}"}, status_code=404)
        return await http_exception_handler(request, e)

    api = "/api/v1"

    @app.get(f"{api}/health")
    def health(deep: bool = False) -> dict:
        out = {"ok": True, "service": "nav-api", "schema_version": MAP_SCHEMA_VERSION}
        if deep:
            out |= {"maps": len(maps.summaries()), "inference": infer_health_fn()}
        return out

    @app.get(f"{api}/maps")
    def list_maps() -> list[MapSummary]:
        """Published maps, newest first (one per place)."""
        return maps.summaries()

    @app.get(f"{api}/maps/{{map_id}}")
    def get_map(map_id: str) -> PlaceMap:
        """The published map: the latest build with its review applied (hidden items left out). Route real users
        over `verified` edges only."""
        return maps.get(map_id)

    @app.get(f"{api}/schema/map")
    def map_schema() -> dict:
        """JSON Schema of a published map (`GET /maps/{map_id}`)."""
        return PlaceMap.model_json_schema(mode="serialization")

    @app.get(f"{api}/maps/{{map_id}}/files/{{path:path}}")
    def map_file(map_id: str, path: str, request: Request):
        """Reference images, candidate frames and the embeddings file of the published build (paths as in the map)."""
        return web.send_file(request, maps.file(map_id, path))

    @app.post(f"{api}/maps/{{map_id}}/route")
    def route(map_id: str, req: RouteRequest) -> Route:
        """The shortest route (by distance) over edges trusted enough, in the direction they may be walked
        (nav/navigation/route.py). `found: false` when trusted edges don't connect the two nodes."""
        pmap = maps.get(map_id)
        ids = {n.id for n in pmap.nodes}
        for node in (req.start, req.goal):
            if node not in ids:
                raise HTTPException(404, f"no node {node!r} in {map_id}")
        hops = plan_route(pmap, req.start, req.goal, req.trust)
        out = [RouteHop(edge=h.edge, source=h.source, target=h.target, forward=h.forward, length_m=h.length_m,
                        bearing_deg=h.bearing_deg, arrival_deg=h.arrival_deg, instruction=h.instruction,
                        steps=h.steps, status=h.status)
               for h in hops or []]
        lengths = [h.length_m for h in out]
        return Route(map_id=map_id, start=req.start, goal=req.goal, trust=req.trust, found=hops is not None, hops=out,
                     length_m=round(sum(lengths), 1) if hops is not None and None not in lengths else None)

    @app.post(f"{api}/maps/{{map_id}}/localize")
    async def localize_frames(
            map_id: str,
            images: list[UploadFile] = File(..., description="1-4 JPEG camera frames, taken within a second or so"),
            previous: str | None = Form(None, description="the last confirmed node: a jump elsewhere stays uncertain"),
            expected: str | None = Form(None, description="the next checkpoint: its score and rank are reported"),
            heading_deg: float | None = Form(None, description="absolute compass heading of the phone, if known"),
            top_k: int = Form(5, ge=1, le=20),
            motion: list[str] | None = Form(None, description="the phone's Motion (JSON, the Orient app's contract): "
                                            "one per image, in the same order; `null` for a frame without one"),
            previous_step_count: int | None = Form(None, description="the phone's stepCount when the user's node "
                                                   "became `previous` (not reset by re-confirming it): with "
                                                   "motion, the walked distance gates confirmations")
            ) -> Localization:
        """Which node of the map the frames show (visual place recognition on the map's reference images): ranked
        nodes with their best matching references, and confirmed / uncertain / lost (nav/navigation/localize.py).
        Stateless: the caller (the orchestrator) keeps the last confirmed node and combines consecutive results."""
        t0 = time.perf_counter()
        if not images or len(images) > MAX_FRAMES:
            raise HTTPException(422, f"send 1..{MAX_FRAMES} images")
        data = [await f.read(MAX_FRAME_BYTES + 1) for f in images]
        if any(len(d) > MAX_FRAME_BYTES for d in data) or not all(data):
            raise HTTPException(413, f"each image must be 1 byte .. {MAX_FRAME_BYTES // 1024 // 1024} MiB")
        motions: list[LiveMotion | None] = []
        for part in motion or []:
            try:
                motions.append(None if part.strip() in ("", "null") else LiveMotion.model_validate_json(part))
            except ValidationError as e:
                raise HTTPException(422, f"motion: {e.errors()[0]['msg']}")
        if motions and len(motions) != len(images):
            raise HTTPException(422, f"send one motion part per image ({len(images)}), `null` for a frame without one")
        index = await run_in_threadpool(maps.index, map_id)
        model = index.pmap.embedding.model if index.pmap.embedding else ""
        took: dict[str, float] = {}
        queries = None
        if index.refs:
            t = time.perf_counter()
            try:
                queries = await run_in_threadpool(place_fn, data, model)
            except InferenceUnavailable as e:
                log.warning("localize %s: %s", map_id, e)
                raise HTTPException(503, f"inference host unavailable: {e}", headers={"Retry-After": "2"})
            except InferenceError as e:
                log.warning("localize %s: %s", map_id, e)
                raise HTTPException(502, f"inference host: {e}")
            took["embed"] = round((time.perf_counter() - t) * 1000, 1)
        t = time.perf_counter()
        res = localize(index, queries if queries is not None else [[0.0]], thresholds, top_k=top_k, previous=previous,
                       expected=expected, heading_abs_deg=heading_deg, model=model, motions=motions or None,
                       previous_step_count=previous_step_count)
        res.frames = len(data)
        took["match"] = round((time.perf_counter() - t) * 1000, 1)
        took["total"] = round((time.perf_counter() - t0) * 1000, 1)
        res.took_ms = took
        return res

    @app.get("/", include_in_schema=False)
    def root():
        if frontend_url:
            return RedirectResponse(f"{frontend_url}/")
        return {"service": "nav-api", "docs": "/docs"}

    if frontend_url:  # old bookmarks: the pages were served here before the split
        for folder in PAGES[1:]:
            for path in (f"/{folder}", f"/{folder}/", f"/{folder}/index.html"):
                app.get(path, include_in_schema=False)(lambda folder=folder: RedirectResponse(f"{frontend_url}/{folder}/"))
    return app
