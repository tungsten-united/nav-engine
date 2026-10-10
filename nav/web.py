"""What both HTTP services share (map-api: nav/map_api/app.py, nav-api: nav/nav_api/app.py): the API token, CORS,
no stale assets behind Cloudflare, error handlers, files under Cloud Run's response cap, cache-busted pages."""

import hashlib
import hmac
import mimetypes
import re
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from nav.recorder.store import Conflict, NotFound

# Cloud Run cuts an HTTP/1 response with a Content-Length over 32 MiB (the browser sees a 500): larger files are
# served in ranges of at most this much, or streamed without a length.
MAX_RESPONSE = 8 * 1024 * 1024
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def setup(app: FastAPI, *, api_token: str | None, cors_origins: list[str], cors_origin_regex: str | None) -> None:
    """Error handlers and middleware, in this order: no-cache headers, the token on /api/* (except health and CORS
    preflights), then CORS outermost, so 401s and preflights get CORS headers too."""

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

    if cors_origins or cors_origin_regex:  # added last: outermost
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_origin_regex=cors_origin_regex,
                           allow_methods=["*"], allow_headers=["Authorization", "Content-Type", "X-Nav-User"],
                           max_age=600)


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


def page(folder: Path, assets: tuple[str, ...]) -> HTMLResponse:
    """folder/index.html with ?v=<content hash> on its assets: Cloudflare forces a 4 h browser TTL on .js/.css, and
    the page itself isn't cached (no_stale_assets)."""
    html = (folder / "index.html").read_text()
    for name in assets:
        digest = hashlib.sha256((folder / name).read_bytes()).hexdigest()[:10]
        html = html.replace(f'"{name}"', f'"{name}?v={digest}"')
    return HTMLResponse(html)
