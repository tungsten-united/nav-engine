"""nav-api's decision log: one JSON line on stdout per /localize and /route call. Cloud Run stores it as a structured
entry whose `message` is the summary line in the Logs tab and whose `severity` makes problems stand out, the same
format as Orient's orchestrator's trace, so one Cloud Logging query shows both sides of a session.

    {"kind": "nav_localize", "service": "nav-api", "mapId": "itnig", "status": "confirmed", "best": "n2", ...,
     "message": "[nav-api] localize itnig · confirmed Kitchen 0.71 (±0.08) · 812 ms", "severity": "INFO"}

The orchestrator may send X-Client-Id / X-Session-Id / X-Request-Id; they come back as clientId / sessionId /
requestId, as in its own entries.
"""

import json
from functools import cache
from importlib.metadata import PackageNotFoundError, version

from fastapi import Request

from nav import config
from nav.schemas.navigation import Localization, Route

SERVICE = "nav-api"
CALLER_HEADERS = {"clientId": "x-client-id", "sessionId": "x-session-id", "requestId": "x-request-id"}
MAX_ID = 64


@cache
def _version() -> str | None:
    try:
        return version("nav-engine")
    except PackageNotFoundError:
        return None


def caller(request: Request) -> dict:
    """Who asked, as far as the orchestrator says (headers, truncated)."""
    return {k: v[:MAX_ID] for k, h in CALLER_HEADERS.items() if (v := request.headers.get(h))}


def entry(kind: str, *, message: str, severity: str = "INFO", **fields) -> dict:
    """A log line: the fields (None left out), which build produced it, and the summary prefixed with the service."""
    out = {"kind": kind, "service": SERVICE, **{k: v for k, v in fields.items() if v is not None},
           "commit": config.COMMIT, "version": _version(),
           "message": f"[{SERVICE}] {message}", "severity": severity}
    return {k: v for k, v in out.items() if v is not None}


def emit(e: dict) -> None:
    print(json.dumps(e, ensure_ascii=False), flush=True)  # one JSON line: Cloud Logging stores it as jsonPayload


def localized(res: Localization, who: dict) -> dict:
    """The line for a localization: what was recognised, how sure, and why."""
    top = res.candidates[0] if res.candidates and res.candidates[0].node == res.best else None
    what = res.status
    if top:
        what += f" {top.name} {top.score:.2f}" + (f" (±{res.margin:.2f})" if res.margin is not None else "")
    if res.status != "confirmed" or not top:
        what += f": {res.reason}"
    if res.expected:
        what += f" · expected {res.expected} #{res.expected_rank}" if res.expected_rank else f" · expected {res.expected}"
    took = res.took_ms.get("total")
    return entry("nav_localize", **who, mapId=res.map_id, status=res.status, reason=res.reason, best=res.best,
                 bestName=top.name if top else None, score=top.score if top else None, margin=res.margin,
                 previous=res.previous, expected=res.expected, expectedScore=res.expected_score,
                 expectedRank=res.expected_rank, headingDeg=res.heading_map_deg, frames=res.frames,
                 framesUsed=res.frames_used, walkedM=res.walked_m, distanceM=top.distance_m if top else None,
                 plausible=top.plausible if top else None, model=res.model or None,
                 poseInliers=res.pose.inliers if res.pose else None, poseNode=res.pose.node if res.pose else None,
                 poseXY=[res.pose.x_m, res.pose.y_m] if res.pose else None,
                 tookMs=res.took_ms, message=f"localize {res.map_id} · {what}" + (f" · {took:.0f} ms" if took else ""),
                 severity="WARNING" if res.status == "lost" else "INFO")


def routed(route: Route, names: dict[str, str], who: dict) -> dict:
    """The line for a route: from where to where, and whether trusted edges connect them."""
    trip = f"{names.get(route.start, route.start)} → {names.get(route.goal, route.goal)} ({route.trust})"
    if not route.found:
        outcome = "no route"
    else:
        outcome = f"{len(route.hops)} hops" + (f", {route.length_m:.1f} m" if route.length_m is not None else "")
    return entry("nav_route", **who, mapId=route.map_id, start=route.start, goal=route.goal, trust=route.trust,
                 found=route.found, hops=len(route.hops), lengthM=route.length_m,
                 message=f"route {route.map_id} · {trip} · {outcome}",
                 severity="INFO" if route.found else "WARNING")


def failed(kind: str, map_id: str, status: int, detail: str, who: dict, **fields) -> dict:
    """The line for a call that ended in an error response: the client's fault is a warning, ours an error."""
    return entry(f"nav_{kind}", **who, mapId=map_id, httpStatus=status, error=detail, **fields,
                 message=f"{kind} {map_id} · error {status}: {detail}",
                 severity="ERROR" if status >= 500 else "WARNING")
