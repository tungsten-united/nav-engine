"""nav-api's own formats (nav/nav_api/app.py): routes and localization results. Maps are `PlaceMap` (map.py).

Headings are degrees clockwise in the map frame (`PlaceMap.frame`). Scores are cosine similarities between place
embeddings (MegaLoc); they are not probabilities, and the thresholds that turn them into a status are tuned on venue
images (docs/navigation_stack.md, 4 and 6).
"""

from typing import Literal

from pydantic import BaseModel

from nav.schemas.map import ImageRole, Status, Step

Trust = Literal["verified", "observed", "any"]
Confidence = Literal["confirmed", "uncertain", "lost"]


class RouteRequest(BaseModel):
    start: str  # node id
    goal: str  # node id
    trust: Trust = "verified"  # real users: verified edges only; "observed" / "any" are for testing and mapping


class RouteHop(BaseModel):
    edge: str
    source: str  # the node you leave
    target: str  # the node you reach
    forward: bool  # True: along the edge (source -> target); False: walking it back
    length_m: float | None
    bearing_deg: float | None  # walking direction when leaving `source`, map frame
    arrival_deg: float | None = None  # walking direction when reaching `target`, map frame
    instruction: str | None
    steps: list[Step] = []
    status: Status


class Route(BaseModel):
    map_id: str
    start: str
    goal: str
    trust: Trust
    found: bool  # False: trusted edges don't connect start and goal
    hops: list[RouteHop] = []
    length_m: float | None = None  # sum of the hops' lengths (None if any is unknown)


class RefMatch(BaseModel):
    """A reference image of the map and how much the live frames look like it."""

    id: str
    score: float
    role: ImageRole
    edge: str | None = None
    toward: str | None = None
    heading_deg: float | None = None  # camera heading of the reference, map frame
    heading_diff_deg: float | None = None  # live heading minus heading_deg, in (-180, 180]; needs the live heading
    image: str  # map-relative path (GET /maps/{map_id}/files/{image})


class NodeMatch(BaseModel):
    node: str
    name: str
    score: float  # its best reference
    refs: list[RefMatch]  # best first


class Thresholds(BaseModel):
    confirm_score: float  # the best node scores at least this ...
    confirm_margin: float  # ... and at least this much more than the second ...
    lost_score: float  # below this nothing is recognised


class Localization(BaseModel):
    map_id: str
    model: str
    frames: int  # live frames in the request (scores are their mean)
    status: Confidence
    reason: str  # why that status, in a few words
    best: str | None = None  # node id
    margin: float | None = None  # best node's score minus the second's
    candidates: list[NodeMatch]  # best first, at most top_k
    previous: str | None = None  # as given: the last confirmed node
    expected: str | None = None  # as given: the next checkpoint
    expected_score: float | None = None  # how the expected node scored (it may not be in candidates)
    expected_rank: int | None = None  # 1 = best
    heading_map_deg: float | None = None  # the live compass heading in the map frame, when the map is compass-referenced
    thresholds: Thresholds
    took_ms: dict[str, float]  # embed (the inference host), match, total
