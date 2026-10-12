"""nav-api's own formats (nav/nav_api/app.py): routes and localization results. Maps are `PlaceMap` (map.py).

Headings are degrees clockwise in the map frame (`PlaceMap.frame`). Scores are cosine similarities between place
embeddings (MegaLoc); they are not probabilities, and the thresholds that turn them into a status are tuned on venue
images (docs/navigation_stack.md, 4 and 6).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
    role: ImageRole | Literal["memory"]  # memory: a frame of the visual memory (PlaceMap.memory), not a reference
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
    distance_m: float | None = None  # shortest walk from `previous` over the map's edges (with previous + motion)
    plausible: bool | None = None  # walked far enough to be there (None: not checked)


class LivePosition(BaseModel):
    """Where the live frames were taken: the position of the memory frame they look most like (map frame, metres,
    the review's scale applied), and how sure that is."""

    x_m: float
    y_m: float
    heading_deg: float | None = None  # that frame's camera heading, map frame
    frame: str  # memory frame id
    rec_id: str
    t_ms: float
    score: float  # place-embedding cosine with it (mean over the burst)
    node: str | None = None  # the node whose region it is in (nearest node within the memory radius)


class LivePose(BaseModel):
    """The camera pose of the live frames from the map's 3D model (nav/mapping/geometry.py): matched against the best
    posed memory frames, their 3D points, PnP. Map frame, metres (the review's scale applied), heading clockwise."""

    x_m: float
    y_m: float
    heading_deg: float  # where the camera faces
    inliers: int  # 2D-3D correspondences that fit the pose: the more, the surer (40+ confirms its node)
    matches: int  # correspondences tried
    focal_px: float  # the camera's focal length the pose implies, at 1024 px on the long side
    node: str | None = None  # the nearest node within the memory radius (None: between nodes)
    frames: list[str] = []  # the memory frames it was matched against


class Thresholds(BaseModel):
    confirm_score: float  # the best node scores at least this ...
    confirm_margin: float  # ... and at least this much more than the second ...
    lost_score: float  # below this nothing is recognised
    # With the phone's motion (each part only applies when its data is sent):
    max_abs_pitch_deg: float = 55.0  # frames looking more up/down than this are left out (the map has none)
    gate_slack_m: float = 1.0  # a node d metres from `previous` is plausible once walked >= d - max(slack_m, ...
    gate_slack_frac: float = 0.35  # ... slack_frac x d)


class LiveOrientation(BaseModel):
    alpha: float
    beta: float
    gamma: float
    absolute: bool = False


class LiveMotion(BaseModel):
    """The phone's motion when a frame was taken, as the Orient app sends it (project-description docs/contracts.md,
    "Motion"): camelCase on the wire. Every field is optional; unknown fields are ignored."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    speed_mps: float | None = Field(None, alias="speedMps")
    cadence_hz: float | None = Field(None, alias="cadenceHz")
    step_count: int | None = Field(None, alias="stepCount")  # since Start, the phone's detector (pdr.StepDetector)
    step_length_m: float | None = Field(None, alias="stepLengthM")  # the phone's guess; the map's is used instead
    heading_deg: float | None = Field(None, alias="headingDeg")  # rear-camera axis, clockwise from north
    heading_source: str | None = Field(None, alias="headingSource")
    heading_accuracy_deg: float | None = Field(None, alias="headingAccuracyDeg")
    orientation: LiveOrientation | None = None
    measured_at: float | None = Field(None, alias="measuredAt")


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
    frames_used: int | None = None  # frames scored, after leaving out those pointed at the floor or ceiling
    pitch_deg: list[float | None] = []  # camera elevation per frame (from the motion's orientation), 0 = level
    walked_m: float | None = None  # since `previous`: (stepCount - previous_step_count) x the map's live step length
    thresholds: Thresholds
    took_ms: dict[str, float]  # embed (the inference host), match, total
    position: LivePosition | None = None  # with a visual memory: where the frame they look most like was taken
    pose: LivePose | None = None  # with a 3D model: where the camera is and which way it faces
