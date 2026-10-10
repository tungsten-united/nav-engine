"""Map format (schema_version 1): the semantic topological map of one place, built from all its recordings.

Two layers:

- `MapDraft` is what the LLM writes. Its JSON Schema (`draft_json_schema()`) is handed to the Claude Code job
  today (`claude -p --json-schema`) and is meant for the Anthropic structured-outputs API later, so it keeps to
  what both accept: no recursion, every object closed (`additionalProperties: false`), every field required
  (unknowns are `null` or `[]`), and no numeric or length constraints.
- `PlaceMap` is what is stored and served: the draft plus what code knows better than the LLM (which frames and
  times back each node, dead-reckoned positions, measured walking distances, provenance of the build).

The LLM works from `Candidate`s: frames that survived sampling and filtering, with their motion and voice context.
Nodes and edges cite candidates by id, which is how evidence is traced back to recordings and times.

Ids are stable across builds of a place (nav/mapping/continuity.py): a node that still shows the same frames keeps
its id, new nodes get new ids, and ids that disappear go to `retired` and are never reused.

On disk (one folder per place, mirrored to object storage under maps/<map_id>/). Builds are immutable folders;
map.json at the top is a copy of the latest one, and its image paths point into that build:

    maps/<map_id>/
      map.json                       PlaceMap of the latest build
      builds/<build_id>/
        map.json                     PlaceMap
        candidates.json              CandidateSet: what the LLM was given
        previous.json                the map before this build, as given to the LLM (continuity.previous_context)
        pairs.json                   same-place checks between keyframes (verify.py), for tuning
        tracks.json                  dead-reckoned path of every walk at 1 Hz: {rec_id: [[t_ms, x_m, y_m], ...]}
        frames/<cand_id>.jpg         one image per candidate
        refs/<ref_id>.jpg            reference images for live localization (approach / at / outgoing, per node)
        embeddings.npy               place embeddings of the reference images, row = ReferenceImage.embedding_row
        job/                         prompt.md, schema.json, claude.json (raw Claude Code result), claude.log
      review.json                    human review overlay (nav/mapping/review.py), keyed by stable ids

The top map.json is the latest build's map.json with review.json applied (verified status, edits, hidden items).

For the navigation side (docs/navigation_stack.md): headings are degrees clockwise in the map frame (`PlaceMap.frame`;
compass-referenced when every walk had a compass, so a live absolute heading H is H - frame.heading_zero_abs_deg).
`status` says how far a fact can be trusted: observed (in a walk), inferred (deduced), verified (a person checked it).
Routing for real users uses verified edges only (nav/navigation/route.py).
"""

import re
import unicodedata
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAP_SCHEMA_VERSION = 1

NodeKind = Literal[
    "entrance",  # way into / out of the place
    "door",  # door between two areas
    "corridor",  # stretch of corridor worth naming (long corridors are usually edges)
    "junction",  # where paths split or cross
    "room",  # enclosed room (meeting room, toilet, kitchen ...)
    "open_area",  # open-plan space, hall, lobby
    "stairs",
    "lift",
    "destination",  # a place people go to: reception, desk, coffee machine, exit ...
    "landmark",  # notable object used to orient: pillar, plant, vending machine ...
    "other",
]
EdgeKind = Literal["walk", "door", "stairs", "lift", "ramp", "other"]
Confidence = Literal["high", "medium", "low"]
Status = Literal["observed", "inferred", "verified"]
Direction = Literal["ahead", "right", "behind", "left"]
ImageRole = Literal["approach", "at", "outgoing"]
InstructionKind = Literal["straight", "turn_left", "turn_right", "turn_around"]


def slugify(place: str) -> str:
    """Map id for a place: "Office, 3rd floor" -> "office-3rd-floor" (case- and accent-insensitive like grouping)."""
    s = unicodedata.normalize("NFKD", " ".join(place.split()).casefold()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "place"


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- what the LLM writes ------------------------------------------------------------------------------------
class DraftRelation(_Closed):
    subject: str = Field(description='A place or fixed object someone may ask for or orient by, as named in the map '
                                     'where it is a node: "Kitchen", "Bar counter", "Stage".')
    candidate: str = Field(description="Id of one of this node's candidates, the photo it is judged from.")
    side: Literal["in_photo", "left", "right", "behind"] = Field(
        description="Where it is from where that photo was taken, facing the way the camera faced: in_photo if the "
                    "photo shows it, else left, right or behind (known from the node's other photos or the walk).")
    x: float | None = Field(description="For in_photo: where it is across the photo, 0 = left edge, 0.5 = centre, "
                                        "1 = right edge. Else null.")
    evidence: Literal["observed", "inferred"] = Field(description="observed: the photo shows it. inferred: deduced.")


class MapNode(_Closed):
    id: str = Field(description='Short unique id: "n1", "n2", ...')
    name: str = Field(description='Short spoken label, 1-4 words, unique in the map: "Main entrance", "Kitchen", '
                                  '"Stage", "Mural tables".')
    kind: NodeKind
    description: str = Field(description="What is here, for a blind person, without visual detail: layout and size, "
                                         "what is on each side within reach, floor surface, sounds, smells, "
                                         "temperature. 1-3 sentences.")
    landmarks: list[str] = Field(description="Non-visual cues that confirm you have arrived: things a cane, feet or "
                                             "hands can find, sounds, smells, floor changes.")
    hazards: list[str] = Field(description="Obstacles or dangers here: steps, glass doors, low furniture ...")
    visual_cues: list[str] = Field(description="What a sighted helper or a camera would recognise here: colours, "
                                               "signs, screens, murals, lights. Kept apart from what is spoken.")
    floor: str | None = Field(description='Floor or level if known ("ground", "1", "-1"), else null.')
    candidates: list[str] = Field(description="Ids of the candidate frames that show this node (at least one).")
    relations: list[DraftRelation] = Field(description="Where other places and fixed objects are from here, seen "
                                                       "from this node's photos (a few of the most useful, or none).")
    confidence: Confidence


StepAction = Literal[
    "straight",  # go ahead / keep going
    "bear_left", "bear_right",  # change direction by roughly 20-60 degrees
    "turn_left", "turn_right",  # roughly 60-135 degrees
    "turn_around",
    "go_through",  # a door or doorway
    "step_up", "step_down",  # a step, a platform edge, a single stair
    "stairs_up", "stairs_down", "lift",
]
Basis = Literal["walked", "inferred"]


class Step(_Closed):
    action: StepAction = Field(description="What to do at the start of this step. The first step's turn is the turn "
                                           "to make when leaving the source.")
    distance_m: float | None = Field(description="Metres walked in this step, from the motion data; null for actions "
                                                 "without walking (a turn on the spot, a step up).")
    along: str | None = Field(description="What to keep beside you while walking, with its side, non-visual: "
                                          "\"stone wall on your left\", \"trail the counter with your right hand\".")
    until: str | None = Field(description="The non-visual cue that ends this step: \"until the floor turns to "
                                          "carpet\", \"until your cane finds the bins\". The last step's cue tells "
                                          "you that you have arrived.")


class MapEdge(_Closed):
    id: str = Field(description='Short unique id: "e1", "e2", ...')
    source: str = Field(description="Node id the instruction starts from.")
    target: str = Field(description="Node id the instruction leads to.")
    kind: EdgeKind
    bidirectional: bool = Field(description="True if it can be walked both ways (almost always).")
    basis: Basis = Field(description="walked: a walk went from source to target without passing another node "
                                     "(consecutive on a timeline). inferred: no walk did, the connection is deduced.")
    steps: list[Step] = Field(description="The route as steps, in order. The instruction is the spoken form of these "
                                          "steps and must agree with them.")
    instruction: str = Field(description="The steps as you would say them to a blind person, egocentric, without "
                                         "visual detail: direction, distance, what you pass and can touch or hear, "
                                         "how you know you have arrived.")
    reverse_instruction: str | None = Field(description="The same route spoken for target -> source, or null if "
                                                        "not bidirectional.")
    landmarks: list[str] = Field(description="Non-visual things passed along the way that help to keep track.")
    hazards: list[str]
    visual_cues: list[str] = Field(description="What a sighted helper or a camera would recognise along the way.")
    candidates: list[str] = Field(description="For a walked edge, the candidate frames of that walk from the source "
                                              "to the target, in walk order (as on the timeline). For an inferred "
                                              "edge, frames that support the connection, or empty.")
    confidence: Confidence


class MapDraft(_Closed):
    place: str
    language: str = Field(description='Language of all the text in the map, BCP 47 ("en", "es").')
    summary: str = Field(description="Overview of the place for a blind person, without visual detail: shape and "
                                     "size, main areas, how they connect, where the entrances are. 2-5 sentences.")
    nodes: list[MapNode]
    edges: list[MapEdge]
    notes: list[str] = Field(description="Uncertainties, places that were merged or left apart and why, "
                                         "and what should be recorded again.")


def draft_json_schema() -> dict:
    """JSON Schema for MapDraft, self-contained ($defs inlined by pydantic as refs at the top level)."""
    return MapDraft.model_json_schema()


# --- what the LLM reads -------------------------------------------------------------------------------------
class Motion(BaseModel):
    """Dead-reckoned motion at a frame, from the IMU (steps x step length) and the fused heading."""

    walked_m: float  # cumulative distance since the start of the recording
    heading_deg: float | None  # camera-axis heading, 0 = magnetic north, clockwise (indoors: only roughly true)
    x_m: float  # dead-reckoned position in the recording's own frame (east, north)
    y_m: float
    pitch_deg: float | None  # camera elevation: 0 = level, + up, - down
    moving: bool
    # the same position in a frame shared by the walks (aligned through frames they share); None when this walk
    # shares none with the others
    map_x_m: float | None = None
    map_y_m: float | None = None


class View(BaseModel):
    """One sighting of a candidate place in one recording."""

    rec_id: str
    t_ms: float  # recording clock
    sharpness: float
    motion: Motion
    marks: list[str] = []  # recorder tags pressed within a few seconds of this frame
    voice: list[str] = []  # transcripts of voice notes spoken around this frame


class Candidate(BaseModel):
    """A frame that survived filtering. Near-identical frames (revisits, other walks) are folded into its views."""

    id: str  # "c01" ...
    image: str  # relative to the build folder, e.g. frames/c01.jpg
    reason: list[str]  # why it was kept: start, end, mark, voice, novel, turn, distance
    views: list[View]  # the first one is the frame shown in the image
    possibly_same_as: list[str] = []  # other candidates that look alike but were not merged


class TimelineStep(BaseModel):
    """A recording as the sequence of candidates it passes, with the motion between consecutive ones."""

    t_ms: float
    candidate: str
    walked_m: float  # since the start of the recording
    from_prev_m: float | None  # walked since the previous step
    turn_deg: float | None  # heading change since the previous step, + = turned right, - = left
    turn_at_deg: float | None = None  # the turn made at this spot (heading 3 s after vs 3 s before), + = right
    heading_deg: float | None


class RecordingInfo(BaseModel):
    rec_id: str
    created_at_utc: datetime
    duration_s: float | None
    notes: str
    step_length_m: float  # used for this walk: re-estimated from the video when it could be, else the recorder's
    live_step_length_m: float | None = None  # video metres per step of the phone's causal detector (live tracker)
    distance: str = "steps"  # steps (x step length) | video (floor odometry, odometry.py) | time (no sensors)
    video_distance_share: float | None = None  # fraction of walked_m measured from the floor in the video
    motion: str = "imu"  # imu: steps and heading from the phone; video: no sensors, heading from the video and
    #                      walked metres guessed from time (~1 m/s): distances are rough
    heading: str | None = None  # compass (north-referenced) | gyro (relative to the start) | video | none
    fov_deg: float | None = None  # horizontal field of view of the video (calibrated, or the default)
    video_offset_ms: float = 0  # subtracted from video times to put frames on the sensor clock (calibrated)
    aligned: bool | None = None  # its views have map_x_m / map_y_m (it shares frames with the other walks)
    walked_m: float
    frames_sampled: int
    frames_usable: int
    marks: list[dict]  # [{t_ms, tag}]
    voice: list[dict]  # [{t_ms, t_end_ms, text}]
    timeline: list[TimelineStep]


class CandidateSet(BaseModel):
    place: str
    map_id: str
    recordings: list[RecordingInfo]
    candidates: list[Candidate]
    params: dict
    embed_model: str
    embed_dim: int | None = None
    place_model: str | None = None  # None: built before MegaLoc (DINOv2 decided merges)


# --- what is stored -----------------------------------------------------------------------------------------
class Evidence(BaseModel):
    candidate: str
    rec_id: str
    t_ms: float
    image: str


class Point(BaseModel):
    x_m: float
    y_m: float


class ItemQuality(BaseModel):
    """Cheap second opinion on one node or edge (TypeSafe Jev, nav/mapping/quality.py). Raw judgments are kept so
    thresholds can change without asking again; `flags` are what the current thresholds make of them."""

    visual_p: float  # probability that the spoken text relies on sight
    usable: float  # 0-3: nodes, how well a blind person can tell they arrived; edges, how followable the instruction is
    turn_agrees_p: float | None = None  # edges with a clear measured turn: does the instruction turn the same way?
    flags: list[str] = []


class QualityReport(BaseModel):
    checker: str = "typesafe"
    model: str | None = None
    checked_at_utc: datetime | None = None
    input_tokens: int | None = None
    flagged: int = 0
    skipped: str | None = None  # why the check didn't run (no API key, error ...)


class SpatialRelation(BaseModel):
    """Where something is as seen from a recorded viewpoint at a node (docs/navigation_stack.md 2.3). With the
    user's heading known at the node, a caller can turn `heading_deg` into ahead / left / right without asking again."""

    id: str  # "<node id>:<subject slug>", stable while the node and the subject keep their names
    subject: str
    candidate: str
    rec_id: str
    t_ms: float
    viewpoint_image: str  # map-relative path of the photo
    viewpoint_heading_deg: float | None  # where the camera faced (map frame); None without a heading
    bearing_deg: float  # from the camera's direction, + = right
    heading_deg: float | None  # direction of the subject from the node, map frame
    arrival_heading_deg: float | None  # how that walk faced when it arrived at the node
    direction: Direction  # relative to the arrival heading (to the camera's direction without one)
    status: Status


class ReferenceImage(BaseModel):
    """A frame kept for live localization: approaching the node, at it, or leaving it along an edge."""

    id: str
    node: str
    edge: str | None = None  # outgoing images: the edge walked next
    toward: str | None = None  # ... and the node it leads to
    role: ImageRole
    image: str  # map-relative path
    rec_id: str
    t_ms: float
    heading_deg: float | None  # camera heading, map frame
    embedding_row: int | None = None  # row in PlaceMap.embedding.file


class EmbeddingInfo(BaseModel):
    model: str  # live frames must be embedded with this same model
    dim: int
    file: str  # map-relative path of the .npy ([N, dim] float32 unit vectors)


class MapFrame(BaseModel):
    """The frame of positions and headings: x east, y north when compass-referenced, metres; headings clockwise."""

    heading_source: str  # compass (every walk), or relative (some walk had no compass: only roughly aligned)
    heading_zero_abs_deg: float | None  # absolute heading of +y; None when not compass-referenced
    step_length_m: float | None  # mean of the walks'
    # The live tracker's starting step length: metres per step of the causal detector the phone runs (pdr.py),
    # mean of the walks'. None for maps built before it, or without floor odometry.
    live_step_length_m: float | None = None
    note: str = "dead reckoning: ~5-10% distance error, drifting a few metres per minute; indoor compass is rough"
    # Distances measured on site (review.json `distances`) rescale every metric quantity of the published map by one
    # factor: edge and step distances, positions, paths, tracks and the step length.
    distance_scale: float = 1.0
    measured_distances: list["MeasuredDistanceCheck"] = []


class MeasuredDistanceCheck(BaseModel):
    """A distance measured on site and how the published map matches it (along the shortest walk between them)."""

    between: tuple[str, str]  # node ids
    meters: float  # measured
    sigma_m: float
    map_m: float | None  # the map's distance after scaling; None if the nodes aren't connected
    residual_m: float | None  # map_m - meters


class PlaceNode(MapNode):
    model_config = ConfigDict(extra="ignore")
    visual_cues: list[str] = []  # default: maps built before the field existed
    relations: list[SpatialRelation] = []  # computed from the draft's DraftRelations
    quality: ItemQuality | None = None
    evidence: list[Evidence] = []
    position: Point | None = None  # dead-reckoned, recordings aligned through shared nodes; approximate
    status: Status = "observed"
    review_flags: list[str] = []  # what a reviewer should check
    references: list[str] = []  # ReferenceImage ids


class PlaceEdge(MapEdge):
    model_config = ConfigDict(extra="ignore")
    # defaults: maps built before these fields existed
    visual_cues: list[str] = []
    basis: Basis = "walked"
    steps: list[Step] = []
    distance_m: float | None = None  # measured if the edge was walked, else the sum of the steps' distances
    measured_distance_m: float | None = None  # walked between the two nodes' frames (centre to centre) in one recording
    reverse_steps: list[Step] = []  # computed from steps: turn around, steps in reverse order, left and right swapped
    path: list[Point] = []  # the dead-reckoned walk between the two nodes, for drawing (same frame as positions)
    walked_at_ms: float | None = None  # when the walk that `path` comes from left the source (recording clock)
    walked_in: str | None = None  # ... and which recording that was
    walked: bool = True  # False: no walk went directly between the two nodes (it passed other nodes in between)
    turn_deg: float | None = None  # heading change when leaving the source, vs arriving at it on the previous edge
    quality: ItemQuality | None = None
    # source -> target was walked (observed) or not (inferred); target -> source likewise, None if not bidirectional
    status: Status = "observed"
    reverse_status: Status | None = None
    review_flags: list[str] = []
    instruction_kind: InstructionKind | None = None  # from the measured turn when leaving the source, else the steps
    departure_heading_deg: float | None = None  # walking direction when leaving the source (map frame)
    arrival_heading_deg: float | None = None  # walking direction when reaching the target
    reference_views: list[str] = []  # outgoing ReferenceImage ids along this edge
    walked_forward: int | None = None  # walks that went source -> target directly (None: built before this was counted)
    walked_back: int | None = None  # ... and target -> source


class LlmRun(BaseModel):
    backend: Literal["claude_code", "anthropic_api"]
    model: str
    effort: str | None = None
    cost_usd: float | None = None
    duration_s: float | None = None
    session_id: str | None = None


class BuildInfo(BaseModel):
    build_id: str  # 20261009T221500Z
    built_at_utc: datetime
    pipeline_version: str
    params: dict
    embed_model: str
    embed_dim: int | None = None
    place_model: str | None = None  # None: built before MegaLoc (DINOv2 decided merges)
    transcribe_model: str | None
    llm: LlmRun
    candidates: int


class Track(BaseModel):
    """A recording's dead-reckoned path at 1 Hz, shifted into the map frame. Points are [t_ms, x_m, y_m]."""

    rec_id: str
    points: list[tuple[float, float, float]]


class Retired(BaseModel):
    """A node or edge id that was in an earlier build of the place and is not any more. Never reused."""

    id: str
    kind: Literal["node", "edge"]
    name: str  # node name, or "source name - target name" for an edge
    build_id: str  # the first build without it


class PlaceMap(MapDraft):
    model_config = ConfigDict(extra="ignore")
    schema_version: int = MAP_SCHEMA_VERSION
    map_id: str
    recordings: list[str]
    build: BuildInfo
    nodes: list[PlaceNode]
    edges: list[PlaceEdge]
    quality: QualityReport | None = None
    tracks: list[Track] = []  # walks that could be placed in the map frame (through shared nodes)
    warnings: list[str] = []  # problems found when checking the draft (dangling ids, unknown candidates ...)
    retired: list[Retired] = []  # ids of earlier builds that are gone, carried from build to build
    frame: MapFrame | None = None
    references: list[ReferenceImage] = []
    embedding: EmbeddingInfo | None = None
    reviewed_at_utc: datetime | None = None  # when review.json was last applied (None: never reviewed)
    hidden: list[str] = []  # node / edge / relation ids hidden by the review (still in the build's map.json)


class MapSummary(BaseModel):
    map_id: str
    place: str
    built_at_utc: datetime
    recordings: int
    nodes: int
    edges: int


def place_map_json_schema() -> str:
    """JSON Schema of the published map, for nav-api's callers (docs/map.schema.json)."""
    import json

    return json.dumps(PlaceMap.model_json_schema(mode="serialization"), indent=1, ensure_ascii=False) + "\n"
