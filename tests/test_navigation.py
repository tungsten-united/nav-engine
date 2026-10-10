"""The navigation-side pieces: calibration, video-only recordings, reference images, relations, status and review,
routing, dead reckoning along a route, the map schema for the app, and their API."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from synthetic import W, make_video, panorama, turn_right, walking_accel

from nav.mapping.build import build_map, finish_map, publish
from nav.mapping.motion import angle_diff
from nav.mapping.calibrate import calibrate, estimate, horizontal_shift, video_yaw
from nav.mapping.review import EdgeReview, NodeReview, RelationReview, Review, apply_review, from_template, template
from nav.mapping.select import SelectParams
from nav.mapping.store import MapStore
from nav.navigation.pdr import GraphTracker, StepDetector, detect_steps
from nav.navigation.route import Hop, plan_route
from nav.recorder.app import create_app
from nav.recorder.store import RecordingStore
from nav.schemas.map import (
    BuildInfo,
    Candidate,
    CandidateSet,
    LlmRun,
    MapDraft,
    Motion,
    PlaceMap,
    RecordingInfo,
    View,
    place_map_json_schema,
)

API = "/api/v1"
ROOT = Path(__file__).resolve().parent.parent
BUILD = BuildInfo(build_id="20261010T000000Z", built_at_utc=datetime(2026, 10, 10, tzinfo=timezone.utc),
                  pipeline_version="t", params={}, embed_model="m", transcribe_model=None,
                  llm=LlmRun(backend="claude_code", model="x"), candidates=3)


# --- calibration -------------------------------------------------------------------------------------------
def test_horizontal_shift_finds_a_pan():
    pano = panorama()
    a, b = pano[:, 300:300 + W], pano[:, 312:312 + W]  # the camera turned right: the scene moved 12 px left
    assert abs(horizontal_shift(a, b) + 12) < 1.5


@pytest.fixture(scope="module")
def pan_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("video") / "pan.mp4"
    make_video(path)  # turns right 90 degrees between 5 and 7 s, 40 degree field of view
    return path


def test_video_yaw_and_calibration_recover_offset_and_fov(pan_video):
    t_v, yaw = video_yaw(pan_video, video_start_ms=0)
    assert abs(yaw[-1] * 40 - 90) < 10  # 90 degrees = 2.25 image widths at 40 degrees
    # The sensors saw the turn 400 ms before the video (stamped late by the encoder start).
    t_s = np.arange(0, 12000, 20.0)
    heading = np.array([(turn_right(t / 1000 + 0.4) + 30) % 360 for t in t_s])
    lag, fov, corr = estimate(t_s, heading, t_v, yaw)
    assert abs(lag - 400) <= 100 and abs(fov - 40) < 6 and corr > 0.8
    cal = calibrate(t_s, heading, t_v, yaw, "handheld_portrait")
    assert cal.applied and cal.offset_ms == lag
    flat = calibrate(t_s, np.full(len(t_s), 30.0), t_v, yaw, "handheld_portrait")  # nothing to correlate
    assert not flat.applied and flat.offset_ms == 0 and flat.fov_deg == 40


# --- a video without sensors, through the whole pipeline -----------------------------------------------------
def fake_models(monkeypatch):
    """Embeddings from the pixels (8x8 thumbnails), no same-place folding: no torch, no inference host."""
    from PIL import Image

    def thumbs(paths, model):
        v = np.stack([np.asarray(Image.open(p).convert("L").resize((8, 8)), np.float32).ravel() for p in paths])
        v -= v.mean(axis=1, keepdims=True)
        return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-6)

    monkeypatch.setattr("nav.mapping.prepare.embed_files", thumbs)
    monkeypatch.setattr("nav.mapping.prepare.place_files", thumbs)
    monkeypatch.setattr("nav.mapping.inference.place_files", thumbs)
    monkeypatch.setattr("nav.mapping.verify.pair_inliers", lambda paths, pairs: [0] * len(pairs))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)


def test_imported_video_builds_a_map_with_references_and_relations(tmp_path, pan_video, monkeypatch):
    fake_models(monkeypatch)
    store = RecordingStore(tmp_path / "recordings")
    m = store.import_video(pan_video, "Video hall", notes="no phone")
    assert m.status == "complete" and m.video and m.created_by == "import"
    assert not (store.dir(m.rec_id) / "raw" / "video").exists() or not any((store.dir(m.rec_id) / "raw" / "video").iterdir())

    maps_root, cache = tmp_path / "maps", tmp_path / "cache"
    res = build_map("video hall", recordings_root=store.root, maps_root=maps_root, cache_root=cache,
                    params=SelectParams(), embed_model="thumb", place_model="thumb", transcribe_model=None,
                    llm_model="x", language="en", budget_usd=0, run_llm=False)
    cset = res.candidates
    rec = cset.recordings[0]
    assert rec.motion == "video" and rec.heading == "video" and rec.fov_deg == 40
    assert rec.walked_m > 5  # guessed from time
    first, last = cset.candidates[0], cset.candidates[-1]
    heading = lambda c: c.views[0].motion.heading_deg  # noqa: E731
    assert abs(angle_diff(0, heading(first))) < 10 and abs(angle_diff(90, heading(last))) < 15  # the video's turn

    node = lambda i, c, rel: {"id": i, "name": f"Spot {i}", "kind": "room", "description": "", "landmarks": ["wall"],  # noqa: E731
                              "hazards": [], "visual_cues": [], "floor": None, "candidates": [c], "relations": rel,
                              "confidence": "high"}
    draft = MapDraft.model_validate({
        "place": "Video hall", "language": "en", "summary": "s", "notes": [],
        "nodes": [node("n1", first.id, [{"subject": "Bar", "candidate": first.id, "side": "in_photo", "x": 1.0,
                                         "evidence": "observed"},
                                        {"subject": "Exit", "candidate": first.id, "side": "behind", "x": None,
                                         "evidence": "inferred"},
                                        {"subject": "Ghost", "candidate": last.id, "side": "left", "x": None,
                                         "evidence": "inferred"}]),
                  node("n2", last.id, [])],
        "edges": [{"id": "e1", "source": "n1", "target": "n2", "kind": "walk", "bidirectional": True,
                   "basis": "walked", "steps": [{"action": "turn_right", "distance_m": None, "along": None,
                                                 "until": None}],
                   "instruction": "go", "reverse_instruction": "back", "landmarks": [], "hazards": [],
                   "visual_cues": [], "candidates": [], "confidence": "high"}]})
    pmap = publish(res.build_dir, cset, draft, LlmRun(backend="claude_code", model="x"), store.root, cache).map

    n1, n2 = pmap.nodes
    bar, exit_ = n1.relations
    assert (bar.id, bar.bearing_deg, bar.heading_deg) == ("n1:bar", 20.0, round((heading(first) + 20) % 360, 1))
    assert exit_.direction == "behind" and exit_.status == "inferred"
    assert any("Ghost" in w for w in pmap.warnings)  # cites a candidate of another node: dropped
    e = pmap.edges[0]
    assert e.measured_distance_m is None  # video-only walks don't measure distances
    # the turn comes 5 s into the edge, not when leaving n1: the measured turn wins over the first step
    assert (e.status, e.reverse_status) == ("observed", "inferred") and e.instruction_kind == "straight"
    assert pmap.frame.heading_source == "relative" and pmap.frame.heading_zero_abs_deg is None

    roles = {r.role for r in pmap.references}
    assert {"at", "outgoing"} <= roles
    out = [r for r in pmap.references if r.role == "outgoing"]
    assert all(r.node == "n1" and r.edge == "e1" and r.toward == "n2" for r in out) and e.reference_views
    assert all((maps_root / "video-hall" / r.image).is_file() for r in pmap.references)
    assert pmap.embedding and pmap.embedding.dim == 64
    vectors = np.load(maps_root / "video-hall" / pmap.embedding.file)
    assert vectors.shape == (len(pmap.references), 64)
    assert sorted(r.embedding_row for r in pmap.references) == list(range(len(pmap.references)))


# --- status, relations and headings from finish_map --------------------------------------------------------
def lshape(heading_east_then_south: bool = True):
    """Walk east 10 m (0-10 s) to c02, then south 10 m to c03, with compass headings."""
    def view(t, x, y, h):
        return View(rec_id="r1", t_ms=t * 1000, sharpness=1,
                    motion=Motion(walked_m=t, heading_deg=h, x_m=x, y_m=y, pitch_deg=0, moving=True))
    pts = [[t * 1000, float(min(t, 10)), float(-max(0, t - 10)), 90.0 if t < 10 else 180.0] for t in range(21)]
    cands = [Candidate(id="c01", image="frames/c01.jpg", reason=[], views=[view(0, 0, 0, 90)]),
             Candidate(id="c02", image="frames/c02.jpg", reason=[], views=[view(10, 10, 0, 90)]),
             Candidate(id="c03", image="frames/c03.jpg", reason=[], views=[view(20, 10, -10, 180)])]
    rec = RecordingInfo(rec_id="r1", created_at_utc=datetime(2026, 10, 10, tzinfo=timezone.utc), duration_s=20,
                        notes="", step_length_m=0.7, heading="compass", fov_deg=50, walked_m=20, frames_sampled=20,
                        frames_usable=20, marks=[], voice=[], timeline=[])
    cset = CandidateSet(place="T", map_id="t", recordings=[rec], candidates=cands, params={}, embed_model="m")
    node = lambda i, rel=(): {"id": f"n{i}", "name": f"N{i}", "kind": "room", "description": "", "landmarks": [],  # noqa: E731
                              "hazards": [], "visual_cues": [], "floor": None, "candidates": [f"c0{i}"],
                              "relations": list(rel), "confidence": "high"}
    edge = lambda i, a, b, basis="walked": {"id": i, "source": a, "target": b, "kind": "walk", "bidirectional": True,  # noqa: E731
                                            "basis": basis, "steps": [], "instruction": f"go {i}",
                                            "reverse_instruction": f"back {i}", "landmarks": [], "hazards": [],
                                            "visual_cues": [], "candidates": [], "confidence": "medium"}
    draft = MapDraft.model_validate({
        "place": "T", "language": "en", "summary": "s", "notes": [],
        "nodes": [node(1), node(2, [{"subject": "Kitchen", "candidate": "c02", "side": "in_photo", "x": 0.0,
                                     "evidence": "observed"}]), node(3)],
        "edges": [edge("e1", "n1", "n2"), edge("e2", "n3", "n2"), edge("e3", "n1", "n3", "inferred")]})
    return draft, cset, {"r1": pts}


def test_status_headings_and_relations():
    draft, cset, tracks = lshape()
    m = finish_map(draft, cset, BUILD, "builds/b", tracks)
    e1, e2, e3 = m.edges
    assert (e1.status, e1.reverse_status) == ("observed", "inferred")
    assert e1.departure_heading_deg == 90 and e1.arrival_heading_deg == 90
    # e2 is n3 -> n2, but the walk went n2 -> n3: only the way back was walked
    assert (e2.status, e2.reverse_status) == ("inferred", "observed")
    assert e2.departure_heading_deg == 0 and e2.arrival_heading_deg == 0  # the southward walk, turned around
    assert (e3.status, e3.reverse_status) == ("inferred", "inferred") and e3.walked is False
    assert "the way back was not walked" in e1.review_flags
    (kitchen,) = m.nodes[1].relations
    # left edge of a 50 degree photo taken facing east: 25 degrees left of east; arrived facing east: ahead
    assert kitchen.bearing_deg == -25 and kitchen.heading_deg == 65 and kitchen.direction == "ahead"
    assert kitchen.arrival_heading_deg == 90 and kitchen.viewpoint_image == "builds/b/frames/c02.jpg"
    assert m.frame.heading_source == "compass" and m.frame.heading_zero_abs_deg == 0


def test_review_overlay_verifies_edits_and_hides_without_deleting():
    draft, cset, tracks = lshape()
    raw = finish_map(draft, cset, BUILD, "builds/b", tracks)
    review = Review(nodes={"n2": NodeReview(approve=True, name="Hall"), "n3": NodeReview(hide=True),
                           "n9": NodeReview(approve=True)},
                    edges={"e1": EdgeReview(approve=True, approve_reverse=True, instruction="Walk 10 m east.")},
                    relations={"n2:kitchen": RelationReview(approve=True)})
    m = apply_review(raw, review)
    assert [n.id for n in m.nodes] == ["n1", "n2"] and m.nodes[1].name == "Hall" and m.nodes[1].status == "verified"
    assert [e.id for e in m.edges] == ["e1"]  # e2 and e3 touch the hidden n3
    assert (m.edges[0].status, m.edges[0].reverse_status) == ("verified", "verified")
    assert m.edges[0].instruction == "Walk 10 m east." and m.nodes[1].relations[0].status == "verified"
    assert m.hidden == ["e2", "e3", "n3"] and m.reviewed_at_utc is not None
    assert any("n9" in w for w in m.warnings)
    assert len(raw.nodes) == 3 and raw.nodes[1].name == "N2"  # the build's own map is untouched

    # The review file round-trips through the editable template; untouched items are not stored as review.
    t = json.loads(json.dumps(template(raw, review), default=str))
    t["edges"]["e2"]["approve_reverse"] = True
    back = from_template(t)
    assert back.edges["e2"].approve_reverse and "e3" not in back.edges and back.nodes["n3"].hide


def test_routes_use_trusted_edges_in_the_walked_direction():
    draft, cset, tracks = lshape()
    m = finish_map(draft, cset, BUILD, "builds/b", tracks)
    assert plan_route(m, "n1", "n3", "verified") is None  # nothing verified yet
    route = plan_route(m, "n1", "n3", "observed")  # e1 forward, e2 walked back (n2 -> n3)
    assert [(h.edge, h.forward, h.source, h.target) for h in route] == [("e1", True, "n1", "n2"),
                                                                        ("e2", False, "n2", "n3")]
    assert route[1].bearing_deg == 180 and route[1].instruction == "back e2"
    assert plan_route(m, "n3", "n1", "observed") is None  # n3 -> n2 was never walked
    assert [h.edge for h in plan_route(m, "n3", "n1", "any")] == ["e3"]  # the inferred shortcut
    m = apply_review(m, Review(edges={"e1": EdgeReview(approve=True), "e2": EdgeReview(approve_reverse=True)}))
    assert [h.edge for h in plan_route(m, "n1", "n3")] == ["e1", "e2"]
    assert plan_route(m, "n1", "n1") == []


# --- a second walk in the opposite direction ----------------------------------------------------------------
def two_walks():
    """Walk 1 goes A -> B -> C eastwards; walk 2 starts at C and goes back westwards (its own frame starts at C).
    The photos of the two walks don't fold (seen from the other side): the LLM joins them in nodes."""
    def view(rec, t, x, h):
        return View(rec_id=rec, t_ms=t * 1000, sharpness=1,
                    motion=Motion(walked_m=abs(x), heading_deg=h, x_m=x, y_m=0, pitch_deg=0, moving=True))
    cands = [Candidate(id="c01", image="f", reason=[], views=[view("r1", 0, 0, 90)]),
             Candidate(id="c02", image="f", reason=[], views=[view("r1", 10, 10, 90)]),
             Candidate(id="c03", image="f", reason=[], views=[view("r1", 20, 20, 90)]),
             Candidate(id="c04", image="f", reason=[], views=[view("r2", 0, 0, 270)]),
             Candidate(id="c05", image="f", reason=[], views=[view("r2", 10, -10, 270)]),
             Candidate(id="c06", image="f", reason=[], views=[view("r2", 20, -20, 270)])]
    tracks = {"r1": [[t * 1000, float(t), 0.0, 90.0] for t in range(21)],
              "r2": [[t * 1000, -float(t), 0.0, 270.0] for t in range(21)]}
    rec = lambda r: RecordingInfo(rec_id=r, created_at_utc=datetime(2026, 10, 10, tzinfo=timezone.utc),  # noqa: E731
                                  duration_s=20, notes="", step_length_m=0.7, heading="compass", walked_m=20,
                                  frames_sampled=20, frames_usable=20, marks=[], voice=[], timeline=[])
    cset = CandidateSet(place="T", map_id="t", recordings=[rec("r1"), rec("r2")], candidates=cands, params={},
                        embed_model="m")
    return cset, tracks


def walk_draft(nodes: list[tuple[str, list[str]]], edges: list[tuple[str, str, str]]) -> MapDraft:
    node = lambda i, c: {"id": i, "name": f"Spot {i}", "kind": "room", "description": "", "landmarks": ["wall"],  # noqa: E731
                         "hazards": [], "visual_cues": [], "floor": None, "candidates": c, "relations": [],
                         "confidence": "high"}
    edge = lambda i, a, b: {"id": i, "source": a, "target": b, "kind": "walk", "bidirectional": True,  # noqa: E731
                            "basis": "walked", "steps": [{"action": "straight", "distance_m": 10, "along": None,
                                                          "until": f"until {b}"}],
                            "instruction": f"{a} to {b}", "reverse_instruction": f"{b} to {a}", "landmarks": [],
                            "hazards": [], "visual_cues": [], "candidates": [], "confidence": "medium"}
    return MapDraft.model_validate({"place": "T", "language": "en", "summary": "s", "notes": [],
                                    "nodes": [node(i, c) for i, c in nodes],
                                    "edges": [edge(i, a, b) for i, a, b in edges]})


def test_a_walk_in_the_opposite_direction_keeps_the_map_and_confirms_it():
    from nav.mapping.build import align_recordings
    from nav.mapping.continuity import previous_context, stabilize

    cset, tracks = two_walks()
    first_set = cset.model_copy(update={"recordings": cset.recordings[:1], "candidates": cset.candidates[:3]})
    first = finish_map(walk_draft([("n1", ["c01"]), ("n2", ["c02"]), ("n3", ["c03"])],
                                  [("e1", "n1", "n2"), ("e2", "n2", "n3")]), first_set, BUILD, "b0",
                       {"r1": tracks["r1"]})
    assert [(e.status, e.reverse_status) for e in first.edges] == [("observed", "inferred")] * 2
    review = Review(edges={"e2": EdgeReview(approve=True)})  # someone checked B -> C

    # The rebuild: the LLM joins walk 2's frames to the nodes, with scratch ids, and writes B-C the other way round.
    d = walk_draft([("x", ["c03", "c04"]), ("y", ["c02", "c05"]), ("z", ["c01", "c06"])],
                   [("k", "x", "y"), ("j", "y", "z")])
    d, retired = stabilize(d, cset, previous_context(first, first_set), "b1")
    assert retired == [] and [n.id for n in d.nodes] == ["n3", "n2", "n1"]
    by = {e.id: e for e in d.edges}
    assert (by["e2"].source, by["e2"].target) == ("n2", "n3")  # turned back to the previous orientation
    assert by["e2"].instruction == "y to x" and by["e2"].reverse_instruction == "x to y"
    assert by["e2"].steps[0].action == "turn_around" and (by["e1"].source, by["e1"].target) == ("n1", "n2")

    m = finish_map(d, cset, BUILD, "b1", tracks)
    pos = {n.id: (n.position.x_m, n.position.y_m) for n in m.nodes}
    assert pos == {"n1": (0, 0), "n2": (10, 0), "n3": (20, 0)}  # walk 2 shifted onto walk 1 through the nodes
    e1, e2 = sorted(m.edges, key=lambda e: e.id)
    assert (e2.walked_forward, e2.walked_back) == (1, 1) and (e2.status, e2.reverse_status) == ("observed",) * 2
    assert e2.measured_distance_m == 10 and e2.departure_heading_deg == 90 and e2.arrival_heading_deg == 90
    assert "the way back was not walked" not in e2.review_flags
    published = apply_review(m, review)
    assert next(e for e in published.edges if e.id == "e2").status == "verified"  # the review kept its direction

    # Before the LLM: walks that share a folded frame get positions in one frame.
    offsets, _ = align_recordings({"c03": [*cset.candidates[2].views, *cset.candidates[3].views]})
    assert offsets["r2"].tolist() == [20, 0] and offsets["r1"].tolist() == [0, 0]


# --- dead reckoning along a route ----------------------------------------------------------------------------
def test_step_detector_counts_a_walk():
    t, acc = walking_accel(seconds=20, cadence=1.8)
    steps = detect_steps(t, acc)
    assert abs(len(steps) - 18 * 1.8) <= 3
    assert len(detect_steps(*walking_accel(amp=0.0))) == 0  # standing still
    det = StepDetector()
    assert sum(det.update(float(ti), tuple(a)) for ti, a in zip(t, acc)) == len(steps)  # the same, causally


def test_graph_tracker_progress_arrival_and_step_length():
    route = [Hop("e1", "a", "b", True, 14.0, 90.0, "go"), Hop("e2", "b", "c", True, 7.0, 180.0, "go")]
    tr = GraphTracker(route, step_length=0.7)
    for _ in range(10):
        st = tr.on_step(90.0)
    assert abs(st.progress_m - 7.0) < 1e-6 and st.arrival_probability < 0.05
    for _ in range(4):
        tr.on_step(180.0)  # walking sideways: no progress
    assert abs(tr.state().progress_m - 7.0) < 1e-6 and tr.state().off_course_deg > 50
    for _ in range(8):
        st = tr.on_step(90.0)
    assert st.remaining_m < 2 and st.arrival_probability > 0.5
    with pytest.raises(ValueError):
        tr.confirm_node("c")
    tr.confirm_node("b")  # walked 12.6 m by its count for a 14 m edge: the user's steps are longer
    assert tr.hop_index == 1 and tr.progress_m == 0 and tr.step_length > 0.7


# --- schema for the app, old maps -------------------------------------------------------------------------------
def test_committed_map_schema_is_current():
    assert (ROOT / "docs" / "map.schema.json").read_text() == place_map_json_schema(), \
        "run: uv run nav map schema --out docs/map.schema.json"


def test_maps_built_before_the_navigation_fields_still_load():
    draft, cset, tracks = lshape()
    stored = json.loads(finish_map(draft, cset, BUILD, "b", tracks).model_dump_json())
    for k in ("frame", "references", "embedding", "reviewed_at_utc", "hidden"):
        stored.pop(k)
    for n in stored["nodes"]:
        for k in ("status", "review_flags", "references", "relations"):
            n.pop(k)
    for e in stored["edges"]:
        for k in ("status", "reverse_status", "review_flags", "instruction_kind", "departure_heading_deg",
                  "arrival_heading_deg", "reference_views"):
            e.pop(k)
    m = PlaceMap.model_validate(stored)
    assert m.nodes[0].status == "observed" and m.edges[0].reverse_status is None and m.references == []


# --- API: review, refinish, schema ------------------------------------------------------------------------------
def test_review_and_refinish_api(tmp_path, monkeypatch):
    fake_models(monkeypatch)
    draft, cset, tracks = lshape()
    build = tmp_path / "maps" / "t" / "builds" / "20261010T000000Z"
    (build / "job").mkdir(parents=True)
    (build / "candidates.json").write_text(cset.model_dump_json())
    (build / "tracks.json").write_text(json.dumps(tracks))
    (build / "job" / "claude.json").write_text(json.dumps({"subtype": "success", "is_error": False,
                                                           "structured_output": draft.model_dump()}))
    from nav.mapping.build import refinish_build

    refinish_build(build, tmp_path / "recordings", tmp_path / "cache")
    app = create_app(RecordingStore(tmp_path / "recordings"), MapStore(tmp_path / "maps"), api_token="s3cret",
                     cache_dir=tmp_path / "cache", web_dir=None)
    auth = {"Authorization": "Bearer s3cret"}
    with TestClient(app) as c:
        assert c.get(f"{API}/maps/t/review").status_code == 401
        assert c.get(f"{API}/schema/map", headers=auth).json()["title"] == "PlaceMap"
        assert c.get(f"{API}/maps/t/review", headers=auth).json()["nodes"] == {}
        r = c.put(f"{API}/maps/t/review", headers={**auth, "X-Nav-User": "tester"},
                  json={"nodes": {"n3": {"hide": True}}, "edges": {"e1": {"approve": True}}})
        assert r.status_code == 200 and r.json()["hidden"] == ["e2", "e3", "n3"]
        assert c.get(f"{API}/maps/t", headers=auth).json()["edges"][0]["status"] == "verified"
        assert c.get(f"{API}/maps/t/review", headers=auth).json()["updated_by"] == "tester"
        # the build's own map keeps everything
        raw = c.get(f"{API}/maps/t/files/builds/20261010T000000Z/map.json", headers=auth).json()
        assert len(raw["nodes"]) == 3
        assert c.put(f"{API}/maps/nope/review", headers=auth, json={}).status_code == 404

        r = c.post(f"{API}/maps/t/refinish", headers=auth)
        assert r.status_code == 202
        for _ in range(100):
            st = c.get(r.json()["status_url"], headers=auth).json()
            if st["status"] != "running":
                break
            import time
            time.sleep(0.05)
        assert st["status"] == "done", st
        published = c.get(f"{API}/maps/t", headers=auth).json()
        assert published["hidden"] == ["e2", "e3", "n3"]  # the review survives a refinish
        assert any("recording r1 not found" in w for w in published["warnings"])
