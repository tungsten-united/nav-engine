import json
import os
import stat
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from nav.mapping.build import finish_map, run_llm_on_build
from nav.mapping.motion import RATE_HZ, angle_diff, load_motion
from nav.mapping.select import Frame, SelectParams, group_frames, mark_usable, select_in_recording
from nav.mapping.store import MapStore
from nav.recorder.app import create_app
from nav.recorder.store import RecordingStore
from nav.schemas.map import (
    BuildInfo,
    Candidate,
    CandidateSet,
    LlmRun,
    MapDraft,
    Motion,
    View,
    draft_json_schema,
    slugify,
)

API = "/api/v1"


# --- schema ------------------------------------------------------------------------------------------------
def test_draft_schema_fits_structured_outputs():
    """Anthropic structured outputs: closed objects, every property required, no numeric/length constraints."""
    schema = draft_json_schema()
    banned = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength",
              "minItems", "maxItems", "pattern"}

    def walk(node, path="$"):
        if isinstance(node, dict):
            assert not banned & node.keys(), f"{path}: {banned & node.keys()}"
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, path
                assert set(node.get("required", [])) == set(node["properties"]), f"{path}: optional properties"
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(schema)
    # no recursion: no definition refers to itself, directly or through another
    defs = schema.get("$defs", {})
    refs = {k: {r.split("/")[-1] for r in json.dumps(v).split('"$ref": "')[1:] for r in [r.split('"')[0]]}
            for k, v in defs.items()}
    for k in defs:
        seen, todo = set(), set(refs[k])
        while todo:
            r = todo.pop()
            assert r != k, f"{k} is recursive"
            if r not in seen:
                seen.add(r)
                todo |= refs.get(r, set())


def test_slugify():
    assert slugify("  Office,  3rd floor ") == "office-3rd-floor"
    assert slugify("Itnig") == "itnig"
    assert slugify("Café Ñandú") == "cafe-nandu"
    assert slugify("!!!") == "place"


# --- motion ------------------------------------------------------------------------------------------------
def write_walk(d, *, seconds=20.0, cadence=1.8, turn_at=10.0, turn_deg=90.0, chrome=True, amp=0.35):
    """Phone upright (gravity on +y), walking with small vertical bounces, one right turn about vertical."""
    t = np.arange(0, seconds * 1000, 1000 / 60)
    ts = t / 1000
    walking = (ts > 1) & (ts < seconds - 1)
    bounce = amp * np.sin(2 * np.pi * cadence * ts) * walking
    # yaw rate about vertical (device y): negative = clockwise = right turn
    yaw = np.where((ts >= turn_at) & (ts < turn_at + 2), -turn_deg / 2, 0.0)
    noise = np.random.default_rng(0).normal(0, 0.02, len(t))
    imu = {"t_ms": t, "ax": noise, "ay": 9.81 + bounce, "az": noise,
           # recorder stores gx/gy/gz = beta/gamma/alpha; Chrome puts alpha/beta/gamma around x/y/z
           "gx": yaw if chrome else noise, "gy": noise if chrome else yaw, "gz": noise}
    pq.write_table(pa.table({k: pa.array(v, pa.float64()) for k, v in imu.items()}), d / "imu.parquet")
    heading = 90 + np.clip((ts - turn_at) / 2, 0, 1) * turn_deg
    pq.write_table(pa.table({"t_ms": t, "heading_deg": heading}), d / "orientation.parquet")
    return t


@pytest.mark.parametrize("chrome", [True, False])
def test_motion_steps_heading_and_gyro_layout(tmp_path, chrome):
    write_walk(tmp_path, chrome=chrome)
    tr = load_motion(tmp_path, step_length_m=0.7, duration_ms=20000)
    assert tr.gyro_layout == ("chrome" if chrome else "spec")
    assert abs(len(tr.step_t_ms) - 18 * 1.8) <= 4  # ~1.8 steps/s while walking
    assert abs(tr.walked_m[-1] - len(tr.step_t_ms) * 0.7) < 1e-6
    assert abs(angle_diff(tr.at(8000)["heading_deg"], tr.at(14000)["heading_deg"]) - 90) < 10
    assert tr.at(500)["moving"] is False and tr.at(5000)["moving"] is True
    # heading starts east, then south: dead reckoning goes +x, then -y
    p = tr.at(19000)
    assert p["x_m"] > 3 and p["y_m"] < -3


def test_motion_without_imu(tmp_path):
    tr = load_motion(tmp_path, 0.7, 5000)
    assert tr.walked_m[-1] == 0 and tr.at(1000)["heading_deg"] is None


def test_angle_diff():
    assert angle_diff(350, 10) == 20 and angle_diff(10, 350) == -20 and abs(angle_diff(0, 180)) == 180


# --- selection ---------------------------------------------------------------------------------------------
def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def frame(rec, t_s, emb, walked, heading=0.0, sharp=100.0):
    return Frame(rec_id=rec, t_ms=t_s * 1000, sharpness=sharp, brightness=120, image_path="x.jpg", emb=unit(emb),
                 motion={"walked_m": walked, "heading_deg": heading, "x_m": walked, "y_m": 0.0, "pitch_deg": 0.0,
                         "moving": True})


def test_select_keeps_new_views_anchors_and_folds_revisits():
    p = SelectParams()
    a, b, c = np.eye(3)
    frames = [frame("r", 0, a, 0), frame("r", 1, a + 0.05 * b, 0.7),       # same view while starting
              frame("r", 2, a, 1.4, sharp=5),                               # blurry: unusable
              frame("r", 3, b, 3.0),                                        # new view after 3 m: novel
              frame("r", 4, b + 0.02 * a, 3.7),                             # voice note here
              frame("r", 5, b, 15.0),                                       # same view, 12 m later: distance
              frame("r", 60, a + 0.01 * c, 40.0),                           # back at the start: revisit
              frame("r", 61, c, 41.0)]
    mark_usable(frames, p)
    assert [f.usable for f in frames].count(False) == 1
    voice = [{"t_ms": 4000, "t_end_ms": 5000, "text": "here is the kitchen"}]
    kept = select_in_recording(frames, None, [], voice, p)
    assert [f.t_ms / 1000 for f in kept] == [0, 3, 4, 5, 60, 61]
    assert kept[1].reasons == ["novel"] and kept[2].voice == ["here is the kitchen"] and "distance" in kept[3].reasons
    groups, hints = group_frames(kept, p)
    # 3 s (novel) is a near-duplicate of the voice frame 1 s later and unanchored: dropped; 60 s folds into 0 s
    times = [[f.t_ms / 1000 for f in g.frames] for g in groups]
    assert [0, 60] in times or [0.0, 60.0] in times
    assert not any(3 in t for t in times)
    assert groups[0].rep.t_ms == 0  # start is anchored, so it represents the group


# --- finish ------------------------------------------------------------------------------------------------
def candidate_set() -> CandidateSet:
    def view(t, walked, x):
        return View(rec_id="r1", t_ms=t, sharpness=1, motion=Motion(walked_m=walked, heading_deg=90, x_m=x, y_m=0,
                                                                      pitch_deg=0, moving=True))
    cands = [Candidate(id="c01", image="frames/c01.jpg", reason=["start"], views=[view(0, 0, 0), view(90_000, 60, 1)]),
             Candidate(id="c02", image="frames/c02.jpg", reason=["novel"], views=[view(10_000, 8, 8)]),
             Candidate(id="c03", image="frames/c03.jpg", reason=["end"], views=[view(20_000, 15, 15)])]
    return CandidateSet(place="Test", map_id="test", recordings=[], candidates=cands, params={}, embed_model="m")


def draft(**over) -> MapDraft:
    node = lambda i, c: {"id": i, "name": i, "kind": "room", "description": "", "landmarks": [], "hazards": [],
                         "visual_cues": [], "floor": None, "candidates": c, "confidence": "high"}
    edge = lambda i, a, b: {"id": i, "source": a, "target": b, "kind": "walk", "bidirectional": True, "distance_m": 8,
                            "instruction": "go", "reverse_instruction": "back", "landmarks": [], "hazards": [], "visual_cues": [],
                            "candidates": [], "confidence": "medium"}
    d = {"place": "Test", "language": "en", "summary": "s", "notes": [],
         "nodes": [node("n1", ["c01"]), node("n2", ["c02", "c99"])],
         "edges": [edge("e1", "n1", "n2"), edge("e2", "n2", "n7"), edge("e1", "n2", "n1")]}
    return MapDraft.model_validate({**d, **over})


BUILD = BuildInfo(build_id="20261010T000000Z", built_at_utc=datetime(2026, 10, 10, tzinfo=timezone.utc),
                  pipeline_version="t", params={}, embed_model="m", transcribe_model=None,
                  llm=LlmRun(backend="claude_code", model="x"), candidates=3)


def test_finish_checks_ids_and_adds_evidence_positions_distances():
    tracks = {"r1": [[t * 1000, t * 0.8, 0.0] for t in range(0, 100)]}
    m = finish_map(draft(), candidate_set(), BUILD, "builds/20261010T000000Z", tracks)
    assert [e.id for e in m.edges] == ["e1"]
    assert any("c99" in w for w in m.warnings) and any("e2" in w for w in m.warnings)
    assert any("duplicate edge id e1" in w for w in m.warnings) and any("c03" in w for w in m.warnings)
    n1, n2 = m.nodes
    assert n2.candidates == ["c02"]
    assert [(e.candidate, e.t_ms) for e in n1.evidence] == [("c01", 0), ("c01", 90_000)]
    assert n1.evidence[0].image == "builds/20261010T000000Z/frames/c01.jpg"
    assert n1.position.x_m == 0  # main visit (the first, both have one view), not the mean with the revisit
    assert m.edges[0].measured_distance_m == 8  # closest visits: c01 at 0 m, c02 at 8 m
    path = m.edges[0].path
    assert path[0].x_m == 0 and path[-1].x_m == 8 and len(path) > 5
    assert m.tracks[0].rec_id == "r1"


# --- Claude Code job ---------------------------------------------------------------------------------------
def test_llm_job_with_fake_claude(tmp_path, monkeypatch):
    build = tmp_path / "maps" / "test" / "builds" / "20261010T000000Z"
    build.mkdir(parents=True)
    (build / "candidates.json").write_text(candidate_set().model_dump_json())
    result = {"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.5, "session_id": "s",
              "modelUsage": {"claude-x": {"costUSD": 0.5}}, "structured_output": draft(edges=[]).model_dump()}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "claude"
    fake.write_text("#!/bin/sh\ncat > \"$PWD/job/stdin.txt\"\necho \"$@\" > \"$PWD/job/args.txt\"\n"
                    f"cat <<'EOF'\n{json.dumps(result)}\nEOF\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    res = run_llm_on_build(build, llm_model="opus", language="es", budget_usd=1)
    assert res.map.build.llm.model == "claude-x" and res.map.build.llm.cost_usd == 0.5
    args = (build / "job" / "args.txt").read_text()
    assert "--json-schema" in args and "--tools Read,Glob" in args and "--safe-mode" in args
    assert "--effort medium" in args and res.map.build.llm.effort == "medium"
    assert "Spanish" in (build / "job" / "stdin.txt").read_text()
    assert json.loads((build / "job" / "schema.json").read_text()) == draft_json_schema()
    published = json.loads((tmp_path / "maps" / "test" / "map.json").read_text())
    assert published["build"]["build_id"] == "20261010T000000Z" and len(published["nodes"]) == 2

    # API + viewer over that map
    store = RecordingStore(tmp_path / "recordings")
    with TestClient(create_app(store, MapStore(tmp_path / "maps"))) as c:
        assert [m["map_id"] for m in c.get(f"{API}/maps").json()] == ["test"]
        assert c.get(f"{API}/maps/test").json()["place"] == "Test"
        assert c.get(f"{API}/maps/test/builds").json() == ["20261010T000000Z"]
        assert c.get(f"{API}/maps/test/files/builds/20261010T000000Z/job/schema.json").status_code == 200
        assert c.get(f"{API}/maps/test/files/../../recordings").status_code == 404
        assert c.get(f"{API}/maps/Test%2F..").status_code == 404
        assert c.get(f"{API}/maps/nope").status_code == 404
        page = c.get("/viz/")
        assert page.status_code == 200 and 'src="viz.js?v=' in page.text
        assert c.get("/viz", follow_redirects=False).headers["location"] == "/viz/"


def test_llm_job_failure_is_reported(tmp_path, monkeypatch):
    from nav.mapping.llm import LlmError

    build = tmp_path / "b" / "builds" / "20261010T000000Z"
    build.mkdir(parents=True)
    (build / "candidates.json").write_text(candidate_set().model_dump_json())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "claude"
    fake.write_text('#!/bin/sh\necho \'{"subtype": "error_max_budget_usd", "is_error": true, "result": "budget"}\'\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(LlmError, match="error_max_budget_usd"):
        run_llm_on_build(build, llm_model="opus", language="en", budget_usd=0.01)


def test_visual_words_in_spoken_text_are_flagged():
    from nav.mapping.build import visual_word_warnings

    d = draft()
    d.nodes[0].description = "A white counter under a big yellow sign; you will see the screen."
    d.nodes[0].visual_cues = ["red fire cabinet"]  # visual detail belongs here: not flagged
    d.nodes[1].description = "A humming fridge against the wall, the floor turns to carpet."
    d.edges[0].instruction = "Walk 5 m with the wall on your left until the floor turns to carpet."
    warns = visual_word_warnings(d)
    assert warns == ["node n1 uses visual words in spoken text: screen, see, sign, white, yellow"]
    m = finish_map(d, candidate_set(), BUILD, "builds/x")
    assert warns[0] in m.warnings
    assert m.nodes[0].visual_cues == ["red fire cabinet"]


def test_old_maps_and_answers_without_visual_cues_still_load(tmp_path):
    from nav.mapping.llm import parse_claude_result
    from nav.schemas.map import PlaceMap

    old = draft().model_dump()
    for x in old["nodes"] + old["edges"]:
        del x["visual_cues"]
    path = tmp_path / "claude.json"
    path.write_text(json.dumps({"subtype": "success", "is_error": False, "structured_output": old}))
    d, _ = parse_claude_result(path)
    assert d.nodes[0].visual_cues == []
    stored = finish_map(d, candidate_set(), BUILD, "b").model_dump()
    for x in stored["nodes"] + stored["edges"]:
        del x["visual_cues"]
    assert PlaceMap.model_validate(stored).nodes[0].visual_cues == []


def test_revisits_are_loop_closed_without_jumps():
    """A node passed twice, far apart by dead reckoning: placed at its first pass, the track bent onto it at the
    second pass, the edge walked then drawn without a jump, and a warning about the gap."""
    def view(t, x):
        return View(rec_id="r1", t_ms=t * 1000, sharpness=1, motion=Motion(walked_m=x, heading_deg=90, x_m=x, y_m=0,
                                                                            pitch_deg=0, moving=True))
    cands = [Candidate(id="c01", image="frames/c01.jpg", reason=[], views=[view(0, 0), view(100, 80)]),
             Candidate(id="c02", image="frames/c02.jpg", reason=[], views=[view(50, 40)]),
             Candidate(id="c03", image="frames/c03.jpg", reason=[], views=[view(90, 70)])]
    cset = CandidateSet(place="T", map_id="t", recordings=[], candidates=cands, params={}, embed_model="m")
    d = draft()
    d.nodes = d.nodes[:2]
    d.nodes[0].candidates, d.nodes[1].candidates = ["c01"], ["c02", "c03"]
    d.edges = [d.edges[0].model_copy(update={"id": "e1", "source": "n2", "target": "n1"})]
    tracks = {"r1": [[t * 1000, t * 0.8, 0.0] for t in range(0, 101)]}
    m = finish_map(d, cset, BUILD, "b", tracks)
    n1, n2 = m.nodes
    assert n1.position.x_m == 0 and n2.position.x_m == 40  # first passes
    path = m.edges[0].path  # walked n2 (90 s) -> n1 (100 s): closest passes
    assert m.edges[0].walked_at_ms == 90_000
    gaps = [abs(b.x_m - a.x_m) for a, b in zip(path, path[1:])]
    assert max(gaps) < 15, gaps  # no jump from the raw track at 80 m back to the node at 0 m
    assert path[0].x_m == 40 and path[-1].x_m == 0
    track = dict((t, x) for t, x, _ in m.tracks[0].points)
    assert abs(track[100_000] - 0) < 1 and abs(track[50_000] - 40) < 1
    assert any("node n1" in w and "80 m" in w for w in m.warnings)


def lshape_set():
    """Walk east 10 m (0-10 s), then south 10 m (10-20 s): a right turn at c02."""
    def view(t, x, y, w):
        return View(rec_id="r1", t_ms=t * 1000, sharpness=1, motion=Motion(walked_m=w, heading_deg=90, x_m=x, y_m=y,
                                                                            pitch_deg=0, moving=True))
    cands = [Candidate(id="c01", image="frames/c01.jpg", reason=[], views=[view(0, 0, 0, 0)]),
             Candidate(id="c02", image="frames/c02.jpg", reason=[], views=[view(10, 10, 0, 10)]),
             Candidate(id="c03", image="frames/c03.jpg", reason=[], views=[view(20, 10, -10, 20)])]
    tracks = {"r1": [[t * 1000, float(min(t, 10)), float(-max(0, t - 10))] for t in range(0, 21)]}
    cset = CandidateSet(place="T", map_id="t", recordings=[], candidates=cands, params={}, embed_model="m")
    d = draft()
    n = d.nodes[0]
    d.nodes = [n.model_copy(update={"id": f"n{i}", "candidates": [f"c0{i}"]}) for i in (1, 2, 3)]
    e = d.edges[0]
    d.edges = [e.model_copy(update={"id": "e1", "source": "n1", "target": "n2"}),
               e.model_copy(update={"id": "e2", "source": "n2", "target": "n3"}),
               e.model_copy(update={"id": "e3", "source": "n1", "target": "n3"})]
    return d, cset, tracks


def test_turns_and_inferred_edges():
    d, cset, tracks = lshape_set()
    m = finish_map(d, cset, BUILD, "b", tracks)
    e1, e2, e3 = m.edges
    assert e1.walked and e2.walked and e2.turn_deg == 90  # east, then south: a right turn
    assert e1.turn_deg is None  # nothing walked before it
    assert e3.walked is False and e3.measured_distance_m is None and len(e3.path) == 2  # passed n2 on the way


class FakeTypeSafe:
    """Stands in for typesafe_sdk.AsyncTypeSafeClient: judges by keywords in the state."""

    def __init__(self, model=None):
        self.model = model

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def system_one(self, state, questions):
        from types import SimpleNamespace as NS
        spoken = json.dumps(state["spoken"]).lower()
        answers = {"visual": NS(noul=0.9 if "yellow" in spoken else 0.05),
                   "usable": NS(score=0.5 if "somewhere" in spoken else 2.6)}
        if "turn" in questions:
            answers["turn"] = NS(noul=0.1 if "left" in spoken else 0.95)
        return NS(answers=answers, usage=NS(input_tokens=100))


def test_quality_check_flags_items(monkeypatch):
    import typesafe_sdk

    from nav.mapping.quality import check_map

    d, cset, tracks = lshape_set()
    d.nodes[0].description = "Under the big yellow letters."
    d.nodes[1].description = "Somewhere in the hall."
    d.edges[1].instruction = "Turn left and walk 10 m until the wall."  # the walk turned right
    m = finish_map(d, cset, BUILD, "b", tracks)

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert check_map(m).quality.skipped == "TYPESAFE_API_KEY is not set"

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", FakeTypeSafe)
    m = check_map(m)
    q = m.quality
    assert q.skipped is None and q.input_tokens == 600 and q.flagged == 3
    assert m.nodes[0].quality.flags == ["relies on sight"]
    assert m.nodes[1].quality.flags == ["no concrete cue to tell you have arrived"]
    assert m.edges[1].quality.turn_agrees_p == 0.1
    assert m.edges[1].quality.flags == ["the walk turned right here but the instruction doesn't say so "
                                        "(one of them is wrong)"]
    assert m.edges[0].quality.turn_agrees_p is None  # no measured turn to compare with
    assert "jev: node n1: relies on sight" in m.warnings
    assert sum(w.startswith("jev: ") for w in check_map(m).warnings) == 3  # rerunning replaces, not appends
