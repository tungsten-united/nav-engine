"""Stable ids from one build to the next (nav/mapping/continuity.py match_previous)."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

from nav.mapping.continuity import match_previous

FIXTURE = Path(__file__).parent / "fixtures" / "itnig-metric" / "continuity.json"


def views(c: dict) -> NS:
    return NS(id=c["id"], views=[NS(rec_id=r, t_ms=t) for r, t in c["views"]])


def test_every_itnig_place_keeps_its_id_where_shared_frames_were_tied():
    """Itnig's first metric-1 build (the fixture's `about`). Its places pass kept all 13 names of the build before,
    but shared frames are few and tied between neighbours: the judges' table shared 3 with the stone wall tables and
    3 with the kitchen, and took the stone wall tables' n9; the hackathon tables took the drinks area's n2. The
    review's merges, keyed by those ids, then moved the judges' table into the centre tables, 6 m away."""
    data = json.loads(FIXTURE.read_text())
    cset = NS(candidates=[views(c) for c in data["candidates"]])
    nodes = [NS(**n) for n in data["places"]]
    old = {n["id"]: n["name"] for n in data["previous"]["nodes"]}
    got = match_previous(nodes, cset, data["previous"])
    assert {n.name: old[got[n.id]] for n in nodes} == {n.name: n.name for n in nodes}
    assert {n.name: got[n.id] for n in nodes if n.name in ("Stone wall tables", "Judges' table", "Drinks area")} == \
        {"Stone wall tables": "n9", "Judges' table": "n13", "Drinks area": "n2"}


def test_a_name_twice_or_a_new_name_goes_by_shared_frames():
    prev = {"nodes": [{"id": "n1", "name": "Door", "evidence": [["r", 0.0]]},
                      {"id": "n2", "name": "Door", "evidence": [["r", 9000.0]]},
                      {"id": "n3", "name": "Hall", "evidence": [["r", 20000.0]]},
                      {"id": "n4", "name": "Bar", "evidence": [["r", 30000.0]]}]}
    cset = NS(candidates=[views({"id": f"c{t}", "views": [["r", t * 1000.0]]}) for t in (0, 9, 20, 30)])
    nodes = [NS(id="a", name="Door", candidates=["c9"]), NS(id="b", name=" door ", candidates=["c0"]),
             NS(id="c", name="Big hall", candidates=["c20"]), NS(id="d", name="bar", candidates=["c20"])]
    # the doors (one name twice) and the renamed hall by their frames; the bar by its name, frames or not
    assert match_previous(nodes, cset, prev) == {"d": "n4", "a": "n2", "b": "n1", "c": "n3"}
