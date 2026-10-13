"""The review's `seen` names (nav/mapping/review.py): its entries stay with the places they were written for."""

import json
from pathlib import Path

from test_clean import edge, node, pmap

from nav.mapping.review import (EdgeReview, MeasuredDistance, NodeReview, Review, apply_review, from_template, moved,
                                save_review, template)
from nav.schemas.map import PlaceMap

METRIC = Path(__file__).parent / "fixtures" / "itnig-metric"


def test_itnig_entries_of_places_that_moved_id_are_held_not_applied():
    """The review decided on site on 2026-10-10, with `seen` as the build it was written for (20261010T140702Z) had
    the names, on Itnig's first metric-1 build. That build gave n9 (the stone wall tables then) to the judges' table
    and n2 (the drinks area) to the hackathon tables, so the merges moved the judges' table into the centre tables,
    6 m away, and the stage corridor into the hackathon tables."""
    build = PlaceMap.model_validate_json((METRIC / "map.json").read_text())
    review = Review.model_validate_json((METRIC / "review-2026-10-10.json").read_text())
    assert moved(build, review) == {"n2": "n14", "n9": "n15"}
    m = apply_review(build, review)
    assert m.merged == {}  # n9 -> n10 (n9 moved) and n4 -> n2 (its target moved) are held
    centre = next(n for n in m.nodes if n.id == "n10")
    assert centre.description == review.nodes["n10"].description  # n10 is still the centre tables: applied
    held = [w for w in m.warnings if "are held" in w]
    assert len(held) == 2
    assert held[0].startswith("review: n2 was 'Drinks area' when reviewed; this build has 'Hackathon tables' there "
                              "and 'Drinks area' at n14")
    assert "'Stone wall tables' at n15" in held[1]


def test_a_renamed_place_keeps_its_entries_a_moved_one_holds_them_with_its_edges_and_distances():
    m = pmap([node("n1"), node("n2"), node("n3")], [edge("e1", "n1", "n2"), edge("e2", "n2", "n3")])
    review = Review(nodes={"n2": NodeReview(description="Checked on site.")}, edges={"e1": EdgeReview(approve=True)},
                    distances=[MeasuredDistance(between=("n1", "n2"), meters=10.0)], seen={"n2": "Old name"})
    renamed = apply_review(m, review)  # "Old name" is nowhere in the build: the LLM renamed n2
    assert renamed.nodes[1].description == "Checked on site." and renamed.edges[0].status == "verified"
    assert renamed.edges[1].distance_m == 10.0  # walked 5 m, measured 10: every distance doubles
    review.seen = {"n2": "N3"}  # what was reviewed as n2 is n3 now
    held = apply_review(m, review)
    assert held.nodes[1].description == "" and held.edges[0].status == "observed" and held.edges[1].distance_m == 5.0
    assert [w for w in held.warnings if "held" in w] == [
        "review: n2 was 'N3' when reviewed; this build has 'N2' there and 'N3' at n3: entries that depend on n2 are "
        "held (move them to n3, or delete seen.n2 to keep them on 'N2')"]


def test_saving_fills_in_the_names_the_review_lacks_and_keeps_the_others(tmp_path):
    m = pmap([node("n1"), node("n2"), node("n3"), node("n4")], [edge("e1", "n1", "n2")])
    build = tmp_path / "builds" / m.build.build_id
    build.mkdir(parents=True)
    for path in (build / "map.json", tmp_path / "map.json"):
        path.write_text(m.model_dump_json())
    review = Review(nodes={"n2": NodeReview(merge_into="n3")}, edges={"e1": EdgeReview(approve=True)},
                    distances=[MeasuredDistance(between=("n1", "n3"), meters=5.0)],
                    seen={"n2": "As reviewed", "n9": "Gone"})
    saved = save_review(tmp_path, review)
    # n1: an end of e1 and of the distance; n2 kept as sent; n3: the merge target; n9: nothing refers to it any more
    assert saved.seen == {"n1": "N1", "n2": "As reviewed", "n3": "N3"}
    assert json.loads((tmp_path / "review.json").read_text())["seen"] == saved.seen
    edited = from_template(json.loads(json.dumps(template(m, saved), default=str)))  # `nav map review` keeps them
    assert edited.seen == saved.seen
