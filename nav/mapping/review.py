"""Human review of a place's map (docs/navigation_stack.md 3, step 5): what a person checked on site.

The review is an overlay, `maps/<map_id>/review.json`, keyed by the map's stable ids (continuity.py), so approvals
and corrections survive rebuilds: a rebuild that keeps a node keeps its review too. Applying it to a build's map
gives the published map.json:

- approve: status -> verified. Routing for real users only uses verified edges (nav/navigation/route.py).
  Edges are approved per direction: `approve` for source -> target, `approve_reverse` for target -> source.
- name / kind / instruction / reverse_instruction: corrections that win over what the LLM wrote.
- hide: leave the item out of the published map (it stays in the build's map.json; nothing is deleted).
  Hiding a node hides its edges, relations and reference images too.

Review entries whose ids are not in the map (retired, or from a later build) are kept and reported.

Adapted from tungsten-united/demo-repository@c5dc26f (navigation, mapping/src/mapper/review.py) by LPA.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from nav.schemas.map import NodeKind, PlaceMap


class NodeReview(BaseModel):
    approve: bool = False
    hide: bool = False
    name: str | None = None
    kind: NodeKind | None = None
    note: str | None = None


class EdgeReview(BaseModel):
    approve: bool = False  # source -> target walked and checked
    approve_reverse: bool = False  # target -> source walked and checked
    hide: bool = False
    instruction: str | None = None
    reverse_instruction: str | None = None
    note: str | None = None


class RelationReview(BaseModel):
    approve: bool = False
    hide: bool = False


class Review(BaseModel):
    updated_at_utc: datetime | None = None
    updated_by: str | None = None
    nodes: dict[str, NodeReview] = {}
    edges: dict[str, EdgeReview] = {}
    relations: dict[str, RelationReview] = {}


def review_path(map_dir: Path) -> Path:
    return map_dir / "review.json"


def load_review(map_dir: Path) -> Review:
    path = review_path(map_dir)
    return Review.model_validate_json(path.read_text()) if path.is_file() else Review()


def save_review(map_dir: Path, review: Review, by: str | None = None) -> Review:
    review = review.model_copy(update={"updated_at_utc": datetime.now(timezone.utc), "updated_by": by})
    tmp = map_dir / ".review.json.tmp"
    tmp.write_text(review.model_dump_json(indent=1, exclude_defaults=False))
    tmp.replace(review_path(map_dir))
    return review


def apply_review(pmap: PlaceMap, review: Review) -> PlaceMap:
    """The map as published: the build's map with the review applied (a copy; the input is unchanged)."""
    m = pmap.model_copy(deep=True)
    if review == Review():
        return m
    hidden: set[str] = set()
    for n in m.nodes:
        r = review.nodes.get(n.id)
        if r is None:
            continue
        if r.hide:
            hidden.add(n.id)
            continue
        if r.name and r.name != n.name:
            n.name = r.name
            n.review_flags = [f for f in n.review_flags if f != "name set in review"] + ["name set in review"]
        if r.kind:
            n.kind = r.kind
        if r.approve:
            n.status = "verified"
            n.review_flags = [f for f in n.review_flags if f == "name set in review"]
    for n in m.nodes:
        for rel in n.relations:
            r = review.relations.get(rel.id)
            if n.id in hidden or (r and r.hide):
                hidden.add(rel.id)
            elif r and r.approve:
                rel.status = "verified"
        n.relations = [rel for rel in n.relations if rel.id not in hidden]
    for e in m.edges:
        r = review.edges.get(e.id)
        if e.source in hidden or e.target in hidden or (r and r.hide):
            hidden.add(e.id)
            continue
        if r is None:
            continue
        if r.instruction:
            e.instruction = r.instruction
        if r.reverse_instruction:
            e.reverse_instruction = r.reverse_instruction
        if r.approve:
            e.status = "verified"
            e.review_flags = [f for f in e.review_flags if f == "the way back was not walked"]
        if r.approve_reverse and e.bidirectional:
            e.reverse_status = "verified"
            e.review_flags = [f for f in e.review_flags if f != "the way back was not walked"]
    m.nodes = [n for n in m.nodes if n.id not in hidden]
    m.edges = [e for e in m.edges if e.id not in hidden]
    m.references = [x for x in m.references if x.node not in hidden]
    for e in m.edges:
        e.reference_views = [x for x in e.reference_views if any(r.id == x for r in m.references)]
    m.hidden = sorted(hidden)
    m.reviewed_at_utc = review.updated_at_utc or datetime.now(timezone.utc)
    known = {n.id for n in pmap.nodes} | {e.id for e in pmap.edges} | {r.id for n in pmap.nodes for r in n.relations}
    stale = sorted(i for i in [*review.nodes, *review.edges, *review.relations] if i not in known)
    if stale:
        m.warnings.append(f"review: entries for ids not in this build (kept for later): {', '.join(stale)}")
    return m


def template(pmap: PlaceMap, review: Review) -> dict:
    """An editable review.json for `nav map review`: every item with what to check, current entries filled in."""
    names = {n.id: n.name for n in pmap.nodes}

    def flags(item, kind: str) -> list[str]:
        q = item.quality.flags if item.quality else []
        warned = [w for w in pmap.warnings if f"{kind} {item.id}:" in w or f"{kind} {item.id} " in w]
        return list(dict.fromkeys([*item.review_flags, *q, *warned]))

    return {
        "_how": "Walk the place with the map. Set approve (edges: approve for source -> target, approve_reverse for "
                "the way back) on what you checked, fix name/kind/instruction, hide what is wrong. Then run "
                "`nav map review <map_id> --apply <this file>` (or PUT it to /api/v1/maps/<map_id>/review). Only "
                "verified edges are used to route real users.",
        "nodes": {n.id: {"_": f"{n.name} ({n.kind}, {n.status})", "_check": flags(n, "node"),
                         **review.nodes.get(n.id, NodeReview()).model_dump()} for n in pmap.nodes},
        "edges": {e.id: {"_": f"{names.get(e.source)} -> {names.get(e.target)} ({e.status}, back: {e.reverse_status})",
                         "_instruction": e.instruction, "_check": flags(e, "edge"),
                         **review.edges.get(e.id, EdgeReview()).model_dump()} for e in pmap.edges},
        "relations": {r.id: {"_": f"at {n.name}: {r.subject} {r.direction} ({r.status})",
                             **review.relations.get(r.id, RelationReview()).model_dump()}
                      for n in pmap.nodes for r in n.relations},
    }


def from_template(data: dict) -> Review:
    """A Review from an edited template (keys starting with "_" are explanations, not review)."""
    def entries(kind: str, model) -> dict:
        out = {k: model(**{f: x for f, x in v.items() if not f.startswith("_")}) for k, v in data.get(kind, {}).items()}
        return {k: v for k, v in out.items() if v != model()}  # untouched items are not review

    return Review(nodes=entries("nodes", NodeReview), edges=entries("edges", EdgeReview),
                  relations=entries("relations", RelationReview))


def dumps(data: dict) -> str:
    return json.dumps(data, indent=1, ensure_ascii=False, default=str)
