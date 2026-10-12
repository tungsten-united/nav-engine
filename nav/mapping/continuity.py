"""Keep a place's map stable from one build to the next, while new walks can still add new spots.

Every build of a place reruns the LLM over all its walks, so without help node ids and names would be reshuffled.

1. `previous_context`: the published map before this build, written into the build as previous.json. For every
   node it lists this build's candidates that show the same frames, so the LLM can keep the node (same name, kind,
   candidates) and only add, merge, split or drop with a reason.
2. `stabilize`: code, not the LLM, decides ids. Each new node is matched to the previous node whose evidence
   frames it shares most (same recording, within MATCH_S), greedily one to one. Matched nodes keep the old id; the
   others get the next unused number. An edge keeps its id if it joins the same two nodes, and its orientation:
   a two-way edge the LLM wrote the other way round (say after a walk in the opposite direction) is turned back,
   so per-direction status and review stay with the right direction. Old ids that are not kept go to `retired` and
   are never handed out again. This works because a rebuild keeps all earlier walks, so their frames come back.
"""

import logging
import re
from itertools import count

from nav.schemas.map import CandidateSet, MapDraft, PlaceMap, Retired

log = logging.getLogger(__name__)
MATCH_S = 3.0  # two views of one recording this close in time are the same frame (keyframes are >= 1 s apart)


def _overlap(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> int:
    """How many views of `a` have a view of `b` in the same recording within MATCH_S."""
    return sum(any(r == r2 and abs(t - t2) <= MATCH_S * 1000 for r2, t2 in b) for r, t in a)


def _nearest(views: list[tuple[str, str, float]], rec_id: str, t_ms: float) -> str | None:
    """The candidate with the view closest to (rec_id, t_ms), within MATCH_S. Only the closest: keyframes are often
    2-3 s apart, and listing every one in range hands each node its neighbours' frames too."""
    best = min(((abs(t - t_ms), c) for c, r, t in views if r == rec_id), default=None)
    return best[1] if best and best[0] <= MATCH_S * 1000 else None


def previous_context(prev: PlaceMap, cset: CandidateSet) -> dict:
    views = [(c.id, v.rec_id, v.t_ms) for c in cset.candidates for v in c.views]
    nodes = []
    for n in prev.nodes:
        evidence = [(e.rec_id, e.t_ms) for e in n.evidence]
        shows = [_nearest(views, r, t) for r, t in evidence]
        nodes.append({"id": n.id, "name": n.name, "kind": n.kind, "description": n.description,
                      "candidates": list(dict.fromkeys(c for c in shows if c)),
                      "evidence": [[r, t] for r, t in evidence]})
    return {"build_id": prev.build.build_id, "summary": prev.summary, "nodes": nodes,
            "edges": [{"id": e.id, "source": e.source, "target": e.target, "kind": e.kind, "basis": e.basis,
                       "instruction": e.instruction, "reverse_instruction": e.reverse_instruction}
                      for e in prev.edges],
            "retired": [r.model_dump() for r in prev.retired]}


def match_previous(draft_nodes: list, cset: CandidateSet, prev: dict) -> dict[str, str]:
    """New node id -> the previous node whose evidence frames it shares most (greedy, one to one), as stabilize."""
    cands = {c.id: c for c in cset.candidates}
    new_views = {n.id: [(v.rec_id, v.t_ms) for c in n.candidates if c in cands for v in cands[c].views]
                 for n in draft_nodes}
    old_views = {n["id"]: [(r, t) for r, t in n["evidence"]] for n in prev["nodes"]}
    scores = sorted(((_overlap(new_views[a], old_views[b]), a, b) for a in new_views for b in old_views),
                    key=lambda s: -s[0])
    out: dict[str, str] = {}
    for score, a, b in scores:
        if score and a not in out and b not in out.values():
            out[a] = b
    return out


def _numbers(prefix: str, ids) -> count:
    nums = [int(m[1]) for i in ids if (m := re.fullmatch(rf"{prefix}(\d+)", i))]
    return count(max(nums, default=0) + 1)


def _turned(e):
    """A two-way edge written the other way round: the same route from the other end."""
    from nav.mapping.build import reverse_steps

    return e.model_copy(update={"instruction": e.reverse_instruction or e.instruction,
                                "reverse_instruction": e.instruction, "steps": reverse_steps(e.steps),
                                "candidates": e.candidates[::-1]})


def stabilize(draft: MapDraft, cset: CandidateSet, prev: dict, build_id: str) -> tuple[MapDraft, list[Retired]]:
    """The draft with ids carried over from the previous map, and the updated retired list."""
    old_views = {n["id"]: [(r, t) for r, t in n["evidence"]] for n in prev["nodes"]}
    node_map = match_previous(draft.nodes, cset, prev)
    retired = [Retired.model_validate(r) for r in prev["retired"]]
    fresh = _numbers("n", list(old_views) + [r.id for r in retired if r.kind == "node"])
    for n in draft.nodes:
        if n.id not in node_map:
            node_map[n.id] = f"n{next(fresh)}"

    def node(i: str) -> str:  # an id the LLM cited but never defined must not collide with a stable one
        return node_map.get(i, f"unknown:{i}")

    old_edges = {frozenset((e["source"], e["target"])): e["id"] for e in prev["edges"]}
    old_ends = {e["id"]: (e["source"], e["target"]) for e in prev["edges"]}
    fresh_e = _numbers("e", [e["id"] for e in prev["edges"]] + [r.id for r in retired if r.kind == "edge"])
    edges, kept_edges = [], set()
    for e in draft.edges:
        s, t = node(e.source), node(e.target)
        eid = old_edges.get(frozenset((s, t)))
        if eid is not None and old_ends[eid] == (t, s) and not e.bidirectional:
            eid = None  # one way, the other way: another edge
        if eid is None or eid in kept_edges:
            eid = f"e{next(fresh_e)}"
        elif old_ends[eid] == (t, s):
            e, (s, t) = _turned(e), (t, s)
            log.info("edge %s: written %s -> %s, kept as before", eid, t, s)
        kept_edges.add(eid)
        edges.append(e.model_copy(update={"id": eid, "source": s, "target": t}))
    nodes = [n.model_copy(update={"id": node_map[n.id]}) for n in draft.nodes]

    names = {n["id"]: n["name"] for n in prev["nodes"]}
    kept_nodes = set(node_map.values())
    retired += [Retired(id=n["id"], kind="node", name=n["name"], build_id=build_id)
                for n in prev["nodes"] if n["id"] not in kept_nodes]
    retired += [Retired(id=e["id"], kind="edge", name=f'{names.get(e["source"], "?")} - {names.get(e["target"], "?")}',
                        build_id=build_id) for e in prev["edges"] if e["id"] not in kept_edges]
    log.info("ids: %d of %d nodes kept from build %s, %d new, %d retired", len(kept_nodes & set(old_views)),
             len(old_views), prev["build_id"], len(kept_nodes - set(old_views)),
             sum(r.build_id == build_id for r in retired))
    return draft.model_copy(update={"nodes": nodes, "edges": edges}), retired
