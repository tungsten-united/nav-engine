"""Visual place recognition against a map's reference images (docs/navigation_stack.md 4.1 and 6): live frames ->
ranked nodes and a confidence status. Stateless: agreement across consecutive results (4.2) is the controller's.

Each live frame is a place embedding (the map's `embedding.model`, MegaLoc), compared by cosine with every reference
image of the map (`embeddings.npy`, unit rows). A burst of frames scores each reference by their mean similarity. A
node scores as its best reference; the margin is the best node's score minus the second's.

Status, preferring a missed confirmation to a false one:
- lost: the best score is under `lost_score` (nothing recognised);
- confirmed: best score >= `confirm_score`, margin >= `confirm_margin`, and, when the last confirmed node is known,
  the best node is that node or one edge away from it (graph consistency);
- uncertain: anything else.
The default thresholds are placeholders until tuned on venue images (env NAV_LOC_*; nav/config.py).
"""

from dataclasses import dataclass

import numpy as np

from nav.schemas.map import PlaceMap
from nav.schemas.navigation import Localization, NodeMatch, RefMatch, Thresholds


def angle_diff(a: float, b: float) -> float:
    """a - b in (-180, 180]."""
    d = (a - b) % 360
    return d - 360 if d > 180 else d


@dataclass
class RefIndex:
    """A map with its reference embeddings, ready to match: row i of `vectors` is `refs[i]`."""

    pmap: PlaceMap
    vectors: np.ndarray  # [N, D] float32, unit rows
    refs: list  # ReferenceImage, aligned with vectors

    @classmethod
    def build(cls, pmap: PlaceMap, embeddings: np.ndarray | None) -> "RefIndex":
        """Keep the references with an embedding row (hidden nodes are already out of a published map)."""
        nodes = {n.id for n in pmap.nodes}
        refs, rows = [], []
        if embeddings is not None and embeddings.ndim == 2:
            for r in pmap.references:
                if r.embedding_row is not None and 0 <= r.embedding_row < len(embeddings) and r.node in nodes:
                    refs.append(r)
                    rows.append(r.embedding_row)
        dim = embeddings.shape[1] if embeddings is not None and embeddings.ndim == 2 else 0
        vectors = embeddings[rows].astype(np.float32) if rows else np.zeros((0, dim), np.float32)
        return cls(pmap, vectors, refs)

    def neighbours(self, node: str) -> set[str]:
        out = {node}
        for e in self.pmap.edges:
            if e.source == node:
                out.add(e.target)
            elif e.target == node and e.bidirectional:
                out.add(e.source)
        return out


def localize(index: RefIndex, queries: np.ndarray, thresholds: Thresholds, *, top_k: int = 5, refs_per_node: int = 3,
             previous: str | None = None, expected: str | None = None, heading_abs_deg: float | None = None,
             model: str = "", took_ms: dict[str, float] | None = None) -> Localization:
    """`queries`: [F, D] unit place embeddings of the live frames (a burst of F >= 1)."""
    pmap = index.pmap
    q = np.atleast_2d(np.asarray(queries, dtype=np.float32))
    names = {n.id: n.name for n in pmap.nodes}
    heading_map = None
    if heading_abs_deg is not None and pmap.frame and pmap.frame.heading_zero_abs_deg is not None:
        heading_map = round((heading_abs_deg - pmap.frame.heading_zero_abs_deg) % 360, 1)
    base = dict(map_id=pmap.map_id, model=model, frames=len(q), previous=previous, expected=expected,
                heading_map_deg=heading_map, thresholds=thresholds, took_ms=took_ms or {})
    if not index.refs:
        return Localization(**base, status="lost", reason="the map has no reference images", candidates=[])

    sims = (index.vectors @ q.T).mean(axis=1)  # [N]
    by_node: dict[str, list[tuple[float, object]]] = {}
    for s, r in zip(sims.tolist(), index.refs):
        by_node.setdefault(r.node, []).append((s, r))
    ranked = sorted(((max(s for s, _ in v), node) for node, v in by_node.items()), reverse=True)

    def node_match(score: float, node: str) -> NodeMatch:
        best_refs = sorted(by_node[node], key=lambda x: -x[0])[:refs_per_node]
        return NodeMatch(node=node, name=names.get(node, node), score=round(score, 4), refs=[
            RefMatch(id=r.id, score=round(s, 4), role=r.role, edge=r.edge, toward=r.toward, heading_deg=r.heading_deg,
                     heading_diff_deg=None if heading_map is None or r.heading_deg is None
                     else round(angle_diff(heading_map, r.heading_deg), 1),
                     image=r.image) for s, r in best_refs])

    best_score, best = ranked[0]
    margin = best_score - ranked[1][0] if len(ranked) > 1 else best_score
    if best_score < thresholds.lost_score:
        status, reason = "lost", f"best score {best_score:.2f} < {thresholds.lost_score:.2f}"
    elif best_score < thresholds.confirm_score:
        status, reason = "uncertain", f"best score {best_score:.2f} < {thresholds.confirm_score:.2f}"
    elif margin < thresholds.confirm_margin:
        status, reason = "uncertain", f"{names.get(ranked[1][1], ranked[1][1])} scores close (margin {margin:.2f})"
    elif previous and previous in names and best not in index.neighbours(previous):
        status, reason = "uncertain", f"not next to the last confirmed node ({names[previous]})"
    else:
        status, reason = "confirmed", f"score {best_score:.2f}, margin {margin:.2f}"
    rank = next((i + 1 for i, (_, n) in enumerate(ranked) if n == expected), None)
    return Localization(**base, status=status, reason=reason, best=best, margin=round(margin, 4),
                        candidates=[node_match(s, n) for s, n in ranked[:top_k]],
                        expected_score=round(ranked[rank - 1][0], 4) if rank else None, expected_rank=rank)
