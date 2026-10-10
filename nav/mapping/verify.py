"""Which keyframes show the same place: retrieve with MegaLoc, confirm with local feature matching.

1. `sim` = MegaLoc cosine between every two keyframes (cheap: one matrix product).
2. Candidate pairs: sim >= hint_sim and one frame is among the other's retrieve_k most similar.
3. Each candidate pair is checked with ALIKED + LightGlue: enough matches consistent with one camera geometry
   (fundamental matrix) means the two frames really see the same structure, not just the same kind of corridor.

Look-alikes that fail the check stay hints (possibly_same_as) for the LLM; they are never folded.
"""

import logging
from collections.abc import Callable

import numpy as np

from nav.mapping.inference import pair_inliers
from nav.mapping.select import Frame, SelectParams

log = logging.getLogger(__name__)


def candidate_pairs(sim: np.ndarray, p: SelectParams) -> list[tuple[int, int]]:
    n = len(sim)
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    top = np.argsort(-s, axis=1)[:, :p.retrieve_k]
    pairs = {(min(i, int(j)), max(i, int(j))) for i in range(n) for j in top[i] if s[i, j] >= p.hint_sim}
    return sorted(pairs)


def match_pairs(kept: list[Frame], p: SelectParams, inliers: Callable[[int, int], int] | None = None
                ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """-> (sim [n, n], same [n, n] bool, checked pairs as rows for pairs.json). `inliers(i, j)` defaults to
    ALIKED + LightGlue on the frames' images (locally, or all pairs in batches on the inference host)."""
    n = len(kept)
    if not n:
        return np.zeros((0, 0)), np.zeros((0, 0), bool), []
    place = np.stack([f.place for f in kept])
    sim = place @ place.T
    same = np.eye(n, dtype=bool)
    pairs = candidate_pairs(sim, p)
    log.info("checking %d of %d keyframe pairs with LightGlue", len(pairs), n * (n - 1) // 2)
    if inliers is None:
        counts = dict(zip(pairs, pair_inliers([f.image_path for f in kept], pairs)))
        inliers = lambda i, j: counts[(i, j)]  # noqa: E731
    rows = []
    for i, j in pairs:
        k = inliers(i, j)
        same[i, j] = same[j, i] = k >= p.min_inliers
        rows.append({"a": {"rec_id": kept[i].rec_id, "t_ms": kept[i].t_ms},
                     "b": {"rec_id": kept[j].rec_id, "t_ms": kept[j].t_ms},
                     "sim": round(float(sim[i, j]), 3), "inliers": k, "same": bool(same[i, j])})
    return sim, same, rows
