"""Compare map builds on what the generation rules are meant to fix (same candidates, different prompt/schema).

    uv run python bench/prompt/compare.py <label>=<map.json> ...   -> table + bench/prompt/results.json
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
from nav.mapping.build import TURN_CHECK_DEG, VISUAL_WORDS  # noqa: E402


def raw_answer(map_path: Path) -> dict | None:
    """The LLM's own answer for the build (before code filled anything in)."""
    f = map_path.parent / "job" / "claude.json"
    return json.loads(f.read_text())["structured_output"] if f.is_file() else None


def llm_distance(e: dict) -> float | None:
    """What the LLM said the distance was: the steps' sum (new schema) or its distance_m (old schema)."""
    if e.get("steps"):
        d = [s["distance_m"] for s in e["steps"] if s["distance_m"] is not None]
        return sum(d) if d else None
    return e.get("distance_m")


def boundary_distance(e: dict, nodes: dict, cands: dict) -> float | None:
    """Walked from the source's last frame to the target's first frame on the closest passes: the distance a
    spoken instruction should give (leaving one spot to reaching the next)."""
    va = [v for c in nodes[e["source"]]["candidates"] for v in cands[c]["views"]]
    vb = [v for c in nodes[e["target"]]["candidates"] for v in cands[c]["views"]]
    pairs = [(abs(x["t_ms"] - y["t_ms"]), x, y) for x in va for y in vb if x["rec_id"] == y["rec_id"]]
    if not pairs:
        return None
    _, x, y = min(pairs, key=lambda p: p[0])
    near_a = [v for v in va if v["rec_id"] == x["rec_id"] and abs(v["t_ms"] - x["t_ms"]) <= 15000]
    near_b = [v for v in vb if v["rec_id"] == y["rec_id"] and abs(v["t_ms"] - y["t_ms"]) <= 15000]
    walked = lambda views: [v["motion"]["walked_m"] for v in views]
    first, then = (near_a, near_b) if x["t_ms"] < y["t_ms"] else (near_b, near_a)
    return round(abs(min(walked(then)) - max(walked(first))), 1)


def metrics(map_path: Path) -> dict:
    m = json.loads(map_path.read_text())
    raw = raw_answer(map_path) or m
    cands = {c["id"]: c for c in json.loads((map_path.parent / "candidates.json").read_text())["candidates"]}
    nodes = {n["id"]: n for n in m["nodes"]}
    edges = m["edges"]
    raw_edges = {e["id"]: e for e in raw["edges"]}
    walked = [e for e in edges if e.get("walked", True)]
    gaps = []
    for e in walked:
        said, ref = llm_distance(raw_edges.get(e["id"], {})), boundary_distance(e, nodes, cands)
        if said is not None and ref is not None:
            gaps.append(abs(said - ref))
    clear = [e for e in walked if e.get("turn_deg") is not None and abs(e["turn_deg"]) >= TURN_CHECK_DEG]
    has_basis = any("basis" in e for e in raw["edges"])
    jev = m.get("quality") or {}
    return {
        "nodes": len(m["nodes"]), "edges": len(edges),
        "edges inferred (code)": sum(not e.get("walked", True) for e in edges),
        "edges claimed walked but not": sum(1 for w in m["warnings"] if " says walked" in w) if has_basis else "n/a",
        "edges with steps": sum(bool(e.get("steps")) for e in raw["edges"]),
        "clear turns checked (>=45 deg)": len(clear),
        "first step contradicts the walk": sum(1 for w in m["warnings"] if "the first step is" in w)
        if any(e.get("steps") for e in raw["edges"]) else "n/a",
        "instruction text contradicts the walk (Jev)": sum(1 for e in edges if (e.get("quality") or {}).get(
            "turn_agrees_p") is not None and e["quality"]["turn_agrees_p"] < 0.3),
        "spoken distance vs walked, mean abs gap (m)": round(sum(gaps) / len(gaps), 1) if gaps else None,
        "spoken distance vs walked, edges compared": len(gaps),
        "far-apart passes merged": sum(1 for w in m["warnings"] if "by dead reckoning" in w),
        "spoken text with visual words (keyword)": sum(1 for w in m["warnings"] if "uses visual words" in w),
        "flagged by Jev": jev.get("flagged"),
        "candidates unused": next((len(w.split(":")[1].split(",")) for w in m["warnings"]
                                   if w.startswith("candidates not used")), 0),
    }


if __name__ == "__main__":
    runs = {}
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        runs[label] = metrics(Path(path))
    out = HERE / "results.json"
    prev = json.loads(out.read_text()) if out.is_file() else {}
    prev.update(runs)
    out.write_text(json.dumps(prev, indent=1))
    labels = list(prev)
    keys = list(next(iter(prev.values())))
    print(f"{'':48}" + "".join(f"{l:>14}" for l in labels))
    for k in keys:
        print(f"{k:48}" + "".join(f"{str(prev[l].get(k, '-')):>14}" for l in labels))
