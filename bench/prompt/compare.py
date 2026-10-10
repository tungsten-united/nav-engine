"""Compare map builds on what the generation rules are meant to fix (same candidates, different prompt/schema).

    uv run python bench/prompt/compare.py <label>=<map.json> ...   -> table + bench/prompt/results.json
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1]))
from nav.mapping.build import TURN_CHECK_DEG, VISUAL_WORDS  # noqa: E402


def llm_distance(e: dict) -> float | None:
    """What the LLM said the distance was: the steps' sum (new schema) or its distance_m (old schema)."""
    if e.get("steps"):
        d = [s["distance_m"] for s in e["steps"] if s["distance_m"] is not None]
        return sum(d) if d else None
    return e.get("distance_m")


def metrics(m: dict) -> dict:
    edges, nodes = m["edges"], m["nodes"]
    walked = [e for e in edges if e.get("walked", True)]
    gaps = [abs(llm_distance(e) - e["measured_distance_m"]) for e in walked
            if llm_distance(e) is not None and e.get("measured_distance_m") is not None]
    clear = [e for e in walked if e.get("turn_deg") is not None and abs(e["turn_deg"]) >= TURN_CHECK_DEG]
    jev = m.get("quality") or {}
    return {
        "nodes": len(nodes), "edges": len(edges),
        "edges inferred (code)": sum(not e.get("walked", True) for e in edges),
        "edges claimed walked but not": sum(1 for w in m["warnings"] if " says walked" in w),
        "edges with steps": sum(bool(e.get("steps")) for e in edges),
        "clear turns checked": len(clear),
        "first step contradicts the walk": sum(1 for w in m["warnings"] if "the first step is" in w),
        "instruction text contradicts the walk (Jev)": sum(1 for e in edges if (e.get("quality") or {}).get(
            "turn_agrees_p") is not None and e["quality"]["turn_agrees_p"] < 0.3),
        "LLM distance vs measured, mean abs gap (m)": round(sum(gaps) / len(gaps), 1) if gaps else None,
        "LLM distance vs measured, edges compared": len(gaps),
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
        runs[label] = metrics(json.loads(Path(path).read_text()))
    out = HERE / "results.json"
    prev = json.loads(out.read_text()) if out.is_file() else {}
    prev.update(runs)
    out.write_text(json.dumps(prev, indent=1))
    labels = list(prev)
    keys = list(next(iter(prev.values())))
    print(f"{'':48}" + "".join(f"{l:>14}" for l in labels))
    for k in keys:
        print(f"{k:48}" + "".join(f"{str(prev[l].get(k, '-')):>14}" for l in labels))
