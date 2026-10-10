"""Benchmark the map quality check: no check vs the free keyword check vs TypeSafe Jev.

    uv run python bench/quality/bench.py collect <map.json> ...   # spoken items -> dataset.json
    uv run python bench/quality/bench.py label                    # reference labels (Claude Code, text only)
    uv run python bench/quality/bench.py run                      # detectors on real + synthetic sets -> results

Two checks have objective enough answers to score:
- visual: does the spoken text point to something only sight can notice? Real items from built maps, labelled by an
  independent reference judge (Opus, high effort, same definition as Jev's question), plus synthetic items: clean
  items with one injected clause, either visual (positive) or a keyword trap that is not visual (negative).
- turn: does the instruction turn the way the walk did? Synthetic: each real instruction whose first turn the
  reference extracted is paired with a measured turn to the same side (agrees) and to the other side (disagrees).
"""

import asyncio
import json
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DATASET = HERE / "dataset.json"
LABELS = HERE / "labels.json"
RESULTS = HERE / "results.json"
JEV_USD_PER_MTOK = 0.042  # docs.typesafe.ai/models, input tokens; output is free

sys.path.insert(0, str(HERE.parents[1]))
from nav.mapping.build import VISUAL_WORDS  # noqa: E402
from nav.mapping import quality  # noqa: E402


# --- collect ---------------------------------------------------------------------------------------------------
def collect(paths: list[str]) -> None:
    items, seen = [], set()
    for p in paths:
        m = json.loads(Path(p).read_text())
        tag = f"{m['build']['build_id']}{'-prev' if 'previous' in p else ''}"
        names = {n["id"]: n["name"] for n in m["nodes"]}
        for n in m["nodes"]:
            spoken = {"description": n["description"], "landmarks": n["landmarks"]}
            key = json.dumps(spoken, sort_keys=True)
            if key not in seen:
                seen.add(key)
                items.append({"id": f"{tag}:{n['id']}", "kind": "node", "place": m["place"], "spot": n["name"],
                              "spoken": spoken})
        for e in m["edges"]:
            spoken = {"instruction": e["instruction"], "along_the_way": e["landmarks"]}
            key = json.dumps(spoken, sort_keys=True)
            if key not in seen:
                seen.add(key)
                items.append({"id": f"{tag}:{e['id']}", "kind": "edge", "place": m["place"],
                              "from": names[e["source"]], "to": names[e["target"]], "spoken": spoken})
    DATASET.write_text(json.dumps(items, indent=1, ensure_ascii=False))
    print(f"{len(items)} items from {len(paths)} maps -> {DATASET}")


# --- reference labels ------------------------------------------------------------------------------------------
LABEL_PROMPT = """\
You are labelling text from a navigation app for blind people, to build a benchmark. Read `items.json` in the current
directory. For every item answer two things about its `spoken` text:

1. visual (true/false): does the text point the listener to something they can only notice by looking: a colour, a
   painted surface or mural, a sign, a screen, a logo, lighting, or how something looks? Objects, furniture, walls,
   doors and materials (glass, wood, metal, stone) are NOT visual: a cane or a hand finds them. Sounds, smells,
   distances and directions are not visual. One visual cue is enough for true.
2. first_turn (edges only; "n/a" for nodes): which way does the instruction first tell the walker to turn or bear
   when leaving the start: "left", "right", "straight" (only go ahead / continue) or "none" (no direction given).

Label carefully, one by one. Answer in the required JSON structure, with every item id exactly once.
"""
LABEL_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["labels"],
    "properties": {"labels": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "visual", "first_turn"],
        "properties": {"id": {"type": "string"}, "visual": {"type": "boolean"},
                       "first_turn": {"enum": ["left", "right", "straight", "none", "n/a"]}}}}},
}


def label() -> None:
    items = json.loads(DATASET.read_text())
    work = HERE / ".label-job"
    work.mkdir(exist_ok=True)
    (work / "items.json").write_text(json.dumps([{k: it[k] for k in ("id", "kind", "spoken")} for it in items],
                                                indent=1, ensure_ascii=False))
    cmd = [shutil.which("claude"), "-p", "--output-format", "json", "--json-schema", json.dumps(LABEL_SCHEMA),
           "--model", "opus", "--effort", "high", "--tools", "Read", "--permission-mode", "dontAsk", "--safe-mode",
           "--strict-mcp-config", "--no-session-persistence", "--max-budget-usd", "2"]
    t0 = time.monotonic()
    out = subprocess.run(cmd, input=LABEL_PROMPT, cwd=work, capture_output=True, text=True, timeout=1800)
    r = json.loads(out.stdout)
    labels = {x["id"]: x for x in r["structured_output"]["labels"]}
    missing = [it["id"] for it in items if it["id"] not in labels]
    assert not missing, f"reference skipped {missing}"
    LABELS.write_text(json.dumps({"judge": "claude-opus (Claude Code, effort high)", "cost_usd": r["total_cost_usd"],
                                  "duration_s": round(time.monotonic() - t0, 1),
                                  "labels": [labels[it["id"]] for it in items]}, indent=1))
    print(f"labelled {len(labels)} items in {time.monotonic() - t0:.0f} s, ${r['total_cost_usd']:.2f} at API list price")


# --- synthetic -------------------------------------------------------------------------------------------------
VISUAL_CLAUSES = [  # positives: only sight can notice these
    "It is next to the blue painted wall.", "It is right below the big screen.",
    "You will see a yellow sign there.", "It is under the bright ceiling lights.",
    "It faces the colourful mural.", "A poster of a mountain hangs on the left.",
    "The wall there has the company logo on it.", "The spot looks brighter than the rest of the room.",
    "You can spot the exit door from there.", "The floor there is marked with green tape.",
]
TRAP_CLAUSES = [  # negatives with keyword-like words a cane, hand or ear can use
    "The heavy glass door swings toward you.", "A light switch is at shoulder height by the door frame.",
    "The sign-in desk is waist high, on your right.", "A radiator there is red-hot; you feel the heat a metre away.",
    "The floor changes to rubber mats you can feel underfoot.", "A screen door rattles when someone passes.",
    "The wall is painted plaster, smooth to the touch.", "A white cane fits under the table edge.",
    "Look out for a low bench at knee height.", "The room is dark at night, so the lights are on a timer you hear click.",
]


def synthetic(items: list[dict], labels: dict[str, dict], seed: int = 7) -> tuple[list[dict], list[dict]]:
    rnd = random.Random(seed)
    clean = [it for it in items if not labels[it["id"]]["visual"]]
    vis = []
    for i, it in enumerate(clean):
        positive = i % 2 == 0
        clause = rnd.choice(VISUAL_CLAUSES if positive else TRAP_CLAUSES)
        spoken = dict(it["spoken"])
        field = "description" if it["kind"] == "node" else "instruction"
        spoken[field] = f"{spoken[field]} {clause}"
        vis.append({**it, "id": f"syn-vis:{it['id']}", "spoken": spoken, "truth": positive, "clause": clause})
    turn = []
    for it in items:
        side = labels[it["id"]]["first_turn"]
        if it["kind"] == "edge" and side in ("left", "right"):
            other = "right" if side == "left" else "left"
            turn.append({**it, "id": f"syn-turn-same:{it['id']}", "measured": side, "truth": True})
            turn.append({**it, "id": f"syn-turn-other:{it['id']}", "measured": other, "truth": False})
    return vis, turn


# --- detectors -------------------------------------------------------------------------------------------------
def keyword_visual(it: dict) -> bool:
    return bool(VISUAL_WORDS.search(json.dumps(it["spoken"], ensure_ascii=False)))


def keyword_turn_agrees(it: dict) -> bool:
    return re.search(rf"\b{it['measured']}\b", it["spoken"]["instruction"], re.I) is not None


async def jev(visual_items: list[dict], turn_items: list[dict]) -> tuple[dict, dict, int, float]:
    from typesafe_sdk import AsyncTypeSafeClient, Noul, NoulCriteria

    vq = Noul(instructions=quality.VISUAL["instructions"], criteria=NoulCriteria(**quality.VISUAL["criteria"]))

    def tq(side):
        return Noul(instructions=f"Does `spoken.instruction` tell the walker to turn or bear to the {side} when "
                                 f"leaving `from`?")

    def state(it):
        base = {"place": it["place"], "spoken": it["spoken"]}
        return {**base, "spot": it["spot"]} if it["kind"] == "node" else {**base, "from": it["from"], "to": it["to"]}

    t0 = time.monotonic()
    async with AsyncTypeSafeClient(model="jev-latest") as client:
        rv, rt = await asyncio.gather(
            asyncio.gather(*(client.system_one(state(it), {"q": vq}) for it in visual_items)),
            asyncio.gather(*(client.system_one(state(it), {"q": tq(it["measured"])}) for it in turn_items)))
    wall = time.monotonic() - t0
    tokens = sum(r.usage.input_tokens for r in [*rv, *rt])
    return ({it["id"]: r.answers["q"].noul for it, r in zip(visual_items, rv)},
            {it["id"]: r.answers["q"].noul for it, r in zip(turn_items, rt)}, tokens, wall)


# --- metrics ---------------------------------------------------------------------------------------------------
def prf(truth: list[bool], pred: list[bool]) -> dict:
    tp = sum(t and p for t, p in zip(truth, pred))
    fp = sum(p and not t for t, p in zip(truth, pred))
    fn = sum(t and not p for t, p in zip(truth, pred))
    tn = len(truth) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": len(truth), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(f1, 3), "accuracy": round((tp + tn) / len(truth), 3)}


def auroc(truth: list[bool], score: list[float]) -> float:
    pos = [s for t, s in zip(truth, score) if t]
    neg = [s for t, s in zip(truth, score) if not t]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def run() -> None:
    items = json.loads(DATASET.read_text())
    ref = json.loads(LABELS.read_text())
    labels = {x["id"]: x for x in ref["labels"]}
    syn_vis, syn_turn = synthetic(items, labels)
    real_vis = [{**it, "truth": labels[it["id"]]["visual"]} for it in items]

    jv, jt, tokens, wall = asyncio.run(jev(real_vis + syn_vis, syn_turn))
    out = {"reference": {k: ref[k] for k in ("judge", "cost_usd", "duration_s")},
           "jev": {"model": "jev-latest", "input_tokens": tokens, "usd": round(tokens / 1e6 * JEV_USD_PER_MTOK, 5),
                   "wall_s": round(wall, 2), "requests": len(jv) + len(jt), "visual_threshold": quality.VISUAL_FLAG,
                   "turn_threshold": quality.TURN_FLAG},
           "sets": {}}
    for name, data in (("visual / real map text", real_vis), ("visual / synthetic", syn_vis)):
        truth = [it["truth"] for it in data]
        out["sets"][name] = {
            "positives": sum(truth),
            "no check": prf(truth, [False] * len(truth)),
            "keyword check": prf(truth, [keyword_visual(it) for it in data]),
            "jev": {**prf(truth, [jv[it["id"]] >= quality.VISUAL_FLAG for it in data]),
                    "auroc": auroc(truth, [jv[it["id"]] for it in data])},
            "errors": {
                "keyword check": [{"id": it["id"], "truth": it["truth"], "text": _txt(it), "clause": it.get("clause")}
                                  for it in data if keyword_visual(it) != it["truth"]],
                "jev": [{"id": it["id"], "truth": it["truth"], "p": round(jv[it["id"]], 3), "text": _txt(it),
                         "clause": it.get("clause")}
                        for it in data if (jv[it["id"]] >= quality.VISUAL_FLAG) != it["truth"]]},
        }
    # turn: the defect to catch is a disagreement, so positives = the instruction contradicts the walk
    truth = [not it["truth"] for it in syn_turn]
    out["sets"]["turn mismatch / synthetic"] = {
        "positives": sum(truth),
        "no check": prf(truth, [False] * len(truth)),
        "keyword check": prf(truth, [not keyword_turn_agrees(it) for it in syn_turn]),
        "jev": {**prf(truth, [jt[it["id"]] < quality.TURN_FLAG for it in syn_turn]),
                "auroc": auroc(truth, [1 - jt[it["id"]] for it in syn_turn])},
        "errors": {
            "keyword check": [{"id": it["id"], "measured": it["measured"], "text": it["spoken"]["instruction"]}
                              for it in syn_turn if keyword_turn_agrees(it) != it["truth"]],
            "jev": [{"id": it["id"], "measured": it["measured"], "p_agrees": round(jt[it["id"]], 3),
                     "text": it["spoken"]["instruction"]}
                    for it in syn_turn if (jt[it["id"]] >= quality.TURN_FLAG) != it["truth"]]},
    }
    out["visual thresholds"] = {}
    for name, data in (("real", real_vis), ("synthetic", syn_vis)):
        truth = [it["truth"] for it in data]
        out["visual thresholds"][name] = {str(th): prf(truth, [jv[it["id"]] >= th for it in data])["f1"]
                                          for th in (0.5, 0.6, 0.7, 0.8)}
    out["probabilities"] = {"visual": {k: round(v, 3) for k, v in jv.items()},
                            "turn": {k: round(v, 3) for k, v in jt.items()}}
    RESULTS.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    summary(out)
    print("\nJev visual F1 by threshold:", out["visual thresholds"])


def _txt(it: dict) -> str:
    return " | ".join(v if isinstance(v, str) else "; ".join(v) for v in it["spoken"].values())


def summary(out: dict) -> None:
    j = out["jev"]
    print(f"Jev: {j['requests']} requests, {j['input_tokens']} input tokens = ${j['usd']}, {j['wall_s']} s wall")
    print(f"reference: {out['reference']}")
    for name, s in out["sets"].items():
        print(f"\n{name}  (n={s['no check']['n']}, positives={s['positives']})")
        for det in ("no check", "keyword check", "jev"):
            m = s[det]
            extra = f"  auroc {m['auroc']}" if "auroc" in m else ""
            print(f"  {det:<14} P {m['precision']:.2f}  R {m['recall']:.2f}  F1 {m['f1']:.2f}  acc {m['accuracy']:.2f}"
                  f"  (fp {m['fp']}, fn {m['fn']}){extra}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "collect":
        collect(sys.argv[2:])
    elif cmd == "label":
        label()
    elif cmd == "run":
        run()
    elif cmd == "summary":
        summary(json.loads(RESULTS.read_text()))
