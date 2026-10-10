"""Cheap quality check of a finished map with TypeSafe's Jev (a System One model: typed judgments, not text).

The map is written by an expensive model; this asks a cheap one a few narrow questions about every node and edge,
the way a reviewer would, and keeps the raw answers next to each item (`quality`) so thresholds can change without
asking again:

- visual_p (Noul, nodes and edges): does the spoken text rely on sight?
- usable (Score 0-3): nodes, can a blind person tell they arrived? edges, can they follow the instruction?
- turn_agrees_p (Noul, walked edges with a clear measured turn): does the instruction turn the way the walk did?

One request per item (its spoken text is the whole state, so answers don't bleed between items), all in parallel.
A map of ~20 items is ~10k input tokens: fractions of a cent. Needs TYPESAFE_API_KEY; without it the check is
skipped and the map says so.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

from nav.schemas.map import ItemQuality, PlaceEdge, PlaceMap, PlaceNode, QualityReport

log = logging.getLogger(__name__)

VISUAL_FLAG = 0.6  # visual_p at or above: flagged
USABLE_FLAG = 1.5  # usable below: flagged
TURN_FLAG = 0.3  # turn_agrees_p below: flagged
TURN_MIN_DEG = 45  # only check turns at least this clear

# Wording matters: "relies on sight anywhere" also fired on materials and furniture a cane finds (0.5-0.8 on fine
# text). This version puts real visual cues at ~0.9 and everything else under ~0.45 on the Itnig map.
VISUAL = {
    "instructions": "Does `spoken` point the listener to something they can only notice by looking: a colour, a "
                    "painted surface or mural, a sign, a screen, a logo, lighting, or how something looks?",
    "criteria": {
        "true": "At least one cue needs sight to notice, such as 'the painted wall', 'the red cabinet', 'the screen "
                "ahead', 'under the yellow letters'",
        "false": "Every cue can be found without sight: objects, furniture, walls, doors and materials (glass, wood, "
                 "metal, stone) by touch or cane, sounds, smells, distances and directions",
    },
}
ARRIVAL = {
    "instructions": "How well can a blind person tell from `spoken` that they have arrived at this spot?",
    "criteria": [
        "No usable cue: nothing a blind person could notice here",
        "Vague: only something general, such as 'tables' or 'a big hall'",
        "One concrete cue a cane, a hand or the ears could find",
        "Several concrete cues a cane, a hand or the ears could find",
    ],
}
FOLLOW = {
    "instructions": "How well can a blind person follow `spoken` to get from `from` to `to`?",
    "criteria": [
        "Unusable: no direction, or no way to know where to stop",
        "A direction only, with no distance and no arrival cue",
        "A direction plus either a distance or an arrival cue they can notice",
        "A direction, a distance and an arrival cue they can notice",
    ],
}


def _node_state(m: PlaceMap, n: PlaceNode) -> dict:
    return {"place": m.place, "spot": n.name, "spoken": {"description": n.description, "landmarks": n.landmarks}}


def _edge_state(m: PlaceMap, e: PlaceEdge, names: dict[str, str]) -> dict:
    return {"place": m.place, "from": names[e.source], "to": names[e.target],
            "spoken": {"instruction": e.instruction, "along_the_way": e.landmarks}}


async def _ask(client, state: dict, questions: dict) -> tuple[dict, int]:
    r = await client.system_one(state, questions)
    return r.answers, (r.usage.input_tokens if r.usage else 0)


async def _check(m: PlaceMap, model: str) -> int:
    from typesafe_sdk import AsyncTypeSafeClient, Noul, NoulCriteria, Score

    visual = Noul(instructions=VISUAL["instructions"], criteria=NoulCriteria(**VISUAL["criteria"]))
    names = {n.id: n.name for n in m.nodes}
    jobs = []
    for n in m.nodes:
        jobs.append((n, _node_state(m, n), {"visual": visual, "usable": Score(**ARRIVAL)}))
    for e in m.edges:
        qs = {"visual": visual, "usable": Score(**FOLLOW)}
        if e.walked and e.turn_deg is not None and abs(e.turn_deg) >= TURN_MIN_DEG:
            side = "right" if e.turn_deg > 0 else "left"
            qs["turn"] = Noul(instructions=f"Does `spoken.instruction` tell the walker to turn or bear to the {side} "
                                           f"when leaving `from`?")
        jobs.append((e, _edge_state(m, e, names), qs))
    async with AsyncTypeSafeClient(model=model) as client:
        results = await asyncio.gather(*(_ask(client, state, qs) for _, state, qs in jobs))
    tokens = 0
    for (item, _, qs), (answers, used) in zip(jobs, results):
        tokens += used
        q = ItemQuality(visual_p=round(answers["visual"].noul, 3), usable=round(answers["usable"].score, 2),
                        turn_agrees_p=round(answers["turn"].noul, 3) if "turn" in answers else None)
        q.flags = flags(q, isinstance(item, PlaceEdge), item)
        item.quality = q
    return tokens


def flags(q: ItemQuality, is_edge: bool, item=None) -> list[str]:
    out = []
    if q.visual_p >= VISUAL_FLAG:
        out.append("relies on sight")
    if q.usable < USABLE_FLAG:
        out.append("hard to follow: missing a distance or an arrival cue" if is_edge
                   else "no concrete cue to tell you have arrived")
    if q.turn_agrees_p is not None and q.turn_agrees_p < TURN_FLAG:
        side = "right" if item is not None and (item.turn_deg or 0) > 0 else "left"
        out.append(f"the walk turned {side} here but the instruction doesn't say so (one of them is wrong)")
    return out


def check_map(m: PlaceMap, model: str = "jev-latest") -> PlaceMap:
    """Fill `quality` on every node and edge and the map's QualityReport; adds a warning per flagged item."""
    m.warnings = [w for w in m.warnings if not w.startswith("jev: ")]
    if not os.environ.get("TYPESAFE_API_KEY"):
        m.quality = QualityReport(model=model, skipped="TYPESAFE_API_KEY is not set")
        return m
    try:
        tokens = asyncio.run(_check(m, model))
    except Exception as e:  # a failed check must not lose the map
        log.warning("quality check failed: %s", e)
        m.quality = QualityReport(model=model, skipped=f"{type(e).__name__}: {e}")
        return m
    flagged = 0
    for kind, items in (("node", m.nodes), ("edge", m.edges)):
        for it in items:
            if it.quality and it.quality.flags:
                flagged += 1
                m.warnings.append(f"jev: {kind} {it.id}: {'; '.join(it.quality.flags)}")
    m.quality = QualityReport(model=model, checked_at_utc=datetime.now(timezone.utc), input_tokens=tokens,
                              flagged=flagged)
    log.info("quality check (%s): %d of %d items flagged, %d input tokens", model, flagged,
             len(m.nodes) + len(m.edges), tokens)
    return m
