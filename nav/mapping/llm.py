"""The LLM step, in two passes that only add meaning (pipeline metric-1): `places` names and describes the nodes,
made of code's spots (candidates.json), and `routes` writes the spoken directions over code's edges (routes.json).
Where things are, how they connect, the turns and the distances are code's (nav/mapping/topology.py).

Today it runs as a headless Claude Code job in the build folder: read-only tools (Read, Glob) so it can open
candidates.json and look at every frame, no settings/hooks/MCP/CLAUDE.md from this machine (--safe-mode), and the
answer constrained to the pass's JSON Schema (--json-schema). The prompts and schemas don't depend on Claude Code,
so the same `build_prompt()` + `stage_schema()` can go to the Messages API with structured outputs later (images as
content blocks instead of files).
"""

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

from nav.schemas.map import MAX_INSTRUCTION_CHARS, LlmRun, MapDraft, PlacesDraft, RoutesDraft

log = logging.getLogger(__name__)

LANGUAGES = {"en": "English", "es": "Spanish", "ca": "Catalan", "fr": "French", "de": "German", "it": "Italian",
             "pt": "Portuguese"}

WORDING = """\
## Wording
- Everything the app will say (`summary`, `description`, `landmarks`, `instruction`, `reverse_instruction`) is for
  someone who cannot see, so it has no visual detail: no colours, signs, screens, murals, lights, "you will see" or
  "looks like". Describe what a cane, feet or hands find (walls and edges you can trail, counters, door frames,
  floor changes from concrete to carpet, a step up), what you hear (echo of a big hall, a humming fridge, the noise
  of people working, a kitchen) and smell, distances and body directions. Sounds and smells are inferred from photos:
  say "likely" where it is a guess. Put the visual details, which help sighted helpers and camera-based localization,
  in `visual_cues` instead.
- People, bags, laptops and other things that move are not landmarks. Fixed furniture, counters, walls, pillars,
  doors and floor changes are. Screens and signs are fixed but visual: they go in `visual_cues`.
- Write every text field in {language_name} (`language`: "{language}"), whatever language the voice notes are in.
"""

PLACES_PROMPT = """\
You are naming the places of the map of "{place}" for a navigation app for blind and visually impaired people. The
app tells people where they are and gives spoken, step-by-step directions.

Code has already worked out where everything is, from a 3D model built from the walks' video, and how the walks
connect the places. Your job is what the places are: which spots deserve a name, what they are called, and how a
blind person recognises them. You don't decide positions, connections or distances.

## Inputs (in the current directory)
- `candidates.json`: read it first, completely.
  - `recordings[]`: one entry per walk through the place: the walker's `notes`, `marks` (buttons pressed while
    walking: door, junction, stairs, lift, destination), `voice` (what the walker said, transcribed, with times on
    the recording clock in ms), and `timeline`: the candidates in the order the walk passed them.
  - `candidates[]`: spots of the walks. `position` is where it is (metres, x east, y north, from the 3D model;
    `posed` false: from dead reckoning, rougher), `nearby` the other candidates within a few metres. `views` lists
    every sighting (recording, time, the marks and voice notes around it); the first view is the photo.
- `frames/<candidate id>.jpg`: the photo of each candidate, taken by a phone held upright at chest height with the
  camera facing where the walker was going. Look at every photo; they are your main evidence of what is there.

## Nodes
- Nodes are spots a person would want to reach or be told they are at: entrances, doors, junctions, rooms, areas
  (kitchen, stage, tables), stairs, lifts, destinations, salient landmarks.
- A node cites the candidates that show it, at least one. Several nearby candidates can be one node (the same area
  from another angle, another walk); they must be within {span:.0f} m of each other (code leaves farther ones out).
  Two nodes are at least {gap:.1f} m apart: candidates closer than that are one place you stand at, photographed
  in different directions (code merges closer nodes into the one with more candidates).
  Candidates far apart are different places even when they look alike (the two ends of one row of tables); nearby
  candidates seen from opposite sides are the same place even when they look different.
- A candidate can show nothing worth naming (blur, a close-up of a person, the middle of a plain corridor): leave
  it out. Prefer fewer, well-supported nodes; put what you are unsure about in `notes`.
- The walker's voice notes and tags are ground truth about what is where: name things with their words, and make
  a tagged door, junction, stairs or lift a node unless the photos clearly contradict it. Transcripts can contain
  speech-recognition mistakes; read them with the photos.
- Names are what the app will say: 1-4 words, unique, using the walker's words where possible. Put the details in
  the description.
- `relations`: fixed things that are not nodes of the map but that someone may ask for or orient by (a counter, a
  fire exit, a pillar), judged from one of the node's own candidates (`candidate`): `side` "in_photo" with `x`
  across the photo (0 = left edge, 1 = right edge), else "left", "right" or "behind" relative to the way that photo
  faces, with `evidence` "inferred". Code adds where the other nodes are. An empty list is fine.

{wording}
## Output
Answer with the places in the required JSON structure. Do not create or edit any files.
"""

PLACES_PREVIOUS = """
## Previous map
`previous.json` is the map built before from this place's walks (new walks may have been added since). For each
previous node, `candidates` lists the candidates that show the same frames now (ignore `evidence`, it is for code).
- Keep the previous nodes where the evidence still supports them: the same names and kinds, citing those
  candidates. People already know the place by these names.
- Add nodes for what is new. Rename, merge, split or drop a previous node only when the photos or voice notes show it
  was wrong, and say why in `notes`.
- Ids are reassigned by code from the evidence, so any unique ids are fine.
"""

ROUTES_PROMPT = """\
You are writing the spoken directions of the map of "{place}" for a navigation app for blind and visually impaired
people. The app speaks one edge's `instruction` at a time while the user walks it.

The places are named, and code has worked out every connection from the walks: which way you turn when leaving,
the bends on the way, and how far each stretch is (from a 3D model of the walks). Your job is what helps a blind
person follow it: what to keep beside them, what they pass and can touch or hear, and how they know each stretch,
and the whole route, is done.

## Inputs (in the current directory)
- `routes.json`: read it first, completely.
  - `nodes[]`: the map's places (id, name, kind, description, landmarks).
  - `edges[]`: the connections to write. `steps` are the route in order: `action` (the turn to make at the start of
    the step; the first is the turn when leaving the source) and `distance_m`. `candidates` are the frames of the
    walk along it, in walk order, with the voice notes and marks around them. `previous` is the instruction the
    last map had for this connection, when there was one.
- `frames/<candidate id>.jpg`: the photos, taken by a phone held upright at chest height facing the way the walker
  went. Look at the photos along each edge.

## Each edge
- `steps`: one per given step, in the same order, with the same action. Fill `along` and `until`; the last step's
  `until` is how you know you have arrived (the target's landmarks help). A straight step that is a door, a step up
  or down, stairs or a lift may say so (go_through, step_up, step_down, stairs_up, stairs_down, lift).
- `instruction`: the steps spoken, in the same order, with the same turns and distances (round them: "about 5
  metres"), plus what helps on the way and how you know you have arrived. Egocentric (ahead, left, right, behind;
  "turn so the wall is on your left"), short and concrete; mention hazards (steps, cables on the floor, low tables,
  glass doors). At most {max_chars} characters: the app cuts a longer one after the last whole sentence that fits,
  and the user stops there. Where `previous` still fits the steps, keep its wording.
- `reverse_instruction`: the same route spoken from the target back to the source (turn around first).
- `kind`: walk, door, stairs, lift, ramp or other. `landmarks`, `hazards` and `visual_cues` along the way.
- Write exactly one entry per edge of `routes.json`, with its `id`.

{wording}
## Output
Answer with the routes in the required JSON structure. Do not create or edit any files.
"""

STAGES = ("places", "routes")


def build_prompt(stage: str, place: str, language: str, previous: bool = False) -> str:
    from nav.mapping.topology import NODE_GAP_M, NODE_SPAN_M

    wording = WORDING.format(language=language, language_name=LANGUAGES.get(language, language))
    if stage == "routes":
        return ROUTES_PROMPT.format(place=place, max_chars=MAX_INSTRUCTION_CHARS, wording=wording)
    prompt = PLACES_PROMPT.format(place=place, span=NODE_SPAN_M, gap=NODE_GAP_M, wording=wording)
    return prompt.replace("## Output", PLACES_PREVIOUS.lstrip() + "\n## Output") if previous else prompt


def stage_schema(stage: str) -> dict:
    return (RoutesDraft if stage == "routes" else PlacesDraft).model_json_schema()


class LlmError(RuntimeError):
    pass


def run_claude_code(build_dir: Path, place: str, language: str, model: str, budget_usd: float,
                    effort: str = "medium", timeout_s: int = 1800, stage: str = "places"):
    """Run one pass in build_dir (candidates.json or routes.json, and frames/); job files go to build_dir/job/<stage>/.
    -> (PlacesDraft | RoutesDraft, LlmRun)."""
    claude = shutil.which("claude")
    if not claude:
        raise LlmError("the `claude` CLI is not on PATH")
    job = build_dir / "job" / stage
    job.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(stage, place, language, previous=(build_dir / "previous.json").is_file())
    schema = json.dumps(stage_schema(stage))
    (job / "prompt.md").write_text(prompt)
    (job / "schema.json").write_text(json.dumps(stage_schema(stage), indent=2))
    cmd = [claude, "-p", "--output-format", "json", "--json-schema", schema, "--model", model, "--effort", effort,
           "--tools", "Read,Glob", "--permission-mode", "dontAsk", "--safe-mode", "--strict-mcp-config",
           "--no-session-persistence", "--max-budget-usd", str(budget_usd)]
    (job / "command.txt").write_text(" ".join(c if c != schema else "<schema.json>" for c in cmd) + "\n")
    log.info("running Claude Code, %s pass (%s, %s effort) in %s", stage, model, effort, build_dir)
    started = time.monotonic()
    with (job / "claude.log").open("w") as err:
        proc = subprocess.run(cmd, input=prompt, cwd=build_dir, stdout=subprocess.PIPE, stderr=err, text=True,
                              timeout=timeout_s)
    duration = time.monotonic() - started
    (job / "claude.json").write_text(proc.stdout)
    if not proc.stdout.strip():
        raise LlmError(f"Claude Code exited {proc.returncode} without a result; see {job / 'claude.log'}")
    draft, run = parse_claude_result(job / "claude.json", model, stage)
    run.duration_s, run.effort = round(duration, 1), effort
    return draft, run


def parse_claude_result(path: Path, model: str = "?", stage: str | None = None):
    """The answer and run info from a saved `claude -p --output-format json` result: a PlacesDraft or RoutesDraft
    (`stage`), or for builds before metric-1 (job/claude.json, one pass) a MapDraft."""
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError:
        raise LlmError(f"no JSON result in {path}; see claude.log next to it")
    if result.get("is_error") or result.get("subtype") != "success" or result.get("structured_output") is None:
        raise LlmError(f"Claude Code failed ({result.get('subtype')}): {str(result.get('result'))[:500]}")
    out = result["structured_output"]
    if stage == "routes":
        draft = RoutesDraft.model_validate(out)
    else:
        for item in out.get("nodes", []) + out.get("edges", []):  # answers saved before these fields existed
            item.setdefault("visual_cues", [])
        for n in out.get("nodes", []):
            n.setdefault("relations", [])
        if stage == "places":
            draft = PlacesDraft.model_validate(out)
        else:
            for e in out.get("edges", []):
                e.pop("distance_m", None)
                e.setdefault("basis", "walked")
                e.setdefault("steps", [])
            draft = MapDraft.model_validate(out)
    usage = result.get("modelUsage") or {}
    main_model = max(usage, key=lambda k: usage[k].get("costUSD") or 0) if usage else model
    return draft, LlmRun(backend="claude_code", model=main_model, cost_usd=result.get("total_cost_usd"),
                         session_id=result.get("session_id"))
