"""The LLM step: turn the candidate set into a MapDraft.

Today it runs as a headless Claude Code job in the build folder: read-only tools (Read, Glob) so it can open
candidates.json and look at every frame, no settings/hooks/MCP/CLAUDE.md from this machine (--safe-mode), and the
answer constrained to MapDraft's JSON Schema (--json-schema). The prompt and schema don't depend on Claude Code,
so the same `build_prompt()` + `draft_json_schema()` can go to the Messages API with structured outputs later
(images as content blocks instead of files).
"""

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

from nav.schemas.map import MAX_INSTRUCTION_CHARS, LlmRun, MapDraft, draft_json_schema

log = logging.getLogger(__name__)

LANGUAGES = {"en": "English", "es": "Spanish", "ca": "Catalan", "fr": "French", "de": "German", "it": "Italian",
             "pt": "Portuguese"}

PROMPT = """\
You are building a semantic topological map of "{place}" for a navigation app for blind and visually impaired
people. The app will use it to tell people where they are and to give spoken, step-by-step directions.

## Inputs (in the current directory)
- `candidates.json`: read it first, completely.
  - `recordings[]`: one entry per walk through the place: the walker's `notes`, `marks` (buttons pressed while
    walking: door, junction, stairs, lift, destination), `voice` (what the walker said, transcribed, with times on
    the recording clock in ms), and `timeline`: the candidates in the order the walk passed them, with metres
    walked since the previous one (`from_prev_m`), heading change since the previous one (`turn_deg`), and the turn
    the walker made at that spot (`turn_at_deg`, heading over the 3 s after vs the 3 s before). For both, + = right,
    - = left, in degrees. A walk with `motion` "video" had no phone sensors: its turns come from the video and its
    metres are guesses from time, so take distances from the other walks where you can.
  - `candidates[]`: frames selected from the walks. `views` lists every sighting (recording, time, motion, the
    marks and voice notes around it); the first view is the photo. `reason` says why it was kept.
    `possibly_same_as` lists look-alike candidates that may be the same spot seen again (a revisit or another walk).
- `frames/<candidate id>.jpg`: the photo of each candidate, taken by a phone held upright at chest height with the
  camera facing where the walker was going. Look at every photo; they are your main evidence.

## Building the map
- Nodes are spots a person would want to reach or be told they are at: entrances, doors, junctions, rooms, areas
  (kitchen, stage, tables), stairs, lifts, destinations, salient landmarks. Edges join nodes that can be walked
  between directly, without passing another node.
- Candidates are evidence, not nodes. Several candidates can show one node (same area from another angle, a revisit,
  another walk); a candidate can show nothing useful (blur, a close-up of a person): leave it out. Every node cites
  the candidates that show it, at least one.
- The walker's voice notes and tags are ground truth about what is where: name things with their words, and make
  a tagged door, junction, stairs or lift a node unless the photos clearly contradict it. Transcripts can contain
  speech-recognition mistakes; read them with the photos.
- A node is one spot you can stand at, not a whole row or zone. Each view has a dead-reckoned position in its own
  walk (`views[].motion.x_m`, `y_m`, in metres from where that walk started, drifting a few metres per minute of
  walking), and in a frame shared by the walks (`map_x_m`, `map_y_m`) when its walk shares frames with the others
  (`recordings[].aligned`); null otherwise. Compare positions within one walk with x/y, across walks only with
  map_x/map_y. Two passes more than about 15 m apart are two nodes joined by an edge, even if they look alike or the
  walker says "the same tables from another angle" (that means the far end of the same row). Merge passes into one
  node only when the photos show the same spot and the positions (where comparable) roughly agree.
- Walks can start anywhere and cross the place in another order or the opposite direction. The same spot is then
  seen from the other side, so its photos rarely look alike and are seldom folded into one candidate. Recognise it
  from the walker's words and tags (each walk says where it starts), the order of places on the timelines (a walk
  in the opposite direction passes them in reverse order), the distances and turns between them, and fixed things
  seen from both sides; then make it one node that cites the candidates of every walk. A walk that goes the
  other way along an existing connection is the same edge walked back, not a new edge.

## Edges
- Connectivity comes from the timelines. `basis` is "walked" when a walk went from the source's frames to the
  target's frames with no other node's frames in between (consecutive on a timeline). Otherwise it is "inferred":
  use that sparingly (say, the two ends of one row), never for a connection no walk or photo supports, and explain
  each inferred edge in `notes`. If you would rate an inferred edge low confidence, leave it out: code drops it
  anyway. Mention the possible connection in `notes` as something to walk next time. Revisits that the photos
  confirm close loops.
- `candidates` of a walked edge: that walk's frames from the source's last frame to the target's first frame, in
  timeline order.
- `steps` are the route in order. The first step's action is the turn when leaving the source: read it from the
  motion data, the `turn_at_deg` of the source's last frame on that walk (and the `turn_deg` of the next timeline
  step). Under 20° is straight, 20-60° bear, 60-135° turn, more is turn_around; + is right, - is left. The motion
  data is usually right about turns even when a photo seems to say otherwise (the walker may have panned the phone);
  if you overrule it, say why in `notes`.
- Step distances come from the motion data: the sum of `from_prev_m` along the walk, rounded to the metre (step
  counting misses slow steps, so it is approximate). A turn on the spot or a step up has no distance.
- `instruction` is the spoken form of the steps: same order, same turns, same distances, plus what helps on the way.
  `reverse_instruction` speaks the same route backwards (code derives the reverse steps from yours).

## Relations
- For each node, `relations` says where a few other things are as seen from there (up to about five, the most
  useful): other nodes of the map, by their name, and fixed things someone may ask for or orient by (a counter, the
  stage, an exit, a pillar). Code turns them into directions with the phone's heading, so the app can say "the
  kitchen is on your right" once it knows which way the user faces.
- Judge each from one of the node's own candidates (`candidate`). If that photo shows it, `side` is "in_photo" and
  `x` is where it is across the photo (0 = left edge, 0.5 = centre, 1 = right edge). If it is out of the photo but
  the node's other photos or the walk show where, `side` is "left", "right" or "behind", relative to the way that
  photo faces, `x` is null and `evidence` is "inferred". Leave out what the photos don't support; an empty list is
  fine.

## Wording
- Everything the app will say (`summary`, `description`, `landmarks`, `instruction`, `reverse_instruction`) is for
  someone who cannot see, so it has no visual detail: no colours, signs, screens, murals, lights, "you will see" or
  "looks like". Describe what a cane, feet or hands find (walls and edges you can trail, counters, door frames,
  floor changes from concrete to carpet, a step up), what you hear (echo of a big hall, a humming fridge, the noise
  of people working, a kitchen) and smell, distances and body directions. Sounds and smells are inferred from photos:
  say "likely" where it is a guess. Put the visual details, which help sighted helpers and camera-based localization,
  in `visual_cues` instead.
- Instructions are egocentric (ahead, left, right, behind; "turn so the wall is on your left"), short and concrete:
  direction, roughly how far, what you pass and can touch or hear, and how you know you have arrived. Mention hazards:
  steps, cables on the floor, low tables, glass doors. At most {max_chars} characters each: the app cuts a longer
  one after the last whole sentence that fits, and the user stops there.
- People, bags, laptops and other things that move are not landmarks. Fixed furniture, counters, walls, pillars,
  doors and floor changes are. Screens and signs are fixed but visual: they go in `visual_cues`.
- Names are what the app will say: 1-4 words, unique, using the walker's words where possible. Put the details in
  the description.
- Prefer fewer, well-supported nodes over many speculative ones. Put what you are unsure about in `notes`.
- Write every text field in {language_name} (`language`: "{language}"), whatever language the voice notes are in.

## Output
Answer with the map in the required JSON structure. Do not create or edit any files.
"""

PREVIOUS = """
## Previous map
`previous.json` is the map built before from this place's walks (new walks may have been added since; all the earlier
walks are still in the candidates). For each previous node, `candidates` lists the candidates that show the same
frames now (ignore `evidence`, it is for code).
- Keep the previous map where the evidence still supports it: the same nodes with the same names and kinds, citing
  those candidates, and the same connections. People already know the place by these names.
- Keep each previous edge's `source` and `target` as they were, even when a new walk went the other way: describe
  that way in `reverse_instruction`. New walks of the same spots go into the existing nodes (add their candidates),
  so the map gains evidence, not duplicates.
- Add nodes and edges for what is new: spots only the new walks reach, or that the previous map missed.
- Merge, split, rename or drop a previous node only when the photos, voice notes or positions show it was wrong, and
  say what changed and why in `notes`.
- Ids are reassigned by code from the evidence, so any unique ids are fine.
"""


def build_prompt(place: str, language: str, previous: bool = False) -> str:
    prompt = PROMPT.format(place=place, language=language, language_name=LANGUAGES.get(language, language),
                           max_chars=MAX_INSTRUCTION_CHARS)
    return prompt.replace("## Output", PREVIOUS.lstrip() + "\n## Output") if previous else prompt


class LlmError(RuntimeError):
    pass


def run_claude_code(build_dir: Path, place: str, language: str, model: str, budget_usd: float,
                    effort: str = "medium", timeout_s: int = 1800) -> tuple[MapDraft, LlmRun]:
    """Run the job in build_dir (which holds candidates.json and frames/); job files go to build_dir/job/."""
    claude = shutil.which("claude")
    if not claude:
        raise LlmError("the `claude` CLI is not on PATH")
    job = build_dir / "job"
    job.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(place, language, previous=(build_dir / "previous.json").is_file())
    schema = json.dumps(draft_json_schema())
    (job / "prompt.md").write_text(prompt)
    (job / "schema.json").write_text(json.dumps(draft_json_schema(), indent=2))
    cmd = [claude, "-p", "--output-format", "json", "--json-schema", schema, "--model", model, "--effort", effort,
           "--tools", "Read,Glob", "--permission-mode", "dontAsk", "--safe-mode", "--strict-mcp-config",
           "--no-session-persistence", "--max-budget-usd", str(budget_usd)]
    (job / "command.txt").write_text(" ".join(c if c != schema else "<schema.json>" for c in cmd) + "\n")
    log.info("running Claude Code (%s, %s effort) in %s", model, effort, build_dir)
    started = time.monotonic()
    with (job / "claude.log").open("w") as err:
        proc = subprocess.run(cmd, input=prompt, cwd=build_dir, stdout=subprocess.PIPE, stderr=err, text=True,
                              timeout=timeout_s)
    duration = time.monotonic() - started
    (job / "claude.json").write_text(proc.stdout)
    if not proc.stdout.strip():
        raise LlmError(f"Claude Code exited {proc.returncode} without a result; see {job / 'claude.log'}")
    draft, run = parse_claude_result(job / "claude.json", model)
    run.duration_s, run.effort = round(duration, 1), effort
    return draft, run


def parse_claude_result(path: Path, model: str = "?") -> tuple[MapDraft, LlmRun]:
    """MapDraft + run info from a saved `claude -p --output-format json` result."""
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError:
        raise LlmError(f"no JSON result in {path}; see claude.log next to it")
    if result.get("is_error") or result.get("subtype") != "success" or result.get("structured_output") is None:
        raise LlmError(f"Claude Code failed ({result.get('subtype')}): {str(result.get('result'))[:500]}")
    out = result["structured_output"]
    for item in out.get("nodes", []) + out.get("edges", []):  # answers saved before these fields existed
        item.setdefault("visual_cues", [])
    for n in out.get("nodes", []):
        n.setdefault("relations", [])
    for e in out.get("edges", []):
        e.pop("distance_m", None)
        e.setdefault("basis", "walked")
        e.setdefault("steps", [])
    draft = MapDraft.model_validate(out)
    usage = result.get("modelUsage") or {}
    main_model = max(usage, key=lambda k: usage[k].get("costUSD") or 0) if usage else model
    return draft, LlmRun(backend="claude_code", model=main_model, cost_usd=result.get("total_cost_usd"),
                         session_id=result.get("session_id"))
