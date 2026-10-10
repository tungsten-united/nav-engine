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

from nav.schemas.map import LlmRun, MapDraft, draft_json_schema

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
    walked (`from_prev_m`) and heading change (`turn_deg`, + = right, - = left) since the previous one.
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
- A node is one spot you can stand at, not a whole row or zone. Each view has a dead-reckoned position
  (`views[].motion.x_m`, `y_m`, in metres, drifting a few metres per minute of walking). Two passes more than
  about 15 m apart are two nodes joined by an edge, even if they look alike or the walker says "the same tables
  from another angle" (that means the far end of the same row). Merge passes into one node only when the photos show
  the same spot and the positions roughly agree.
- Connectivity comes from the timelines: consecutive candidates of a walk are joined by what was walked between them.
  Revisits (`possibly_same_as` that the photos confirm, or a voice note such as "the same ... from another angle")
  close loops. Do not invent connections that no walk supports.
- `distance_m` on an edge is the sum of `from_prev_m` between its nodes along a walk, rounded to the metre. Step
  counting misses slow steps, so it is approximate; null if unknown. Use `turn_deg` for left/right, but trust the
  photos where they disagree (the walker may have panned the phone).
- Everything the app will say (`summary`, `description`, `landmarks`, `instruction`, `reverse_instruction`) is for
  someone who cannot see, so it has no visual detail: no colours, signs, screens, murals, lights, "you will see" or
  "looks like". Describe what a cane, feet or hands find (walls and edges you can trail, counters, door frames,
  floor changes from concrete to carpet, a step up), what you hear (echo of a big hall, a humming fridge, the noise
  of people working, a kitchen) and smell, distances and body directions. Sounds and smells are inferred from photos:
  say "likely" where it is a guess. Put the visual details, which help sighted helpers and camera-based localization,
  in `visual_cues` instead.
- Instructions are egocentric (ahead, left, right, behind; "turn so the wall is on your left"), short and concrete:
  direction, roughly how far, what you pass and can touch or hear, and how you know you have arrived. Mention hazards:
  steps, cables on the floor, low tables, glass doors.
- People, bags, laptops and other things that move are not landmarks. Fixed furniture, counters, walls, pillars,
  screens, signs and floor changes are.
- Names are what the app will say: 1-4 words, unique, using the walker's words where possible. Put the details in
  the description.
- Prefer fewer, well-supported nodes over many speculative ones. Put what you are unsure about in `notes`.
- Write every text field in {language_name} (`language`: "{language}"), whatever language the voice notes are in.

## Output
Answer with the map in the required JSON structure. Do not create or edit any files.
"""


def build_prompt(place: str, language: str) -> str:
    return PROMPT.format(place=place, language=language, language_name=LANGUAGES.get(language, language))


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
    prompt = build_prompt(place, language)
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
    for item in out.get("nodes", []) + out.get("edges", []):
        item.setdefault("visual_cues", [])  # answers saved before the field existed
    draft = MapDraft.model_validate(out)
    usage = result.get("modelUsage") or {}
    main_model = max(usage, key=lambda k: usage[k].get("costUSD") or 0) if usage else model
    return draft, LlmRun(backend="claude_code", model=main_model, cost_usd=result.get("total_cost_usd"),
                         session_id=result.get("session_id"))
