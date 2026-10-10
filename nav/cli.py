"""nav: command-line entry points."""

import typer

from nav import config

app = typer.Typer(no_args_is_help=True, add_completion=False)
rec = typer.Typer(no_args_is_help=True, help="Inspect and finalize recordings.")
app.add_typer(rec, name="rec")
maps = typer.Typer(no_args_is_help=True, help="Build, inspect and share the map of each place.")
app.add_typer(maps, name="map")


@app.command()
def serve(host: str = config.HOST, port: int = config.PORT, reload: bool = False) -> None:
    """Run the recorder (phone web app + /api/v1)."""
    import uvicorn

    uvicorn.run("nav.recorder.app:create_app", factory=True, host=host, port=port, reload=reload,
                proxy_headers=True, forwarded_allow_ips="127.0.0.1")


def _store():
    from nav.recorder.store import RecordingStore

    return RecordingStore(config.RECORDINGS_DIR)


@rec.command("list")
def rec_list() -> None:
    for s in _store().summaries():
        dur = f"{s.duration_ms / 1000:6.0f}s" if s.duration_ms else "      -"
        print(f"{s.rec_id}  {s.status:<10} {dur}  marks={s.counts.marks:<3} {s.place}")


@rec.command("show")
def rec_show(rec_id: str) -> None:
    print(_store().get(rec_id).model_dump_json(indent=2))


@rec.command("finalize")
def rec_finalize(rec_id: str) -> None:
    m = _store().finalize(rec_id)
    print(f"{m.rec_id}: {m.status}" + (f"  gaps: {', '.join(m.gaps)}" if m.gaps else "") + (f"  error: {m.error}" if m.error else ""))


def _gcloud_rsync(src: str, dst: str, *flags: str, dry_run: bool) -> None:
    import subprocess

    cmd = ["gcloud", "storage", "rsync", "--recursive", *flags, *(["--dry-run"] if dry_run else []), src, dst]
    print("$", " ".join(cmd))
    if subprocess.run(cmd).returncode:
        raise typer.Exit(1)


@rec.command("push")
def rec_push(rec_ids: list[str] = typer.Argument(None, help="Default: every finished recording."),
             raw: bool = typer.Option(False, help="Also upload raw/ (the uploads as received)."),
             dry_run: bool = False) -> None:
    """Upload recordings to the team bucket (NAV_GCS_URL). Only adds or updates objects, never deletes."""
    store = _store()
    ids = rec_ids or [s.rec_id for s in store.summaries() if s.status in ("complete", "incomplete", "failed")]
    for rec_id in ids:
        _gcloud_rsync(str(store.dir(rec_id)), f"{config.GCS_URL}/{rec_id}", *([] if raw else ["--exclude", "^raw/"]),
                      dry_run=dry_run)


@rec.command("pull")
def rec_pull(rec_ids: list[str] = typer.Argument(None, help="Default: everything in the bucket."),
             dry_run: bool = False) -> None:
    """Download recordings from the team bucket into NAV_DATA_DIR/recordings. Never deletes local files."""
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if not rec_ids:
        _gcloud_rsync(config.GCS_URL, str(config.RECORDINGS_DIR), dry_run=dry_run)
    for rec_id in rec_ids or []:
        _gcloud_rsync(f"{config.GCS_URL}/{rec_id}", str(config.RECORDINGS_DIR / rec_id), dry_run=dry_run)


def _logging() -> None:
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("nav").setLevel(logging.INFO)


def _map_store():
    from nav.mapping.store import MapStore

    return MapStore(config.MAPS_DIR)


def _print_build(result) -> None:
    c = result.candidates
    print(f"build: {result.build_dir}")
    for r in c.recordings:
        print(f"  {r.rec_id}  walked {r.walked_m:5.0f} m  frames {r.frames_usable}/{r.frames_sampled} usable  "
              f"steps on timeline {len(r.timeline)}  voice {len(r.voice)}  marks {len(r.marks)}")
    print(f"  {len(c.candidates)} candidates")
    if m := result.map:
        llm = m.build.llm
        print(f"map: {len(m.nodes)} nodes, {len(m.edges)} edges  ({llm.model} {llm.effort or ''}, {llm.duration_s} s, "
              f"${llm.cost_usd or 0:.2f} at API list price)")
        for w in m.warnings:
            print(f"  warning: {w}")
        if m.quality and m.quality.skipped:
            print(f"  quality check skipped: {m.quality.skipped}")
        print(f"view: /viz/#{m.map_id}")


@maps.command("setup")
def map_setup(transcribe: bool = True) -> None:
    """Download and check the local models: frame and place embeddings (NAV_EMBED_MODEL, NAV_PLACE_MODEL), local
    feature matching (ALIKED + LightGlue) and voice (NAV_TRANSCRIBE_MODEL)."""
    from nav.mapping.models import check

    _logging()
    for k, v in check(config.EMBED_MODEL, config.PLACE_MODEL, config.TRANSCRIBE_MODEL if transcribe else None).items():
        print(f"{k:>16}: {v}")


@maps.command("build")
def map_build(place: str = typer.Argument(..., help="Place name as recorded (case-insensitive)."),
              llm: bool = typer.Option(True, help="Run the LLM step; --no-llm stops after selecting candidates."),
              model: str = typer.Option(config.MAP_LLM_MODEL, help="Claude Code model (alias or id)."),
              effort: str = typer.Option(config.MAP_LLM_EFFORT, help="Claude Code effort: low, medium, high, xhigh, max."),
              language: str = typer.Option(config.MAP_LANGUAGE, help="Language of the map's text."),
              budget: float = typer.Option(config.MAP_LLM_BUDGET_USD, help="Max USD for the Claude Code job."),
              max_candidates: int = typer.Option(60, help="Most frames shown to the LLM."),
              transcribe: bool = typer.Option(True, help="Transcribe voice notes.")) -> None:
    """Build (or rebuild) the map of a place from all its finished recordings."""
    from nav.mapping.build import build_map
    from nav.mapping.select import SelectParams

    _logging()
    result = build_map(place, recordings_root=config.RECORDINGS_DIR, maps_root=config.MAPS_DIR,
                       cache_root=config.CACHE_DIR, params=SelectParams(max_candidates=max_candidates),
                       embed_model=config.EMBED_MODEL, place_model=config.PLACE_MODEL,
                       transcribe_model=config.TRANSCRIBE_MODEL if transcribe else None,
                       llm_model=model, language=language, budget_usd=budget, run_llm=llm, effort=effort)
    _print_build(result)


@maps.command("llm")
def map_llm(map_id: str, build_id: str = typer.Argument(None, help="Default: the latest build."),
            model: str = typer.Option(config.MAP_LLM_MODEL), effort: str = typer.Option(config.MAP_LLM_EFFORT),
            language: str = typer.Option(config.MAP_LANGUAGE),
            budget: float = typer.Option(config.MAP_LLM_BUDGET_USD)) -> None:
    """Rerun only the LLM step with the candidates of an existing build, e.g. after changing the prompt. The result
    is a new build (candidates copied), so earlier answers are kept."""
    from nav.mapping.build import fork_build, run_llm_on_build

    _logging()
    builds = _map_store().dir(map_id) / "builds"
    build_dir = fork_build(builds / build_id if build_id else max(builds.glob("*/candidates.json")).parent)
    _print_build(run_llm_on_build(build_dir, llm_model=model, language=language, budget_usd=budget, effort=effort))


@maps.command("refinish")
def map_refinish(map_id: str, build_id: str = typer.Argument(None, help="Default: the latest build with an LLM result.")) -> None:
    """Redo post-processing from the saved LLM answer (no new LLM call) and publish that build's map."""
    from nav.mapping.build import refinish_build

    _logging()
    builds = _map_store().dir(map_id) / "builds"
    build_dir = builds / build_id if build_id else max(builds.glob("*/job/claude.json")).parent.parent
    _print_build(refinish_build(build_dir))


@maps.command("check")
def map_check(map_id: str) -> None:
    """Rerun the cheap quality check (TypeSafe Jev) on the published map and save the result."""
    from nav.mapping.build import write_map
    from nav.mapping.quality import check_map

    _logging()
    store = _map_store()
    m = check_map(store.get(map_id), config.QUALITY_MODEL)
    write_map(store.dir(map_id) / "builds" / m.build.build_id, m)
    q = m.quality
    if q.skipped:
        print(f"skipped: {q.skipped}")
    else:
        print(f"{q.flagged} of {len(m.nodes) + len(m.edges)} items flagged ({q.model}, {q.input_tokens} input tokens)")
    for w in m.warnings:
        if w.startswith("jev: "):
            print(f"  {w}")


@maps.command("list")
def map_list() -> None:
    for s in _map_store().summaries():
        print(f"{s.map_id:<24} {s.built_at_utc:%Y-%m-%d %H:%M}  recs={s.recordings:<3} nodes={s.nodes:<3} "
              f"edges={s.edges:<3} {s.place}")


@maps.command("show")
def map_show(map_id: str) -> None:
    print(_map_store().get(map_id).model_dump_json(indent=2))


@maps.command("push")
def map_push(map_ids: list[str] = typer.Argument(None, help="Default: every map."), dry_run: bool = False) -> None:
    """Upload maps to the team bucket (NAV_GCS_MAPS_URL, next to recordings/). Never deletes."""
    store = _map_store()
    for map_id in map_ids or [s.map_id for s in store.summaries()]:
        _gcloud_rsync(str(store.dir(map_id)), f"{config.GCS_MAPS_URL}/{map_id}", "--exclude", r"(^|/)\.",
                      dry_run=dry_run)


@maps.command("pull")
def map_pull(map_ids: list[str] = typer.Argument(None, help="Default: everything in the bucket."),
             dry_run: bool = False) -> None:
    """Download maps from the team bucket into NAV_DATA_DIR/maps. Never deletes local files."""
    config.MAPS_DIR.mkdir(parents=True, exist_ok=True)
    if not map_ids:
        _gcloud_rsync(config.GCS_MAPS_URL, str(config.MAPS_DIR), dry_run=dry_run)
    for map_id in map_ids or []:
        _gcloud_rsync(f"{config.GCS_MAPS_URL}/{map_id}", str(config.MAPS_DIR / map_id), dry_run=dry_run)


if __name__ == "__main__":
    app()
