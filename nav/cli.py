"""nav: command-line entry points."""

from pathlib import Path

import typer

from nav import config

app = typer.Typer(no_args_is_help=True, add_completion=False)
rec = typer.Typer(no_args_is_help=True, help="Inspect and finalize recordings.")
app.add_typer(rec, name="rec")
maps = typer.Typer(no_args_is_help=True, help="Build, inspect and share the map of each place.")
app.add_typer(maps, name="map")


APPS = {"map-api": "nav.map_api.app:create_app", "nav-api": "nav.nav_api.app:create_nav_app"}


@app.command()
def serve(app_name: str = typer.Option(config.APP, "--app", help="map-api: recordings, map builds, review and the "
                                       "debugging frontend. nav-api: the live API (maps, routes, localization)."),
          host: str = config.HOST, port: int = typer.Option(None, help="Default: NAV_PORT (8450) for map-api, "
                                                            "NAV_NAV_API_PORT (8452) for nav-api."),
          reload: bool = False) -> None:
    """Run one of the two HTTP services (default NAV_APP, else map-api)."""
    import uvicorn

    if app_name not in APPS:
        raise typer.BadParameter(f"--app must be one of {', '.join(APPS)}")
    _logging()
    port = port or (config.NAV_API_PORT if app_name == "nav-api" else config.PORT)
    uvicorn.run(APPS[app_name], factory=True, host=host, port=port, reload=reload,
                proxy_headers=True, forwarded_allow_ips="127.0.0.1")


infer = typer.Typer(no_args_is_help=True, help="Inference host: the local models and the map job over HTTP.")
app.add_typer(infer, name="infer")


@infer.command("serve")
def infer_serve(host: str = config.INFER_HOST, port: int = config.INFER_PORT) -> None:
    """Serve the models and the Claude Code map job to workers (nav/infer/server.py). One process: one copy of the
    models on the GPU."""
    import uvicorn

    _logging()
    uvicorn.run("nav.infer.server:create_infer_app", factory=True, host=host, port=port, workers=1,
                limit_concurrency=32, timeout_keep_alive=30, proxy_headers=True, forwarded_allow_ips="127.0.0.1")


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


@rec.command("import")
def rec_import(video: Path = typer.Argument(..., exists=True, dir_okay=False, help="A video file (mp4, webm, mov ...)."),
               place: str = typer.Option(..., help="Place it shows (groups it with the place's other walks)."),
               notes: str = "", step_length: float = typer.Option(0.7, help="Only used for the time-based guess."),
               holding: str = "handheld_portrait") -> None:
    """Make a recording from a video alone (no phone sensors): the map pipeline takes its heading from the video and
    guesses distances from time. Lands in NAV_DATA_DIR/recordings; share it with `nav rec push`."""
    m = _store().import_video(video, place, notes=notes, step_length_m=step_length, holding=holding)
    print(f"{m.rec_id}: {m.status}  {m.video.duration_s if m.video else '?'} s  place={m.place!r}"
          + (f"  error: {m.error}" if m.error else ""))


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
             raw: bool = typer.Option(False, help="Also download raw/ (the cloud worker keeps the uploads there)."),
             dry_run: bool = False) -> None:
    """Download recordings from the team bucket into NAV_DATA_DIR/recordings. Never deletes local files."""
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    skip_raw = [] if raw else ["--exclude", "(^|/)raw/"]
    if not rec_ids:
        _gcloud_rsync(config.GCS_URL, str(config.RECORDINGS_DIR), *skip_raw, dry_run=dry_run)
    for rec_id in rec_ids or []:
        _gcloud_rsync(f"{config.GCS_URL}/{rec_id}", str(config.RECORDINGS_DIR / rec_id), *skip_raw, dry_run=dry_run)


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
                       transcribe_language=config.TRANSCRIBE_LANGUAGE,
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
    from nav.mapping.build import build_map_of, write_map
    from nav.mapping.quality import check_map

    _logging()
    build_dir, m = build_map_of(_map_store().dir(map_id))
    m = check_map(m, config.QUALITY_MODEL)
    write_map(build_dir, m)
    q = m.quality
    if q.skipped:
        print(f"skipped: {q.skipped}")
    else:
        print(f"{q.flagged} of {len(m.nodes) + len(m.edges)} items flagged ({q.model}, {q.input_tokens} input tokens)")
    for w in m.warnings:
        if w.startswith("jev: "):
            print(f"  {w}")


@maps.command("review")
def map_review(map_id: str,
               apply: Path = typer.Option(None, exists=True, dir_okay=False, help="An edited review file to apply."),
               out: Path = typer.Option(None, help="Where to write the review file (default ./<map_id>-review.json).")
               ) -> None:
    """Human review: without --apply, write an editable review file (every node, edge and relation with what to
    check); with --apply, save it as the place's review.json and republish the map (no LLM, no recomputation)."""
    import json

    from nav.mapping.build import build_map_of, republish
    from nav.mapping.review import dumps, from_template, load_review, save_review, template

    map_dir = _map_store().dir(map_id)
    if apply:
        save_review(map_dir, from_template(json.loads(apply.read_text())), by="cli")
        m = republish(map_dir)
        count = lambda items, attr="status": sum(getattr(i, attr) == "verified" for i in items)
        print(f"published {map_id}: verified {count(m.nodes)}/{len(m.nodes)} nodes, {count(m.edges)}/{len(m.edges)} "
              f"edges ({count(m.edges, 'reverse_status')} the way back), hidden {len(m.hidden)}")
        for w in m.warnings:
            if w.startswith("review: "):
                print(f"  {w}")
        return
    _, raw = build_map_of(map_dir)
    out = out or Path(f"{map_id}-review.json")
    out.write_text(dumps(template(raw, load_review(map_dir))))
    print(f"wrote {out}; edit it, then: nav map review {map_id} --apply {out}")


@maps.command("clean")
def map_clean(map_id: str, apply: bool = typer.Option(False, help="Republish the map with the cleaner on.")) -> None:
    """What the cleaner (nav/mapping/clean.py) leaves out of the published map, and why: only the nodes and edges
    the walks back up are published. Without --apply nothing is written."""
    from nav.mapping.build import build_map_of, published_map, republish

    map_dir = _map_store().dir(map_id)
    _, raw = build_map_of(map_dir)
    m = republish(map_dir) if apply else published_map(map_dir, raw, clean=True)
    names = {n.id: n.name for n in raw.nodes}
    c = m.clean
    print(f"{'published' if apply else 'would publish'} {map_id}: {c.nodes}/{c.nodes_before} nodes, "
          f"{c.edges}/{c.edges_before} edges")
    for i, why in m.pruned.items():
        print(f"  {i:5} {names.get(i, '')}{' - ' if i in names else ''}{why}")


@maps.command("republish")
def map_republish(map_id: str) -> None:
    """Publish the published build again with the current review and cleaner (no LLM, no recomputation)."""
    from nav.mapping.build import republish

    m = republish(_map_store().dir(map_id))
    print(f"published {map_id}: {len(m.nodes)} nodes, {len(m.edges)} edges"
          + (f" (cleaned from {m.clean.nodes_before} / {m.clean.edges_before})" if m.clean else ""))


@maps.command("measure")
def map_measure(map_id: str, start: str, goal: str, meters: float,
                sigma: float = typer.Option(0.5, help="How sure the measurement is, metres."),
                note: str = typer.Option(None, help="How it was measured (tape, pacing, floor plan).")) -> None:
    """Record a distance measured on site between two nodes (ids or names) in the review and republish: one scale
    factor fitted to all measurements rescales every distance, position and path of the map."""
    from nav.mapping.build import build_map_of, republish
    from nav.mapping.review import MeasuredDistance, load_review, save_review

    map_dir = _map_store().dir(map_id)
    _, raw = build_map_of(map_dir)
    ids = {n.name.casefold(): n.id for n in raw.nodes} | {n.id: n.id for n in raw.nodes}
    a, b = ids.get(start.casefold(), start), ids.get(goal.casefold(), goal)
    review = load_review(map_dir)
    review.distances = [d for d in review.distances if set(d.between) != {a, b}]
    review.distances.append(MeasuredDistance(between=(a, b), meters=meters, sigma_m=sigma, note=note))
    save_review(map_dir, review, by="cli")
    m = republish(map_dir)
    names = {n.id: n.name for n in m.nodes}
    print(f"published {map_id}: distance scale x{m.frame.distance_scale if m.frame else 1:.3f}")
    for c in (m.frame.measured_distances if m.frame else []):
        got = "not connected" if c.map_m is None else f"map {c.map_m:.2f} m, residual {c.residual_m:+.2f} m"
        print(f"  {names.get(c.between[0], c.between[0])} - {names.get(c.between[1], c.between[1])}: "
              f"measured {c.meters} m (±{c.sigma_m}), {got}")


@maps.command("route")
def map_route(map_id: str, start: str, goal: str,
              trust: str = typer.Option("verified", help="verified (real users), observed or any.")) -> None:
    """Shortest route between two nodes (ids or names) over edges trusted enough."""
    from nav.navigation.route import plan_route

    m = _map_store().get(map_id)
    ids = {n.name.casefold(): n.id for n in m.nodes} | {n.id: n.id for n in m.nodes}
    names = {n.id: n.name for n in m.nodes}
    a, b = ids.get(start.casefold(), start), ids.get(goal.casefold(), goal)
    route = plan_route(m, a, b, trust)
    if route is None:
        print(f"no route from {start} to {goal} over {trust} edges")
        raise typer.Exit(1)
    for h in route:
        dist = f"{h.length_m:.0f} m" if h.length_m is not None else "? m"
        head = f"{h.bearing_deg:.0f}°" if h.bearing_deg is not None else "?"
        print(f"{h.edge:>5} {'' if h.forward else '(back) '}{names[h.source]} -> {names[h.target]}  {dist}  heading "
              f"{head}  [{h.status}]\n      {h.instruction}")


@maps.command("pdr-replay")
def map_pdr_replay(map_id: str, rec_id: str,
                   step_scale: float = typer.Option(1.0, help="0.85 = a user with 15% shorter steps."),
                   heading_bias: float = typer.Option(0.0, help="Compass bias of the user's phone, degrees."),
                   step_length: float = typer.Option(None, help="The tracker's starting step length.")) -> None:
    """Replay a mapping walk as a user along the map: how early or late the dead-reckoning tracker would announce
    each node (nav/navigation/pdr.py)."""
    from nav.navigation.pdr import replay

    rows = replay(_map_store().get(map_id), config.RECORDINGS_DIR / rec_id, step_scale, heading_bias, step_length)
    if not rows:
        print(f"{rec_id} doesn't walk any edge of {map_id}")
        raise typer.Exit(1)
    print(f"{'edge':>5} {'to':<22} {'len':>6} {'steps':>5} {'left':>7} {'sigma':>6} {'P':>5} {'alert':>6} {'step':>6}")
    for r in rows:
        alert = f"{r.first_alert_s:+.1f}" if r.first_alert_s is not None else "-"
        length = f"{r.length_m:.1f}" if r.length_m is not None else "-"
        print(f"{r.edge:>5} {r.to[:22]:<22} {length:>6} {r.steps:>5} {r.predicted_remaining_m:>7.2f} "
              f"{r.sigma_m:>6.2f} {r.arrival_probability:>5.2f} {alert:>6} {r.step_length_after:>6.3f}")


@maps.command("nav-replay")
def map_nav_replay(map_id: str, rec_id: str,
                   step_scale: float = typer.Option(1.0, help="0.85 = a user with 15% shorter steps."),
                   heading_bias: float = typer.Option(0.0, help="Compass bias of the user's phone, degrees."),
                   burst: int = typer.Option(1, help="Frames per /localize call, as the app sends them."),
                   every: float = typer.Option(1.0, help="... every this many seconds."),
                   k: int = typer.Option(None, help="Votes needed (ArrivalParams.k)."),
                   n: int = typer.Option(None, help="... among the last n localizations (ArrivalParams.n)."),
                   heading_tol: float = typer.Option(None, help="Heading tolerance when passing, degrees."),
                   score: float = typer.Option(None, help="A vote needs this score (ArrivalParams.vote_score) ..."),
                   margin: float = typer.Option(None, help="... and this lead (ArrivalParams.vote_margin)."),
                   confirm_score: float = typer.Option(None, help="Override NAV_LOC_CONFIRM_SCORE (nav-api's status)."),
                   confirm_margin: float = typer.Option(None, help="Override NAV_LOC_CONFIRM_MARGIN.")) -> None:
    """Replay a mapping walk as a user navigating its route with the whole live loop: steps -> dead reckoning,
    frames -> localization against the map's references (frames near the walk's own references left out), and the
    node confirmer (nav/navigation/arrival.py). Reports how each node got confirmed and how late."""
    from dataclasses import replace

    from nav.navigation.arrival import ArrivalParams
    from nav.navigation.replay import nav_replay
    from nav.schemas.navigation import Thresholds

    _logging()
    store = _map_store()
    params = ArrivalParams()
    params = replace(params, **{f: v for f, v in (("k", k), ("n", n), ("heading_tol_deg", heading_tol),
                                                  ("vote_score", score), ("vote_margin", margin)) if v is not None})
    th = Thresholds(confirm_score=confirm_score if confirm_score is not None else config.LOC_CONFIRM_SCORE,
                    confirm_margin=confirm_margin if confirm_margin is not None else config.LOC_CONFIRM_MARGIN,
                    lost_score=config.LOC_LOST_SCORE)
    res = nav_replay(store.get(map_id), store.dir(map_id), config.RECORDINGS_DIR / rec_id, config.CACHE_DIR, th,
                     params, step_scale, heading_bias, burst=burst, every_s=every)
    if res is None:
        print(f"{rec_id} doesn't walk any edge of {map_id}, or the map has no reference embeddings")
        raise typer.Exit(1)
    print(f"frames {res.frames_used} (+{res.frames_excluded} left out near the walk's own references) · "
          f"bursts of {burst} every {every:g} s · k={params.k}/n={params.n} · vote: score>={params.vote_score} "
          f"lead>={params.vote_margin}")
    print(f"{'#':>2} {'to':<24} {'len':>5} {'reached':>8} {'how':<8} {'delay':>6} {'unc':>4} {'off':>4} "
          f"{'score':>6} {'step':>6}")
    for r in res.rows:
        length = f"{r.length_m:.1f}" if r.length_m is not None else "-"
        delay = f"{r.delay_s:+.1f}" if r.delay_s is not None else "-"
        score = f"{r.best_score:.2f}" if r.best_score is not None else "-"
        print(f"{r.hop_index + 1:>2} {r.name[:24]:<24} {length:>5} {r.arrival_s:>7.1f}s {r.kind:<8} {delay:>6} "
              f"{r.uncertain:>4} {r.off_route:>4} {score:>6} {r.step_length_after:>6.3f}")
    print(f"confirmed by the camera: {res.confirmed}/{len(res.rows)} · a person had to confirm: "
          f"{sum(r.kind == 'manual' for r in res.rows)} · premature (false): {res.premature} · "
          f"off-route alarms: {sum(r.off_route for r in res.rows)}")


@maps.command("orchestrator-replay")
def map_orchestrator_replay(map_id: str, rec_id: str,
                            motion: bool = typer.Option(True, help="Send the phone's motion with each frame."),
                            burst: int = typer.Option(1, help="New frames per /localize call."),
                            k: int = typer.Option(3, help="Votes needed ..."),
                            n: int = typer.Option(4, help="... among the last n calls."),
                            margin: float = typer.Option(0.04, help="A result votes when its best node leads by this."),
                            lost_margin: float = typer.Option(0.06, help="Locating, a lost result votes when it leads "
                                                              "by this (the orchestrator's NAV_LOCATE_LOST_MARGIN)."),
                            lost_s: float = typer.Option(6.0, help="Seconds with no result pointing at the hop "
                                                         "(lost, or another node leading) that start over."),
                            fps: float = typer.Option(1.5, help="Frames the phone sends per second."),
                            step_scale: float = typer.Option(1.0, help="0.85 = a user with 15% shorter steps.")) -> None:
    """Replay a mapping walk as a user the Orient orchestrator guides to the walk's last node: localize, one route,
    then along it, each /localize result a vote on reaching a node ahead (the hop's target, or one further on);
    lost, start over. Reports what the
    loop decided against where the walk was, and when each node of the walk was reached."""
    from nav.navigation.replay import LoopParams, orchestrator_replay

    _logging()
    store = _map_store()
    params = LoopParams(burst=burst, k=k, n=n, margin=margin, lost_margin=lost_margin, lost_s=lost_s)
    res = orchestrator_replay(store.get(map_id), store.dir(map_id), config.RECORDINGS_DIR / rec_id, config.CACHE_DIR,
                              config.loc_thresholds(), params=params, use_motion=motion, step_scale=step_scale, fps=fps)
    if res is None:
        print(f"{rec_id} doesn't walk any edge of {map_id}, or the map has no reference embeddings")
        raise typer.Exit(1)
    print(f"motion {'on' if motion else 'off'} · {params} · {res.evaluations} localizations, goal {res.goal}")
    for e in res.events:
        where = ', '.join(e['truth']) or '?'
        mark = f"  WRONG: the walk was at {where}" if not e["ok"] else f"  (late: the walk was at {where})" \
            if e["late"] else ""
        print(f"  {e['t_s']:>6.1f}s {e['kind']:<8} {e['name'] or '-'}{mark}")
    for x in res.nodes:
        how = "-" if x["delay_s"] is None else f"{x['delay_s']:+.1f} s"
        print(f"  {x['node']:>4} {x['name'][:24]:<24} walk there {x['arrival_s']:>6.1f}s  reached {how}")
    delays = sorted(x["delay_s"] for x in res.nodes if x["delay_s"] is not None)
    median = f"{delays[len(delays) // 2]:+.1f} s" if delays else "-"
    print(f"nodes reached: {res.found}/{len(res.nodes)} · wrong: {len(res.wrong)} · relocations: {res.relocations} "
          f"· median delay {median}")


@maps.command("schema")
def map_schema(out: Path = typer.Option(None, help="Write here instead of printing.")) -> None:
    """JSON Schema of the published map (PlaceMap), for nav-api's callers (committed as docs/map.schema.json)."""
    from nav.schemas.map import place_map_json_schema

    text = place_map_json_schema()
    if out:
        out.write_text(text)
    else:
        print(text, end="")


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
