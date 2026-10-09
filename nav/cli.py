"""nav: command-line entry points."""

import typer

from nav import config

app = typer.Typer(no_args_is_help=True, add_completion=False)
rec = typer.Typer(no_args_is_help=True, help="Inspect and finalize recordings.")
app.add_typer(rec, name="rec")


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


if __name__ == "__main__":
    app()
