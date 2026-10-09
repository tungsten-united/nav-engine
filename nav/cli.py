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


if __name__ == "__main__":
    app()
