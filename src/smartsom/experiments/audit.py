"""Read-only execution audit for the sole grid core and its experiment records."""

from pathlib import Path


def audit_run(directory: str | Path) -> dict:
    """Replay semantic grid commands; historical matrix logs need their source."""
    from smartsom.trace.production import audit

    path = Path(directory).expanduser().resolve()
    if not (path / "run.json").is_file():
        raise ValueError(
            "historical matrix evidence requires its recorded source checkout for replay"
        )
    return audit(path)
