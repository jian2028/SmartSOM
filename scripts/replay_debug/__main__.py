"""CLI: ``python -m scripts.replay_debug <command> ...`` (run from the repo root).

Commands:
  mock      generate a synthetic trace with the mock factory (rule or marl controller)
  validate  check a trace against smartsom.somarl.replay.v1
  metrics   print the slide-4 episode metrics for one or more traces
  render    embed a trace into viewer.html -> one self-contained HTML file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .metrics import compute_metrics
from .mock_episode import generate_episode
from .render import render_html
from .schema import load_trace, validate_trace, write_trace

TABLE_KEYS = (
    ("total_throughput", "Total thru."),
    ("passed_throughput", "Passed"),
    ("defect_throughput", "Defect"),
    ("passing_rate", "Passing rate"),
    ("total_agv_conflicts", "AGV conflicts"),
    ("movement_conflicts", "Movement"),
    ("pickup_dropoff_conflicts", "Pick/drop"),
    ("loaded_path_ratio", "Loaded-path ratio"),
)


def _default_out_root() -> Path:
    return Path("runs") / "replay_debug"


def _fmt(key: str, value: object) -> str:
    if value is None:
        return "-"
    if key == "passing_rate":
        return f"{100 * float(value):.1f}%"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def cmd_mock(args: argparse.Namespace) -> int:
    manifest, frames = generate_episode(
        args.controller,
        args.seed,
        args.horizon,
        args.jobs,
        args.release_every,
        args.agvs,
    )
    out = (
        Path(args.out)
        if args.out
        else _default_out_root() / f"mock_{args.controller}_seed{args.seed}"
    )
    write_trace(out, manifest, frames)
    print(f"wrote {len(frames)} frames -> {out}")
    if args.html:
        html = render_html(manifest, frames, out / "replay.html")
        print(f"viewer -> {html}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    manifest, frames = load_trace(args.trace)
    errors = validate_trace(manifest, frames)
    if errors:
        print(f"{len(errors)} problem(s) in {args.trace}:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"ok: {args.trace} ({len(frames)} frames)")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    rows = []
    for trace in args.traces:
        manifest, frames = load_trace(trace)
        rows.append(
            (
                manifest.get("controller", "?"),
                manifest.get("trace_id", trace),
                compute_metrics(manifest, frames),
            )
        )
    if args.json:
        print(
            json.dumps(
                [{"controller": c, "trace_id": t, **m} for c, t, m in rows],
                indent=2,
                default=str,
            )
        )
        return 0
    header = ["controller", "trace"] + [label for _, label in TABLE_KEYS]
    body = [[c, t] + [_fmt(k, m[k]) for k, _ in TABLE_KEYS] for c, t, m in rows]
    widths = [max(len(str(r[i])) for r in [header] + body) for i in range(len(header))]
    for r in [header] + body:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    manifest, frames = load_trace(args.trace)
    errors = validate_trace(manifest, frames)
    if errors:
        print(
            f"warning: {len(errors)} schema problem(s); first: {errors[0]}",
            file=sys.stderr,
        )
    trace = Path(args.trace)
    out = (
        Path(args.out)
        if args.out
        else (trace / "replay.html" if trace.is_dir() else trace.with_suffix(".html"))
    )
    print(f"viewer -> {render_html(manifest, frames, out)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.replay_debug",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("mock", help="generate a synthetic trace")
    p.add_argument("--controller", choices=["rule", "marl"], default="rule")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=600)
    p.add_argument("--jobs", type=int, default=60)
    p.add_argument("--release-every", type=int, default=10)
    p.add_argument("--agvs", type=int, default=4)
    p.add_argument("--out", help="trace directory (default runs/replay_debug/...)")
    p.add_argument(
        "--html", action="store_true", help="also write a self-contained replay.html"
    )
    p.set_defaults(func=cmd_mock)

    p = sub.add_parser("validate", help="validate a trace")
    p.add_argument("trace")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("metrics", help="episode metrics table")
    p.add_argument("traces", nargs="+")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("render", help="embed a trace into a standalone HTML viewer")
    p.add_argument("trace")
    p.add_argument("-o", "--out")
    p.set_defaults(func=cmd_render)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
