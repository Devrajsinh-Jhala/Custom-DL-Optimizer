from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .cache import PlanCache
from .research import export_paper_artifacts
from .runtime import inspect_runtime


def _version() -> str:
    try:
        return version("custom-dl-optimizer")
    except PackageNotFoundError:
        return "source"


def _number(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _report_summary(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"workload: {payload.get('workload_name', 'workload')}")
    print(f"selected: {payload.get('selected_plan', '-')}")
    print(f"basis: {payload.get('selection_basis', '-')}")
    print(f"baseline: {payload.get('baseline_plan', '-') or '-'}")
    print(f"confidence: {float(payload.get('confidence_level', 0.95)):.1%}")
    print(f"confidence_gate_passed: {bool(payload.get('confidence_gate_passed', False))}")
    print(f"cache_hit: {bool(payload.get('cache_hit', False))}")
    print(
        "candidate                 median ms    p99 ms  cost CI low cost CI high  parity"
    )
    for candidate in payload.get("candidates", []):
        print(
            f"{candidate.get('name', '-')[:24]:24} "
            f"{_number(candidate.get('latency_ms')):>10} "
            f"{_number(candidate.get('latency_p99_ms')):>9} "
            f"{_number(candidate.get('selection_cost_ci_low_ms')):>11} "
            f"{_number(candidate.get('selection_cost_ci_high_ms')):>12} "
            f"{str(bool(candidate.get('parity', False))):>7}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="custom-dl-optimizer",
        description="Inspect runtimes, reports, and persistent optimization decisions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version", help="Print the installed package version.")

    inspect_parser = subparsers.add_parser("inspect", help="Print runtime capabilities.")
    inspect_parser.add_argument("--device", default=None)

    report_parser = subparsers.add_parser("report", help="Summarize a JSON report.")
    report_parser.add_argument("path", type=Path)

    export_parser = subparsers.add_parser(
        "paper-export",
        help="Export CSV, LaTeX, and figure artifacts from JSON reports.",
    )
    export_parser.add_argument("reports", type=Path, nargs="+")
    export_parser.add_argument("--output-dir", type=Path, required=True)
    export_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib figure generation.",
    )

    cache_parser = subparsers.add_parser("cache", help="Inspect or clear plan-cache records.")
    cache_parser.add_argument("action", choices=("list", "clear"))
    cache_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "custom-dl-optimizer",
    )
    cache_parser.add_argument("--yes", action="store_true", help="Confirm cache deletion.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "version":
        print(_version())
        return 0
    if args.command == "inspect":
        print(json.dumps(inspect_runtime(args.device).as_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "report":
        _report_summary(args.path)
        return 0
    if args.command == "paper-export":
        outputs = export_paper_artifacts(
            args.reports,
            args.output_dir,
            plots=not args.no_plots,
        )
        for output in outputs:
            print(output)
        return 0
    cache = PlanCache(args.cache_dir)
    if args.action == "list":
        records = cache.records()
        if not records:
            print("No cached plans.")
            return 0
        for record in records:
            print(
                f"{record.key[:12]}  {record.selected_plan:24} "
                f"{record.latency_ms:.3f} ms  {record.created_at}"
            )
        return 0
    if not args.yes:
        raise SystemExit("Refusing to clear the cache without --yes")
    print(f"Removed {cache.clear()} cached plan record(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
