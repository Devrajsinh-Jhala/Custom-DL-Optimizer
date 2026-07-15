from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError(f"{path} is not a Custom-DL-Optimizer report")
    return payload


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _candidate_rows(source: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "source_report": source.name,
        "workload": report.get("workload_name", "workload"),
        "device": report.get("device", ""),
        "selected_plan": report.get("selected_plan", ""),
        "selection_basis": report.get("selection_basis", ""),
        "expected_calls": report.get("expected_calls"),
        "cache_hit": bool(report.get("cache_hit", False)),
    }
    rows: list[dict[str, Any]] = []
    for candidate in report["candidates"]:
        rows.append(
            {
                **common,
                "candidate": candidate.get("name", ""),
                "selected": bool(candidate.get("selected", False)),
                "parity": bool(candidate.get("parity", False)),
                "median_ms": candidate.get("latency_ms"),
                "mean_ms": candidate.get("latency_mean_ms"),
                "p95_ms": candidate.get("latency_p95_ms"),
                "p99_ms": candidate.get("latency_p99_ms"),
                "ci95_low_ms": candidate.get("latency_ci95_low_ms"),
                "ci95_high_ms": candidate.get("latency_ci95_high_ms"),
                "setup_time_s": candidate.get("setup_time_s"),
                "first_call_time_s": candidate.get("first_call_time_s"),
                "projected_total_ms": candidate.get("projected_total_ms"),
                "peak_memory_mb": candidate.get("peak_memory_mb"),
                "calls_per_second": candidate.get("calls_per_second"),
                "speedup_vs_eager": candidate.get("speedup_vs_eager"),
                "speedup_vs_native": candidate.get("speedup_vs_native"),
                "break_even_calls_vs_baseline": candidate.get(
                    "break_even_calls_vs_baseline"
                ),
                "constraint_violations": ";".join(
                    candidate.get("constraint_violations", [])
                ),
                "error": candidate.get("error", ""),
                "raw_samples_json": json.dumps(candidate.get("latency_samples_ms", [])),
            }
        )
    return rows


def _case_rows(source: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in report["candidates"]:
        for case in candidate.get("workload_cases", []):
            rows.append(
                {
                    "source_report": source.name,
                    "workload": report.get("workload_name", "workload"),
                    "device": report.get("device", ""),
                    "candidate": candidate.get("name", ""),
                    "selected": bool(candidate.get("selected", False)),
                    "case": case.get("name", ""),
                    "weight": case.get("weight"),
                    "input_signature": case.get("input_signature", ""),
                    "parity": bool(case.get("parity", False)),
                    "median_ms": case.get("latency_ms"),
                    "mean_ms": case.get("latency_mean_ms"),
                    "p95_ms": case.get("latency_p95_ms"),
                    "p99_ms": case.get("latency_p99_ms"),
                    "ci95_low_ms": case.get("latency_ci95_low_ms"),
                    "ci95_high_ms": case.get("latency_ci95_high_ms"),
                    "first_call_time_s": case.get("first_call_time_s"),
                    "peak_memory_mb": case.get("peak_memory_mb"),
                    "max_abs_error": case.get("max_abs_error"),
                    "mean_abs_error": case.get("mean_abs_error"),
                    "error": case.get("error", ""),
                    "raw_samples_json": json.dumps(case.get("latency_samples_ms", [])),
                }
            )
    return rows


def _tex(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _format_number(value: Any) -> str:
    return "--" if value is None else f"{float(value):.3f}"


def _write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Workload & Candidate & Median (ms) & P95 (ms) & P99 (ms) & Speedup & Parity \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                (
                    _tex(row["workload"]),
                    _tex(row["candidate"]),
                    _format_number(row["median_ms"]),
                    _format_number(row["p95_ms"]),
                    _format_number(row["p99_ms"]),
                    _format_number(row["speedup_vs_eager"]),
                    "True" if row["parity"] else "False",
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(directory: Path, rows: list[dict[str, Any]]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as error:
        raise RuntimeError(
            "Plot export requires the 'research' extra: "
            "pip install 'custom-dl-optimizer[research]'"
        ) from error

    valid = [row for row in rows if row["median_ms"] is not None]
    if not valid:
        return []
    outputs: list[Path] = []
    for key, ylabel, filename in (
        ("median_ms", "Median serial latency (ms)", "paper_median_latency.png"),
        ("p99_ms", "P99 serial latency (ms)", "paper_p99_latency.png"),
    ):
        plot_rows = [row for row in valid if row[key] is not None]
        if not plot_rows:
            continue
        figure_width = max(8.0, len(plot_rows) * 1.25)
        figure, axis = plt.subplots(figsize=(figure_width, 5.2))
        plot_labels = [f"{row['workload']}\n{row['candidate']}" for row in plot_rows]
        plot_colors = ["#087f5b" if row["selected"] else "#4c6ef5" for row in plot_rows]
        bars = axis.bar(plot_labels, [row[key] for row in plot_rows], color=plot_colors)
        axis.set_ylabel(ylabel)
        axis.set_title("Custom-DL-Optimizer candidate evidence")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelrotation=25)
        axis.legend(
            handles=(
                Patch(facecolor="#087f5b", label="Selected plan"),
                Patch(facecolor="#4c6ef5", label="Other valid candidate"),
            ),
            frameon=False,
        )
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        figure.tight_layout()
        output = directory / filename
        figure.savefig(output, dpi=220, bbox_inches="tight")
        plt.close(figure)
        outputs.append(output)
    return outputs


def export_paper_artifacts(
    report_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    plots: bool = True,
) -> list[Path]:
    """Export auditable paper tables and optional figures from JSON reports."""

    sources = [Path(path).resolve() for path in report_paths]
    if not sources:
        raise ValueError("At least one report path is required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for source in sources:
        report = _load_report(source)
        candidate_rows.extend(_candidate_rows(source, report))
        case_rows.extend(_case_rows(source, report))

    candidate_fields = list(candidate_rows[0]) if candidate_rows else []
    case_fields = list(case_rows[0]) if case_rows else []
    candidate_csv = destination / "paper_candidates.csv"
    case_csv = destination / "paper_workload_cases.csv"
    latex = destination / "paper_results.tex"
    manifest = destination / "paper_artifacts.json"
    _write_csv(candidate_csv, candidate_rows, candidate_fields)
    _write_csv(case_csv, case_rows, case_fields)
    _write_latex(latex, candidate_rows)

    outputs = [candidate_csv, case_csv, latex]
    if plots:
        outputs.extend(_write_plots(destination, candidate_rows))
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source_reports": [str(path) for path in sources],
                "candidate_rows": len(candidate_rows),
                "workload_case_rows": len(case_rows),
                "artifacts": [path.name for path in outputs],
                "latency_semantics": "serial invocation measurements",
                "memory_semantics": "incremental CUDA allocation during a warmed invocation",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs.append(manifest)
    return outputs
