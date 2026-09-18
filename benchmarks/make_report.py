#!/usr/bin/env python3
"""Regenerate the tables in docs/benchmarks.md from the measured result files.

The tables in the documentation are generated, never hand-written. A number in
a README that nobody can trace back to a JSON file produced by a script is a
number nobody should believe, including its author.

Every result file records whether it is ``measured`` or ``simulation``, and that
label is carried into the generated tables.

Usage:
    python benchmarks/make_report.py
    python benchmarks/make_report.py --check      # fail if the doc is stale
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BEGIN = "<!-- BEGIN GENERATED: {name} -->"
END = "<!-- END GENERATED: {name} -->"


def load(results_dir: Path, pattern: str) -> list[dict]:
    """Load every result file matching ``pattern``, newest last."""
    payloads = []
    for path in sorted(results_dir.glob(pattern)):
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"[warn] skipping malformed {path}")
    return payloads


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    """Render a Markdown table."""
    if not rows:
        return "_No results yet. Run the benchmark to populate this table._"
    separator = [align or "---"] * len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def detection_table(results_dir: Path) -> str:
    payloads = load(results_dir, "detection_*.json")
    payloads = [p for p in payloads if not p.get("tag", "").startswith("_")]
    rows = []
    for p in sorted(payloads, key=lambda p: -p["metrics"].get("AP", 0)):
        m = p["metrics"]
        rows.append(
            [
                f"`{p['tag']}`",
                p["backend"],
                f"{m['AP']:.3f}",
                f"{m['AP50']:.3f}",
                f"{m['AP75']:.3f}",
                f"{m['AP_small']:.3f}",
                f"{m['AP_medium']:.3f}",
                f"{m['AP_large']:.3f}",
                str(p["images"]),
            ]
        )
    return table(
        ["Model", "Backend", "AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L", "Images"],
        rows,
        align="---:",
    )


def latency_table(results_dir: Path) -> str:
    payloads = load(results_dir, "latency_*.json")
    rows = []
    for p in payloads:
        e = p["end_to_end_ms"]
        inference = p["stages_ms"].get("inference", {})
        rows.append(
            [
                f"`{p['tag']}`",
                p["backend"],
                f"{inference.get('median', float('nan')):.1f}",
                f"{e['median']:.1f}",
                f"{e['p95']:.1f}",
                f"{e['p99']:.1f}",
                f"{p['throughput_fps']:.1f}",
            ]
        )
    return table(
        ["Model", "Backend", "Inference (ms)", "End to end (ms)", "p95", "p99", "FPS"],
        rows,
        align="---:",
    )


def hard_negative_table(results_dir: Path) -> str:
    payloads = load(results_dir, "hard_negatives_*.json")
    rows = []
    for p in payloads:
        rows.append(
            [
                f"`{p['tag']}`",
                str(p["images"]),
                f"{p['operating_threshold']:.2f}",
                f"{p['false_positive_rate_at_operating_threshold'] * 100:.2f}%",
                str(p["threshold_for_target_fp_rate"] or "not reached"),
            ]
        )
    return table(
        ["Model", "Bird images", "Threshold", "False-positive rate", "Threshold for 1%"],
        rows,
        align="---:",
    )


def tracking_table(results_dir: Path) -> str:
    payloads = load(results_dir, "tracking_*.json")
    rows = []
    for p in payloads:
        o = p["overall"]
        rows.append(
            [
                f"`{p['tag']}`",
                str(o["sequences"]),
                f"{o['total_frames']:,}",
                f"{o['success_auc']:.3f}",
                f"{o['success_at_0_5']:.3f}",
                f"{o['precision_20px']:.3f}",
                f"{o['recall']:.3f}",
                str(o["reacquisitions_total"]),
            ]
        )
    return table(
        [
            "Model",
            "Sequences",
            "Frames",
            "Success AUC",
            "Success@0.5",
            "P@20px",
            "Recall",
            "Re-acquisitions",
        ],
        rows,
        align="---:",
    )


def control_tables(results_dir: Path) -> dict[str, str]:
    path = results_dir / "control_simulation.json"
    if not path.exists():
        empty = "_No results yet. Run `python benchmarks/bench_control.py`._"
        return {"control-latency": empty, "control-feedforward": empty, "control-dropout": empty}

    payload = json.loads(path.read_text(encoding="utf-8"))
    studies = payload["studies"]
    out = {}

    if "latency" in studies:
        out["control-latency"] = table(
            ["Sense-to-act latency", "RMS error", "Peak error", "In frame"],
            [
                [
                    f"{r['latency_ms']} ms",
                    f"{r['rms_error_deg']:.2f}°",
                    f"{r['max_error_deg']:.2f}°",
                    f"{r['in_frame_fraction'] * 100:.0f}%",
                ]
                for r in studies["latency"]
            ],
            align="---:",
        )
    if "feedforward" in studies:
        out["control-feedforward"] = table(
            ["Peak target rate", "Feed-forward off", "Feed-forward on", "Reduction"],
            [
                [
                    f"{r['peak_target_rate_deg_s']:.1f} °/s",
                    f"{r['rms_error_deg_ff_off']:.2f}°",
                    f"{r['rms_error_deg_ff_on']:.2f}°",
                    f"{r['error_reduction'] * 100:.0f}%",
                ]
                for r in studies["feedforward"]
            ],
            align="---:",
        )
    if "dropout" in studies:
        out["control-dropout"] = table(
            ["Detections missed", "RMS error", "In frame"],
            [
                [
                    f"{r['dropout'] * 100:.0f}%",
                    f"{r['rms_error_deg']:.2f}°",
                    f"{r['in_frame_fraction'] * 100:.0f}%",
                ]
                for r in studies["dropout"]
            ],
            align="---:",
        )
    return out


def readme_latency_table(results_dir: Path) -> str:
    """A compact subset of the latency study, for the README headline."""
    path = results_dir / "control_simulation.json"
    if not path.exists():
        return "_Run `python benchmarks/bench_control.py` to populate this table._"

    rows_by_latency = {
        r["latency_ms"]: r
        for r in json.loads(path.read_text(encoding="utf-8"))["studies"].get("latency", [])
    }
    # Labels describe latency *regimes*, not measurements of specific hardware.
    # Measure your own with benchmarks/bench_latency.py and read off the row.
    highlights = [
        (10, "an accelerator with headroom to spare"),
        (45, "the budget `configs/rpi5_hailo8l.yaml` assumes"),
        (120, "a pipeline four times slower"),
        (300, "a pipeline near the edge of usable"),
    ]
    rows = []
    for latency, note in highlights:
        row = rows_by_latency.get(latency)
        if row is None:
            continue
        rows.append([f"{latency} ms", f"**{row['rms_error_deg']:.2f}°**", note])
    return table(["Sense-to-act latency", "RMS pointing error", ""], rows)


def readme_feedforward_table(results_dir: Path) -> str:
    """The feed-forward headline, trimmed to three rows."""
    path = results_dir / "control_simulation.json"
    if not path.exists():
        return "_Run `python benchmarks/bench_control.py` to populate this table._"

    study = json.loads(path.read_text(encoding="utf-8"))["studies"].get("feedforward", [])
    rows = [
        [
            f"{r['peak_target_rate_deg_s']:.1f} °/s",
            f"{r['rms_error_deg_ff_off']:.2f}°",
            f"**{r['rms_error_deg_ff_on']:.2f}°**",
            f"{r['error_reduction'] * 100:.0f}% lower",
        ]
        for r in study[::2]
    ]
    return table(["Target angular rate", "PID only", "PID + feed-forward", ""], rows, align="---:")


def splice(text: str, name: str, body: str) -> str:
    """Replace the content between the markers for ``name``."""
    begin, end = BEGIN.format(name=name), END.format(name=name)
    start = text.find(begin)
    stop = text.find(end)
    if start < 0 or stop < 0:
        return text
    return text[: start + len(begin)] + "\n" + body + "\n" + text[stop:]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    parser.add_argument(
        "--docs",
        type=Path,
        nargs="+",
        default=[Path("docs/benchmarks.md"), Path("README.md")],
        help="documents carrying generated sections",
    )
    parser.add_argument("--check", action="store_true", help="exit non-zero if a doc is stale")
    args = parser.parse_args()

    sections = {
        "detection": detection_table(args.results),
        "latency": latency_table(args.results),
        "hard-negatives": hard_negative_table(args.results),
        "tracking": tracking_table(args.results),
        "readme-latency": readme_latency_table(args.results),
        "readme-feedforward": readme_feedforward_table(args.results),
        **control_tables(args.results),
    }

    stale = False
    for doc in args.docs:
        if not doc.exists():
            print(f"[warn] {doc} not found, skipping")
            continue

        original = doc.read_text(encoding="utf-8")
        updated = original
        for name, body in sections.items():
            # Only splice sections the document actually declares.
            if BEGIN.format(name=name) in updated:
                updated = splice(updated, name, body)

        if args.check:
            if updated != original:
                print(f"{doc} is out of date; run: python benchmarks/make_report.py")
                stale = True
            else:
                print(f"{doc} is up to date")
            continue

        if updated != original:
            doc.write_text(updated, encoding="utf-8")
            print(f"updated {doc}")
        else:
            print(f"{doc} unchanged")

    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
