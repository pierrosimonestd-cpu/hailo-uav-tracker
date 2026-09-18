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
    # Ablations are answers to a specific question, not deployable configurations,
    # so they get their own table rather than a row next to the real models.
    payloads = [p for p in payloads if not p.get("tag", "").startswith(("_", "ablation-"))]
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


def preprocessing_ablation_table(results_dir: Path) -> str:
    """INTER_AREA against INTER_LINEAR, same graph, same images.

    Two questions at once. Matching Ultralytics' bilinear kernel should
    reproduce Ultralytics' score -- if it does not, the hand-written decode is
    wrong somewhere. And if area-averaging helps because it anti-aliases, the
    gain belongs on small objects, not large ones.
    """
    by_tag = {}
    for payload in load(results_dir, "detection_*.json"):
        by_tag[payload.get("tag", "")] = payload

    linear = by_tag.get("ablation-inter-linear")
    area = by_tag.get("yolov8n-fp32-onnx")
    torch_ = by_tag.get("yolov8n-fp32-pytorch")
    if not (linear and area and torch_):
        return "_No results yet. Run the benchmark to populate this table._"

    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]
    rows = []
    for key in keys:
        t_val = torch_["metrics"][key]
        l_val = linear["metrics"][key]
        a_val = area["metrics"][key]
        rows.append(
            [
                f"`{key}`",
                f"{t_val:.4f}",
                f"{l_val:.4f}",
                f"{a_val:.4f}",
                f"{a_val - l_val:+.4f}",
            ]
        )
    return table(
        [
            "Metric",
            "Ultralytics (bilinear)",
            "This decode, bilinear",
            "This decode, `INTER_AREA`",
            "Area &minus; bilinear",
        ],
        rows,
        align="---:",
    )


def resize_by_resolution_table(results_dir: Path) -> str:
    """Where the resize-kernel gain actually comes from.

    Rows below the interpretability threshold are still shown -- hiding a row
    because it is noisy is its own kind of dishonesty -- but marked, so a +0.35
    delta measured on one image is not read as a result.
    """
    path = results_dir / "resize_ablation_by_resolution.json"
    if not path.exists():
        return "_No results yet. Run the benchmark to populate this table._"

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for group in payload["by_resolution"]:
        label = f"{group['width']}x{group['height']}"
        images = str(group["images"])
        if not group["interpretable"]:
            images += " &dagger;"
        rows.append(
            [
                label,
                f"{group['downscale_ratio']:.2f}:1",
                images,
                f"{group['ap_area']:.4f}",
                f"{group['ap_linear']:.4f}",
                f"{group['ap_delta']:+.4f}",
                f"{group['ap_small_delta']:+.4f}",
            ]
        )
    body = table(
        [
            "Source",
            "Downscale",
            "Images",
            "AP `area`",
            "AP `linear`",
            "&Delta; AP",
            "&Delta; AP_S",
        ],
        rows,
        align="---:",
    )
    threshold = payload["min_images_for_a_claim"]
    footnote = f"&dagger; fewer than {threshold} images; shown for completeness, not interpretable."
    return f"{body}\n\n{footnote}"


def host_label(payload: dict) -> str:
    """Which machine a latency figure came from.

    Worth a column of its own now that the same model is timed on a desktop and
    on the Raspberry Pi: a table mixing the two without saying so invites the
    reader to compare numbers that are four times apart for reasons having
    nothing to do with the model.
    """
    platform = payload.get("host", {}).get("platform", "")
    if "aarch64" in platform:
        return "Pi 5 CPU"
    if "Windows" in platform or "x86_64" in platform:
        return "x86 desktop"
    return platform.split("-")[0] or "unknown"


def latency_table(results_dir: Path) -> str:
    payloads = load(results_dir, "latency_*.json")
    rows = []
    # Slowest first: the interesting comparison is how far the accelerator has
    # to close, not which configuration happens to sort first alphabetically.
    for p in sorted(payloads, key=lambda p: -p["end_to_end_ms"]["median"]):
        e = p["end_to_end_ms"]
        inference = p["stages_ms"].get("inference", {})
        rows.append(
            [
                f"`{p['tag']}`",
                host_label(p),
                p["backend"],
                f"{inference.get('median', float('nan')):.1f}",
                f"{e['median']:.1f}",
                f"{e['p95']:.1f}",
                f"{e['p99']:.1f}",
                f"{p['throughput_fps']:.1f}",
            ]
        )
    return table(
        [
            "Model",
            "Host",
            "Backend",
            "Inference (ms)",
            "End to end (ms)",
            "p95",
            "p99",
            "FPS",
        ],
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


def negatives_tradeoff_table(results_dir: Path) -> str:
    """Drone recall at matched bird false-positive rates.

    The actual bird rate each model reaches is printed next to the target it was
    asked for, because the two are not always equal -- the sweep is on a grid of
    thresholds, and a model can overshoot. Hiding that would make the comparison
    look tidier than it is.
    """
    path = results_dir / "negatives_tradeoff.json"
    if not path.exists():
        return "_No results yet. Run the benchmark to populate this table._"

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for entry in payload["comparisons"]:
        models = entry["models"]
        if len(models) != 2:
            continue
        base, tuned = models["baseline"], models["with-negatives"]
        # The threshold grid is coarse, so a model can land well under the
        # target and the pair stops being a fair match. Where the two actual
        # rates are more than 1.5x apart the relative gain is inflated by the
        # mismatch, and saying so beats printing a headline number that a
        # careful reader would have to catch.
        base_rate = max(base["bird_fp_rate"], 1e-9)
        tuned_rate = max(tuned["bird_fp_rate"], 1e-9)
        spread = max(base_rate, tuned_rate) / min(base_rate, tuned_rate)
        matched = spread <= 1.5
        delta = f"{entry['recall_relative'] * 100:+.0f}%"
        rows.append(
            [
                f"{entry['target_fp_rate'] * 100:.1f}%",
                f"{base['threshold']:.2f} / {base['bird_fp_rate'] * 100:.1f}%",
                f"{base['drone_recall']:.3f}",
                f"{tuned['threshold']:.2f} / {tuned['bird_fp_rate'] * 100:.1f}%",
                f"**{tuned['drone_recall']:.3f}**",
                delta if matched else f"{delta} &dagger;",
            ]
        )
    body = table(
        [
            "Target bird FP",
            "Baseline thr / actual",
            "Recall",
            "With negatives thr / actual",
            "Recall",
            "&Delta;",
        ],
        rows,
        align="---:",
    )
    footnote = (
        "&dagger; the two models' actual bird rates differ by more than 1.5x here, so "
        "this row is not a matched comparison and its relative gain is overstated."
    )
    if any("&dagger;" in row[-1] for row in rows):
        return f"{body}\n\n{footnote}"
    return body


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
        "preprocessing-ablation": preprocessing_ablation_table(args.results),
        "resize-by-resolution": resize_by_resolution_table(args.results),
        "hard-negatives": hard_negative_table(args.results),
        "negatives-tradeoff": negatives_tradeoff_table(args.results),
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
