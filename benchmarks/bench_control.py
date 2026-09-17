#!/usr/bin/env python3
"""Closed-loop pointing benchmarks, in simulation.

Three studies, each answering a question the detection metrics cannot:

``latency``
    Steady-state pointing error as a function of sense-to-act latency. This is
    the study that decides whether an NPU is worth its price: inference time
    enters the control loop as dead time, dead time caps the achievable loop
    bandwidth, and bandwidth is what keeps a manoeuvring target centred.

``feedforward``
    Steady-state error with and without velocity feed-forward, across target
    speeds. Quantifies the velocity-lag term the feed-forward exists to cancel.

``dropout``
    Tracking quality against the fraction of frames in which the detector finds
    nothing, exercising the tracker's ability to coast on prediction.

Every number produced here is a *simulation* result and is labelled as such in
``docs/benchmarks.md``. The servo model is documented in
:mod:`uavtrack.control.plant`.

Usage:
    python benchmarks/bench_control.py --study all --out benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uavtrack.control import (  # noqa: E402
    PIDGains,
    SimulationConfig,
    TargetTrajectory,
    simulate,
)

# Tuned by the sweep in docs/control-design.md; latency-limited, not actuator-limited.
DEFAULT_GAINS = PIDGains(kp=4.5, ki=0.5, kd=0.25)

# Sense-to-act latencies worth comparing, in milliseconds. The annotations say
# what each one roughly corresponds to; the mapping from model to latency comes
# from docs/benchmarks.md, not from this file.
LATENCIES_MS = [10, 20, 30, 45, 60, 80, 120, 160, 200, 300]

SEEDS = [0, 1, 2, 3, 4]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _run(kind: str, seed: int, **overrides) -> dict:
    config = SimulationConfig(
        duration_s=8.0,
        pan_gains=DEFAULT_GAINS,
        tilt_gains=DEFAULT_GAINS,
        seed=seed,
        **overrides,
    )
    trajectory = TargetTrajectory(kind=kind)
    result = simulate(trajectory, config)
    return result.summary(steady_state_from_s=2.0)


def study_latency() -> list[dict]:
    """Steady-state error against sense-to-act latency."""
    rows = []
    for latency_ms in LATENCIES_MS:
        runs = [_run("circular", seed, detection_latency_s=latency_ms / 1000.0) for seed in SEEDS]
        rows.append(
            {
                "latency_ms": latency_ms,
                "effective_fps": round(1000.0 / latency_ms, 1),
                "rms_error_deg": round(_mean([r["rms_error_steady_deg"] for r in runs]), 4),
                "max_error_deg": round(_mean([r["max_error_steady_deg"] for r in runs]), 4),
                "in_frame_fraction": round(_mean([r["in_frame_fraction"] for r in runs]), 4),
                "seeds": len(SEEDS),
            }
        )
        print(
            f"[latency] {latency_ms:4d} ms -> "
            f"rms {rows[-1]['rms_error_deg']:.3f} deg, "
            f"in frame {rows[-1]['in_frame_fraction'] * 100:.1f}%"
        )
    return rows


def study_feedforward() -> list[dict]:
    """Steady-state error with and without feed-forward, across target speeds."""
    rows = []
    for frequency in (0.05, 0.10, 0.15, 0.20, 0.30):
        # Peak angular rate of the circular trajectory: 2*pi*f*radius.
        peak_rate = 2.0 * 3.141592653589793 * frequency * TargetTrajectory().radius
        entry: dict = {
            "frequency_hz": frequency,
            "peak_target_rate_deg_s": round(peak_rate, 2),
        }
        for gain, label in ((0.0, "off"), (0.9, "on")):
            runs = []
            for seed in SEEDS:
                config = SimulationConfig(
                    duration_s=8.0,
                    pan_gains=DEFAULT_GAINS,
                    tilt_gains=DEFAULT_GAINS,
                    feedforward_gain=gain,
                    seed=seed,
                )
                result = simulate(TargetTrajectory(kind="circular", frequency=frequency), config)
                runs.append(result.summary(steady_state_from_s=2.0))
            entry[f"rms_error_deg_ff_{label}"] = round(
                _mean([r["rms_error_steady_deg"] for r in runs]), 4
            )
        reduction = 1.0 - entry["rms_error_deg_ff_on"] / max(entry["rms_error_deg_ff_off"], 1e-9)
        entry["error_reduction"] = round(reduction, 4)
        rows.append(entry)
        print(
            f"[feedforward] {peak_rate:5.1f} deg/s -> "
            f"off {entry['rms_error_deg_ff_off']:.3f} deg, "
            f"on {entry['rms_error_deg_ff_on']:.3f} deg "
            f"({reduction * 100:.0f}% lower)"
        )
    return rows


def study_dropout() -> list[dict]:
    """Tracking quality against the detector's miss rate."""
    rows = []
    for dropout in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7):
        runs = [_run("circular", seed, detection_dropout=dropout) for seed in SEEDS]
        rows.append(
            {
                "dropout": dropout,
                "rms_error_deg": round(_mean([r["rms_error_steady_deg"] for r in runs]), 4),
                "max_error_deg": round(_mean([r["max_error_steady_deg"] for r in runs]), 4),
                "in_frame_fraction": round(_mean([r["in_frame_fraction"] for r in runs]), 4),
            }
        )
        print(
            f"[dropout] {dropout * 100:3.0f}% missed -> "
            f"rms {rows[-1]['rms_error_deg']:.3f} deg, "
            f"in frame {rows[-1]['in_frame_fraction'] * 100:.1f}%"
        )
    return rows


STUDIES = {
    "latency": study_latency,
    "feedforward": study_feedforward,
    "dropout": study_dropout,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--study", default="all", choices=[*STUDIES, "all"])
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    selected = list(STUDIES) if args.study == "all" else [args.study]
    results = {name: STUDIES[name]() for name in selected}

    payload = {
        "kind": "simulation",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "gains": {"kp": DEFAULT_GAINS.kp, "ki": DEFAULT_GAINS.ki, "kd": DEFAULT_GAINS.kd},
        "seeds": SEEDS,
        "note": (
            "Closed-loop simulation with the servo model in uavtrack.control.plant. "
            "Not a hardware measurement."
        ),
        "studies": results,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "control_simulation.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
