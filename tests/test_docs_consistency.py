"""The prose must not drift away from the measurements.

`docs/benchmarks.md` is generated, so it cannot drift. `docs/control-design.md`
is written by hand around the same numbers, because the argument it makes needs
the figures inline rather than in a table at the end -- and hand-written numbers
go stale the moment a benchmark is re-run.

These tests re-derive every figure quoted in that document from
`benchmarks/results/control_simulation.json` and fail if any has moved. They
also check that no documentation link points at a file that does not exist.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results" / "control_simulation.json"
CONTROL_DOC = ROOT / "docs" / "control-design.md"


@pytest.fixture(scope="module")
def simulation() -> dict:
    if not RESULTS.exists():
        pytest.skip("control_simulation.json not present; run benchmarks/bench_control.py")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def control_doc() -> str:
    return CONTROL_DOC.read_text(encoding="utf-8")


def test_latency_table_matches_the_measurements(simulation, control_doc):
    for row in simulation["studies"]["latency"]:
        match = re.search(rf"\|\s*{row['latency_ms']} ms\s*\|\s*([0-9.]+)°", control_doc)
        assert match, f"no row for {row['latency_ms']} ms in control-design.md"
        assert float(match.group(1)) == pytest.approx(row["rms_error_deg"], abs=0.005), (
            f"{row['latency_ms']} ms: document says {match.group(1)}°"
        )


def test_feedforward_table_matches_the_measurements(simulation, control_doc):
    for row in simulation["studies"]["feedforward"]:
        rate = f"{row['peak_target_rate_deg_s']:.1f}"
        pattern = rf"\|\s*{re.escape(rate)} °/s\s*\|\s*([0-9.]+)°\s*\|\s*([0-9.]+)°\s*\|\s*(\d+)%"
        match = re.search(pattern, control_doc)
        assert match, f"no row for {rate} °/s in control-design.md"

        assert float(match.group(1)) == pytest.approx(row["rms_error_deg_ff_off"], abs=0.005)
        assert float(match.group(2)) == pytest.approx(row["rms_error_deg_ff_on"], abs=0.005)
        assert int(match.group(3)) == pytest.approx(round(row["error_reduction"] * 100), abs=1)


def test_dropout_table_matches_the_measurements(simulation, control_doc):
    for row in simulation["studies"]["dropout"]:
        percentage = f"{row['dropout'] * 100:.0f}%"
        match = re.search(rf"\|\s*{re.escape(percentage)}\s*\|\s*([0-9.]+)°", control_doc)
        assert match, f"no row for {percentage} dropout in control-design.md"
        assert float(match.group(1)) == pytest.approx(row["rms_error_deg"], abs=0.05)


def test_quoted_gains_match_the_benchmark_configuration(simulation, control_doc):
    gains = simulation["gains"]
    quoted = re.search(r"kp\s*=\s*([0-9.]+)\s+ki\s*=\s*([0-9.]+)\s+kd\s*=\s*([0-9.]+)", control_doc)
    assert quoted, "control-design.md no longer states the tuned gains"
    assert float(quoted.group(1)) == pytest.approx(gains["kp"])
    assert float(quoted.group(2)) == pytest.approx(gains["ki"])
    assert float(quoted.group(3)) == pytest.approx(gains["kd"])


def _markdown_files() -> list[Path]:
    return [
        *ROOT.glob("*.md"),
        *(ROOT / "docs").glob("*.md"),
        *(ROOT / "models").glob("*.md"),
    ]


def test_no_documentation_link_is_broken():
    broken = []
    for doc in _markdown_files():
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            target = match.group(1).split("#")[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (doc.parent / target).exists():
                broken.append(f"{doc.relative_to(ROOT)} -> {target}")
    assert not broken, "broken links: " + "; ".join(broken)


def test_referenced_scripts_exist():
    """Every `python tools/x.py` or `python benchmarks/x.py` in the docs must resolve."""
    missing = []
    for doc in _markdown_files():
        text = doc.read_text(encoding="utf-8")
        for match in re.finditer(r"python (tools/[\w.]+\.py|benchmarks/[\w.]+\.py)", text):
            if not (ROOT / match.group(1)).exists():
                missing.append(f"{doc.relative_to(ROOT)} -> {match.group(1)}")
    assert not missing, "missing scripts: " + "; ".join(missing)
