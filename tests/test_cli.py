"""The command-line interface.

The commands that need no hardware -- `check` and `simulate` -- are exercised
end to end. `run` needs a camera and a model, so only its argument handling is
covered here; the loop itself is covered by tests/test_pipeline.py.

`check` earns its tests: it is the command people reach for when nothing works,
so it has to be right about which backend a config resolves to and whether the
model file is actually there.
"""

from __future__ import annotations

import pytest

from uavtrack import __version__
from uavtrack.cli import main


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code != 0


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["frobnicate"])


# --------------------------------------------------------------------- check


def write_config(tmp_path, model: str, **overrides) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"detector:\n"
        f"  model: {model}\n"
        f"  backend: {overrides.get('backend', 'onnx')}\n"
        f"camera:\n"
        f"  source: '0'\n"
        f"link:\n"
        f"  enabled: {str(overrides.get('link', False)).lower()}\n",
        encoding="utf-8",
    )
    return str(path)


def test_check_reports_a_missing_model_and_fails(tmp_path, capsys):
    code = main(["check", "--config", write_config(tmp_path, "models/absent.onnx")])
    output = capsys.readouterr().out

    assert code == 1, "a missing model must be a non-zero exit, not a warning"
    assert "MISSING" in output
    assert "onnx" in output


def test_check_succeeds_when_the_model_exists(tmp_path, capsys):
    model = tmp_path / "present.onnx"
    model.write_bytes(b"not a real graph, but it exists")

    code = main(["check", "--config", write_config(tmp_path, str(model))])
    output = capsys.readouterr().out

    assert code == 0
    assert "found" in output
    assert "MISSING" not in output


def test_check_reports_the_link_state(tmp_path, capsys):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")

    main(["check", "--config", write_config(tmp_path, str(model), link=True)])
    assert "enabled" in capsys.readouterr().out

    main(["check", "--config", write_config(tmp_path, str(model), link=False)])
    assert "disabled" in capsys.readouterr().out


def test_check_reports_the_compensated_latency(tmp_path, capsys):
    path = tmp_path / "c.yaml"
    path.write_text("detector:\n  model: m.onnx\ncontrol:\n  latency_s: 0.075\n", encoding="utf-8")
    main(["check", "--config", str(path)])
    assert "75 ms" in capsys.readouterr().out


def test_check_on_a_bad_config_fails_cleanly(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("detector:\n  conf_treshold: 0.5\n", encoding="utf-8")

    code = main(["check", "--config", str(path)])
    assert code == 1, "a config typo must not traceback"


def test_check_on_a_missing_config_fails_cleanly(tmp_path):
    assert main(["check", "--config", str(tmp_path / "nope.yaml")]) == 1


def test_check_infers_the_backend_from_the_extension(tmp_path, capsys):
    path = tmp_path / "c.yaml"
    path.write_text("detector:\n  model: models/x.hef\n", encoding="utf-8")
    main(["check", "--config", str(path)])
    assert "hailo" in capsys.readouterr().out


# ------------------------------------------------------------------ simulate


def test_simulate_prints_the_summary_metrics(capsys):
    assert main(["simulate", "--duration", "2"]) == 0
    output = capsys.readouterr().out

    for metric in ("rms_error_deg", "in_frame_fraction", "rms_error_steady_deg"):
        assert metric in output


@pytest.mark.parametrize("trajectory", ["static", "step", "linear", "circular", "crossing"])
def test_simulate_accepts_every_trajectory(trajectory, capsys):
    assert main(["simulate", "--trajectory", trajectory, "--duration", "2"]) == 0
    assert "rms_error_deg" in capsys.readouterr().out


def test_simulate_rejects_an_unknown_trajectory():
    with pytest.raises(SystemExit):
        main(["simulate", "--trajectory", "spiral"])


def test_simulate_latency_changes_the_result(capsys):
    main(["simulate", "--duration", "6", "--latency-ms", "10"])
    fast = capsys.readouterr().out
    main(["simulate", "--duration", "6", "--latency-ms", "300"])
    slow = capsys.readouterr().out
    assert fast != slow, "latency must reach the simulation"


def test_verbose_flag_is_accepted(capsys):
    assert main(["-v", "simulate", "--duration", "2"]) == 0
