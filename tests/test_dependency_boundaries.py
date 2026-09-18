"""The Raspberry Pi install must stay light.

The README claims the runtime dependencies are NumPy, OpenCV, PyYAML and
pyserial, and that PyTorch, ONNX Runtime and HailoRT are optional extras behind
backends. That claim is one careless `import torch` away from being false, and
nothing about the failure would be obvious -- it would just make `pip install`
on a Pi pull a 200 MB wheel, or fail outright on a platform with no build.

So it is enforced rather than asserted. These tests are cheap and they are the
only thing standing between the stated contract and its quiet erosion.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "uavtrack"

#: Packages that must never be imported when a module is loaded. Each is either
#: large, platform-specific, or unavailable on the deployment target.
OPTIONAL_PACKAGES = {
    "torch",
    "torchvision",
    "ultralytics",
    "onnx",
    "onnxruntime",
    "hailo_platform",
    "picamera2",
    "fiftyone",
    "pycocotools",
    "gdown",
    "matplotlib",
    "pandas",
    "scipy",
}

#: Core runtime dependencies, declared in pyproject and installed on the Pi.
CORE_PACKAGES = {"numpy", "cv2", "yaml", "serial"}


def module_level_imports(path: Path) -> set[str]:
    """Top-level package names imported when ``path`` is loaded.

    Only statements at module scope count. An import inside a function or a
    ``try`` block within one is deferred to first call, which is the pattern the
    backends use deliberately.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Try):
            # A module-level try/except ImportError is a guarded optional import:
            # allowed, because the module still loads without the package.
            continue
    return names


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_no_module_imports_an_optional_package_eagerly(path: Path):
    offending = module_level_imports(path) & OPTIONAL_PACKAGES
    assert not offending, (
        f"{path.relative_to(ROOT)} imports {sorted(offending)} at module level. "
        "Optional dependencies belong inside the function that needs them, or "
        "behind a module-level try/except ImportError."
    )


#: Installs a meta-path hook that refuses the named packages.
#:
#: `find_spec`, not the `find_module`/`load_module` pair: the latter was removed
#: in Python 3.12, so a blocker written that way is silently a no-op and every
#: test using it passes without proving anything. That happened here once.
BLOCKER_PREAMBLE = """
import sys

BLOCKED = set(%r)

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is blocked for this test")
        return None

sys.meta_path.insert(0, Blocker())
"""


def _run_blocked(blocked: list[str], body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a subprocess with ``blocked`` packages unimportable.

    A subprocess because the optional packages are already imported in this
    session; blocking them in-process would prove nothing.
    """
    import os

    return subprocess.run(
        [sys.executable, "-c", (BLOCKER_PREAMBLE % (sorted(blocked),)) + body],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_the_import_blocker_actually_blocks():
    """Guards the test below from quietly becoming vacuous."""
    result = _run_blocked(["numpy"], "import numpy\nprint('not blocked')\n")
    assert result.returncode != 0, "the blocker let a blocked package through"
    assert "blocked for this test" in result.stderr


def test_the_core_package_imports_with_only_its_declared_dependencies():
    body = """
import uavtrack
import uavtrack.config
import uavtrack.pipeline
import uavtrack.viz
import uavtrack.cli
import uavtrack.data.voc
import uavtrack.detect
import uavtrack.detect.preprocess
import uavtrack.detect.postprocess
import uavtrack.detect.factory
import uavtrack.track
import uavtrack.track.bytetrack
import uavtrack.control
import uavtrack.io
import uavtrack.io.protocol
import uavtrack.io.serial_link
print("ok")
"""
    result = _run_blocked(sorted(OPTIONAL_PACKAGES), body)
    assert result.returncode == 0, (
        f"the package failed to import without its optional dependencies:\n{result.stderr[-2000:]}"
    )
    assert "ok" in result.stdout


def test_the_tracker_falls_back_to_greedy_matching_without_scipy():
    """The SciPy-free path is what a minimal Raspberry Pi install runs."""
    body = """
from uavtrack.track.bytetrack import ByteTracker, _linear_sum_assignment
from uavtrack.detect.base import Detection

assert _linear_sum_assignment is None, "SciPy was not actually blocked"

tracker = ByteTracker(min_hits=2)
for i in range(6):
    x = 300.0 + 8 * i
    tracks = tracker.update([Detection(x, 300, x + 40, 340, 0.9)], 0.05)
assert len(tracks) == 1, f"expected one track, got {len(tracks)}"
print("ok")
"""
    result = _run_blocked(["scipy"], body)
    assert result.returncode == 0, f"tracking failed without SciPy:\n{result.stderr[-2000:]}"
    assert "ok" in result.stdout


def test_declared_core_dependencies_match_what_the_package_imports():
    """Every core package the library imports must be declared in pyproject."""
    declared = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    mapping = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "yaml": "pyyaml",
        "serial": "pyserial",
    }

    imported: set[str] = set()
    for path in SRC.rglob("*.py"):
        imported |= module_level_imports(path) & CORE_PACKAGES

    missing = [
        mapping[name] for name in sorted(imported) if mapping[name].lower() not in declared.lower()
    ]
    assert not missing, f"imported but not declared in pyproject: {missing}"
