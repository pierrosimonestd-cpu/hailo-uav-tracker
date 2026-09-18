"""What is on disk is not what gets published.

Everything in this repository ran correctly on the machine it was written on
while `src/uavtrack/data/` was missing from git entirely: a bare `data/` line in
.gitignore matched that directory as well as the dataset directory it was meant
for. Local imports resolved from the working tree, the test suite passed, and a
clone was broken.

These tests compare the working tree against what git actually tracks, which is
the only thing a user receives.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Directories whose Python sources must all be tracked. Anything importable or
#: runnable belongs here; scratch directories deliberately do not.
SOURCE_TREES = ("src", "tests", "tools", "benchmarks")


def tracked_files() -> set[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return {Path(line) for line in result.stdout.splitlines() if line}


def working_tree_sources() -> set[Path]:
    found: set[Path] = set()
    for tree in SOURCE_TREES:
        for path in (ROOT / tree).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            found.add(path.relative_to(ROOT))
    return found


@pytest.fixture(scope="module")
def tracked() -> set[Path]:
    try:
        return tracked_files()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        pytest.skip("not a git checkout, or git is unavailable")


def test_every_python_source_is_tracked_by_git(tracked):
    """A source file that exists only in the working tree ships to nobody."""
    untracked = sorted(str(p) for p in working_tree_sources() - {Path(t) for t in tracked})
    assert not untracked, (
        "these Python files are not tracked by git, so a clone will not have them:\n  "
        + "\n  ".join(untracked)
        + "\nCheck .gitignore: an unanchored pattern such as `data/` matches at any depth."
    )


def test_every_package_directory_has_an_init(tracked):
    """A package directory without `__init__.py` is not importable once installed."""
    package_root = ROOT / "src" / "uavtrack"
    missing = []
    for directory in package_root.rglob("*"):
        if not directory.is_dir() or "__pycache__" in directory.parts:
            continue
        has_modules = any(child.suffix == ".py" for child in directory.iterdir())
        if has_modules and not (directory / "__init__.py").exists():
            missing.append(str(directory.relative_to(ROOT)))
    assert not missing, f"package directories without __init__.py: {missing}"


def test_the_dataset_and_run_directories_stay_ignored():
    """The anchoring fix must not have un-ignored what the pattern was for.

    Datasets and training runs are fetched and produced by scripts, never
    committed; docs/dataset.md says so and the repository would be hundreds of
    megabytes if it were not true.
    """
    for path in ("data/dut_antiuav", "runs/train"):
        result = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True)
        assert result.returncode == 0, f"{path} is no longer ignored"
