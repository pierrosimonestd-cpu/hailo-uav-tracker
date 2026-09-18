#!/usr/bin/env bash
# Provision a Raspberry Pi 5 + AI HAT+ (Hailo-8L) for this project.
#
# Idempotent: safe to re-run. It installs the Hailo runtime from apt, creates a
# virtualenv that can still see the apt-installed packages, installs this
# project into it, and then proves the accelerator is actually there rather
# than assuming the install worked.
#
# Two details that are easy to get wrong and expensive to debug:
#
#   * `hailo_platform` and `picamera2` are apt packages, not pip packages. A
#     plain `python3 -m venv` cannot see them, so the venv is created with
#     --system-site-packages. Without it, the Hailo backend fails to import on
#     a machine where the runtime is correctly installed.
#   * Raspberry Pi OS Bookworm marks its Python as externally managed (PEP 668),
#     so a bare `pip install` into the system interpreter is refused. The venv
#     is the fix; --break-system-packages is not.
#
# Usage, on the Pi:
#   ./tools/provision_pi.sh              # install and verify
#   ./tools/provision_pi.sh --verify     # verify only, change nothing

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_DIR}/.venv"
VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mOK\033[0m   %s\n' "$*"; }
warn() { printf '    \033[33mWARN\033[0m %s\n' "$*"; }
bad()  { printf '    \033[31mFAIL\033[0m %s\n' "$*"; }

failures=0
note_failure() { bad "$*"; failures=$((failures + 1)); }

# --------------------------------------------------------------- environment

say "Host"
model="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
printf '    model   %s\n' "$model"
printf '    kernel  %s\n' "$(uname -srm)"
printf '    os      %s\n' "$(. /etc/os-release && echo "$PRETTY_NAME")"

case "$model" in
  *"Raspberry Pi 5"*) ok "Raspberry Pi 5" ;;
  *) warn "not a Pi 5 — the AI HAT+ needs the Pi 5 PCIe connector" ;;
esac

if [ "$(uname -m)" != "aarch64" ]; then
  note_failure "not a 64-bit userland; HailoRT packages are arm64 only"
fi

# ------------------------------------------------------------------- install

if [ "$VERIFY_ONLY" -eq 0 ]; then
  say "Installing the Hailo runtime and camera stack"
  sudo apt-get update
  # hailo-all: PCIe driver, firmware, HailoRT, Python bindings and TAPPAS.
  # python3-opencv from apt rather than pip: the wheel builds from source on
  # this platform and takes the better part of an hour.
  sudo apt-get install -y hailo-all python3-picamera2 python3-opencv python3-venv git

  say "Creating the virtualenv"
  if [ ! -d "$VENV" ]; then
    python3 -m venv --system-site-packages "$VENV"
    ok "created $VENV with --system-site-packages"
  else
    ok "$VENV already exists"
  fi

  say "Installing the project"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install --upgrade pip >/dev/null
  pip install -e "$REPO_DIR"
else
  say "Verify only — nothing will be installed"
  # shellcheck disable=SC1091
  [ -d "$VENV" ] && source "$VENV/bin/activate"
fi

# -------------------------------------------------------------- verification

say "PCIe link"
if lspci 2>/dev/null | grep -qi hailo; then
  ok "$(lspci | grep -i hailo)"
else
  note_failure "no Hailo device on the PCIe bus (lspci). Check the ribbon cable seating and that the HAT is powered."
  dmesg 2>/dev/null | grep -i hailo | tail -5 || true
fi

say "HailoRT"
if command -v hailortcli >/dev/null 2>&1; then
  if identify="$(hailortcli fw-control identify 2>&1)"; then
    printf '%s\n' "$identify" | sed 's/^/    /'
    if printf '%s' "$identify" | grep -qi "HAILO8L"; then
      ok "device reports HAILO8L"
    else
      warn "device found but it does not report HAILO8L; the HEF must match the part"
    fi
  else
    note_failure "hailortcli could not talk to the device"
    printf '%s\n' "$identify" | sed 's/^/    /'
  fi
else
  note_failure "hailortcli not on PATH — is hailo-all installed?"
fi

say "Python bindings"
if python3 -c "import hailo_platform; print('   hailo_platform', hailo_platform.__version__)" 2>/dev/null; then
  ok "hailo_platform imports"
else
  note_failure "hailo_platform does not import. If the venv was created without --system-site-packages, delete $VENV and re-run."
fi

say "Project"
if python3 -c "import uavtrack; print('   uavtrack', uavtrack.__version__)" 2>/dev/null; then
  ok "uavtrack imports"
  python3 -m uavtrack.cli check --config "$REPO_DIR/configs/rpi5_hailo8l.yaml" || true
else
  note_failure "uavtrack does not import"
fi

say "Camera"
if command -v rpicam-hello >/dev/null 2>&1; then
  if rpicam-hello --list-cameras 2>&1 | grep -qi "Available cameras"; then
    rpicam-hello --list-cameras 2>&1 | sed 's/^/    /' | head -12
  else
    warn "no camera detected — the tracker can still run from a video file or an image directory"
  fi
else
  warn "rpicam-hello not installed; skipping the camera check"
fi

# ------------------------------------------------------------------- summary

say "Summary"
if [ "$failures" -eq 0 ]; then
  ok "everything the accelerator path needs is present"
  printf '\n    Next: put a compiled .hef in models/ and run\n'
  printf '      source %s/bin/activate\n' "$VENV"
  printf '      python -m uavtrack.cli check --config configs/rpi5_hailo8l.yaml\n'
  printf '      python benchmarks/bench_latency.py --model models/<name>.hef --tag yolov8n-int8-hailo8l\n\n'
  exit 0
fi

bad "$failures check(s) failed — see docs/hailo-deployment.md"
exit 1
