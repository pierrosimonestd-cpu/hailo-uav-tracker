#!/usr/bin/env python3
"""Drive the turret by hand from a USB gamepad.

Useful for more than play. Aiming the rig, checking that the travel limits are
where the mechanics actually stop, and — the reason this exists — identifying
the servo model: hold a stick over, watch how long the turret takes to get
there, and the time constant the controller currently guesses at becomes a
measured number.

The sticks command a *rate*, not a position. A pan/tilt turret has no absolute
reference a stick can map onto, and rate control is what every camera gimbal
does for the same reason: the stick says "keep turning this fast", which is
also exactly what the tracking loop commands, so the two share a path.

The firmware's 500 ms failsafe still applies, so this keeps sending even when
the sticks are centred. Letting go of the pad must not make the turret go limp.

Usage, on the Pi:
    python tools/manual_control.py --config configs/rpi5_face_hailo8l.yaml
    python tools/manual_control.py --list        # show the input devices seen

Requires `evdev` (pip install evdev) and membership of the `input` group:
    sudo usermod -aG input $USER    # then log out and back in
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uavtrack.config import PipelineConfig  # noqa: E402
from uavtrack.io.protocol import EffectorState  # noqa: E402
from uavtrack.io.serial_link import SerialLink  # noqa: E402

logger = logging.getLogger("manual")

#: Sticks report a signed range; anything under this fraction of full scale is
#: treated as centred. Cheap pads rest a few percent off zero and would
#: otherwise creep the turret across the room while nobody is touching it.
DEADZONE = 0.12

#: Full stick deflection, in degrees per second. Deliberately well under the
#: MG90S's 600 deg/s: a human aiming wants control, not speed, and the servos
#: draw their worst current at full slew.
MAX_RATE_DEG_S = 90.0

#: How often commands go out. The firmware interpolates at its own rate; this
#: only has to beat the 500 ms failsafe comfortably and feel continuous.
SEND_HZ = 50.0


def find_gamepad(explicit: str | None = None):
    """The first device that looks like a gamepad, or the one named."""
    try:
        import evdev
    except ImportError:
        sys.exit(
            "evdev is required: pip install evdev\n"
            "If the import succeeds but no device is found, check that you are in the "
            "'input' group: sudo usermod -aG input $USER"
        )

    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    if explicit:
        for device in devices:
            if explicit in (device.path, device.name):
                return device
        sys.exit(f"no input device matching {explicit!r}; try --list")

    for device in devices:
        capabilities = device.capabilities()
        # A gamepad has absolute axes and buttons. A keyboard has neither, and
        # a touchpad has axes but no gamepad buttons.
        axes = capabilities.get(evdev.ecodes.EV_ABS, [])
        keys = capabilities.get(evdev.ecodes.EV_KEY, [])
        has_stick = any(code in (evdev.ecodes.ABS_X, evdev.ecodes.ABS_Y) for code, _ in axes)
        has_buttons = any(
            evdev.ecodes.BTN_JOYSTICK <= key <= evdev.ecodes.BTN_THUMBR for key in keys
        )
        if has_stick and has_buttons:
            return device
    return None


def list_devices() -> int:
    try:
        import evdev
    except ImportError:
        sys.exit("evdev is required: pip install evdev")

    paths = evdev.list_devices()
    if not paths:
        print("no input devices are readable.")
        print("Either nothing is plugged in, or this user is not in the 'input' group:")
        print("  sudo usermod -aG input $USER   # then log out and back in")
        return 1

    print(f"{'path':<20}{'name':<40}{'looks like a gamepad'}")
    for path in paths:
        device = evdev.InputDevice(path)
        capabilities = device.capabilities()
        axes = capabilities.get(evdev.ecodes.EV_ABS, [])
        keys = capabilities.get(evdev.ecodes.EV_KEY, [])
        has_stick = any(code in (evdev.ecodes.ABS_X, evdev.ecodes.ABS_Y) for code, _ in axes)
        has_buttons = any(
            evdev.ecodes.BTN_JOYSTICK <= key <= evdev.ecodes.BTN_THUMBR for key in keys
        )
        print(f"{path:<20}{device.name[:38]:<40}{'yes' if has_stick and has_buttons else 'no'}")
    return 0


class Axis:
    """One analogue axis, normalised to -1..1 with a deadzone applied."""

    def __init__(self, info) -> None:
        self.min = info.min
        self.max = info.max
        self.centre = (info.min + info.max) / 2.0
        self.span = max((info.max - info.min) / 2.0, 1.0)
        self.value = 0.0

    def set_raw(self, raw: int) -> None:
        normalised = (raw - self.centre) / self.span
        normalised = max(-1.0, min(1.0, normalised))
        if abs(normalised) < DEADZONE:
            self.value = 0.0
            return
        # Rescale what is left of the range so the stick does not jump to
        # DEADZONE the instant it leaves the dead band.
        sign = 1.0 if normalised > 0 else -1.0
        self.value = sign * (abs(normalised) - DEADZONE) / (1.0 - DEADZONE)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/rpi5_face_hailo8l.yaml"))
    parser.add_argument("--device", default=None, help="input device path or name")
    parser.add_argument("--list", action="store_true", help="list input devices and exit")
    parser.add_argument("--max-rate", type=float, default=MAX_RATE_DEG_S)
    parser.add_argument("--invert-tilt", action="store_true", help="push forward to look down")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")

    if args.list:
        return list_devices()

    from evdev import ecodes

    device = find_gamepad(args.device)
    if device is None:
        print("no gamepad found.")
        print("  - `--list` shows what this machine can see")
        print("  - many controllers ship with a charge-only USB cable; try another")
        print("  - Bigben/Nacon pads often have a PC/PS3 mode switch that must be on PC")
        return 1
    logger.info("using %s (%s)", device.name, device.path)

    config = PipelineConfig.load(args.config)
    pan_limits = config.control.pan
    tilt_limits = config.control.tilt

    capabilities = dict(device.capabilities().get(ecodes.EV_ABS, []))
    axes = {
        "pan": Axis(capabilities[ecodes.ABS_X]) if ecodes.ABS_X in capabilities else None,
        "tilt": Axis(capabilities[ecodes.ABS_Y]) if ecodes.ABS_Y in capabilities else None,
    }
    if axes["pan"] is None or axes["tilt"] is None:
        sys.exit(f"{device.name} has no ABS_X/ABS_Y sticks")

    pan = config.control.home_pan_deg
    tilt = config.control.home_tilt_deg
    effector = False

    link = SerialLink(
        config.link.port,
        config.link.baudrate,
        heartbeat_interval_s=config.link.heartbeat_interval_s,
    )
    logger.info("link open on %s", config.link.port)
    logger.info("left stick: pan/tilt   A/cross: toggle effector   B/circle: home   Ctrl+C: quit")

    device.grab()  # stop the pad also acting as a mouse or keyboard on the desktop
    period = 1.0 / SEND_HZ
    last = time.monotonic()
    reported = ""

    try:
        while True:
            # Drain whatever the pad has said since the last tick.
            while True:
                event = device.read_one()
                if event is None:
                    break
                if event.type == ecodes.EV_ABS:
                    if event.code == ecodes.ABS_X:
                        axes["pan"].set_raw(event.value)
                    elif event.code == ecodes.ABS_Y:
                        axes["tilt"].set_raw(event.value)
                elif event.type == ecodes.EV_KEY and event.value == 1:
                    if event.code in (ecodes.BTN_SOUTH, ecodes.BTN_A):
                        effector = not effector
                        logger.info("effector %s", "on" if effector else "off")
                    elif event.code in (ecodes.BTN_EAST, ecodes.BTN_B):
                        pan = config.control.home_pan_deg
                        tilt = config.control.home_tilt_deg
                        logger.info("home")

            now = time.monotonic()
            dt = now - last
            if dt < period:
                time.sleep(max(0.0, period - dt))
                continue
            last = now

            # ABS_Y is positive downwards on every pad; tilt is positive up.
            tilt_sign = 1.0 if args.invert_tilt else -1.0
            pan += axes["pan"].value * args.max_rate * dt
            tilt += tilt_sign * axes["tilt"].value * args.max_rate * dt
            pan = min(max(pan, pan_limits.min_deg), pan_limits.max_deg)
            tilt = min(max(tilt, tilt_limits.min_deg), tilt_limits.max_deg)

            link.send_angles(pan, tilt, EffectorState.ON if effector else EffectorState.OFF)
            statuses = link.poll()

            line = f"pan {pan:6.1f}  tilt {tilt:6.1f}"
            if statuses:
                latest = statuses[-1]
                line += f"   reported {latest.pan_deg:6.1f} / {latest.tilt_deg:6.1f}"
            if line != reported:
                print(f"\r{line}   ", end="", flush=True)
                reported = line
    except KeyboardInterrupt:
        print()
        logger.info("stopping")
    finally:
        with contextlib.suppress(OSError):
            device.ungrab()
        link.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
