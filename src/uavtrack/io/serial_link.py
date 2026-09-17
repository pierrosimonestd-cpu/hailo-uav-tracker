"""Serial transport to the ESP32 pan/tilt controller."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from uavtrack.io.protocol import (
    EffectorState,
    FrameDecoder,
    MessageType,
    SetAngles,
    Status,
    encode,
)

logger = logging.getLogger(__name__)


@dataclass
class LinkStats:
    """Counters for link health, surfaced in the HUD and the logs."""

    frames_sent: int = 0
    frames_received: int = 0
    crc_errors: int = 0
    resyncs: int = 0
    write_errors: int = 0


class SerialLink:
    """Framed, non-blocking link to the turret firmware.

    Every write is a complete frame and every read is drained into a
    resynchronising decoder, so neither end can be left interpreting a partial
    message. Reads never block the control loop: whatever has arrived is
    consumed and the loop moves on.

    Args:
        port: Serial device, e.g. ``/dev/ttyUSB0`` or ``COM5``.
        baudrate: Must match the firmware. 500000 leaves a 12-byte frame at
            about 0.24 ms on the wire, which is negligible next to inference.
        timeout: Read timeout in seconds. Zero means a non-blocking read.
        heartbeat_interval_s: How often to send a heartbeat when no angle
            command has gone out. The firmware trips its failsafe if it hears
            nothing, so silence must be deliberate, never incidental.

    Raises:
        ImportError: If ``pyserial`` is not installed.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 500_000,
        timeout: float = 0.0,
        heartbeat_interval_s: float = 0.2,
    ) -> None:
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("pyserial is required for SerialLink: pip install pyserial") from exc

        self._serial = serial.Serial(port, baudrate, timeout=timeout, write_timeout=0.05)
        self._decoder = FrameDecoder()
        self._sequence = 0
        self._last_send = 0.0
        self.heartbeat_interval_s = heartbeat_interval_s
        self.stats = LinkStats()
        self.last_status: Status | None = None

        # The ESP32 resets when the serial port opens; anything written during
        # the bootloader window is lost or, worse, interpreted by it.
        time.sleep(2.0)
        self._serial.reset_input_buffer()

    def _next_sequence(self) -> int:
        self._sequence = (self._sequence + 1) & 0xFF
        return self._sequence

    def _write(self, frame: bytes) -> bool:
        try:
            self._serial.write(frame)
            self.stats.frames_sent += 1
            self._last_send = time.monotonic()
            return True
        except Exception:  # pragma: no cover - hardware dependent
            self.stats.write_errors += 1
            logger.exception("serial write failed")
            return False

    def send_angles(
        self, pan_deg: float, tilt_deg: float, effector: EffectorState = EffectorState.OFF
    ) -> bool:
        """Command the turret to an absolute pan/tilt position.

        Returns:
            True if the frame was written.
        """
        payload = SetAngles(pan_deg, tilt_deg, effector).payload()
        return self._write(encode(MessageType.SET_ANGLES, self._next_sequence(), payload))

    def send_heartbeat(self) -> bool:
        """Tell the firmware the link is alive without moving anything."""
        return self._write(encode(MessageType.HEARTBEAT, self._next_sequence()))

    def maintain(self) -> None:
        """Send a heartbeat if nothing else has gone out recently.

        Call once per control cycle. It is a no-op while commands are flowing.
        """
        if time.monotonic() - self._last_send >= self.heartbeat_interval_s:
            self.send_heartbeat()

    def poll(self, max_bytes: int = 256) -> list[Status]:
        """Read and decode whatever the firmware has sent.

        Args:
            max_bytes: Upper bound on bytes consumed, so a firmware that has
                gone chatty cannot stall the control loop.

        Returns:
            Any status frames received since the last poll.
        """
        waiting = min(getattr(self._serial, "in_waiting", 0), max_bytes)
        data = self._serial.read(waiting) if waiting else b""

        statuses: list[Status] = []
        for frame in self._decoder.feed(data):
            self.stats.frames_received += 1
            if frame.message_type is MessageType.STATUS:
                try:
                    status = Status.from_payload(frame.payload)
                except ValueError:
                    logger.warning("malformed STATUS payload: %s", frame.payload.hex())
                    continue
                self.last_status = status
                statuses.append(status)

        self.stats.crc_errors = self._decoder.crc_errors
        self.stats.resyncs = self._decoder.resyncs
        return statuses

    def close(self) -> None:
        """Disable the effector and close the port.

        Ordering matters: the effector is commanded off *before* the port is
        closed, so a crash-and-restart cycle cannot leave it energised.
        """
        try:
            self.send_angles(90.0, 90.0, EffectorState.OFF)
            self._serial.flush()
        except Exception:  # pragma: no cover - best effort on shutdown
            logger.exception("failed to send shutdown command")
        finally:
            self._serial.close()

    def __enter__(self) -> SerialLink:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
