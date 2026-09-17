"""Wire protocol between the Raspberry Pi and the ESP32 servo controller.

The obvious protocol -- write two raw bytes, pan then tilt, and have the
firmware read them whenever two bytes are available -- has a failure mode that
is worth spelling out, because it is silent and permanent. Drop or corrupt a
single byte anywhere on the link and every subsequent pair is read off by one:
pan is interpreted as tilt and tilt as the next pan, for as long as the link
stays up. There is no way for either end to notice, and the turret simply
behaves wrongly forever.

This protocol fixes that with three things:

* A two-byte sync word, so a receiver that loses framing can find it again.
* A length byte and a CRC-8, so a corrupt frame is dropped rather than acted on.
* A sequence number, so the firmware can detect and report gaps.

Frame layout, little-endian::

    0xA5 0x5A  LEN  TYPE  SEQ  PAYLOAD[LEN-2]  CRC8

``LEN`` counts ``TYPE``, ``SEQ`` and the payload -- everything the CRC covers.

The CRC is an 8-bit CRC over the polynomial ``0x31``
(:math:`x^8 + x^5 + x^4 + 1`), MSB-first, initial value ``0x00``, no final XOR
and no bit reflection. That is *not* CRC-8/MAXIM-DOW, which uses the same
polynomial but reflects both input and output -- the two disagree on almost
every message, so do not substitute one for the other. Its check value over
``b"123456789"`` is ``0xA2``. It was chosen because it is eight lines of C, it
detects every single-bit error and every burst up to eight bits, and frames here
are at most a dozen bytes.

Angles travel as centidegrees in a uint16, which resolves 0.01 degrees over the
full 0-180 degree travel -- roughly fifty times finer than the servos can
actually position, so quantisation here is never the limiting factor.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

SYNC_0 = 0xA5
SYNC_1 = 0x5A

#: Longest frame this protocol can express, used to size the firmware's buffer.
MAX_FRAME_LEN = 64


class MessageType(IntEnum):
    """Frame types understood by both ends."""

    SET_ANGLES = 0x01  # Pi -> ESP32: commanded pan/tilt and effector state
    HEARTBEAT = 0x02  # Pi -> ESP32: keep the failsafe from tripping
    STATUS = 0x81  # ESP32 -> Pi: acknowledgement and link statistics


class EffectorState(IntEnum):
    """Requested state of the deterrent output."""

    OFF = 0
    ON = 1


def crc8(data: bytes) -> int:
    """CRC-8 over ``data``: polynomial 0x31, MSB-first, init 0x00, no reflection.

    Must stay byte-for-byte identical to ``crc8`` in
    ``firmware/esp32_pantilt/protocol.h``. The fixed vectors in
    ``tests/test_protocol.py`` pin the exact values both sides must produce.

    Args:
        data: Bytes to checksum.

    Returns:
        The CRC as an integer in ``0..255``.
    """
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            # Written as an explicit branch rather than a ternary so it stays
            # line-for-line comparable with the C in firmware/esp32_pantilt.
            if crc & 0x80:  # noqa: SIM108
                crc = ((crc << 1) ^ 0x31) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


@dataclass(frozen=True)
class SetAngles:
    """Commanded pan/tilt angles and effector state."""

    pan_deg: float
    tilt_deg: float
    effector: EffectorState = EffectorState.OFF

    def payload(self) -> bytes:
        pan = _to_centidegrees(self.pan_deg)
        tilt = _to_centidegrees(self.tilt_deg)
        return struct.pack("<HHB", pan, tilt, int(self.effector))

    @classmethod
    def from_payload(cls, payload: bytes) -> SetAngles:
        if len(payload) != 5:
            raise ValueError(f"SET_ANGLES payload must be 5 bytes, got {len(payload)}")
        pan, tilt, effector = struct.unpack("<HHB", payload)
        return cls(pan / 100.0, tilt / 100.0, EffectorState(effector))


@dataclass(frozen=True)
class Status:
    """Link statistics reported by the firmware."""

    last_seq: int
    dropped_frames: int
    crc_errors: int
    effector: EffectorState

    def payload(self) -> bytes:
        return struct.pack(
            "<BHHB", self.last_seq, self.dropped_frames, self.crc_errors, int(self.effector)
        )

    @classmethod
    def from_payload(cls, payload: bytes) -> Status:
        if len(payload) != 6:
            raise ValueError(f"STATUS payload must be 6 bytes, got {len(payload)}")
        last_seq, dropped, crc_errors, effector = struct.unpack("<BHHB", payload)
        return cls(last_seq, dropped, crc_errors, EffectorState(effector))


def _to_centidegrees(angle_deg: float) -> int:
    """Clamp an angle to the servo range and convert to centidegrees."""
    clamped = min(max(angle_deg, 0.0), 180.0)
    return int(round(clamped * 100.0))


def encode(message_type: MessageType, sequence: int, payload: bytes = b"") -> bytes:
    """Build a complete frame.

    Args:
        message_type: Frame type.
        sequence: Sequence number; wraps at 256.
        payload: Type-specific payload bytes.

    Returns:
        The framed bytes, ready to write to the serial port.

    Raises:
        ValueError: If the frame would exceed :data:`MAX_FRAME_LEN`.
    """
    body = bytes([int(message_type), sequence & 0xFF]) + payload
    if len(body) + 4 > MAX_FRAME_LEN:
        raise ValueError(f"frame too long: {len(body) + 4} > {MAX_FRAME_LEN}")
    return bytes([SYNC_0, SYNC_1, len(body)]) + body + bytes([crc8(body)])


@dataclass
class DecodedFrame:
    """One successfully decoded frame."""

    message_type: MessageType
    sequence: int
    payload: bytes


class FrameDecoder:
    """Incremental, resynchronising frame decoder.

    Feed it whatever bytes arrive; it yields complete, CRC-checked frames and
    silently discards anything else. Being a state machine rather than a
    blocking reader is what lets the caller keep a bounded read budget per
    control cycle instead of stalling the loop on a partial frame.

    Attributes:
        crc_errors: Frames discarded because the CRC did not match.
        resyncs: Times the decoder had to hunt for a new sync word.
    """

    def __init__(self, max_frame_len: int = MAX_FRAME_LEN) -> None:
        self.max_frame_len = max_frame_len
        self._buffer = bytearray()
        self.crc_errors = 0
        self.resyncs = 0

    def reset(self) -> None:
        """Discard any partial frame."""
        self._buffer.clear()

    def feed(self, data: bytes) -> list[DecodedFrame]:
        """Add received bytes and return every complete frame found.

        Args:
            data: Newly received bytes, possibly empty, possibly spanning
                several frames or a fraction of one.

        Returns:
            Decoded frames in arrival order.
        """
        self._buffer.extend(data)
        frames: list[DecodedFrame] = []

        while True:
            start = self._buffer.find(bytes([SYNC_0, SYNC_1]))
            if start < 0:
                # Keep a trailing byte: it may be the first half of a sync word
                # split across two reads.
                if len(self._buffer) > 1:
                    self._buffer = self._buffer[-1:]
                break
            if start > 0:
                del self._buffer[:start]
                self.resyncs += 1

            if len(self._buffer) < 3:
                break
            length = self._buffer[2]
            if length + 4 > self.max_frame_len or length < 2:
                # Not a plausible header: this sync word was payload data.
                del self._buffer[:2]
                self.resyncs += 1
                continue

            total = 3 + length + 1
            if len(self._buffer) < total:
                break

            body = bytes(self._buffer[3 : 3 + length])
            received_crc = self._buffer[3 + length]

            if crc8(body) != received_crc:
                # Drop only the false sync word, not the whole candidate frame.
                # A truncated frame followed immediately by a good one is a real
                # pattern on a link that glitches; consuming the full candidate
                # length here would swallow the good frame with it.
                self.crc_errors += 1
                del self._buffer[:2]
                self.resyncs += 1
                continue

            del self._buffer[:total]

            try:
                message_type = MessageType(body[0])
            except ValueError:
                # Well-formed frame, unknown type: drop it but stay in sync.
                continue
            frames.append(DecodedFrame(message_type, body[1], body[2:]))

        return frames
