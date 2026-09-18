"""The serial link, against a fake port.

What is worth testing here is not that bytes go out, but the safety behaviour:
that heartbeats keep the firmware's failsafe from tripping while the loop is
open, that the effector is commanded off before the port closes, and that a
chatty or corrupt peer cannot stall or confuse the control loop.
"""

from __future__ import annotations

import pytest

from uavtrack.io.protocol import (
    EffectorState,
    FrameDecoder,
    MessageType,
    SetAngles,
    Status,
    encode,
)
from uavtrack.io.serial_link import SerialLink


class FakePort:
    """A serial port that records writes and replays scripted reads."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
        self.written = bytearray()
        self.to_read = bytearray()
        self.closed = False
        self.flushed = 0
        self.input_buffer_resets = 0
        self.fail_writes = False

    @property
    def in_waiting(self) -> int:
        return len(self.to_read)

    def write(self, data: bytes) -> int:
        if self.fail_writes:
            raise OSError("device disconnected")
        self.written.extend(data)
        return len(data)

    def read(self, count: int) -> bytes:
        chunk = bytes(self.to_read[:count])
        del self.to_read[:count]
        return chunk

    def flush(self) -> None:
        self.flushed += 1

    def reset_input_buffer(self) -> None:
        self.input_buffer_resets += 1

    def close(self) -> None:
        self.closed = True

    # -- helpers for the tests ------------------------------------------------

    def frames(self):
        """Decode everything written so far."""
        return FrameDecoder().feed(bytes(self.written))

    def queue(self, data: bytes) -> None:
        self.to_read.extend(data)


@pytest.fixture
def link():
    """A link over a fake port, with the bootloader wait skipped."""
    port = FakePort()
    link = SerialLink("FAKE", port_factory=lambda *a, **k: port, reset_delay_s=0.0)
    link.port = port  # type: ignore[attr-defined]
    return link


# ----------------------------------------------------------------- transmit


def test_open_clears_the_input_buffer(link):
    """Whatever the ESP32 emitted while resetting must not be parsed as a frame."""
    assert link.port.input_buffer_resets == 1


def test_send_angles_writes_one_well_formed_frame(link):
    assert link.send_angles(91.25, 45.5, EffectorState.ON)

    frames = link.port.frames()
    assert len(frames) == 1
    assert frames[0].message_type is MessageType.SET_ANGLES

    command = SetAngles.from_payload(frames[0].payload)
    assert command.pan_deg == pytest.approx(91.25, abs=0.005)
    assert command.tilt_deg == pytest.approx(45.5, abs=0.005)
    assert command.effector is EffectorState.ON


def test_sequence_numbers_increment_so_the_firmware_can_spot_gaps(link):
    for _ in range(5):
        link.send_angles(90.0, 90.0)
    assert [f.sequence for f in link.port.frames()] == [1, 2, 3, 4, 5]


def test_sequence_wraps_at_a_byte(link):
    link._sequence = 254
    link.send_heartbeat()
    link.send_heartbeat()
    link.send_heartbeat()
    assert [f.sequence for f in link.port.frames()] == [255, 0, 1]


def test_write_failure_is_reported_not_raised(link):
    link.port.fail_writes = True
    assert link.send_angles(90.0, 90.0) is False
    assert link.stats.write_errors == 1


# ---------------------------------------------------------------- heartbeat


def test_maintain_sends_a_heartbeat_when_the_link_has_been_quiet(link):
    link.heartbeat_interval_s = 0.0
    link.maintain()

    frames = link.port.frames()
    assert len(frames) == 1
    assert frames[0].message_type is MessageType.HEARTBEAT


def test_maintain_is_a_no_op_while_commands_are_flowing(link):
    link.heartbeat_interval_s = 60.0
    link.send_angles(90.0, 90.0)
    link.maintain()
    assert len(link.port.frames()) == 1, "a recent command already proves the link is alive"


# ------------------------------------------------------------------- receive


def test_status_frames_are_decoded(link):
    status = Status(last_seq=42, dropped_frames=3, crc_errors=1, effector=EffectorState.OFF)
    link.port.queue(encode(MessageType.STATUS, 1, status.payload()))

    received = link.poll()
    assert received == [status]
    assert link.last_status == status
    assert link.stats.frames_received == 1


def test_poll_on_a_silent_port_returns_nothing(link):
    assert link.poll() == []
    assert link.last_status is None


def test_poll_reassembles_a_frame_split_across_reads(link):
    frame = encode(MessageType.STATUS, 1, Status(1, 0, 0, EffectorState.OFF).payload())
    link.port.queue(frame[:4])
    assert link.poll() == []
    link.port.queue(frame[4:])
    assert len(link.poll()) == 1


def test_poll_bounds_how_much_it_reads(link):
    """A firmware gone chatty must not be able to stall the control loop."""
    link.port.queue(b"\x00" * 10_000)
    link.poll(max_bytes=256)
    assert link.port.in_waiting == 10_000 - 256


def test_corrupt_status_is_counted_and_dropped(link):
    frame = bytearray(encode(MessageType.STATUS, 1, Status(1, 0, 0, EffectorState.OFF).payload()))
    frame[6] ^= 0xFF
    link.port.queue(bytes(frame))

    assert link.poll() == []
    assert link.stats.crc_errors == 1
    assert link.last_status is None


def test_link_resynchronises_after_garbage(link):
    good = encode(MessageType.STATUS, 7, Status(7, 0, 0, EffectorState.OFF).payload())
    link.port.queue(b"\xde\xad\xbe\xef" + good)

    received = link.poll(max_bytes=4096)
    assert len(received) == 1
    assert received[0].last_seq == 7


# ------------------------------------------------------------------ shutdown


def test_close_disarms_the_effector_before_closing_the_port(link):
    port = link.port
    link.close()

    frames = port.frames()
    assert frames, "close must send a final command"
    final = SetAngles.from_payload(frames[-1].payload)
    assert final.effector is EffectorState.OFF, "the effector must never be left energised"
    assert port.closed


def test_close_still_closes_the_port_when_the_write_fails(link):
    port = link.port
    port.fail_writes = True
    link.close()
    assert port.closed, "a dead link must not leak the file handle"


def test_context_manager_closes_the_port():
    port = FakePort()
    with SerialLink("FAKE", port_factory=lambda *a, **k: port, reset_delay_s=0.0):
        pass
    assert port.closed
