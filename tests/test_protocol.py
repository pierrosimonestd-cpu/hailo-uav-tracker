"""Wire protocol framing, CRC and resynchronisation.

These tests exist because the failure they guard against is silent. A framing
bug does not crash anything; it points the turret somewhere wrong, forever.
"""

from __future__ import annotations

import random

import pytest

from uavtrack.io.protocol import (
    MAX_FRAME_LEN,
    SYNC_0,
    SYNC_1,
    EffectorState,
    FrameDecoder,
    MessageType,
    SetAngles,
    Status,
    crc8,
    encode,
)

# ------------------------------------------------------------------------ CRC


def test_crc8_known_vectors():
    """Fixed vectors the firmware must reproduce byte for byte.

    Verified against two independent formulations of the same CRC -- a
    256-entry table and textbook GF(2) long division -- before being written
    down, so this pins the algorithm rather than merely restating the
    implementation. 0xA2 over b"123456789" is the conventional check value for
    this polynomial and parameter set.
    """
    assert crc8(b"") == 0x00
    assert crc8(b"\x00") == 0x00
    assert crc8(b"\x01") == 0x31
    assert crc8(b"\x01\x02\x03\x04") == 0xFE
    assert crc8(bytes(range(16))) == 0xF0
    assert crc8(b"123456789") == 0xA2


def test_crc8_detects_every_single_bit_flip_in_a_command_frame():
    payload = bytes([int(MessageType.SET_ANGLES), 42]) + SetAngles(91.25, 45.5).payload()
    reference = crc8(payload)
    for index in range(len(payload)):
        for bit in range(8):
            corrupted = bytearray(payload)
            corrupted[index] ^= 1 << bit
            assert crc8(bytes(corrupted)) != reference


# ------------------------------------------------------------------- encoding


def test_angles_round_trip_within_quantisation():
    original = SetAngles(91.25, 45.5, EffectorState.ON)
    restored = SetAngles.from_payload(original.payload())
    assert restored.pan_deg == pytest.approx(91.25, abs=0.005)
    assert restored.tilt_deg == pytest.approx(45.5, abs=0.005)
    assert restored.effector is EffectorState.ON


@pytest.mark.parametrize(
    ("given", "expected"), [(-30.0, 0.0), (0.0, 0.0), (180.0, 180.0), (250.0, 180.0)]
)
def test_angles_are_clamped_to_the_servo_range(given, expected):
    assert SetAngles.from_payload(SetAngles(given, 90.0).payload()).pan_deg == pytest.approx(
        expected
    )


def test_status_round_trips():
    original = Status(last_seq=200, dropped_frames=7, crc_errors=3, effector=EffectorState.OFF)
    assert Status.from_payload(original.payload()) == original


def test_malformed_payload_lengths_are_rejected():
    with pytest.raises(ValueError):
        SetAngles.from_payload(b"\x00\x01")
    with pytest.raises(ValueError):
        Status.from_payload(b"\x00")


def test_frame_layout():
    frame = encode(MessageType.SET_ANGLES, 7, SetAngles(90.0, 90.0).payload())
    assert frame[0] == SYNC_0 and frame[1] == SYNC_1
    assert frame[2] == len(frame) - 4  # LEN covers TYPE, SEQ and payload
    assert frame[3] == int(MessageType.SET_ANGLES)
    assert frame[4] == 7
    assert frame[-1] == crc8(frame[3:-1])


def test_oversized_frames_are_refused():
    with pytest.raises(ValueError, match="too long"):
        encode(MessageType.SET_ANGLES, 0, b"\x00" * MAX_FRAME_LEN)


def test_sequence_wraps_at_a_byte():
    assert encode(MessageType.HEARTBEAT, 256)[4] == 0
    assert encode(MessageType.HEARTBEAT, 257)[4] == 1


# ------------------------------------------------------------------- decoding


def test_decode_round_trip():
    frame = encode(MessageType.SET_ANGLES, 7, SetAngles(91.25, 45.5, EffectorState.ON).payload())
    frames = FrameDecoder().feed(frame)
    assert len(frames) == 1
    assert frames[0].message_type is MessageType.SET_ANGLES
    assert frames[0].sequence == 7
    assert SetAngles.from_payload(frames[0].payload).pan_deg == pytest.approx(91.25, abs=0.005)


def test_decode_tolerates_byte_at_a_time_delivery():
    frame = encode(MessageType.SET_ANGLES, 1, SetAngles(90.0, 90.0).payload())
    decoder = FrameDecoder()
    got = [f for b in frame for f in decoder.feed(bytes([b]))]
    assert len(got) == 1


def test_decode_handles_several_frames_in_one_read():
    stream = b"".join(
        encode(MessageType.SET_ANGLES, i, SetAngles(90.0 + i, 90.0).payload()) for i in range(5)
    )
    frames = FrameDecoder().feed(stream)
    assert [f.sequence for f in frames] == [0, 1, 2, 3, 4]


def test_corrupt_frames_are_rejected_not_delivered():
    frame = bytearray(encode(MessageType.SET_ANGLES, 1, SetAngles(90.0, 90.0).payload()))
    frame[5] ^= 0xFF
    decoder = FrameDecoder()
    assert decoder.feed(bytes(frame)) == []
    assert decoder.crc_errors == 1


def test_decoder_resynchronises_after_leading_garbage():
    frame = encode(MessageType.SET_ANGLES, 3, SetAngles(90.0, 90.0).payload())
    decoder = FrameDecoder()
    frames = decoder.feed(b"\x00\x11\x22rubbish" + frame)
    assert len(frames) == 1
    assert decoder.resyncs >= 1


def test_a_truncated_frame_does_not_swallow_the_next_good_one():
    """The regression this decoder was rewritten for."""
    frame = encode(MessageType.SET_ANGLES, 9, SetAngles(100.0, 50.0).payload())
    frames = FrameDecoder().feed(frame[:5] + frame)
    assert len(frames) == 1
    assert frames[0].sequence == 9


def test_unknown_message_types_are_dropped_without_losing_sync():
    unknown = encode(0x7F, 1, b"\x00\x00")  # type: ignore[arg-type]
    good = encode(MessageType.SET_ANGLES, 2, SetAngles(90.0, 90.0).payload())
    frames = FrameDecoder().feed(unknown + good)
    assert len(frames) == 1
    assert frames[0].sequence == 2


def test_random_bytes_never_decode_as_frames():
    """A false positive here would mean acting on noise."""
    rng = random.Random(0)
    accepted = 0
    for _ in range(20_000):
        blob = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 40)))
        accepted += len(FrameDecoder().feed(blob))
    assert accepted == 0


def test_recovers_most_frames_and_corrupts_none_under_byte_loss():
    """The property the old two-byte protocol could not offer at any loss rate."""
    rng = random.Random(1)
    stream = b"".join(
        encode(MessageType.SET_ANGLES, i & 0xFF, SetAngles(90.0 + (i % 50) * 0.1, 45.0).payload())
        for i in range(200)
    )
    damaged = bytes(b for b in stream if rng.random() > 0.01)

    decoder = FrameDecoder()
    frames = decoder.feed(damaged)

    assert len(frames) > 150, "should recover the large majority of frames"
    # Every frame that *is* delivered must be intact: tilt was constant.
    for frame in frames:
        assert SetAngles.from_payload(frame.payload).tilt_deg == pytest.approx(45.0, abs=0.005)


def test_decoder_does_not_grow_without_bound_on_junk():
    decoder = FrameDecoder()
    for _ in range(100):
        decoder.feed(bytes([SYNC_0]) * 64)
    frame = encode(MessageType.SET_ANGLES, 5, SetAngles(90.0, 90.0).payload())
    assert len(decoder.feed(frame)) == 1
