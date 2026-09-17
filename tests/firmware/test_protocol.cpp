// Host-side conformance test for firmware/esp32_pantilt/protocol.h.
//
// The firmware and the Python host must agree byte for byte on framing and on
// the CRC. They are two implementations of one specification, written in two
// languages, and nothing but a test keeps them in step.
//
// This compiles the *actual firmware header* with a normal C++ compiler -- no
// Arduino, no ESP32 -- and checks it against the same vectors and the same
// adversarial streams as tests/test_protocol.py.
//
// Build and run:
//   c++ -std=c++17 -Wall -Wextra -Werror -I firmware/esp32_pantilt \
//       tests/firmware/test_protocol.cpp -o /tmp/test_protocol && /tmp/test_protocol

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "protocol.h"

static int failures = 0;

#define CHECK(cond, ...)                                     \
  do {                                                       \
    if (!(cond)) {                                           \
      std::printf("FAIL %s:%d  ", __FILE__, __LINE__);       \
      std::printf(__VA_ARGS__);                              \
      std::printf("\n");                                     \
      failures++;                                            \
    }                                                        \
  } while (0)

// A minimal Stream stand-in so sendFrame() can be exercised on the host.
struct BufferStream {
  std::vector<uint8_t> data;
  void write(const uint8_t *bytes, uint8_t length) {
    data.insert(data.end(), bytes, bytes + length);
  }
};

// Build a frame the way the Python encoder does, for decoding tests.
static std::vector<uint8_t> encodeFrame(uint8_t type, uint8_t seq,
                                        const std::vector<uint8_t> &payload) {
  std::vector<uint8_t> body{type, seq};
  body.insert(body.end(), payload.begin(), payload.end());

  std::vector<uint8_t> frame{SYNC_0, SYNC_1, static_cast<uint8_t>(body.size())};
  frame.insert(frame.end(), body.begin(), body.end());
  frame.push_back(crc8(body.data(), static_cast<uint8_t>(body.size())));
  return frame;
}

static std::vector<uint8_t> setAnglesPayload(double pan, double tilt, uint8_t effector) {
  SetAnglesPayload payload{};
  payload.pan_centideg = static_cast<uint16_t>(pan * 100.0 + 0.5);
  payload.tilt_centideg = static_cast<uint16_t>(tilt * 100.0 + 0.5);
  payload.effector = effector;

  std::vector<uint8_t> bytes(sizeof(payload));
  std::memcpy(bytes.data(), &payload, sizeof(payload));
  return bytes;
}

// ---------------------------------------------------------------------- CRC

static void testCrcVectors() {
  // These must match tests/test_protocol.py::test_crc8_known_vectors exactly.
  const uint8_t empty[] = {0};
  CHECK(crc8(empty, 0) == 0x00, "crc8(empty)");

  const uint8_t zero[] = {0x00};
  CHECK(crc8(zero, 1) == 0x00, "crc8(00)");

  const uint8_t one[] = {0x01};
  CHECK(crc8(one, 1) == 0x31, "crc8(01) = %02x", crc8(one, 1));

  const uint8_t four[] = {0x01, 0x02, 0x03, 0x04};
  CHECK(crc8(four, 4) == 0xFE, "crc8(01020304) = %02x", crc8(four, 4));

  uint8_t sixteen[16];
  for (uint8_t i = 0; i < 16; i++) sixteen[i] = i;
  CHECK(crc8(sixteen, 16) == 0xF0, "crc8(0..15) = %02x", crc8(sixteen, 16));

  const uint8_t check[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  CHECK(crc8(check, 9) == 0xA2, "crc8(123456789) = %02x", crc8(check, 9));
}

// ------------------------------------------------------------------ structs

static void testPayloadLayout() {
  // The Python side packs these with struct '<HHB' and '<BHHB'. If the compiler
  // inserted padding, the two would silently disagree about every field.
  CHECK(sizeof(SetAnglesPayload) == 5, "sizeof(SetAnglesPayload) = %zu",
        sizeof(SetAnglesPayload));
  CHECK(sizeof(StatusPayload) == 6, "sizeof(StatusPayload) = %zu", sizeof(StatusPayload));

  const std::vector<uint8_t> bytes = setAnglesPayload(91.25, 45.5, 1);
  CHECK(bytes.size() == 5, "payload size");
  // 9125 = 0x23A5 little-endian, 4550 = 0x11C6 little-endian.
  CHECK(bytes[0] == 0xA5 && bytes[1] == 0x23, "pan little-endian");
  CHECK(bytes[2] == 0xC6 && bytes[3] == 0x11, "tilt little-endian");
  CHECK(bytes[4] == 1, "effector");
}

// ----------------------------------------------------------------- encoding

static void testSendFrameMatchesLayout() {
  BufferStream stream;
  const std::vector<uint8_t> payload = setAnglesPayload(90.0, 90.0, 0);
  sendFrame(stream, MSG_SET_ANGLES, 7, payload.data(), static_cast<uint8_t>(payload.size()));

  const std::vector<uint8_t> expected = encodeFrame(MSG_SET_ANGLES, 7, payload);
  CHECK(stream.data == expected, "sendFrame output differs from the reference layout");
}

// ----------------------------------------------------------------- decoding

// Feed a whole buffer through the decoder, collecting every frame it yields.
static std::vector<std::vector<uint8_t>> decodeAll(const std::vector<uint8_t> &stream) {
  FrameDecoder decoder;
  std::vector<std::vector<uint8_t>> frames;
  for (uint8_t byte : stream) {
    if (decoder.feed(byte)) {
      frames.emplace_back(decoder.body(), decoder.body() + decoder.bodyLength());
    }
  }
  return frames;
}

static void testRoundTrip() {
  const std::vector<uint8_t> payload = setAnglesPayload(91.25, 45.5, 1);
  const auto frames = decodeAll(encodeFrame(MSG_SET_ANGLES, 7, payload));

  CHECK(frames.size() == 1, "expected 1 frame, got %zu", frames.size());
  if (frames.size() == 1) {
    CHECK(frames[0][0] == MSG_SET_ANGLES, "type");
    CHECK(frames[0][1] == 7, "sequence");

    SetAnglesPayload decoded{};
    std::memcpy(&decoded, frames[0].data() + 2, sizeof(decoded));
    CHECK(decoded.pan_centideg == 9125, "pan = %u", decoded.pan_centideg);
    CHECK(decoded.tilt_centideg == 4550, "tilt = %u", decoded.tilt_centideg);
    CHECK(decoded.effector == 1, "effector");
  }
}

static void testMultipleFramesInOneBurst() {
  std::vector<uint8_t> stream;
  for (uint8_t i = 0; i < 5; i++) {
    const auto frame = encodeFrame(MSG_SET_ANGLES, i, setAnglesPayload(90.0 + i, 45.0, 0));
    stream.insert(stream.end(), frame.begin(), frame.end());
  }
  CHECK(decodeAll(stream).size() == 5, "expected 5 frames");
}

static void testLeadingGarbageIsSkipped() {
  std::vector<uint8_t> stream{0x00, 0x11, 0x22, 0xA5, 0x33, 'j', 'u', 'n', 'k'};
  const auto frame = encodeFrame(MSG_SET_ANGLES, 3, setAnglesPayload(90.0, 90.0, 0));
  stream.insert(stream.end(), frame.begin(), frame.end());
  CHECK(decodeAll(stream).size() == 1, "should resynchronise past garbage");
}

static void testCorruptFrameIsRejected() {
  auto frame = encodeFrame(MSG_SET_ANGLES, 1, setAnglesPayload(90.0, 90.0, 0));
  frame[5] ^= 0xFF;
  CHECK(decodeAll(frame).empty(), "a corrupt frame must not be delivered");
}

static void testTruncatedFrameDoesNotSwallowTheNext() {
  // The regression the decoder was rewritten for: on a CRC failure, drop only
  // the false sync word, not the whole candidate frame.
  const auto frame = encodeFrame(MSG_SET_ANGLES, 9, setAnglesPayload(100.0, 50.0, 0));
  std::vector<uint8_t> stream(frame.begin(), frame.begin() + 5);
  stream.insert(stream.end(), frame.begin(), frame.end());

  const auto frames = decodeAll(stream);
  CHECK(frames.size() == 1, "expected the good frame to survive, got %zu", frames.size());
  if (frames.size() == 1) CHECK(frames[0][1] == 9, "sequence should be 9");
}

static void testRandomBytesNeverDecode() {
  std::mt19937 rng(0);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<int> length(1, 40);

  size_t accepted = 0;
  for (int trial = 0; trial < 20000; trial++) {
    std::vector<uint8_t> blob(static_cast<size_t>(length(rng)));
    for (auto &b : blob) b = static_cast<uint8_t>(byte(rng));
    accepted += decodeAll(blob).size();
  }
  CHECK(accepted == 0, "%zu random blobs decoded as frames", accepted);
}

static void testRecoversAfterASyncFlood() {
  // A long run with no valid frame must not wedge the fixed-size buffer.
  FrameDecoder decoder;
  for (int i = 0; i < 500; i++) decoder.feed(SYNC_0);

  const auto frame = encodeFrame(MSG_SET_ANGLES, 3, setAnglesPayload(10.0, 20.0, 0));
  size_t decoded = 0;
  for (uint8_t byte : frame) {
    if (decoder.feed(byte)) decoded++;
  }
  CHECK(decoded == 1, "should recover after a sync flood, decoded %zu", decoded);
}

static void testSurvivesByteLossWithoutCorruption() {
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> chance(0.0, 1.0);

  std::vector<uint8_t> stream;
  for (int i = 0; i < 200; i++) {
    const auto frame = encodeFrame(MSG_SET_ANGLES, static_cast<uint8_t>(i & 0xFF),
                                   setAnglesPayload(90.0 + (i % 50) * 0.1, 45.0, 0));
    stream.insert(stream.end(), frame.begin(), frame.end());
  }

  std::vector<uint8_t> damaged;
  for (uint8_t byte : stream) {
    if (chance(rng) > 0.01) damaged.push_back(byte);
  }

  const auto frames = decodeAll(damaged);
  CHECK(frames.size() > 150, "expected most frames to survive, got %zu", frames.size());

  // Tilt was constant: any frame delivered with a different value is corrupt.
  for (const auto &frame : frames) {
    SetAnglesPayload decoded{};
    std::memcpy(&decoded, frame.data() + 2, sizeof(decoded));
    CHECK(decoded.tilt_centideg == 4500, "corrupt frame delivered: tilt = %u",
          decoded.tilt_centideg);
  }
}

int main() {
  testCrcVectors();
  testPayloadLayout();
  testSendFrameMatchesLayout();
  testRoundTrip();
  testMultipleFramesInOneBurst();
  testLeadingGarbageIsSkipped();
  testCorruptFrameIsRejected();
  testTruncatedFrameDoesNotSwallowTheNext();
  testRandomBytesNeverDecode();
  testRecoversAfterASyncFlood();
  testSurvivesByteLossWithoutCorruption();

  if (failures == 0) {
    std::printf("firmware protocol conformance: all checks passed\n");
    return 0;
  }
  std::printf("firmware protocol conformance: %d check(s) failed\n", failures);
  return 1;
}
