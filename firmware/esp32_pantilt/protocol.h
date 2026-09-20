// Wire protocol shared with src/uavtrack/io/protocol.py.
//
// Keep the two in step: the Python side is the reference, and
// tests/test_protocol.py checks the CRC vectors this header must reproduce.
//
// Frame layout, little-endian:
//
//   0xA5 0x5A  LEN  TYPE  SEQ  PAYLOAD[LEN-2]  CRC8
//
// LEN counts TYPE, SEQ and the payload: everything the CRC covers.
//
// CRC-8 over polynomial 0x31 (x^8 + x^5 + x^4 + 1), MSB-first, init 0x00, no
// final XOR and no bit reflection. This is NOT CRC-8/MAXIM-DOW, which uses the
// same polynomial but reflects input and output. Check value over "123456789"
// is 0xA2; tests/test_protocol.py pins the vectors both sides must reproduce.

#pragma once

#include <stdint.h>

static const uint8_t SYNC_0 = 0xA5;
static const uint8_t SYNC_1 = 0x5A;
static const uint8_t MAX_FRAME_LEN = 64;

enum MessageType : uint8_t {
  MSG_SET_ANGLES = 0x01,  // host -> device: pan, tilt, effector
  MSG_HEARTBEAT  = 0x02,  // host -> device: link is alive
  MSG_STATUS     = 0x81,  // device -> host: acknowledgement and counters
};

// SET_ANGLES payload: pan and tilt in centidegrees, then the effector flag.
struct __attribute__((packed)) SetAnglesPayload {
  uint16_t pan_centideg;
  uint16_t tilt_centideg;
  uint8_t effector;
};

// STATUS payload: last sequence seen, the two error counters, then the angles
// the firmware is currently driving, in centidegrees.
//
// The angles are what makes camera ego-motion cancellable on the host. With a
// camera bolted to the turret, apparent target motion in the image is the sum
// of the target's motion and the turret's own; subtracting the turret's angle
// at capture time leaves the target's. Doing that from the host's *model* of
// the servos leaves whatever the model gets wrong, so the firmware reports
// what it is actually commanding instead.
struct __attribute__((packed)) StatusPayload {
  uint8_t last_seq;
  uint16_t dropped_frames;
  uint16_t crc_errors;
  uint8_t effector;
  int16_t pan_centideg;
  int16_t tilt_centideg;
};

inline uint8_t crc8(const uint8_t *data, uint8_t length) {
  uint8_t crc = 0x00;
  for (uint8_t i = 0; i < length; i++) {
    crc ^= data[i];
    for (uint8_t bit = 0; bit < 8; bit++) {
      crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x31) : (uint8_t)(crc << 1);
    }
  }
  return crc;
}

// Incremental decoder. Mirrors FrameDecoder in the Python module, including the
// behaviour on a CRC failure: drop only the false sync word and rescan, so a
// truncated frame cannot swallow a good one that follows it.
class FrameDecoder {
 public:
  void reset() { length_ = 0; }

  // Feed one received byte. Returns true when a complete, CRC-checked frame is
  // available in body() / bodyLength().
  bool feed(uint8_t byte) {
    if (length_ < MAX_FRAME_LEN) {
      buffer_[length_++] = byte;
    } else {
      // Buffer full without a valid frame: the stream is not what we expect.
      // Drop the oldest byte and keep hunting rather than deadlocking.
      shift(1);
      buffer_[length_++] = byte;
      resyncs_++;
    }
    return tryDecode();
  }

  const uint8_t *body() const { return body_; }
  uint8_t bodyLength() const { return bodyLength_; }
  uint16_t crcErrors() const { return crcErrors_; }
  uint16_t resyncs() const { return resyncs_; }

 private:
  void shift(uint8_t count) {
    if (count >= length_) {
      length_ = 0;
      return;
    }
    for (uint8_t i = 0; i + count < length_; i++) {
      buffer_[i] = buffer_[i + count];
    }
    length_ = (uint8_t)(length_ - count);
  }

  bool tryDecode() {
    while (true) {
      // Hunt for the sync word.
      uint8_t start = 0;
      bool found = false;
      while (start + 1 < length_) {
        if (buffer_[start] == SYNC_0 && buffer_[start + 1] == SYNC_1) {
          found = true;
          break;
        }
        start++;
      }
      if (!found) {
        // Keep at most one trailing byte: it may be a split sync word.
        if (length_ > 1) {
          shift((uint8_t)(length_ - 1));
          resyncs_++;
        }
        return false;
      }
      if (start > 0) {
        shift(start);
        resyncs_++;
      }

      if (length_ < 3) return false;
      const uint8_t payloadLen = buffer_[2];
      if (payloadLen < 2 || payloadLen + 4 > MAX_FRAME_LEN) {
        shift(2);
        resyncs_++;
        continue;
      }

      const uint8_t total = (uint8_t)(3 + payloadLen + 1);
      if (length_ < total) return false;

      if (crc8(&buffer_[3], payloadLen) != buffer_[3 + payloadLen]) {
        crcErrors_++;
        shift(2);
        resyncs_++;
        continue;
      }

      for (uint8_t i = 0; i < payloadLen; i++) body_[i] = buffer_[3 + i];
      bodyLength_ = payloadLen;
      shift(total);
      return true;
    }
  }

  uint8_t buffer_[MAX_FRAME_LEN];
  uint8_t length_ = 0;
  uint8_t body_[MAX_FRAME_LEN];
  uint8_t bodyLength_ = 0;
  uint16_t crcErrors_ = 0;
  uint16_t resyncs_ = 0;
};

// Write a framed message to a Stream-like object.
template <typename StreamT>
void sendFrame(StreamT &stream, uint8_t type, uint8_t seq, const uint8_t *payload,
               uint8_t payloadLen) {
  uint8_t body[MAX_FRAME_LEN];
  body[0] = type;
  body[1] = seq;
  for (uint8_t i = 0; i < payloadLen; i++) body[2 + i] = payload[i];
  const uint8_t bodyLen = (uint8_t)(2 + payloadLen);

  uint8_t header[3] = {SYNC_0, SYNC_1, bodyLen};
  const uint8_t crc = crc8(body, bodyLen);
  stream.write(header, 3);
  stream.write(body, bodyLen);
  stream.write(&crc, 1);
}
