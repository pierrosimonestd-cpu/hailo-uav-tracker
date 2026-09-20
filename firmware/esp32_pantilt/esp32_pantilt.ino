// Pan/tilt turret controller for the uavtrack project.
//
// Listens for framed commands from a Raspberry Pi over USB serial and drives
// two MG90S servos plus one effector output (relay or logic-level MOSFET).
//
// The firmware's job is to be the part of the system that is boring and safe.
// Everything clever happens on the Pi; everything here exists so that a crash,
// a yanked cable or a corrupted byte leaves the turret in a known state:
//
//   * Failsafe. No valid frame for FAILSAFE_TIMEOUT_MS and the effector is cut
//     and the servos are detached. A stalled host must not leave the turret
//     energised and pointing somewhere.
//   * Framing. Commands are CRC-checked (see protocol.h). A corrupt frame is
//     counted and dropped, never acted on.
//   * Rate limiting. Commanded angles are slew-limited here as well as on the
//     Pi, so a bad command cannot ask the servos for a step they would answer
//     with a current spike big enough to brown out the board.
//
// Wiring and power notes are in docs/hardware.md. The short version: do not
// power the servos from the ESP32's regulator, and tie all grounds together.

#include <ESP32Servo.h>

#include "protocol.h"

// ---------------------------------------------------------------- pin mapping
static const int PIN_PAN = 13;
static const int PIN_TILT = 12;
static const int PIN_EFFECTOR = 21;

// ------------------------------------------------------------------ behaviour
static const uint32_t BAUD_RATE = 500000;
static const uint32_t FAILSAFE_TIMEOUT_MS = 500;
static const uint32_t STATUS_INTERVAL_MS = 200;
static const uint32_t CONTROL_INTERVAL_MS = 10;  // 100 Hz servo update

// MG90S at 4.8 V manages 60 degrees in 0.1 s. Commanding faster than that
// achieves nothing except current draw.
static const float MAX_RATE_DEG_PER_S = 600.0f;

// Servo::write() takes whole degrees, so commanding through it quantises every
// position to one degree. On this rig one degree is about 32 pixels of frame,
// and the turret steps between them visibly instead of tracking smoothly.
// writeMicroseconds() carries the fractional angle through to the pulse, which
// over 500-2400 us across 180 degrees is roughly 10.6 us per degree -- finer
// than the servo's own resolution, which is where the limit belongs.
static const int PULSE_MIN_US = 500;
static const int PULSE_MAX_US = 2400;
static const float SERVO_RANGE_DEG = 180.0f;

static const float PAN_MIN_DEG = 0.0f;
static const float PAN_MAX_DEG = 180.0f;
// The camera mount fouls the base below 20 degrees; see docs/hardware.md.
static const float TILT_MIN_DEG = 20.0f;
static const float TILT_MAX_DEG = 160.0f;

static const float HOME_PAN_DEG = 90.0f;
static const float HOME_TILT_DEG = 90.0f;

// ---------------------------------------------------------------------- state
Servo servoPan;
Servo servoTilt;
FrameDecoder decoder;

static float targetPan = HOME_PAN_DEG;
static float targetTilt = HOME_TILT_DEG;
static float currentPan = HOME_PAN_DEG;
static float currentTilt = HOME_TILT_DEG;

static bool effectorOn = false;
static bool servosAttached = false;
static bool failsafeActive = true;

static uint32_t lastFrameMs = 0;
static uint32_t lastStatusMs = 0;
static uint32_t lastControlMs = 0;

static uint8_t lastSeq = 0;
static bool haveSeq = false;
static uint16_t droppedFrames = 0;
static uint8_t statusSeq = 0;

static float clampf(float value, float low, float high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

static void attachServos() {
  if (servosAttached) return;
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  servoPan.setPeriodHertz(50);
  servoTilt.setPeriodHertz(50);
  // 500-2400 us rather than the 1000-2000 default: MG90S servos use the wider
  // range, and clipping it throws away roughly a third of the travel.
  servoPan.attach(PIN_PAN, PULSE_MIN_US, PULSE_MAX_US);
  servoTilt.attach(PIN_TILT, PULSE_MIN_US, PULSE_MAX_US);
  servosAttached = true;
}

static void detachServos() {
  if (!servosAttached) return;
  servoPan.detach();
  servoTilt.detach();
  servosAttached = false;
}

static void enterFailsafe() {
  if (failsafeActive) return;
  failsafeActive = true;
  effectorOn = false;
  digitalWrite(PIN_EFFECTOR, LOW);
  // Detaching stops the PWM: the servos go limp rather than holding a stale
  // position against whatever they are pressed against.
  detachServos();
}

static void leaveFailsafe() {
  if (!failsafeActive) return;
  failsafeActive = false;
  attachServos();
  // Resume from where the servos actually are, not from a stale command.
  targetPan = currentPan;
  targetTilt = currentTilt;
}

static void handleSetAngles(uint8_t seq, const uint8_t *payload, uint8_t length) {
  if (length != sizeof(SetAnglesPayload)) return;

  SetAnglesPayload command;
  memcpy(&command, payload, sizeof(command));

  if (haveSeq) {
    const uint8_t expected = (uint8_t)(lastSeq + 1);
    if (seq != expected) {
      // Sequence gap: frames were lost somewhere. Report it, do not refuse the
      // command -- a fresh position is more useful than a stale one.
      droppedFrames = (uint16_t)(droppedFrames + (uint8_t)(seq - expected));
    }
  }
  lastSeq = seq;
  haveSeq = true;

  targetPan = clampf(command.pan_centideg / 100.0f, PAN_MIN_DEG, PAN_MAX_DEG);
  targetTilt = clampf(command.tilt_centideg / 100.0f, TILT_MIN_DEG, TILT_MAX_DEG);

  leaveFailsafe();
  effectorOn = command.effector != 0;
  digitalWrite(PIN_EFFECTOR, effectorOn ? HIGH : LOW);
}

static void sendStatus() {
  StatusPayload status;
  status.last_seq = lastSeq;
  status.dropped_frames = droppedFrames;
  status.crc_errors = decoder.crcErrors();
  status.effector = effectorOn ? 1 : 0;
  sendFrame(Serial, MSG_STATUS, statusSeq++, (const uint8_t *)&status, sizeof(status));
}

static int angleToMicroseconds(float angleDeg) {
  const float span = (float)(PULSE_MAX_US - PULSE_MIN_US);
  float pulse = PULSE_MIN_US + (angleDeg / SERVO_RANGE_DEG) * span;
  return (int)clampf(pulse + 0.5f, (float)PULSE_MIN_US, (float)PULSE_MAX_US);
}

static void stepServos(float dt) {
  const float maxStep = MAX_RATE_DEG_PER_S * dt;

  float deltaPan = targetPan - currentPan;
  deltaPan = clampf(deltaPan, -maxStep, maxStep);
  currentPan += deltaPan;

  float deltaTilt = targetTilt - currentTilt;
  deltaTilt = clampf(deltaTilt, -maxStep, maxStep);
  currentTilt += deltaTilt;

  if (servosAttached) {
    servoPan.writeMicroseconds(angleToMicroseconds(currentPan));
    servoTilt.writeMicroseconds(angleToMicroseconds(currentTilt));
  }
}

void setup() {
  Serial.begin(BAUD_RATE);

  pinMode(PIN_EFFECTOR, OUTPUT);
  digitalWrite(PIN_EFFECTOR, LOW);

  attachServos();
  servoPan.write((int)HOME_PAN_DEG);
  servoTilt.write((int)HOME_TILT_DEG);

  // Start in failsafe: the turret stays inert until the host proves it is
  // there. Anything else means a power cycle can move the turret on its own.
  failsafeActive = false;
  enterFailsafe();

  lastFrameMs = millis();
  lastControlMs = millis();
}

void loop() {
  const uint32_t now = millis();

  while (Serial.available() > 0) {
    if (decoder.feed((uint8_t)Serial.read())) {
      const uint8_t *body = decoder.body();
      const uint8_t type = body[0];
      const uint8_t seq = body[1];

      if (type == MSG_SET_ANGLES) {
        handleSetAngles(seq, body + 2, (uint8_t)(decoder.bodyLength() - 2));
        lastFrameMs = now;
      } else if (type == MSG_HEARTBEAT) {
        lastSeq = seq;
        haveSeq = true;
        lastFrameMs = now;
        leaveFailsafe();
      }
    }
  }

  if (now - lastFrameMs > FAILSAFE_TIMEOUT_MS) {
    enterFailsafe();
  }

  if (now - lastControlMs >= CONTROL_INTERVAL_MS) {
    stepServos((now - lastControlMs) / 1000.0f);
    lastControlMs = now;
  }

  if (now - lastStatusMs >= STATUS_INTERVAL_MS) {
    sendStatus();
    lastStatusMs = now;
  }
}
