# Architecture

## The loop, end to end

```
┌────────────┐   BGR frame    ┌──────────────┐   detections   ┌─────────────┐
│  Camera    │───────────────▶│   Detector   │───────────────▶│   Tracker   │
│ Picamera2  │  1280×720@30   │  Hailo-8L    │  boxes+scores  │  ByteTrack  │
└────────────┘                │  YOLOv8n     │                │  + Kalman   │
      ▲                       └──────────────┘                └─────────────┘
      │                                                              │
      │ camera is rigidly mounted on the turret                      │ tracks
      │                                                              ▼
┌────────────┐   pan/tilt    ┌──────────────┐   target px    ┌─────────────┐
│  Servos    │◀──────────────│  ESP32       │◀───────────────│  Controller │
│  MG90S ×2  │     PWM       │  framed UART │  pan/tilt deg  │ PID + FF    │
└────────────┘               └──────────────┘                └─────────────┘
                                                                     ▲
                                                             ┌─────────────┐
                                                             │  Selector   │
                                                             │  hysteresis │
                                                             └─────────────┘
```

Where the work happens:

| Stage | Runs on | Why there |
|---|---|---|
| Capture | Pi 5, libcamera | Direct to Picamera2, no V4L2 shim |
| Detection | **Hailo-8L NPU** | 13 TOPS; the Pi CPU cannot do this in real time |
| NMS | **Hailo-8L NPU** | Compiled into the HEF, so the CPU never sees raw head tensors |
| Tracking | Pi 5 CPU | Microseconds; NumPy only |
| Control | Pi 5 CPU | Microseconds |
| Servo drive | ESP32 | Hard-real-time PWM, and a failsafe independent of the Pi |

## Package layout

```
src/uavtrack/
├── detect/            Detection backends behind one interface
│   ├── base.py          Detection, DetectorBackend
│   ├── preprocess.py    Letterbox + its inverse
│   ├── postprocess.py   YOLO head decode + NMS, NumPy only
│   ├── hailo_backend.py HailoRT async API, on-chip NMS
│   ├── onnx_backend.py  ONNX Runtime (reference implementation)
│   ├── ultralytics_backend.py  PyTorch (training-time reference)
│   └── factory.py       Backend selection
├── track/             Multi-object tracking
│   ├── kalman.py        Constant-velocity filter, DWNA process noise
│   ├── bytetrack.py     Two-stage IoU association
│   └── selector.py      Primary-target policy with hysteresis
├── control/           Pointing
│   ├── pid.py           Anti-windup, derivative-on-measurement, filtered
│   ├── gimbal.py        Pixel→angle, bearing reconstruction, feed-forward
│   ├── plant.py         Servo model and generic discrete transfer functions
│   └── simulator.py     Closed-loop simulation
├── io/                The outside world
│   ├── camera.py        Picamera2 / OpenCV / image-sequence sources
│   ├── protocol.py      Framed wire protocol with CRC-8
│   └── serial_link.py   Transport, heartbeat, statistics
├── data/voc.py        VOC → YOLO and VOC → COCO conversion
├── config.py          Typed YAML config that rejects unknown keys
├── pipeline.py        Wires it all together
├── viz.py             Overlays
└── cli.py             `uavtrack run | check | simulate`
```

## Design decisions

### One `Detection` type, three backends

Every backend returns identical `Detection` objects in **original-frame pixel
coordinates**. Nothing downstream knows whether inference ran on a Hailo-8L, on
ONNX Runtime or on PyTorch. This is what makes the accuracy benchmarks
meaningful: `benchmarks/eval_detection.py` runs the same code path the live
pipeline runs, so it measures the system rather than a benchmark-only wrapper.

It is also what makes development possible at all — the NPU exists on exactly
one machine, and the tests need to run everywhere.

### The Raspberry Pi install has no PyTorch

Runtime dependencies are NumPy, OpenCV, PyYAML and pyserial. Training, export
and benchmarking dependencies are extras (`[train]`, `[onnx]`, `[bench]`). The
decode path in `postprocess.py` is NumPy for this reason.

### Timestamps come from capture, not from a nominal frame rate

Inference time on a shared NPU varies. A loop assuming a fixed `dt` accumulates
timing error the Kalman filter cannot see. Real elapsed times are threaded
through every stage, and `dt <= 0` is rejected rather than propagated.

### Failsafe lives in the firmware, not in Python

If the Pi crashes, the Python process cannot clean up — that is what crashing
means. The ESP32 trips its own failsafe after 500 ms of silence: effector cut,
servos detached. The Pi sends heartbeats to prove it is alive, so silence is
always deliberate.

### Config is a dataclass tree, and typos are fatal

`conf_treshold: 0.5` in a YAML dict would be silently ignored. Here it raises at
startup. On a system where a wrong gain means a turret hitting its end stop,
failing loudly at load time is the cheap option.

## Data flow per frame

1. **Capture** — `(monotonic_timestamp, bgr_frame)`. Timestamped at capture.
2. **Letterbox** — aspect-preserving resize to 640×640, centre-padded with 114.
   The `Letterbox` object carries its own inverse.
3. **Inference** — uint8 straight to the NPU; normalisation happens on chip.
4. **NMS** — on chip. Returns per-class `[y_min, x_min, y_max, x_max, score]`,
   normalised to the letterboxed input.
5. **Inverse transform** — back to original-frame pixels, clipped.
6. **Association** — two-stage IoU: confident detections first, then a rescue
   pass with weak ones. Kalman-predicted boxes are what is matched against.
7. **Selection** — score by confidence, apparent size and centrality, with an
   incumbent bonus and a dwell time so the turret cannot oscillate between
   targets.
8. **Control** — reconstruct absolute bearing through the servo model, fit a
   rate over a sliding window, extrapolate to the aim time, PID plus feed-forward.
9. **Command** — one framed, CRC-checked `SET_ANGLES`.

## Failure modes and what handles them

| Failure | Handled by |
|---|---|
| Detector misses frames | Kalman coasting; graceful to ~50% loss |
| Two UAVs in frame | Selector hysteresis and dwell time |
| Target occluded | Track kept `LOST` for `max_age` frames |
| Serial byte dropped | Sync word + CRC-8; frame dropped, never misread |
| Pi crashes or unplugged | ESP32 failsafe after 500 ms |
| Target beyond travel | Axis limits clamp; anti-windup prevents the overshoot on return |
| Target lost entirely | Controller integrators reset, so no stale error on re-acquisition |
| Bad config | Rejected at load |

---

**See also:** [control-design.md](control-design.md) ·
[hailo-deployment.md](hailo-deployment.md) · [hardware.md](hardware.md)
