# hailo-uav-tracker

**Real-time UAV detection and pan/tilt tracking on a Raspberry Pi 5 with a 13 TOPS Hailo-8L NPU.**

[![CI](https://github.com/pierrosimonestd-cpu/hailo-uav-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/pierrosimonestd-cpu/hailo-uav-tracker/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

A camera on a two-axis turret finds a drone in the sky, decides which one to
follow, and keeps it centred while it moves. Detection runs on the NPU; the
tracker and a latency-compensating controller run on the Pi's CPU; an ESP32
drives the servos and holds a failsafe that does not depend on the Pi being
alive.

Accuracy is reported on the public [DUT Anti-UAV](https://github.com/wangdongdut/DUT-Anti-UAV)
benchmark (IEEE T-ITS 2022) under the COCO protocol. Pointing performance is
reported from a closed-loop simulation, clearly labelled as such. Every number
in this repository is produced by a script in `benchmarks/` and written to a
JSON file you can inspect.

---

## What is interesting here

Most edge-vision projects stop at "the model runs at N FPS". The question this
one is built around is different:

> **What does inference latency cost you in degrees of pointing error?**

It turns out that is the *only* question that matters for a tracking turret, and
it has a clean answer. Inference time enters the control loop as dead time. Dead
time caps the achievable loop bandwidth at roughly

$$\omega_c \approx \frac{0.8}{T_d + \tau}$$

which for this hardware is under 1 Hz — while the servos themselves could slew at
600 °/s. **The loop is latency-limited, not actuator-limited.** That is what the
accelerator buys, and it is measurable:

<!-- BEGIN GENERATED: readme-latency -->
| Sense-to-act latency | RMS pointing error |  |
| --- | --- | --- |
| 10 ms | **0.68°** | an accelerator with headroom to spare |
| 45 ms | **0.83°** | the budget `configs/rpi5_hailo8l.yaml` assumes |
| 120 ms | **1.21°** | a pipeline four times slower |
| 300 ms | **2.77°** | a pipeline near the edge of usable |
<!-- END GENERATED: readme-latency -->

Along the way, three things turned out to be more subtle than expected, and each
is written up with the wrong answers included:

- **Velocity feed-forward has two plausible formulations, and both are unstable.**
  Reconstructing the target's absolute rate from "apparent rate plus my own rate"
  is positive feedback with loop gain equal to the feed-forward gain.
  Reconstructing it from the commanded *angle* looks safe but puts a positive
  real pole at `1/τ_servo` into the loop. What works is reconstructing through a
  forward model of the actuator. [Write-up →](docs/control-design.md#5-velocity-feed-forward-and-two-ways-to-get-it-wrong)
- **The aim-time lead is the actuator's lag, not another detection latency.**
  Getting this wrong is invisible: it is correct at one particular frame rate.
  It produced a spurious minimum in the latency curve that looked like a real
  result. [Write-up →](docs/control-design.md#6-aim-time-a-subtle-but-expensive-detail)
- **The obvious serial protocol fails silently and permanently.** Two raw bytes
  per command means one dropped byte swaps pan and tilt for as long as the link
  stays up, and neither end can tell. [Write-up →](docs/hardware.md#why-not-two-raw-bytes)

---

## Results

### Detection — DUT Anti-UAV test split, COCO protocol

<!-- BEGIN GENERATED: detection -->
_No results yet. Run the benchmark to populate this table._
<!-- END GENERATED: detection -->

Read `AP_S` first: **52% of the objects in this dataset are smaller than 32×32
pixels**, and the median object covers under one tenth of one percent of the
frame. An aggregate AP can look respectable while the model has stopped seeing
distant targets, which are the ones worth detecting early.

### Latency

<!-- BEGIN GENERATED: latency -->
_No results yet. Run the benchmark to populate this table._
<!-- END GENERATED: latency -->

### Velocity feed-forward — closed-loop simulation

<!-- BEGIN GENERATED: readme-feedforward -->
| Target angular rate | PID only | PID + feed-forward |  |
| ---: | ---: | ---: | ---: |
| 4.7 °/s | 1.02° | **0.34°** | 67% lower |
| 14.1 °/s | 2.89° | **0.83°** | 71% lower |
| 28.3 °/s | 5.48° | **2.86°** | 48% lower |
<!-- END GENERATED: readme-feedforward -->

Full tables, methodology and the reproduction commands: **[docs/benchmarks.md](docs/benchmarks.md)**

---

## Quick start

### On a laptop, no hardware needed

```bash
git clone https://github.com/pierrosimonestd-cpu/hailo-uav-tracker
cd hailo-uav-tracker
pip install -e ".[dev,onnx,bench]"

pytest tests                                    # the whole library, no NPU required
python benchmarks/bench_control.py --study all  # reproduce the pointing tables
uavtrack simulate --trajectory crossing --latency-ms 45
```

### On the Raspberry Pi

```bash
sudo apt install -y hailo-all python3-picamera2
pip install -e .                                # NumPy, OpenCV, PyYAML, pyserial

uavtrack check --config configs/rpi5_hailo8l.yaml
uavtrack run   --config configs/rpi5_hailo8l.yaml --display
```

Full deployment guide, including compiling the model:
**[docs/hailo-deployment.md](docs/hailo-deployment.md)**

### Train your own detector

```bash
python tools/fetch_dut_antiuav.py --out data/dut_antiuav   # 10,000 annotated images
python tools/prepare_dataset.py   --root data/dut_antiuav  # VOC → YOLO + COCO
python tools/train_uav.py --epochs 20 --cache ram
python tools/export_onnx.py --weights runs/train/uav_yolov8n/weights/best.pt
bash tools/compile_hailo.sh models/uav_yolov8n_640.onnx    # needs the Hailo DFC
```

---

## How it works

```
┌────────────┐   BGR frame    ┌──────────────┐   detections   ┌─────────────┐
│  Camera    │───────────────▶│   Detector   │───────────────▶│   Tracker   │
│ Picamera2  │  1280×720@30   │  Hailo-8L    │  boxes+scores  │  ByteTrack  │
└────────────┘                │  YOLOv8n     │                │  + Kalman   │
      ▲                       └──────────────┘                └─────────────┘
      │                                                              │
      │ camera is rigidly mounted on the turret                      │
      │                                                              ▼
┌────────────┐   pan/tilt    ┌──────────────┐   target px    ┌─────────────┐
│  Servos    │◀──────────────│  ESP32       │◀───────────────│  Controller │
│  MG90S ×2  │     PWM       │ framed UART  │  pan/tilt deg  │  PID + FF   │
└────────────┘               └──────────────┘                └─────────────┘
```

| Stage | Runs on | Why |
|---|---|---|
| Detection + NMS | **Hailo-8L NPU** | 13 TOPS; NMS is compiled into the HEF so the CPU never sees raw head tensors |
| Tracking + control | Pi 5 CPU | Microseconds, NumPy only |
| Servo drive + failsafe | ESP32 | Hard-real-time PWM, and a failsafe independent of the Pi |

**Design notes worth reading:** [architecture.md](docs/architecture.md) ·
[control-design.md](docs/control-design.md) · [hardware.md](docs/hardware.md) ·
[dataset.md](docs/dataset.md)

### A few decisions and their reasons

**The Raspberry Pi install has no PyTorch.** Runtime dependencies are NumPy,
OpenCV, PyYAML and pyserial. The YOLO decode path is written in NumPy for this
reason. Training and benchmarking live in optional extras.

**One `Detection` type, three backends.** Hailo, ONNX Runtime and PyTorch all
return identical objects in original-frame pixels. The accuracy benchmark runs
the *same code path* the live pipeline runs, so it measures the system rather
than a benchmark-only wrapper — and the tests run on machines with no NPU.

**Timestamps come from capture, not from a nominal frame rate.** Inference time
on a shared NPU varies; a loop that assumes fixed `dt` accumulates a timing error
the Kalman filter cannot see.

**The failsafe is in the firmware.** If the Pi crashes, Python cannot clean up —
that is what crashing means. The ESP32 cuts the effector and detaches the servos
after 500 ms of silence, and the Pi sends heartbeats so silence is always
deliberate.

**Config typos are fatal.** `conf_treshold: 0.5` raises at startup instead of
being silently ignored.

---

## Repository layout

```
src/uavtrack/      detect/ · track/ · control/ · io/ · data/   the library
firmware/          ESP32 pan/tilt controller, framed protocol
hardware/cad/      printable STL/3MF plus Creo sources
benchmarks/        accuracy, latency, hard negatives, control studies
tools/             dataset fetch/prepare, train, export, quantise, Hailo compile
configs/           Pi + Hailo, and a desktop ONNX config
docs/              architecture, control design, benchmarks, hardware, dataset
tests/             132 tests, plus a C++ conformance test for the firmware
```

## Testing

```bash
pytest tests                       # runs anywhere; no NPU, no PyTorch, no turret
```

CI additionally compiles the **actual firmware header** with a host C++ compiler
and runs the same protocol vectors and adversarial byte streams against it, so
the ESP32 and Python implementations cannot drift apart silently. It also builds
the firmware with `arduino-cli`, and fails if `docs/benchmarks.md` no longer
matches the JSON files in `benchmarks/results/`.

The tests that earn their keep are the ones pinning down specific failures: a
byte dropped on the serial link, a frame missed by the detector, an integrator
charging against a travel limit. Two real bugs found this way are documented in
[control-design.md](docs/control-design.md) §6 and [dataset.md](docs/dataset.md).

## Credits and licence

MIT, see [LICENSE](LICENSE).

- **DUT Anti-UAV** — Zhao, Zhang, Li, Wang, *Vision-based Anti-UAV Detection and
  Tracking*, IEEE T-ITS 2022. [Dataset](https://github.com/wangdongdut/DUT-Anti-UAV)
- **ByteTrack** — Zhang et al., ECCV 2022,
  [arXiv:2110.06864](https://arxiv.org/abs/2110.06864). The association strategy;
  this is an independent implementation, not a port.
- **Ultralytics YOLOv8** — training and export.
- **Hailo** — HailoRT, the Dataflow Compiler and the published Model Zoo figures
  quoted in [benchmarks.md](docs/benchmarks.md) §4.
- **Open Images V7** — bird images for the hard-negative benchmark.

Hailo throughput figures quoted in this repository are Hailo's own published
COCO measurements on an Intel host, attributed as such, and are not measurements
from this project.
