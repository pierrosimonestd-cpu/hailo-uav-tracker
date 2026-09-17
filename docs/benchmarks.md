# Benchmarks

Every table below is generated from JSON files in
[`benchmarks/results/`](../benchmarks/results) by
`python benchmarks/make_report.py`. Nothing here is typed in by hand, and every
number traces back to a script you can re-run.

Results are labelled by kind:

- **measured** — produced by running code on real data on the stated hardware.
- **simulation** — produced by the closed-loop model in
  [`simulator.py`](../src/uavtrack/control/simulator.py). Useful for comparing
  configurations, not a substitute for a hardware measurement.
- **published** — figures from a vendor or a paper, cited and attributed.

---

## 1. Detection accuracy — DUT Anti-UAV test split

COCO protocol, 2,200 images, single class, IoU 0.50:0.95. See
[dataset.md](dataset.md) for the benchmark and the evaluation protocol.

```bash
python benchmarks/eval_detection.py \
    --model runs/train/uav_yolov8n/weights/best.pt \
    --tag yolov8n-fp32-pytorch
```

<!-- BEGIN GENERATED: detection -->
_No results yet. Run the benchmark to populate this table._
<!-- END GENERATED: detection -->

`AP_S` is the column to read first. Over half the objects in this dataset are
below 32×32 pixels, and an aggregate AP can look respectable while the model has
stopped seeing distant targets — which are the ones worth detecting early.

### Quantisation

The Hailo-8L is an INT8 accelerator, so the model that runs on the turret is not
the model that was trained. Reporting only FP32 numbers would be reporting a
model this project does not run.

Compiling a HEF needs the Hailo Dataflow Compiler, which requires a developer
account and does not run in CI. Static INT8 quantisation through ONNX Runtime
does run anywhere, and shares the mechanism that matters: per-tensor affine
quantisation calibrated on a sample of training images. It is a **proxy** for
the Hailo quantiser, not a substitute, and it is labelled as one in the table
above (`-int8-onnx` tags).

For scale, Hailo's own published float-to-hardware gap on COCO is 0.6–1.5 mAP
points for the YOLOv8/YOLO11 family (§4), which is the right order of magnitude
to expect.

---

## 2. Latency and throughput

```bash
python benchmarks/bench_latency.py --model models/uav_yolov8n_640.hef --frames 300
```

<!-- BEGIN GENERATED: latency -->
_No results yet. Run the benchmark to populate this table._
<!-- END GENERATED: latency -->

Quote the **p95**, not the mean. A control loop is hurt by the tail: one 200 ms
frame in fifty does more damage to a pointing solution than a mean that is 5 ms
worse, because the tracker's extrapolation error grows with the gap and the
controller cannot know a frame is late until it arrives.

`control.latency_s` in your config should be set from the median-to-p95 range
measured on **your** hardware.

---

## 3. Bird discrimination

Birds are the canonical false positive for ground-to-air UAV detection: similar
apparent size at range, same sky background, similar motion. DUT Anti-UAV
contains no labelled birds, so this is measured separately on Open Images V7
bird images with aircraft classes excluded.

```bash
python benchmarks/eval_hard_negatives.py --model runs/train/uav_yolov8n/weights/best.pt --fetch
```

<!-- BEGIN GENERATED: hard-negatives -->
_No results yet. Run the benchmark to populate this table._
<!-- END GENERATED: hard-negatives -->

The metric is the fraction of bird images that produce at least one UAV
detection at the deployed threshold — that is, how often the turret would swing
onto a pigeon.

---

## 4. Published Hailo-8L throughput

Not measurements from this project. These are **Hailo's published Model Zoo
figures on COCO**, measured on an Intel Core i5-9400 over PCIe Gen 3 ×4 with
Dataflow Compiler v2.19.0 — a different host and a different link from a
Raspberry Pi. Included because they set the expectation for what the accelerator
can do:

| Model | Input | mAP (float) | mAP (hardware) | FPS (batch 1) | FPS (batch 8) |
|---|---|---:|---:|---:|---:|
| yolov8n | 640×640 | 37.0 | 36.4 | 202 | 438 |
| yolov8s | 640×640 | 44.6 | 43.9 | 110 | 208 |
| yolov11n | 640×640 | 39.0 | 37.5 | 157 | 371 |
| yolov11s | 640×640 | 46.3 | 45.1 | 92.0 | 192 |

Source: [Hailo Model Zoo, HAILO8L object detection](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst)

yolov8n at 202 FPS gives the control loop roughly an order of magnitude more
frames than it can use — §5 shows the loop bandwidth is capped near 1 Hz by
latency. The useful question for this system is latency, not frame rate.

---

## 5. Closed-loop pointing — simulation

**These are simulation results.** The servo model — transport delay, first-order
lag, hard slew-rate limit — is documented in
[`plant.py`](../src/uavtrack/control/plant.py), and what it does *not* model
(backlash, stiction, boresight misalignment, structural flex) is listed in
[control-design.md](control-design.md) §9.

```bash
python benchmarks/bench_control.py --study all
```

### 5.1 Pointing error against sense-to-act latency

Target orbiting at 14 °/s; mean of five seeds; steady state after 2 s.

<!-- BEGIN GENERATED: control-latency -->
| Sense-to-act latency | RMS error | Peak error | In frame |
| ---: | ---: | ---: | ---: |
| 10 ms | 0.68° | 1.03° | 100% |
| 20 ms | 0.69° | 1.05° | 100% |
| 30 ms | 0.73° | 1.14° | 100% |
| 45 ms | 0.83° | 1.21° | 100% |
| 60 ms | 0.87° | 1.24° | 100% |
| 80 ms | 1.01° | 1.40° | 100% |
| 120 ms | 1.21° | 1.63° | 100% |
| 160 ms | 1.45° | 1.93° | 100% |
| 200 ms | 1.83° | 2.37° | 100% |
| 300 ms | 2.77° | 4.48° | 100% |
<!-- END GENERATED: control-latency -->

This is the table that justifies the accelerator. Inference time enters the
control loop as dead time, dead time caps the achievable loop bandwidth, and
bandwidth is what keeps a manoeuvring target centred.

### 5.2 Velocity feed-forward

<!-- BEGIN GENERATED: control-feedforward -->
| Peak target rate | Feed-forward off | Feed-forward on | Reduction |
| ---: | ---: | ---: | ---: |
| 4.7 °/s | 1.02° | 0.34° | 67% |
| 9.4 °/s | 1.93° | 0.41° | 79% |
| 14.1 °/s | 2.89° | 0.83° | 71% |
| 18.9 °/s | 3.81° | 1.36° | 64% |
| 28.3 °/s | 5.48° | 2.86° | 48% |
<!-- END GENERATED: control-feedforward -->

A rate-commanded loop has a steady velocity-lag error of `target_rate / kp`.
Feed-forward cancels it open-loop, removing roughly two thirds of the pointing
error on a moving target. Getting the reconstruction right is subtle — two
plausible approaches are unstable, and
[control-design.md](control-design.md) §5 works through why.

### 5.3 Robustness to missed detections

<!-- BEGIN GENERATED: control-dropout -->
| Detections missed | RMS error | In frame |
| ---: | ---: | ---: |
| 0% | 0.83° | 100% |
| 10% | 0.84° | 100% |
| 20% | 0.84° | 100% |
| 30% | 0.85° | 100% |
| 50% | 1.07° | 100% |
| 70% | 48.09° | 78% |
<!-- END GENERATED: control-dropout -->

Flat to 50% loss, then a cliff as gaps start exceeding the tracker's coasting
budget. Past that point the answer is a better detector, not a longer `max_age`.

---

## Reproducing everything

```bash
pip install -e ".[train,onnx,bench]"

python tools/fetch_dut_antiuav.py --out data/dut_antiuav
python tools/prepare_dataset.py   --root data/dut_antiuav
python tools/train_uav.py --data data/dut_antiuav/dut_antiuav.yaml --epochs 20
python tools/export_onnx.py   --weights runs/train/uav_yolov8n/weights/best.pt
python tools/quantize_onnx.py --model models/uav_yolov8n_640.onnx

python benchmarks/eval_detection.py --model runs/train/uav_yolov8n/weights/best.pt --tag yolov8n-fp32-pytorch
python benchmarks/eval_detection.py --model models/uav_yolov8n_640.onnx      --tag yolov8n-fp32-onnx
python benchmarks/eval_detection.py --model models/uav_yolov8n_640.int8.onnx --tag yolov8n-int8-onnx
python benchmarks/bench_latency.py  --model models/uav_yolov8n_640.onnx      --tag yolov8n-fp32-onnx
python benchmarks/eval_hard_negatives.py --model runs/train/uav_yolov8n/weights/best.pt --fetch
python benchmarks/bench_control.py --study all

python benchmarks/make_report.py
```

Host details are recorded inside each result JSON, so the numbers stay
interpretable when read on a different machine.
