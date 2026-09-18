# Deploying to the Raspberry Pi AI HAT+ (Hailo-8L, 13 TOPS)

Two machines are involved, and confusing them wastes an afternoon:

| | Compile | Run |
|---|---|---|
| Machine | x86_64 Linux | Raspberry Pi 5 |
| Software | Hailo **Dataflow Compiler** (DFC) | **HailoRT** + `hailo_platform` |
| Input | `.onnx` | `.hef` |
| Output | `.hef` | detections |

The Pi does not compile. The DFC is distributed through the Hailo Developer Zone
and needs an account, which is why compilation is a script you run rather than a
CI job.

---

## 1. Hardware

- Raspberry Pi 5 (4 GB or 8 GB)
- Raspberry Pi **AI HAT+ 13 TOPS** (Hailo-8L). The 26 TOPS version carries a
  Hailo-8; set `HW_ARCH=hailo8` when compiling and everything else is identical.
- Raspberry Pi OS **Bookworm, 64-bit**

The HAT connects over PCIe Gen 3. Confirm Gen 3 is enabled — Gen 2 roughly halves
the achievable throughput:

```bash
grep -q "^dtparam=pciex1_gen=3" /boot/firmware/config.txt || \
  echo "dtparam=pciex1_gen=3" | sudo tee -a /boot/firmware/config.txt
sudo reboot
```

## 2. Runtime install on the Pi

One command does all of section 2 and checks its own work:

```bash
./tools/provision_pi.sh
```

It installs the packages below, builds the virtualenv correctly, installs the
project and then verifies the PCIe link, HailoRT, the Python bindings and the
camera -- reporting which of those failed rather than exiting on the first one.
Re-running it is safe. `./tools/provision_pi.sh --verify` checks without
changing anything.

The manual equivalent:

```bash
sudo apt update
sudo apt install -y hailo-all python3-picamera2 python3-opencv python3-venv
sudo reboot
```

`hailo-all` pulls in the PCIe driver, the firmware, HailoRT and the Python
bindings. Verify:

```bash
hailortcli fw-control identify
```

The output must name **HAILO8L** and report a firmware version. If the device is
not found, it is nearly always the PCIe link: check `dmesg | grep -i hailo`.

Then install this project:

```bash
git clone https://github.com/pierrosimonestd-cpu/hailo-uav-tracker
cd hailo-uav-tracker
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e .            # NumPy, OpenCV, PyYAML, pyserial. No PyTorch.
```

Both flags on that venv line are load-bearing:

- **`--system-site-packages`.** `hailo_platform` and `picamera2` are apt
  packages, not pip packages. A venv without this cannot see them, and the
  Hailo backend fails to import on a machine where the runtime is installed
  perfectly.
- **The venv itself.** Bookworm marks its system Python as externally managed
  (PEP 668), so `pip install -e .` outside a venv is refused. The venv is the
  fix; `--break-system-packages` is not -- it installs this project's
  dependencies over apt-managed ones and the next `apt upgrade` gets to
  arbitrate.

## 3. Compiling the model (on x86_64)

```bash
# 1. Train, or download a release checkpoint
python tools/train_uav.py --data data/dut_antiuav/dut_antiuav.yaml --epochs 20

# 2. Export to ONNX: opset 13, static shapes, no NMS
python tools/export_onnx.py --weights runs/train/uav_yolov8n/weights/best.pt

# 3. Build the calibration tensor from TRAINING images
python tools/make_calibration_set.py --out models/calibration.npy --count 1024

# 4. Parse, quantise, compile
bash tools/compile_hailo.sh models/uav_yolov8n_640.onnx models/calibration.npy
```

### Things that will bite you

**Opset.** The DFC parses opsets 11–13 reliably. `tools/export_onnx.py` pins
**13**, not 11: opset 11's `QuantizeLinear` has no `axis` attribute, so the INT8
study in [benchmarks.md](benchmarks.md) cannot use per-channel weights and fails
at load with `INVALID_GRAPH`. Opset 13 satisfies both the compiler and the
quantiser, so one export serves both paths.

**End node names.** The parser needs to be told where the graph ends, because
the NMS node is attached by the compiler rather than exported. The defaults in
`compile_hailo.sh` match a stock Ultralytics YOLOv8 export; confirm against your
own:

```bash
python -c "import onnx; m=onnx.load('models/uav_yolov8n_640.onnx'); print([n.name for n in m.graph.node][-12:])"
```

**The sigmoid on the classification branches.** Cutting the graph at the head
convolutions cuts *before* the activation, and the NMS node expects
probabilities. The layer names are assigned by the parser and differ per model,
so they are deliberately not hardcoded. List them from the parsed HAR and pass
them in:

```bash
hailo profiler models/uav_yolov8n_640.har | grep conv
CLS_OUTPUT_LAYERS="conv42,conv53,conv63" bash tools/compile_hailo.sh models/uav_yolov8n_640.onnx
```

Skip this and every score comes out as a logit, every detection is filtered out,
and the pipeline runs perfectly while detecting nothing.

**Calibration set.** Hailo recommends at least 1024 images, pre-processed
exactly as at inference. `make_calibration_set.py` reuses the pipeline's own
`letterbox()` for this reason, and draws from the **training** split —
calibrating on test images leaks the test set.

**Normalisation on chip.** The generated model script puts the `/255` on the
device, so the runtime feeds it uint8. That saves a float conversion and a 4×
larger PCIe transfer per frame. The `HailoDetector` matches this by setting
`FormatType.UINT8` on the input; changing one without the other silently
destroys accuracy.

## 4. Running

```bash
scp models/uav_yolov8n_640.hef pi@raspberrypi:~/hailo-uav-tracker/models/

# on the Pi
uavtrack check --config configs/rpi5_hailo8l.yaml
uavtrack run   --config configs/rpi5_hailo8l.yaml --display
```

`check` validates the config, resolves the backend and confirms the model file
exists — run it before wondering why nothing happens.

## 5. Measure your own latency

The single most important number in `configs/rpi5_hailo8l.yaml` is
`control.latency_s`. It is loop dead time, and [control-design.md](control-design.md)
shows what it costs. Measure it rather than trusting the default:

```bash
python benchmarks/bench_latency.py --model models/uav_yolov8n_640.hef --frames 300
```

Set `control.latency_s` from the median-to-p95 range. Quote the tail, not the
mean: one late frame in fifty hurts a pointing solution more than a mean that is
5 ms worse.

## 6. Published Hailo-8L throughput

For reference, from the Hailo Model Zoo — these are **Hailo's published figures
on COCO**, measured on an Intel Core i5-9400 over PCIe Gen 3 ×4 with Dataflow
Compiler v2.19.0. They are not measurements from this project and not from a
Raspberry Pi, where the host and the PCIe link differ:

| Model | Input | mAP (float) | mAP (hardware) | FPS (batch 1) | FPS (batch 8) |
|---|---|---:|---:|---:|---:|
| yolov8n | 640×640 | 37.0 | 36.4 | 202 | 438 |
| yolov8s | 640×640 | 44.6 | 43.9 | 110 | 208 |
| yolov11n | 640×640 | 39.0 | 37.5 | 157 | 371 |
| yolov11s | 640×640 | 46.3 | 45.1 | 92.0 | 192 |

Source: [hailo_model_zoo/docs/public_models/HAILO8L](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO8L/HAILO8L_object_detection.rst)

Two things to take from this table. First, **yolov8n at 202 FPS leaves the
control loop far more headroom than it can use** — §3 of the control design
shows the loop bandwidth is capped near 1 Hz by latency, so the useful question
is latency, not frame rate. Second, Hailo's own float-to-hardware gap on COCO is
0.6–1.5 mAP points, which is the right order of magnitude to expect from INT8
quantisation and a useful sanity check on the numbers in
[benchmarks.md](benchmarks.md).

## 7. Troubleshooting

| Symptom | Cause |
|---|---|
| `hailo_platform` import fails | Installed via apt, not pip; venv needs `--system-site-packages` |
| `hailortcli` finds no device | PCIe link; check `dmesg \| grep -i hailo` |
| Pipeline runs, zero detections | Missing sigmoid on the classification branches (§3) |
| Detections in the wrong place | Input format mismatch: on-chip normalisation expects uint8 |
| Throughput about half of published | PCIe Gen 2; set `dtparam=pciex1_gen=3` |
| Latency spikes every few seconds | Thermal throttling; check `vcgencmd measure_temp` |

---

**See also:** [architecture.md](architecture.md) ·
[benchmarks.md](benchmarks.md) · [hardware.md](hardware.md)
