# Model artifacts

This directory is where the build products land. None of them are committed —
they are derived files, and a repository that carries 25 MB of weights per
experiment stops being clonable.

| File | Produced by | Used by |
|---|---|---|
| `uav_yolov8n_640.onnx` | `tools/export_onnx.py` | ONNX Runtime backend; input to the Hailo compiler |
| `uav_yolov8n_640.int8.onnx` | `tools/quantize_onnx.py` | The INT8 accuracy study |
| `calibration.npy` | `tools/make_calibration_set.py` | The Hailo optimiser |
| `uav_yolov8n_640.har` | `tools/compile_hailo.sh` (parse step) | Intermediate; inspect with `hailo profiler` |
| `uav_yolov8n_640.hef` | `tools/compile_hailo.sh` (compile step) | What actually runs on the Pi |

## Getting a model

Train one:

```bash
python tools/fetch_dut_antiuav.py --out data/dut_antiuav
python tools/prepare_dataset.py   --root data/dut_antiuav
python tools/train_uav.py --epochs 20 --cache ram
python tools/export_onnx.py --weights runs/train/uav_yolov8n/weights/best.pt
```

Then, on an x86_64 machine with the Hailo Dataflow Compiler:

```bash
python tools/make_calibration_set.py --out models/calibration.npy
bash tools/compile_hailo.sh models/uav_yolov8n_640.onnx models/calibration.npy
```

See [docs/hailo-deployment.md](../docs/hailo-deployment.md) for the parts of
that flow that will bite you.
