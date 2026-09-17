#!/usr/bin/env bash
# Compile the exported ONNX model into a Hailo-8L HEF.
#
# This runs on a x86_64 Linux machine with the Hailo Dataflow Compiler (DFC)
# installed, NOT on the Raspberry Pi. The Pi runs HailoRT and the resulting
# .hef; it does not compile. The DFC is distributed through the Hailo Developer
# Zone and needs an account, which is why this step is a script you run rather
# than a CI job.
#
# Pipeline: ONNX -> parse (HAR) -> optimise/quantise with calibration data ->
# compile (HEF).
#
# Usage:
#   bash tools/compile_hailo.sh models/uav_yolov8n_640.onnx [calibration.npy]
#
# Produce the calibration set first:
#   python tools/make_calibration_set.py --out models/calibration.npy

set -euo pipefail

ONNX_PATH="${1:-models/uav_yolov8n_640.onnx}"
CALIB_PATH="${2:-models/calibration.npy}"
HW_ARCH="${HW_ARCH:-hailo8l}"          # hailo8l = AI HAT+ 13 TOPS; hailo8 = 26 TOPS
MODEL_NAME="$(basename "${ONNX_PATH}" .onnx)"
OUT_DIR="${OUT_DIR:-models}"

# End node names for a YOLOv8 head exported with nms=False. These are the three
# detection branches, and they are what the Hailo NMS postprocess node attaches
# to. Confirm them against your own export with:
#   python -c "import onnx; m=onnx.load('${ONNX_PATH}'); print([n.name for n in m.graph.node][-12:])"
END_NODES="${END_NODES:-/model.22/cv2.0/cv2.0.2/Conv,/model.22/cv3.0/cv3.0.2/Conv,/model.22/cv2.1/cv2.1.2/Conv,/model.22/cv3.1/cv3.1.2/Conv,/model.22/cv2.2/cv2.2.2/Conv,/model.22/cv3.2/cv3.2.2/Conv}"

if ! command -v hailo >/dev/null 2>&1; then
  echo "error: the 'hailo' CLI is not on PATH." >&2
  echo "Install the Hailo Dataflow Compiler from the Hailo Developer Zone and" >&2
  echo "activate its virtualenv. See docs/hailo-deployment.md." >&2
  exit 1
fi

for required in "${ONNX_PATH}" "${CALIB_PATH}"; do
  if [[ ! -f "${required}" ]]; then
    echo "error: ${required} not found" >&2
    exit 1
  fi
done

mkdir -p "${OUT_DIR}"
HAR_PATH="${OUT_DIR}/${MODEL_NAME}.har"
QUANTIZED_HAR="${OUT_DIR}/${MODEL_NAME}_quantized.har"
HEF_PATH="${OUT_DIR}/${MODEL_NAME}.hef"
ALLS_PATH="${OUT_DIR}/${MODEL_NAME}.alls"

echo "==> [1/3] parsing ${ONNX_PATH} for ${HW_ARCH}"
hailo parser onnx "${ONNX_PATH}" \
  --hw-arch "${HW_ARCH}" \
  --har-path "${HAR_PATH}" \
  --end-node-names ${END_NODES//,/ } \
  -y

# The model script tells the optimiser to (a) take uint8 input and do the
# normalisation on chip, which saves a float conversion and a 4x larger PCIe
# transfer per frame on the Pi, and (b) attach the YOLOv8 NMS node so the device
# returns final boxes instead of raw head tensors.
{
  echo "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])"

  # The three classification branches need a sigmoid the ONNX export left out:
  # cutting the graph at the head convolutions cuts before the activation, and
  # the NMS node expects probabilities, not logits.
  #
  # The layer names are assigned by the parser and differ per model, so they are
  # NOT hardcoded here. List them from the parsed HAR with:
  #
  #   hailo profiler "${HAR_PATH}" | grep conv
  #
  # then re-run with, for example:
  #
  #   CLS_OUTPUT_LAYERS="conv42,conv53,conv63" bash tools/compile_hailo.sh ...
  #
  if [[ -n "${CLS_OUTPUT_LAYERS:-}" ]]; then
    IFS=',' read -ra layers <<< "${CLS_OUTPUT_LAYERS}"
    for layer in "${layers[@]}"; do
      echo "change_output_activation(${layer}, sigmoid)"
    done
  else
    echo "# WARNING: CLS_OUTPUT_LAYERS was not set, so no sigmoid was applied to" >&2
    echo "# the classification branches. Scores from the resulting HEF will be"   >&2
    echo "# logits and every detection will be filtered out. See the comment in"  >&2
    echo "# tools/compile_hailo.sh."                                              >&2
  fi

  echo "nms_postprocess(\"${OUT_DIR}/${MODEL_NAME}_nms_config.json\", meta_arch=yolov8, engine=cpu)"
} > "${ALLS_PATH}"

echo "==> [2/3] optimising and quantising with ${CALIB_PATH}"
hailo optimize "${HAR_PATH}" \
  --hw-arch "${HW_ARCH}" \
  --calib-set-path "${CALIB_PATH}" \
  --model-script "${ALLS_PATH}" \
  --output-har-path "${QUANTIZED_HAR}"

echo "==> [3/3] compiling to ${HEF_PATH}"
hailo compiler "${QUANTIZED_HAR}" \
  --hw-arch "${HW_ARCH}" \
  --output-dir "${OUT_DIR}"

echo
echo "Built ${HEF_PATH}"
echo "Copy it to the Raspberry Pi and point configs/rpi5_hailo8l.yaml at it, then:"
echo "  uavtrack check --config configs/rpi5_hailo8l.yaml"
echo "  python benchmarks/bench_latency.py --model ${HEF_PATH}"
