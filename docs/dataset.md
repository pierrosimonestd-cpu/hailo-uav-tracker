# Dataset

## DUT Anti-UAV

Accuracy in this project is reported on the **DUT Anti-UAV** detection
benchmark: a public, peer-reviewed, single-class UAV detection dataset.

> Jie Zhao, Jingshu Zhang, Dongdong Li, Dong Wang.
> *Vision-based Anti-UAV Detection and Tracking.*
> IEEE Transactions on Intelligent Transportation Systems, 2022.
> <https://github.com/wangdongdut/DUT-Anti-UAV>

It was chosen over the easy alternatives for three reasons: it is published with
a paper and a baseline table, so results are comparable to something; it is
ground-to-air, which is the geometry this turret actually sees, unlike
drone-mounted datasets such as VisDrone; and its object-size distribution is
brutal in exactly the way real UAV detection is.

No images are vendored in this repository. Everything is fetched by script.

```bash
pip install -e ".[bench]"
python tools/fetch_dut_antiuav.py --out data/dut_antiuav
python tools/prepare_dataset.py   --root data/dut_antiuav
```

The archives are hosted on Google Drive by the authors, so `fetch` needs
`gdown`. If Google rate-limits the download, the script points you at the Baidu
mirror in the upstream README. The test split alone (271 MB) is enough to
reproduce every accuracy number here.

## What is in it

Measured by `tools/prepare_dataset.py` and written to
`data/dut_antiuav/stats.json`:

| Split | Images | Objects | Background images |
|---|---:|---:|---:|
| train | 5,200 | 5,243 | 3 |
| val | 2,600 | 2,620 | 0 |
| test | 2,200 | 2,245 | 0 |

Split sizes match those published in the paper, and `fetch_dut_antiuav.py`
verifies the counts after extraction.

### Object size — the whole problem in one table

Using the COCO thresholds (small `< 32²` px, large `≥ 96²` px):

| Split | Small | Medium | Large | Median object area |
|---|---:|---:|---:|---:|
| train | 2,723 (52%) | 1,857 (35%) | 663 (13%) | 0.047% of the frame |
| val | 1,401 (53%) | 892 (34%) | 327 (12%) | 0.046% of the frame |
| test | 848 (38%) | 836 (37%) | 561 (25%) | 0.091% of the frame |

The median object covers **under one tenth of one percent of the frame**. The
smallest is 1.9 × 10⁻⁵ of the frame — about 25 pixels in a 1280 × 720 image.
The largest covers 70%.

Three consequences that shape the rest of the project:

1. **`AP_small` is close to the whole story.** An aggregate AP can look
   respectable while the model has quietly stopped seeing distant targets, which
   are precisely the ones worth detecting early. Every result table here breaks
   AP down by size.
2. **Input resolution is not negotiable.** Dropping from 640 to 416 to buy frame
   rate throws away the small objects first. The NPU exists so this trade does
   not have to be made.
3. **Quantisation is riskier than usual.** Small, low-contrast objects produce
   small activations, which is where INT8's coarse quantisation levels hurt
   most. This is why [benchmarks.md](benchmarks.md) reports the quantised model
   separately rather than assuming the FP32 result carries over.

## Format conversion

`tools/prepare_dataset.py` converts Pascal VOC XML into both layouts:

```
data/dut_antiuav/
├── train/ val/ test/         img/*.jpg + xml/*.xml   (as downloaded)
├── yolo/
│   ├── images/{train,val,test}/   symlinks, not copies
│   └── labels/{train,val,test}/   normalised centre-form
├── coco/{train,val,test}.json     COCO ground truth
├── dut_antiuav.yaml               Ultralytics descriptor
└── stats.json
```

Images are symlinked so the 1.2 GB of JPEGs is not duplicated; on Windows
without Developer Mode this falls back to hardlinks, then to copies.

Two conversion details that are easy to get wrong and invisible on large objects:

- **VOC is inclusive corners; COCO is `[x, y, w, h]`.** Mixing them shifts every
  box by a pixel. On a 20-pixel UAV that is a 5% error in every box.
- **Empty label files are required.** A background image with no `.txt` is
  reported by Ultralytics as *unlabelled* and silently skipped, rather than
  learned from as a negative.

Both are covered by `tests/test_data_and_config.py`.

## Evaluation protocol

`benchmarks/eval_detection.py` scores with `pycocotools` at the standard COCO
IoU sweep (0.50:0.95). Two choices worth stating:

- **Detections are collected at a confidence threshold of 0.001.** COCO AP
  integrates over the precision–recall curve; filtering before scoring truncates
  the curve and *understates* AP. The operating threshold used in the live
  pipeline is a separate decision, made in `configs/`.
- **The scorer is restricted to exactly the images that were run.** With
  `--limit`, scoring a subset of detections against the full ground truth
  collapses every metric in proportion to the subset size. (This was a real bug
  during development: it made a working model look like it had an AP of 0.024.)

## The tracking subset

The same benchmark ships 20 annotated video sequences, used by
`benchmarks/eval_tracking.py`:

```bash
python tools/fetch_dut_antiuav.py --out data/dut_antiuav_tracking --splits tracking
python benchmarks/eval_tracking.py --model runs/train/uav_yolov8n/weights/best.pt
```

Worth stating plainly: **this is a harder task than the single-object-tracking
baselines published with the dataset.** A SOT tracker is handed the ground-truth
box in frame one and only has to follow it. This pipeline is never told where
the target is -- it detects, associates and picks a primary target with no
initialisation. The numbers are therefore not comparable to a SOT leaderboard
and are not presented as if they were.

## Bird discrimination

Birds are the canonical false positive for ground-to-air UAV detection: similar
apparent size, similar sky background, similar motion at a distance. DUT
Anti-UAV contains no labelled birds, so it cannot measure this.

`benchmarks/eval_hard_negatives.py` measures it separately, using bird images
from **Open Images V7** — properly licensed, scriptable to download, and with
per-image labels that let images containing aircraft be excluded. The metric is
a false-positive rate on images guaranteed to contain no UAV, which is the
number that decides whether the turret spends its day chasing pigeons.

---

**See also:** [benchmarks.md](benchmarks.md) · [architecture.md](architecture.md)
