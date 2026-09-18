"""Pascal VOC XML parsing and conversion to YOLO and COCO formats.

DUT Anti-UAV ships VOC XML. Training wants YOLO text files; scoring wants COCO
JSON. Both conversions live here, behind one parser, so a coordinate convention
can only be got wrong in one place.

Convention notes that matter for correctness:

* VOC boxes are inclusive pixel indices; COCO boxes are ``[x, y, w, h]`` with
  ``w = xmax - xmin``. Mixing the two shifts every box by one pixel, which is
  invisible on large objects and material on a 20-pixel UAV.
* YOLO labels are normalised centre-form, clipped to the image.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VocBox:
    """One annotated object from a VOC XML file."""

    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    difficult: bool = False

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class VocAnnotation:
    """The contents of one VOC XML file."""

    filename: str
    width: int
    height: int
    boxes: list[VocBox] = field(default_factory=list)


def _require(node: ET.Element, tag: str, source: Path) -> ET.Element:
    child = node.find(tag)
    if child is None:
        raise ValueError(f"{source}: missing required <{tag}> element")
    return child


def parse_voc_xml(path: Path) -> VocAnnotation:
    """Parse a single Pascal VOC annotation file.

    Args:
        path: Path to the ``.xml`` file.

    Returns:
        The parsed annotation. Images with no objects yield an empty box list,
        which is legitimate: DUT Anti-UAV includes pure-background frames.

    Raises:
        ValueError: If the file is missing size information or a box is
            degenerate after clipping.
    """
    root = ET.parse(path).getroot()
    size = _require(root, "size", path)
    width = int(float(_require(size, "width", path).text or 0))
    height = int(float(_require(size, "height", path).text or 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: non-positive image size {width}x{height}")

    filename_node = root.find("filename")
    filename = (filename_node.text if filename_node is not None else None) or f"{path.stem}.jpg"

    boxes: list[VocBox] = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        label = (name_node.text if name_node is not None else "").strip()
        bnd = _require(obj, "bndbox", path)

        xmin = max(0.0, float(_require(bnd, "xmin", path).text or 0))
        ymin = max(0.0, float(_require(bnd, "ymin", path).text or 0))
        xmax = min(float(width), float(_require(bnd, "xmax", path).text or 0))
        ymax = min(float(height), float(_require(bnd, "ymax", path).text or 0))

        if xmax <= xmin or ymax <= ymin:
            # A box that survives clipping with zero extent is corrupt rather
            # than merely small; skipping it beats poisoning the label file.
            continue

        difficult_node = obj.find("difficult")
        difficult = bool(int(difficult_node.text or 0)) if difficult_node is not None else False
        boxes.append(VocBox(label, xmin, ymin, xmax, ymax, difficult))

    return VocAnnotation(filename=filename, width=width, height=height, boxes=boxes)


def to_yolo_lines(annotation: VocAnnotation, class_to_id: dict[str, int]) -> list[str]:
    """Render an annotation as YOLO label-file lines.

    Args:
        annotation: Parsed VOC annotation.
        class_to_id: Mapping from VOC label text to contiguous class index.

    Returns:
        One ``"cls cx cy w h"`` line per box, all values normalised to ``[0, 1]``.
    """
    lines: list[str] = []
    for box in annotation.boxes:
        if box.label not in class_to_id:
            continue
        cx = (box.xmin + box.xmax) / 2.0 / annotation.width
        cy = (box.ymin + box.ymax) / 2.0 / annotation.height
        w = box.width / annotation.width
        h = box.height / annotation.height
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
        lines.append(f"{class_to_id[box.label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def to_coco(
    annotations: list[VocAnnotation],
    class_to_id: dict[str, int],
    description: str = "",
) -> dict:
    """Build a COCO-format ground-truth dictionary.

    ``pycocotools`` requires 1-based category ids, so the internal 0-based class
    index is offset by one here and only here; :mod:`benchmarks.eval_detection`
    applies the same offset to predictions.

    Args:
        annotations: Parsed annotations, in a stable order.
        class_to_id: Mapping from VOC label text to 0-based class index.
        description: Free text stored in the ``info`` block.

    Returns:
        A dict ready for ``json.dump`` and ``pycocotools.coco.COCO``.
    """
    images = []
    coco_annotations = []
    annotation_id = 1

    for image_id, ann in enumerate(annotations, start=1):
        images.append(
            {"id": image_id, "file_name": ann.filename, "width": ann.width, "height": ann.height}
        )
        for box in ann.boxes:
            if box.label not in class_to_id:
                continue
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_to_id[box.label] + 1,
                    "bbox": [box.xmin, box.ymin, box.width, box.height],
                    "area": box.area,
                    "iscrowd": 0,
                    "ignore": int(box.difficult),
                }
            )
            annotation_id += 1

    categories = [
        {"id": index + 1, "name": name, "supercategory": "none"}
        for name, index in sorted(class_to_id.items(), key=lambda kv: kv[1])
    ]
    return {
        "info": {"description": description},
        "images": images,
        "annotations": coco_annotations,
        "categories": categories,
    }


def write_coco(path: Path, coco: dict) -> None:
    """Write a COCO dict to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco), encoding="utf-8")
