"""VOC parsing, format conversion and configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from uavtrack.config import PipelineConfig
from uavtrack.data.voc import parse_voc_xml, to_coco, to_yolo_lines

VOC_TEMPLATE = """<annotation>
  <folder>test</folder>
  <filename>{filename}</filename>
  <size><width>{width}</width><height>{height}</height><depth>3</depth></size>
  <segmented>0</segmented>
  {objects}
</annotation>
"""

OBJECT_TEMPLATE = """<object>
    <name>{name}</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>{difficult}</difficult>
    <bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox>
  </object>"""


def write_voc(tmp_path: Path, objects: str, width: int = 1280, height: int = 720) -> Path:
    path = tmp_path / "sample.xml"
    path.write_text(
        VOC_TEMPLATE.format(filename="sample.jpg", width=width, height=height, objects=objects),
        encoding="utf-8",
    )
    return path


def one_object(**kwargs) -> str:
    defaults = {"name": "UAV", "difficult": 0, "xmin": 616, "ymin": 540, "xmax": 684, "ymax": 601}
    defaults.update(kwargs)
    return OBJECT_TEMPLATE.format(**defaults)


# --------------------------------------------------------------- VOC parsing


def test_parses_a_single_object(tmp_path):
    annotation = parse_voc_xml(write_voc(tmp_path, one_object()))
    assert annotation.filename == "sample.jpg"
    assert (annotation.width, annotation.height) == (1280, 720)
    assert len(annotation.boxes) == 1

    box = annotation.boxes[0]
    assert box.label == "UAV"
    assert (box.xmin, box.ymin, box.xmax, box.ymax) == (616, 540, 684, 601)
    assert box.width == 68 and box.height == 61


def test_background_image_yields_no_boxes(tmp_path):
    """DUT Anti-UAV contains genuine background frames; they are not an error."""
    annotation = parse_voc_xml(write_voc(tmp_path, ""))
    assert annotation.boxes == []


def test_boxes_are_clipped_to_the_image(tmp_path):
    annotation = parse_voc_xml(
        write_voc(tmp_path, one_object(xmin=-40, ymin=-10, xmax=5000, ymax=5000))
    )
    box = annotation.boxes[0]
    assert box.xmin == 0 and box.ymin == 0
    assert box.xmax == 1280 and box.ymax == 720


def test_degenerate_boxes_are_dropped(tmp_path):
    annotation = parse_voc_xml(
        write_voc(tmp_path, one_object(xmin=100, ymin=100, xmax=100, ymax=200))
    )
    assert annotation.boxes == []


def test_difficult_flag_is_preserved(tmp_path):
    annotation = parse_voc_xml(write_voc(tmp_path, one_object(difficult=1)))
    assert annotation.boxes[0].difficult is True


def test_missing_size_is_an_error(tmp_path):
    path = tmp_path / "broken.xml"
    path.write_text("<annotation><filename>a.jpg</filename></annotation>", encoding="utf-8")
    with pytest.raises(ValueError, match="size"):
        parse_voc_xml(path)


def test_zero_size_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="non-positive"):
        parse_voc_xml(write_voc(tmp_path, one_object(), width=0, height=720))


# -------------------------------------------------------------- YOLO output


def test_yolo_conversion_is_normalised_centre_form(tmp_path):
    annotation = parse_voc_xml(
        write_voc(tmp_path, one_object(xmin=640, ymin=360, xmax=740, ymax=460))
    )
    lines = to_yolo_lines(annotation, {"UAV": 0})
    cls, cx, cy, w, h = lines[0].split()

    assert cls == "0"
    assert float(cx) == pytest.approx(690 / 1280, abs=1e-6)
    assert float(cy) == pytest.approx(410 / 720, abs=1e-6)
    assert float(w) == pytest.approx(100 / 1280, abs=1e-6)
    assert float(h) == pytest.approx(100 / 720, abs=1e-6)


def test_yolo_values_are_within_unit_range(tmp_path):
    annotation = parse_voc_xml(write_voc(tmp_path, one_object(xmin=0, ymin=0, xmax=1280, ymax=720)))
    for value in to_yolo_lines(annotation, {"UAV": 0})[0].split()[1:]:
        assert 0.0 <= float(value) <= 1.0


def test_unknown_labels_are_skipped(tmp_path):
    annotation = parse_voc_xml(write_voc(tmp_path, one_object(name="Helicopter")))
    assert to_yolo_lines(annotation, {"UAV": 0}) == []


# -------------------------------------------------------------- COCO output


def test_coco_conversion_shape(tmp_path):
    annotation = parse_voc_xml(write_voc(tmp_path, one_object()))
    coco = to_coco([annotation], {"UAV": 0})

    assert len(coco["images"]) == 1
    assert coco["images"][0]["id"] == 1
    assert len(coco["annotations"]) == 1
    assert coco["categories"] == [{"id": 1, "name": "UAV", "supercategory": "none"}]


def test_coco_bbox_is_xywh_not_xyxy(tmp_path):
    """The conversion that is easy to get wrong and invisible on large objects."""
    annotation = parse_voc_xml(
        write_voc(tmp_path, one_object(xmin=100, ymin=200, xmax=150, ymax=260))
    )
    coco = to_coco([annotation], {"UAV": 0})
    assert coco["annotations"][0]["bbox"] == [100.0, 200.0, 50.0, 60.0]
    assert coco["annotations"][0]["area"] == pytest.approx(3000.0)


def test_coco_category_ids_are_one_based(tmp_path):
    """pycocotools requires it; predictions apply the same offset."""
    annotation = parse_voc_xml(write_voc(tmp_path, one_object()))
    coco = to_coco([annotation], {"UAV": 0})
    assert coco["annotations"][0]["category_id"] == 1


def test_coco_image_ids_are_unique_and_sequential(tmp_path):
    annotations = [parse_voc_xml(write_voc(tmp_path, one_object())) for _ in range(3)]
    coco = to_coco(annotations, {"UAV": 0})
    assert [image["id"] for image in coco["images"]] == [1, 2, 3]


# ----------------------------------------------------------------- config


def test_shipped_configs_load():
    root = Path(__file__).resolve().parents[1]
    for name in ("rpi5_hailo8l.yaml", "desktop_onnx.yaml"):
        config = PipelineConfig.load(root / "configs" / name)
        assert config.camera.width > 0
        assert config.control.pan.kp > 0


def test_nested_sections_become_dataclasses(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("control:\n  pan:\n    kp: 9.5\n", encoding="utf-8")
    config = PipelineConfig.load(path)
    assert config.control.pan.kp == 9.5
    assert config.control.tilt.kp == 4.5, "unspecified sections keep their defaults"


def test_unknown_top_level_key_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("detectors:\n  model: x.onnx\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown option"):
        PipelineConfig.load(path)


def test_unknown_nested_key_is_rejected(tmp_path):
    """A typo in a gain must fail loudly, not be silently ignored."""
    path = tmp_path / "c.yaml"
    path.write_text("control:\n  pan:\n    kpp: 9.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="control.pan"):
        PipelineConfig.load(path)


def test_empty_config_is_all_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("", encoding="utf-8")
    config = PipelineConfig.load(path)
    assert config.detector.conf_threshold == 0.35


def test_non_mapping_section_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("camera: 12\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a mapping"):
        PipelineConfig.load(path)
