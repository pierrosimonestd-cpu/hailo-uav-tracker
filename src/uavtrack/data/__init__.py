"""Dataset parsing and format conversion."""

from uavtrack.data.voc import (
    VocAnnotation,
    VocBox,
    parse_voc_xml,
    to_coco,
    to_yolo_lines,
    write_coco,
)

__all__ = [
    "VocAnnotation",
    "VocBox",
    "parse_voc_xml",
    "to_coco",
    "to_yolo_lines",
    "write_coco",
]
