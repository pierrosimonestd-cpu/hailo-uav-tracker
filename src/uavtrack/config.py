"""Typed configuration loaded from YAML.

Configuration is a dataclass tree rather than a dict so that a typo in a YAML
key is an error at load time instead of a silently ignored setting. On a system
where a wrong gain means a turret slamming into its end stop, failing loudly at
startup is the cheap option.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class DetectorConfig:
    """Which model to run and how to threshold it."""

    model: str = "models/uav_yolov8n.hef"
    backend: str | None = None
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    labels: list[str] = field(default_factory=lambda: ["uav"])
    #: Class names to keep, or None for all of them. A model trained on many
    #: classes will happily hand the tracker every chair in the room, and the
    #: controller will dutifully point at one; naming the classes that matter
    #: is what makes a general detector usable as a turret's eyes.
    classes: list[str] | None = None


@dataclass
class CameraConfig:
    """Frame source and lens geometry."""

    source: str = "picamera"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    horizontal_fov_deg: float = 66.0
    vertical_fov_deg: float | None = None


@dataclass
class TrackerConfig:
    """Association and track lifecycle thresholds."""

    high_threshold: float = 0.5
    low_threshold: float = 0.1
    match_threshold: float = 0.3
    second_match_threshold: float = 0.2
    max_age: int = 30
    min_hits: int = 3


@dataclass
class AxisConfig:
    """Limits for one servo axis."""

    min_deg: float = 0.0
    max_deg: float = 180.0
    max_rate_deg_s: float = 600.0
    deadband_deg: float = 0.5
    #: Reverse this axis, for a servo mounted mirrored. See AxisLimits.invert.
    invert: bool = False
    kp: float = 4.5
    ki: float = 0.5
    kd: float = 0.25


@dataclass
class ControlConfig:
    """Controller configuration for both axes."""

    pan: AxisConfig = field(default_factory=AxisConfig)
    tilt: AxisConfig = field(default_factory=AxisConfig)
    latency_s: float = 0.045
    feedforward_gain: float = 0.9
    estimator_window: int = 5
    home_pan_deg: float = 90.0
    home_tilt_deg: float = 90.0


@dataclass
class LinkConfig:
    """Serial link to the ESP32."""

    enabled: bool = False
    port: str = "/dev/ttyUSB0"
    baudrate: int = 500_000
    heartbeat_interval_s: float = 0.2


@dataclass
class PipelineConfig:
    """Top-level configuration."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    link: LinkConfig = field(default_factory=LinkConfig)

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from a YAML file.

        Raises:
            ValueError: If the file contains a key this schema does not define.
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return _from_dict(cls, data, path=str(path))


def _from_dict(cls: type[T], data: dict[str, Any], path: str, prefix: str = "") -> T:
    """Recursively build a dataclass from a dict, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected a mapping at {prefix or '<root>'}, got {type(data).__name__}"
        )

    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"{path}: unknown option(s) {sorted(unknown)} at {prefix or '<root>'}; "
            f"expected one of {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for name, spec in known.items():
        if name not in data:
            continue
        value = data[name]

        # `from __future__ import annotations` makes `spec.type` a string, so
        # the nested-dataclass test goes through the default instead. Every
        # nested section in this schema has a dataclass default_factory, which
        # makes this both reliable and free of runtime type resolution.
        nested_type = None
        if spec.default_factory is not MISSING:  # type: ignore[misc]
            default = spec.default_factory()
            if is_dataclass(default):
                nested_type = type(default)

        if nested_type is not None:
            kwargs[name] = _from_dict(nested_type, value, path, f"{prefix}{name}.")
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[return-value]
