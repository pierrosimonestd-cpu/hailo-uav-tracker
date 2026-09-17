"""Pointing control: PID, pixel-to-angle geometry, servo models, simulation."""

from uavtrack.control.gimbal import (
    AxisLimits,
    CameraGeometry,
    GimbalCommand,
    PanTiltController,
)
from uavtrack.control.pid import PID, PIDGains, PIDState
from uavtrack.control.plant import DiscreteTransferFunction, ServoModel
from uavtrack.control.simulator import (
    SimulationConfig,
    SimulationResult,
    TargetTrajectory,
    simulate,
)

__all__ = [
    "PID",
    "AxisLimits",
    "CameraGeometry",
    "DiscreteTransferFunction",
    "GimbalCommand",
    "PIDGains",
    "PIDState",
    "PanTiltController",
    "ServoModel",
    "SimulationConfig",
    "SimulationResult",
    "TargetTrajectory",
    "simulate",
]
