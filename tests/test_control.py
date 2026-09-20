"""PID behaviour, pixel-to-angle geometry, servo model and the closed loop."""

from __future__ import annotations

import importlib.util
import math

import pytest

from uavtrack.control.gimbal import (
    AxisLimits,
    BearingEstimator,
    CameraGeometry,
    PanTiltController,
)
from uavtrack.control.pid import PID, PIDGains
from uavtrack.control.plant import DiscreteTransferFunction, ServoModel
from uavtrack.control.simulator import SimulationConfig, TargetTrajectory, simulate

# ------------------------------------------------------------------------ PID


def test_proportional_output():
    pid = PID(PIDGains(kp=2.0))
    assert pid(10.0, 0.0, 0.1) == pytest.approx(20.0)


def test_integral_accumulates_over_time():
    pid = PID(PIDGains(ki=1.0))
    pid(1.0, 0.0, 1.0)
    assert pid.state.integral == pytest.approx(1.0)
    pid(1.0, 0.0, 1.0)
    assert pid.state.integral == pytest.approx(2.0)


def test_output_is_clamped_to_limits():
    pid = PID(PIDGains(kp=100.0), output_limits=(-5.0, 5.0))
    assert pid(10.0, 0.0, 0.1) == 5.0
    assert pid(-10.0, 0.0, 0.1) == -5.0


def test_anti_windup_stops_the_integrator_running_away():
    """While saturated, the integral must not keep charging."""
    limited = PID(PIDGains(kp=1.0, ki=10.0), output_limits=(-1.0, 1.0))
    unlimited = PID(PIDGains(kp=1.0, ki=10.0))

    for _ in range(50):
        limited(10.0, 0.0, 0.05)
        unlimited(10.0, 0.0, 0.05)

    assert abs(limited.state.integral) < abs(unlimited.state.integral) / 10


def _closed_loop_overshoot(anti_windup_gain: float) -> float:
    """Drive an integrator plant to a far setpoint, then measure the overshoot.

    A rate-limited actuator plus a distant setpoint is exactly the situation
    that charges an integrator it cannot act on. The overshoot on arrival is
    the visible symptom, so that is what is measured -- in a closed loop, since
    an integrator can only discharge if the output is allowed to move the
    measurement.
    """
    pid = PID(
        PIDGains(kp=1.0, ki=5.0),
        output_limits=(-1.0, 1.0),
        anti_windup_gain=anti_windup_gain,
    )
    position = 0.0
    setpoint = 5.0
    dt = 0.02
    peak = 0.0
    for _ in range(1500):
        position += pid(setpoint, position, dt) * dt
        peak = max(peak, position)
    return peak - setpoint


def test_anti_windup_reduces_overshoot_after_saturation():
    """The symptom anti-windup exists to prevent."""
    with_aw = _closed_loop_overshoot(1.0)
    without_aw = _closed_loop_overshoot(0.0)
    assert without_aw > 0.5, "the test setup must actually provoke windup"
    assert with_aw < without_aw / 4


def test_derivative_on_measurement_avoids_setpoint_kick():
    """A step in the setpoint must not produce a derivative impulse."""
    on_measurement = PID(PIDGains(kd=1.0), derivative_filter_hz=None)
    on_measurement(0.0, 0.0, 0.1)
    output = on_measurement(100.0, 0.0, 0.1)  # setpoint jumps, measurement does not
    assert output == pytest.approx(0.0)


def test_derivative_responds_to_measurement_change():
    pid = PID(PIDGains(kd=1.0), derivative_filter_hz=None)
    pid(0.0, 0.0, 0.1)
    assert pid(0.0, 1.0, 0.1) == pytest.approx(-10.0)


def test_derivative_filter_attenuates_noise():
    filtered = PID(PIDGains(kd=1.0), derivative_filter_hz=2.0)
    unfiltered = PID(PIDGains(kd=1.0), derivative_filter_hz=None)

    filtered_peak = unfiltered_peak = 0.0
    for i in range(20):
        noise = 1.0 if i % 2 else -1.0
        filtered_peak = max(filtered_peak, abs(filtered(0.0, noise, 0.033)))
        unfiltered_peak = max(unfiltered_peak, abs(unfiltered(0.0, noise, 0.033)))

    assert filtered_peak < unfiltered_peak / 2


def test_non_positive_dt_returns_the_previous_output():
    pid = PID(PIDGains(kp=1.0))
    first = pid(5.0, 0.0, 0.1)
    assert pid(5.0, 0.0, 0.0) == first
    assert pid(5.0, 0.0, -1.0) == first


def test_reset_clears_state():
    pid = PID(PIDGains(kp=1.0, ki=1.0, kd=1.0))
    for _ in range(10):
        pid(1.0, 0.0, 0.1)
    pid.reset()
    assert pid.state.integral == 0.0
    assert pid.state.last_measurement is None


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError):
        PID(PIDGains(), output_limits=(5.0, -5.0))


# ------------------------------------------------------------------- geometry


def test_focal_length_from_field_of_view():
    geometry = CameraGeometry(1280, 720, 90.0)
    # At 90 degrees horizontal FOV, half the width subtends 45 degrees.
    assert geometry.focal_px_x == pytest.approx(640.0)


def test_centre_pixel_is_zero_angle():
    geometry = CameraGeometry(1280, 720, 66.0)
    pan, tilt = geometry.pixel_offset_to_angles(0.0, 0.0)
    assert pan == pytest.approx(0.0)
    assert tilt == pytest.approx(0.0)


def test_frame_edge_maps_to_half_the_field_of_view():
    geometry = CameraGeometry(1280, 720, 66.0)
    pan, _ = geometry.pixel_offset_to_angles(640.0, 0.0)
    assert pan == pytest.approx(33.0, abs=0.01)


def test_tilt_sign_is_inverted_relative_to_image_rows():
    """Image rows increase downwards; a tilt axis increases upwards."""
    geometry = CameraGeometry(1280, 720, 66.0)
    _, tilt = geometry.pixel_offset_to_angles(0.0, 100.0)
    assert tilt < 0


def test_angle_conversion_is_not_the_small_angle_approximation():
    """The arctangent must actually be used, or edge gain is wrong."""
    geometry = CameraGeometry(1280, 720, 66.0)
    half, _ = geometry.pixel_offset_to_angles(320.0, 0.0)
    full, _ = geometry.pixel_offset_to_angles(640.0, 0.0)
    assert full < 2 * half, "angle must grow sub-linearly with pixel offset"


# ------------------------------------------------------------ rate estimation


def test_bearing_estimator_recovers_a_constant_rate():
    estimator = BearingEstimator(window=5)
    for i in range(6):
        estimator.update(i * 0.1, 10.0 + 20.0 * i * 0.1)
    bearing, rate = estimator.predict(0.5)
    assert rate == pytest.approx(20.0, rel=1e-6)
    assert bearing == pytest.approx(20.0, rel=1e-6)


def test_bearing_estimator_is_robust_to_uneven_sample_spacing():
    """Fitting over timestamps, not indices, is what makes this work."""
    estimator = BearingEstimator(window=5)
    for t in (0.0, 0.03, 0.11, 0.14, 0.27):
        estimator.update(t, 5.0 + 12.0 * t)
    _, rate = estimator.predict(0.3)
    assert rate == pytest.approx(12.0, rel=1e-6)


def test_bearing_estimator_discards_history_across_a_long_gap():
    estimator = BearingEstimator(window=5, max_gap_s=0.2)
    for i in range(5):
        estimator.update(i * 0.05, i * 10.0)
    estimator.update(10.0, 0.0)  # re-acquisition long afterwards
    _, rate = estimator.predict(10.0)
    assert rate == 0.0


def test_bearing_estimator_is_not_ready_with_one_sample():
    estimator = BearingEstimator()
    assert not estimator.ready
    estimator.update(0.0, 1.0)
    assert not estimator.ready
    estimator.update(0.1, 2.0)
    assert estimator.ready


def test_bearing_estimator_rejects_a_degenerate_window():
    with pytest.raises(ValueError):
        BearingEstimator(window=1)


# ---------------------------------------------------------------- servo model


def test_servo_approaches_its_command():
    servo = ServoModel(tau_s=0.05, delay_s=0.0)
    servo.reset(90.0)
    for _ in range(200):
        servo.step(120.0, 0.005)
    assert servo.angle_deg == pytest.approx(120.0, abs=0.5)


def test_servo_respects_its_slew_rate_limit():
    servo = ServoModel(tau_s=0.001, max_rate_deg_s=100.0, delay_s=0.0)
    servo.reset(0.0)
    servo.step(180.0, 0.1)
    assert servo.angle_deg <= 10.0 + 1e-6


def test_servo_respects_its_travel_limits():
    servo = ServoModel(tau_s=0.01, delay_s=0.0, min_deg=20.0, max_deg=160.0)
    servo.reset(90.0)
    for _ in range(500):
        servo.step(400.0, 0.01)
    assert servo.angle_deg <= 160.0


def test_servo_transport_delay_defers_motion():
    servo = ServoModel(tau_s=0.01, delay_s=0.05)
    servo.reset(90.0)
    servo.step(150.0, 0.01)
    assert servo.angle_deg == pytest.approx(90.0, abs=1e-6)
    for _ in range(20):
        servo.step(150.0, 0.01)
    assert servo.angle_deg > 100.0


def test_servo_rejects_a_non_positive_time_constant():
    with pytest.raises(ValueError):
        ServoModel(tau_s=0.0)


# ------------------------------------------------------- transfer function
#
# DiscreteTransferFunction is the one part of the control stack that needs
# SciPy, which is deliberately not a runtime dependency: a minimal Raspberry Pi
# install does not have it, and the tracker and controller are built to work
# without it. These tests therefore skip rather than fail there -- as they did
# on the Pi, where SciPy is genuinely absent and the code was behaving exactly
# as designed by raising a clear ImportError.

pytestmark_scipy = pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None,
    reason="scipy is an optional dependency and is not installed on a minimal Pi image",
)


@pytestmark_scipy
def test_discrete_transfer_function_unit_gain_passthrough():
    tf = DiscreteTransferFunction([1.0], [1.0], dt=0.01)
    assert tf.step(1.0) == pytest.approx(1.0)


@pytestmark_scipy
def test_first_order_lag_settles_at_the_input():
    tf = DiscreteTransferFunction([1.0], [0.1, 1.0], dt=0.001)
    output = 0.0
    for _ in range(2000):
        output = tf.step(1.0)
    assert output == pytest.approx(1.0, abs=0.01)


@pytestmark_scipy
def test_transfer_function_rejects_bad_arguments():
    with pytest.raises(ValueError):
        DiscreteTransferFunction([], [1.0], dt=0.01)
    with pytest.raises(ValueError):
        DiscreteTransferFunction([1.0], [1.0], dt=0.0)


# --------------------------------------------------------------- closed loop


def _controller(**kwargs) -> PanTiltController:
    return PanTiltController(
        geometry=CameraGeometry(1280, 720, 66.0),
        pan_gains=PIDGains(4.5, 0.5, 0.25),
        tilt_gains=PIDGains(4.5, 0.5, 0.25),
        pan_limits=AxisLimits(max_rate_deg_s=600.0, deadband_deg=0.2),
        tilt_limits=AxisLimits(max_rate_deg_s=600.0, deadband_deg=0.2),
        **kwargs,
    )


def test_controller_turns_towards_an_off_centre_target():
    controller = _controller()
    command = controller.update((900.0, 360.0), 0.033)
    assert command.pan_deg > 90.0, "a target to the right must increase pan"


def test_controller_tilts_up_for_a_target_above_centre():
    controller = _controller()
    command = controller.update((640.0, 100.0), 0.033)
    assert command.tilt_deg > 90.0


def test_controller_holds_still_inside_the_deadband():
    controller = _controller()
    for _ in range(10):
        command = controller.update((641.0, 361.0), 0.033)
    assert command.in_deadband
    assert command.pan_deg == pytest.approx(90.0, abs=0.05)


def test_controller_never_exceeds_travel_limits():
    controller = PanTiltController(
        geometry=CameraGeometry(1280, 720, 66.0),
        pan_gains=PIDGains(50.0, 0.0, 0.0),
        tilt_gains=PIDGains(50.0, 0.0, 0.0),
        pan_limits=AxisLimits(min_deg=0.0, max_deg=180.0, max_rate_deg_s=600.0),
        tilt_limits=AxisLimits(min_deg=20.0, max_deg=160.0, max_rate_deg_s=600.0),
    )
    for _ in range(500):
        command = controller.update((1279.0, 0.0), 0.033)
        assert 0.0 <= command.pan_deg <= 180.0
        assert 20.0 <= command.tilt_deg <= 160.0


def test_controller_respects_the_slew_rate_limit():
    controller = PanTiltController(
        geometry=CameraGeometry(1280, 720, 66.0),
        pan_gains=PIDGains(100.0, 0.0, 0.0),
        tilt_gains=PIDGains(100.0, 0.0, 0.0),
        pan_limits=AxisLimits(max_rate_deg_s=30.0),
        tilt_limits=AxisLimits(max_rate_deg_s=30.0),
    )
    dt = 0.05
    previous = controller.pan_deg
    for _ in range(20):
        command = controller.update((1279.0, 360.0), dt)
        assert abs(command.pan_deg - previous) <= 30.0 * dt + 1e-6
        previous = command.pan_deg


def test_controller_reset_returns_home():
    controller = _controller(home=(90.0, 90.0))
    for _ in range(20):
        controller.update((1200.0, 200.0), 0.033)
    assert controller.pan_deg != pytest.approx(90.0)
    controller.reset(to_home=True)
    assert controller.pan_deg == pytest.approx(90.0)
    assert controller.tilt_deg == pytest.approx(90.0)


# ---------------------------------------------------------------- simulation


def test_simulation_keeps_a_circular_target_in_frame():
    result = simulate(TargetTrajectory(kind="circular"), SimulationConfig(duration_s=6.0))
    assert result.in_frame_fraction > 0.99
    assert result.rms_error_after(2.0) < 2.0


def test_feedforward_reduces_steady_state_error_on_a_moving_target():
    """The quantitative claim the control design rests on."""
    trajectory = TargetTrajectory(kind="circular")
    without = simulate(trajectory, SimulationConfig(duration_s=8.0, feedforward_gain=0.0))
    with_ff = simulate(trajectory, SimulationConfig(duration_s=8.0, feedforward_gain=0.9))
    assert with_ff.rms_error_after(2.0) < 0.6 * without.rms_error_after(2.0)


def test_higher_latency_degrades_pointing():
    """If this ever inverts, the lead compensation has a sign error."""
    trajectory = TargetTrajectory(kind="circular")
    fast = simulate(trajectory, SimulationConfig(duration_s=8.0, detection_latency_s=0.02))
    slow = simulate(trajectory, SimulationConfig(duration_s=8.0, detection_latency_s=0.25))
    assert slow.rms_error_after(2.0) > fast.rms_error_after(2.0)


def test_tracker_coasts_through_moderate_detection_dropout():
    trajectory = TargetTrajectory(kind="circular")
    clean = simulate(trajectory, SimulationConfig(duration_s=8.0, detection_dropout=0.0))
    lossy = simulate(trajectory, SimulationConfig(duration_s=8.0, detection_dropout=0.3))
    assert lossy.rms_error_after(2.0) < clean.rms_error_after(2.0) * 1.5


def test_simulation_is_reproducible_for_a_fixed_seed():
    trajectory = TargetTrajectory(kind="circular")
    a = simulate(trajectory, SimulationConfig(duration_s=3.0, seed=7))
    b = simulate(trajectory, SimulationConfig(duration_s=3.0, seed=7))
    assert a.rms_error_deg == b.rms_error_deg


def test_step_response_settles():
    result = simulate(TargetTrajectory(kind="step"), SimulationConfig(duration_s=6.0))
    settling = result.settling_time_s(1.0)
    assert settling is not None and settling < 3.0


def test_unknown_trajectory_is_rejected():
    with pytest.raises(ValueError):
        TargetTrajectory(kind="spiral").at(0.0)  # type: ignore[arg-type]


def test_trajectories_are_finite():
    for kind in ("static", "step", "linear", "circular", "crossing"):
        trajectory = TargetTrajectory(kind=kind)  # type: ignore[arg-type]
        for t in (0.0, 1.0, 5.0):
            azimuth, elevation = trajectory.at(t)
            assert math.isfinite(azimuth) and math.isfinite(elevation)


# ------------------------------------------------------------ axis inversion


def _reversible(invert_pan: bool = False, invert_tilt: bool = False) -> PanTiltController:
    """Like _controller above, but with the axis travel and reversal pinned.

    Explicit limits because the inversion mirrors about the centre of travel,
    so the test would be asserting against defaults it did not choose.
    """
    return PanTiltController(
        geometry=CameraGeometry(1280, 720, 66.0),
        pan_gains=PIDGains(4.5, 0.5, 0.25),
        tilt_gains=PIDGains(4.5, 0.5, 0.25),
        pan_limits=AxisLimits(0.0, 180.0, 600.0, 0.2, invert=invert_pan),
        tilt_limits=AxisLimits(20.0, 160.0, 600.0, 0.2, invert=invert_tilt),
    )


def test_inversion_mirrors_the_command_about_the_centre_of_travel():
    """A servo fitted mirrored drives away from the target without this."""
    straight = _reversible()
    reversed_ = _reversible(invert_pan=True, invert_tilt=True)

    # Same target, off-centre so the controller actually moves.
    for _ in range(20):
        plain = straight.update((900.0, 250.0), 0.05)
        flipped = reversed_.update((900.0, 250.0), 0.05)

    assert plain.pan_deg == pytest.approx((0.0 + 180.0) - flipped.pan_deg, abs=1e-6)
    assert plain.tilt_deg == pytest.approx((20.0 + 160.0) - flipped.tilt_deg, abs=1e-6)


def test_inversion_keeps_the_command_inside_the_axis_limits():
    """Mirroring about the centre, not negating: 20-160 must stay 20-160."""
    controller = _reversible(invert_tilt=True)
    for _ in range(80):
        command = controller.update((640.0, 719.0), 0.05)
        assert 20.0 <= command.tilt_deg <= 160.0


def test_inversion_does_not_touch_the_control_mathematics():
    """The errors and rates are computed in the optical frame either way.

    If inversion leaked into the loop rather than staying an output mapping,
    the reported error would change sign and the integrator would wind the
    wrong way -- which is a far worse bug than a mirrored servo.
    """
    straight = _reversible()
    reversed_ = _reversible(invert_pan=True, invert_tilt=True)

    for _ in range(15):
        plain = straight.update((900.0, 250.0), 0.05)
        flipped = reversed_.update((900.0, 250.0), 0.05)

    assert plain.pan_error_deg == pytest.approx(flipped.pan_error_deg, abs=1e-9)
    assert plain.tilt_error_deg == pytest.approx(flipped.tilt_error_deg, abs=1e-9)
    assert plain.pan_rate_deg_s == pytest.approx(flipped.pan_rate_deg_s, abs=1e-9)


def test_an_uninverted_axis_is_unchanged():
    """The default must be a no-op, or every existing rig changes behaviour."""
    controller = _reversible()
    for _ in range(10):
        command = controller.update((900.0, 250.0), 0.05)
    assert command.pan_deg == pytest.approx(controller.pan_deg, abs=1e-9)
    assert command.tilt_deg == pytest.approx(controller.tilt_deg, abs=1e-9)


# ------------------------------------------------- camera ego-motion removal
#
# The camera is bolted to the turret, so apparent motion in the image is the
# sum of the target's motion and the turret's own. Everything here is about
# the second term not being mistaken for the first.


def _pixels_for_bearing(geometry: CameraGeometry, bearing_deg: float, boresight_deg: float):
    """Where a target at ``bearing_deg`` lands when the turret points at ``boresight_deg``."""
    offset_rad = math.radians(bearing_deg - boresight_deg)
    dx = geometry.focal_px_x * math.tan(offset_rad)
    return geometry.width / 2.0 + dx, geometry.height / 2.0


def test_reported_angle_sets_the_reconstructed_bearing():
    """Bearing must be turret angle plus in-frame offset, using the real angle."""
    controller = _reversible()
    geometry = controller.geometry

    controller.report_angles(120.0, 90.0)
    target_px = _pixels_for_bearing(geometry, bearing_deg=130.0, boresight_deg=120.0)
    controller.update(target_px, 0.05)

    bearing, _ = controller.pan_estimator.predict(controller.pan_estimator._samples[-1][0])
    assert bearing == pytest.approx(130.0, abs=0.5)


def test_a_static_target_reads_as_static_while_the_turret_sweeps():
    """The whole point of reconstructing an absolute bearing.

    The turret sweeps 40 degrees; the target never moves. In image coordinates
    it races across the frame, and a loop that fitted a rate to *that* would
    feed forward a velocity the target does not have -- which is exactly the
    instability this cancellation exists to prevent.
    """
    controller = _reversible()
    geometry = controller.geometry
    bearing = 110.0

    for step in range(10):
        boresight = 70.0 + 4.0 * step  # the turret sweeping past
        controller.report_angles(boresight, 90.0)
        controller.update(_pixels_for_bearing(geometry, bearing, boresight), 0.05)

    _, rate = controller.pan_estimator.predict(controller._time)
    assert abs(rate) < 2.0, f"a stationary target was fitted a rate of {rate:.2f} deg/s"


def test_without_cancellation_the_same_sweep_looks_like_motion():
    """Guards the test above from passing for the wrong reason.

    Feeding the raw in-frame offsets, with no turret angle added, must produce
    a large apparent rate. If it did not, the test above would prove nothing.
    """
    controller = _reversible()
    geometry = controller.geometry
    bearing = 110.0

    estimator = type(controller.pan_estimator)(window=controller.pan_estimator.window)
    for step in range(10):
        boresight = 70.0 + 4.0 * step
        px, _ = _pixels_for_bearing(geometry, bearing, boresight)
        offset_deg, _ = geometry.pixel_offset_to_angles(px - geometry.width / 2.0, 0.0)
        estimator.update(step * 0.05, offset_deg)  # no turret angle added

    _, naive_rate = estimator.predict(9 * 0.05)
    assert abs(naive_rate) > 40.0, (
        f"expected the uncancelled signal to look fast, saw {naive_rate:.1f} deg/s"
    )


def test_reporting_angles_snaps_the_internal_model():
    """A report must correct the model, not sit alongside it drifting."""
    controller = _reversible()
    controller.update((900.0, 300.0), 0.05)

    controller.report_angles(45.0, 100.0)
    pan_then, tilt_then = controller._angle_at(controller._time)
    assert pan_then == pytest.approx(45.0)
    assert tilt_then == pytest.approx(100.0)
