"""A PID controller that survives contact with real actuators.

The textbook three-line PID is wrong in three specific ways once it drives a
servo, and each one has a visible failure mode on a turret:

* **Integral windup.** While the servo is saturated at its travel limit the
  error keeps integrating, and when the target comes back the accumulated term
  drives a long overshoot. Fixed here by back-calculation: the integrator is
  corrected by however much the output had to be clamped.
* **Derivative kick.** Differentiating the error means a step change in setpoint
  produces an impulse in the output. Fixed by differentiating the *measurement*
  instead, which is mathematically equivalent for a constant setpoint and far
  better behaved when the setpoint jumps -- which it does every time the target
  selector switches targets.
* **Derivative noise.** A detector's box centre jitters by a pixel or two per
  frame; differentiating that amplifies it. Fixed with a first-order filter on
  the derivative term.

The gains are in the caller's units: feed it pixels and it returns whatever
``output_limits`` are expressed in.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIDGains:
    """Proportional, integral and derivative gains."""

    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0


@dataclass
class PIDState:
    """Internal state, exposed for logging and for the control-design notebook."""

    integral: float = 0.0
    last_measurement: float | None = None
    derivative: float = 0.0
    output: float = 0.0
    saturated: bool = False


class PID:
    """Discrete PID with anti-windup, derivative-on-measurement and filtering.

    Args:
        gains: Controller gains.
        output_limits: ``(low, high)`` clamp on the output. Use the actuator's
            real limits; the anti-windup term depends on them being honest.
        derivative_filter_hz: Cut-off of the first-order low-pass on the
            derivative term. ``None`` disables filtering.
        anti_windup_gain: Back-calculation gain. ``0`` disables anti-windup;
            ``1 / ki`` is the classic choice and the default scales with the
            sample time instead, which behaves better at variable frame rates.
    """

    def __init__(
        self,
        gains: PIDGains,
        output_limits: tuple[float, float] = (-float("inf"), float("inf")),
        derivative_filter_hz: float | None = 8.0,
        anti_windup_gain: float = 1.0,
    ) -> None:
        low, high = output_limits
        if low >= high:
            raise ValueError(
                f"output_limits must be (low, high) with low < high, got {output_limits}"
            )

        self.gains = gains
        self.output_limits = output_limits
        self.derivative_filter_hz = derivative_filter_hz
        self.anti_windup_gain = anti_windup_gain
        self.state = PIDState()

    def reset(self) -> None:
        """Clear the integrator and derivative history.

        Call this whenever the loop is opened -- a lost target, a target switch,
        a manual override -- so stale state cannot leak into the next lock.
        """
        self.state = PIDState()

    def __call__(self, setpoint: float, measurement: float, dt: float) -> float:
        """Compute one control output.

        Args:
            setpoint: Desired value.
            measurement: Current value.
            dt: Seconds since the previous call. Non-positive values return the
                previous output unchanged rather than dividing by zero.

        Returns:
            The clamped control output.
        """
        if dt <= 0:
            return self.state.output

        error = setpoint - measurement

        # Derivative on measurement, negated so the sign matches derivative-on-
        # error for a constant setpoint.
        if self.state.last_measurement is None:
            raw_derivative = 0.0
        else:
            raw_derivative = -(measurement - self.state.last_measurement) / dt
        self.state.last_measurement = measurement

        if self.derivative_filter_hz is None:
            self.state.derivative = raw_derivative
        else:
            # One-pole IIR: alpha = dt / (tau + dt), tau = 1 / (2*pi*fc).
            tau = 1.0 / (2.0 * 3.141592653589793 * self.derivative_filter_hz)
            alpha = dt / (tau + dt)
            self.state.derivative += alpha * (raw_derivative - self.state.derivative)

        self.state.integral += error * dt

        unclamped = (
            self.gains.kp * error
            + self.gains.ki * self.state.integral
            + self.gains.kd * self.state.derivative
        )

        low, high = self.output_limits
        output = min(max(unclamped, low), high)
        self.state.saturated = output != unclamped

        # Back-calculation anti-windup: unwind the integrator by exactly the
        # amount the actuator refused to deliver.
        if self.state.saturated and self.gains.ki != 0.0 and self.anti_windup_gain > 0.0:
            self.state.integral += self.anti_windup_gain * (output - unclamped) / self.gains.ki

        self.state.output = output
        return output
