"""Models of the thing the controller is pushing against.

Two models live here:

* :class:`DiscreteTransferFunction` -- a general continuous transfer function
  discretised with Tustin's method, for exploring controller designs.
* :class:`ServoModel` -- a concrete model of the MG90S hobby servos on this
  turret: transport delay, first-order lag, and a hard slew-rate limit.

The servo model is deliberately simple but *not* linear. The slew-rate limit is
the dominant nonlinearity in a pan/tilt turret and the reason a controller tuned
purely against a linear model overshoots on real hardware: during a large step
the actuator is rate-limited, the loop is effectively open, and any integrator
keeps charging. Modelling the limit is what makes the simulation predictive
rather than decorative.

Parameter provenance: the MG90S datasheet quotes 0.1 s per 60 degrees at 4.8 V
(600 deg/s) and 0.08 s per 60 degrees at 6 V (750 deg/s). The defaults here use
the conservative 4.8 V figure. The lag and delay defaults are placeholders until
identified on your own hardware -- ``tools/identify_servo.py`` does that.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class DiscreteTransferFunction:
    """A continuous transfer function stepped in discrete time.

    Args:
        num: Numerator coefficients, highest power of ``s`` first.
        den: Denominator coefficients, highest power of ``s`` first.
        dt: Sample interval in seconds.
        method: Discretisation method passed to SciPy. Tustin (bilinear) is the
            default because it preserves stability and, unlike zero-order hold,
            does not assume the input is piecewise constant between samples.

    Raises:
        ImportError: If SciPy is not installed.
    """

    def __init__(
        self,
        num: list[float],
        den: list[float],
        dt: float,
        method: str = "tustin",
    ) -> None:
        try:
            import scipy.signal as signal
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "scipy is required for DiscreteTransferFunction: pip install scipy"
            ) from exc

        if not num or not den:
            raise ValueError("numerator and denominator must be non-empty")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        discrete = signal.TransferFunction(num, den).to_discrete(dt, method=method)
        state_space = discrete.to_ss()

        self.A = np.atleast_2d(state_space.A)
        self.B = np.atleast_2d(state_space.B)
        self.C = np.atleast_2d(state_space.C)
        self.D = np.atleast_2d(state_space.D)
        self.dt = dt
        self.x = np.zeros((self.A.shape[0], 1))

    def reset(self) -> None:
        """Return the internal state to zero."""
        self.x = np.zeros((self.A.shape[0], 1))

    def step(self, u: float) -> float:
        """Advance one sample with input ``u`` and return the output."""
        if self.x.size == 0:
            return float(self.D[0, 0] * u)
        y = self.C @ self.x + self.D * u
        self.x = self.A @ self.x + self.B * u
        return float(np.asarray(y).ravel()[0])


class ServoModel:
    """Rate-limited first-order servo with transport delay.

    Args:
        tau_s: First-order time constant of the closed servo loop.
        max_rate_deg_s: Slew-rate limit. 600 deg/s is the MG90S figure at 4.8 V.
        delay_s: Pure transport delay: serial transmission plus the servo's
            command-to-motion latency.
        min_deg: Lower mechanical limit.
        max_deg: Upper mechanical limit.
        initial_deg: Starting angle.
    """

    def __init__(
        self,
        tau_s: float = 0.08,
        max_rate_deg_s: float = 600.0,
        delay_s: float = 0.02,
        min_deg: float = 0.0,
        max_deg: float = 180.0,
        initial_deg: float = 90.0,
    ) -> None:
        if tau_s <= 0:
            raise ValueError(f"tau_s must be positive, got {tau_s}")

        self.tau_s = tau_s
        self.max_rate_deg_s = max_rate_deg_s
        self.delay_s = delay_s
        self.min_deg = min_deg
        self.max_deg = max_deg

        self.angle_deg = initial_deg
        self._command_queue: deque[tuple[float, float]] = deque()
        self._time = 0.0

    def reset(self, initial_deg: float = 90.0) -> None:
        """Return the servo to ``initial_deg`` and drop any queued commands."""
        self.angle_deg = initial_deg
        self._command_queue.clear()
        self._time = 0.0

    def step(self, command_deg: float, dt: float) -> float:
        """Apply ``command_deg`` and advance the model by ``dt`` seconds.

        Args:
            command_deg: Commanded angle.
            dt: Time step in seconds.

        Returns:
            The servo's new actual angle.
        """
        if dt <= 0:
            return self.angle_deg

        self._time += dt
        self._command_queue.append((self._time + self.delay_s, command_deg))

        # Apply the most recent command whose transport delay has elapsed.
        active = None
        while self._command_queue and self._command_queue[0][0] <= self._time:
            active = self._command_queue.popleft()[1]
        if active is None:
            active = self.angle_deg

        target = min(max(active, self.min_deg), self.max_deg)

        # First-order approach, then clamp to the slew-rate limit. Doing it in
        # this order means small moves are lag-dominated and large moves are
        # rate-dominated, which is what a real servo does.
        alpha = 1.0 - math.exp(-dt / self.tau_s)
        desired_step = (target - self.angle_deg) * alpha
        max_step = self.max_rate_deg_s * dt
        step = min(max(desired_step, -max_step), max_step)

        self.angle_deg = min(max(self.angle_deg + step, self.min_deg), self.max_deg)
        return self.angle_deg
