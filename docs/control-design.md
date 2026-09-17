# Control design

How the turret decides where to point, why it is built this way, and what each
decision is worth in measured degrees.

Everything here is reproducible with:

```bash
python benchmarks/bench_control.py --study all
```

Results are written to `benchmarks/results/control_simulation.json`. They are
**simulation** results — the servo model is described below and in
[`plant.py`](../src/uavtrack/control/plant.py). They are not a substitute for
measuring on your own hardware, they are how you decide what is worth measuring.

---

## 1. The loop

```
frame ──▶ detector ──▶ tracker ──▶ selector ──▶ controller ──▶ serial ──▶ servos
  ▲                                                                          │
  └──────────────────── camera is bolted to the turret ──────────────────────┘
```

The feedback path is mechanical: the camera moves with the turret, so a pixel
measurement is an angle *relative to wherever the turret currently points*. That
single fact drives most of the design.

## 2. Pixels to angles

A pixel offset is not proportional to an angle. Through a pinhole model:

$$\theta = \arctan\left(\frac{\Delta x}{f}\right), \qquad f = \frac{W/2}{\tan(\text{HFOV}/2)}$$

Using the small-angle approximation $\theta \approx \Delta x / f$ instead makes
the effective loop gain fall off towards the frame edges — a controller tuned on
a centred target goes sluggish exactly when the target is about to leave frame.
At the edge of a 66° lens the approximation is off by about 10%.

The lens field of view therefore sets the loop gain. **Measure it.** Getting
`horizontal_fov_deg` wrong is indistinguishable from mistuning every gain.

## 3. Why the loop is latency-limited, not actuator-limited

The MG90S servos slew at 600 °/s. The loop cannot use anything close to that.

The plant seen by the controller is an integrator (commanded rate integrates to
angle) behind a dead time $T_d$ and a lag $\tau$. For a 45° phase margin the
crossover $\omega_c$ must satisfy

$$\omega_c T_d + \arctan(\omega_c \tau) = 45° $$

With $T_d \approx 65$ ms (detection latency plus serial and servo transport) and
$\tau \approx 80$ ms, that gives $\omega_c \approx 5.6$ rad/s — under 1 Hz — and
a proportional gain of roughly

$$k_p = \omega_c\sqrt{1 + (\tau\omega_c)^2} \approx 6$$

So $k_p$ is capped near 6 by *latency*, while the actuator could support far
more. This is the single most important fact about the system, and it is the
reason an NPU matters: **inference time is loop dead time.**

Measured, by sweeping the sense-to-act latency and tracking a target orbiting at
14 °/s (mean of five seeds, steady state after 2 s):

| Sense-to-act latency | RMS pointing error |
|---------------------:|-------------------:|
| 10 ms | 0.68° |
| 20 ms | 0.69° |
| 30 ms | 0.73° |
| 45 ms | 0.83° |
| 60 ms | 0.87° |
| 80 ms | 1.01° |
| 120 ms | 1.21° |
| 160 ms | 1.45° |
| 200 ms | 1.83° |
| 300 ms | 2.77° |

Pointing error grows roughly four-fold from 10 ms to 300 ms. Latency is the
budget the whole perception stack spends from.

## 4. The PID, and the three ways the textbook version fails

[`pid.py`](../src/uavtrack/control/pid.py) is a PID with three modifications,
each fixing a failure with a visible symptom on a turret:

| Problem | Symptom | Fix |
|---|---|---|
| Integral windup | Long overshoot after the target returns from beyond a travel limit | Back-calculation: unwind the integrator by exactly what the actuator refused to deliver |
| Derivative kick | Violent jerk each time the selector switches target | Differentiate the *measurement*, not the error |
| Derivative noise | Servo buzz tracking box jitter | First-order low-pass at 8 Hz on the derivative term |

Anti-windup is worth measuring rather than asserting. Driving an integrator
plant to a distant setpoint through a saturating actuator
(`tests/test_control.py::test_anti_windup_reduces_overshoot_after_saturation`),
back-calculation cuts overshoot by more than 4×.

Tuned gains, from the sweep in `benchmarks/bench_control.py`:

```
kp = 4.5    ki = 0.5    kd = 0.25
```

`kp` sits below the stability-limited ≈6 for margin against an unidentified
servo. `ki` is small: it exists to remove static bias, not to chase motion —
that is the feed-forward's job.

## 5. Velocity feed-forward, and two ways to get it wrong

A rate-commanded loop of this shape has a steady velocity-lag error of

$$e_{ss} = \frac{\dot\theta_{\text{target}}}{k_p}$$

At 14 °/s with $k_p = 4.5$ that is **3.1° of permanent lag**. Raising $k_p$ is
not available (§3), and integral action large enough to remove it destabilises
the loop. Feed-forward cancels it open-loop instead: add the target's angular
rate directly to the commanded rate.

That needs the target's rate in *world* coordinates, and the camera only
measures relative angles. Reconstructing it is where this gets interesting.

**Wrong approach 1 — add back your own commanded rate.** "Apparent rate plus my
own rate" seems to reconstruct the absolute rate. It is a positive feedback path
with loop gain equal to the feed-forward gain: the fixed point requires
$g_{ff} = 1$ exactly, and any bias is amplified by $1/(1-g_{ff})$. Measured
before it was removed: unstable within a second, RMS error rising from 3.6° to
over 90°.

**Wrong approach 2 — reconstruct from the commanded angle.** Safer-looking, and
still wrong. These servos have no encoder, so the commanded angle leads the real
one by the servo lag:

$$\hat\theta_{\text{target}} = \theta_{\text{cmd}} + \theta_{\text{offset}} = \theta_{\text{target}} + \tau_{\text{eff}}\,\dot\theta_{\text{cmd}}$$

Differentiating, $\dot\theta_{cmd} - \tau_{\text{eff}}\ddot\theta_{cmd} =
\dot\theta_{\text{target}}$ — a **positive real pole at $1/\tau_{\text{eff}}
\approx 12\ \text{s}^{-1}$**. It diverges in under a second. Measured: RMS error
of 88° at $g_{ff} = 0.9$.

**What works — reconstruct through a model of the servo.** The controller keeps
a forward model of its own actuator ([`ServoModel`](../src/uavtrack/control/plant.py))
and reconstructs the bearing against the *model's output* rather than against
the raw command. The lag term cancels, the feedback path disappears, and gains
near 1.0 are stable.

The reconstructed bearings then go into a
[`BearingEstimator`](../src/uavtrack/control/gimbal.py): a least-squares line fit
over a five-sample sliding window, fitted against *timestamps* rather than
sample indices so a dropped frame or a jittering inference time does not bias the
slope. The fit gives both the smoothed bearing and its rate, and extrapolating it
is what performs latency compensation.

Measured effect (steady state, five seeds, circular target):

| Peak target rate | FF off | FF on | Reduction |
|---:|---:|---:|---:|
| 4.7 °/s | 1.02° | 0.34° | 67% |
| 9.4 °/s | 1.93° | 0.41° | 79% |
| 14.1 °/s | 2.89° | 0.83° | 71% |
| 18.8 °/s | 3.81° | 1.36° | 64% |
| 28.3 °/s | 5.48° | 2.86° | 48% |

Feed-forward removes roughly two thirds of the pointing error on a moving
target. The benefit tapers at high rates, where the linear fit stops describing
the motion within its own window.

## 6. Aim time: a subtle but expensive detail

The command should aim at where the target will be when the turret *gets there*.
Measured from now, that is the **actuator's** lag — not another detection
latency. The detection latency is already accounted for by timestamping the
observation at capture time.

Getting this wrong is not obviously wrong: leading by `2 × latency` happens to be
correct at one particular frame rate. During development this produced a
spurious minimum in the latency sweep at 80 ms — the over-lead accidentally
cancelling the under-compensation — which looked like a real result until the
test on a constant-velocity target (where a linear extrapolator should be exact)
showed a residual that no amount of latency should cause. After the fix, error
on a constant-velocity target dropped from 1.18° to 0.11° at 10 ms latency, and
the latency sweep became monotonic.

The lesson generalises: if a performance curve has a minimum somewhere other
than the physical limit, look for two errors cancelling.

## 7. Robustness to dropped detections

The Kalman filter coasts a lost track on its velocity estimate, so missed
detections degrade gracefully rather than opening the loop:

| Detections missed | RMS error | Target in frame |
|---:|---:|---:|
| 0% | 0.83° | 100% |
| 10% | 0.84° | 100% |
| 20% | 0.84° | 100% |
| 30% | 0.85° | 100% |
| 50% | 1.07° | 100% |
| 70% | 48.1° | 78% |

Flat to 50%, then a cliff. The cliff is the `max_age` limit: past roughly half
the frames missing, gaps regularly exceed the coasting budget and the track is
dropped. If your detector misses more than half the frames, the answer is a
better detector, not a longer `max_age` — coasting for a second on a two-second-old
velocity estimate points at empty sky.

## 8. The servo model

```python
ServoModel(tau_s=0.08, max_rate_deg_s=600.0, delay_s=0.02)
```

Transport delay, then a first-order lag, then a hard slew-rate limit. The rate
limit is the dominant nonlinearity in a pan/tilt turret and the reason a
controller tuned purely against a linear model overshoots on real hardware:
during a large step the actuator is rate-limited, the loop is effectively open,
and any integrator keeps charging.

`max_rate_deg_s = 600` is the MG90S datasheet figure at 4.8 V (0.1 s per 60°).
`tau_s` and `delay_s` are **placeholders until identified on your hardware.**
Until you do, keep `feedforward_gain` at 0.9 rather than 1.0 — the feed-forward
is only as good as this model.

## 9. What is not modelled

Stated plainly, because a simulation's honesty is in its omissions:

- **Backlash and stiction.** Present in MG90S gear trains, and the reason for
  the deadband. Not modelled; expect a real limit cycle the simulation does not
  show.
- **Camera-to-turret misalignment.** Assumed zero. A real build has a boresight
  offset that shows as a constant pointing bias.
- **Lens distortion.** A pure pinhole model. Real wide lenses need calibration.
- **Structural flexibility.** The turret is rigid here. A 3D-printed yoke is not.
- **Target range.** Everything is angular. Two targets at the same bearing and
  different ranges are indistinguishable, which is exactly true of a monocular
  camera.

---

**See also:** [architecture.md](architecture.md) ·
[benchmarks.md](benchmarks.md) · [hardware.md](hardware.md)
