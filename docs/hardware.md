# Hardware

A two-axis pan/tilt turret carrying a camera, driven by an ESP32, commanded by a
Raspberry Pi 5 with a Hailo-8L NPU.

## Bill of materials

| Part | Notes |
|---|---|
| Raspberry Pi 5, 4 GB or 8 GB | 8 GB only helps if you also train on it, which you should not |
| Raspberry Pi AI HAT+ 13 TOPS (Hailo-8L) | The 26 TOPS Hailo-8 variant works unchanged; set `HW_ARCH=hailo8` when compiling |
| Raspberry Pi Camera Module 3 | 66° horizontal FOV. Any camera works — **measure the FOV** |
| ESP32 devkit (e.g. DOIT DevKit v1) | Any ESP32 with USB serial |
| 2 × MG90S metal-gear servo | Plastic-gear SG90s strip under the side loads from a camera mount |
| 5 V 2 A+ supply for the servos | **Separate from the Pi.** See below |
| Logic-level MOSFET module or 3.3 V relay | Drives the effector output |
| 608ZZ bearing | Takes the radial load off the pan servo's output shaft |
| M3 hardware, USB-C cable | |

Printed parts are in [`../hardware/cad`](../hardware/cad) — STL and 3MF for
printing, Creo sources under `creo/`.

| File | Part |
|---|---|
| `base_plate.stl` | Fixed base, pan servo mount |
| `base_rpi5_tray.stl` | Pi 5 + AI HAT+ tray |
| `yoke_pan.stl` | Rotating yoke |
| `tilt_arm.stl` | Tilt arm |
| `camera_mount_v2.stl` | Camera mount (v2; v1 kept for reference) |
| `bearing_holder.stl` | 608ZZ seat |
| `yoke_mg90s_fit_test.stl` | Print this first — checks the servo pocket fit |

Print `yoke_mg90s_fit_test.stl` before committing to a four-hour print. Servo
body tolerances vary between suppliers, and a pocket 0.3 mm undersized means
either a broken part or a servo you cannot remove.

The Creo sources under `hardware/cad/creo/` keep the native `.prt.N` versioning,
where the **highest** `N` is the current revision -- `base.prt.6` supersedes
`base.prt.5`. Earlier revisions are kept because they are the design history;
only the STL and 3MF exports are what you print.

## Power — read this part

**Do not power the servos from the Pi or from the ESP32.** MG90S stall current
is around 700 mA each, with startup transients well above that. An ESP32's
regulator supplies a few hundred milliamps, so the board browns out and resets
the instant both servos move together. The symptom is a turret that works
perfectly until it has to track something.

```
5 V 2 A+ supply ──┬── servo V+ (red)
                  └── (do NOT connect to the ESP32 5 V pin)

        GND ──────┬── servo GND (brown/black)
                  ├── ESP32 GND
                  └── MOSFET / relay module GND        ← all grounds common
```

The common ground is not optional. Without it the PWM signal has no reference
and the servos jitter or ignore commands entirely.

A 470–1000 µF electrolytic across the servo supply, physically close to the
servos, absorbs the switching transients. Cheap, and it removes a whole class of
intermittent faults.

## Wiring

| Signal | ESP32 pin | Goes to |
|---|---|---|
| Pan servo PWM | GPIO 13 | Servo signal (orange/yellow) |
| Tilt servo PWM | GPIO 12 | Servo signal |
| Effector | GPIO 21 | MOSFET gate or relay input |
| Serial | USB-C | Raspberry Pi USB port |

Pin assignments are at the top of
[`esp32_pantilt.ino`](../firmware/esp32_pantilt/esp32_pantilt.ino).

The effector pin **cannot drive a load directly** — an ESP32 GPIO supplies 3.3 V
at a few tens of milliamps. Use a logic-level MOSFET (IRLZ44N or similar) or a
relay module rated for 3.3 V logic.

### Servo pulse range

The firmware attaches the servos with a 500–2400 µs pulse range rather than the
1000–2000 µs Arduino default. MG90S servos use the wider range; clipping it
throws away roughly a third of the mechanical travel, which shows up as a turret
that mysteriously cannot reach the edges of its own field of regard.

## Mechanical limits

Configured in both `configs/rpi5_hailo8l.yaml` and the firmware, deliberately:

| Axis | Range | Why |
|---|---|---|
| Pan | 0–180° | Full servo travel |
| Tilt | 20–160° | The camera mount fouls the base below 20° |

**Check these against your own build before powering on.** Both ends enforce
them so a bug on the Pi cannot drive the mechanism into itself, but neither end
knows your print tolerances.

## Firmware

```bash
arduino-cli config init
arduino-cli config set board_manager.additional_urls \
  https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32
arduino-cli lib install ESP32Servo

arduino-cli compile --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_pantilt
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32doit-devkit-v1 firmware/esp32_pantilt
```

If upload fails at `Connecting...`, hold the BOOT button until it starts.

### What the firmware guarantees

The Pi does the thinking; the firmware's entire job is to be the part that stays
safe when the thinking stops.

- **Failsafe.** No valid frame for 500 ms and the effector is cut and the servos
  are *detached* — PWM stops, so they go limp rather than straining against
  whatever they are pressed into. The Pi sends heartbeats to prove it is alive,
  so silence is always deliberate.
- **Starts inert.** The turret does not move on power-up until the host has
  spoken to it. A power cycle cannot make it move on its own.
- **Framed commands.** Sync word, length, CRC-8. A corrupt frame is counted and
  dropped, never acted on.
- **Independent rate limiting.** Commands are slew-limited in the firmware as
  well as on the Pi, so a bad command cannot ask for a step big enough to brown
  out the supply.
- **Telemetry.** A `STATUS` frame every 200 ms reporting the last sequence
  number, dropped frames and CRC errors — so link quality is observable rather
  than guessed at.

### Why not two raw bytes

The obvious protocol is to write pan and tilt as two bytes and have the firmware
read them whenever two are available. It has a failure mode that is silent and
permanent: drop one byte anywhere and every subsequent pair is read off by one,
so pan is interpreted as tilt for as long as the link stays up. Neither end can
tell.

Measured on the framed protocol with 1% of bytes randomly dropped: **173 of 200
frames recovered, and zero frames delivered with wrong content**
(`tests/test_protocol.py`). Frames are lost, never misread. That is the property
worth having, and it costs seven bytes of overhead per command.

The full frame layout is in
[`protocol.py`](../src/uavtrack/io/protocol.py), and
[`protocol.h`](../firmware/esp32_pantilt/protocol.h) must stay byte-identical —
`tests/test_protocol.py` pins the CRC vectors both sides have to reproduce.

## Assembly notes

1. Print and test-fit `yoke_mg90s_fit_test.stl` first.
2. Centre both servos at 90° **before** attaching the horns, or the mechanical
   range will be offset from the commanded range and the limits will be wrong.
3. Seat the 608ZZ bearing in `bearing_holder.stl` — it carries the radial load
   the pan servo's output shaft should not.
4. Route the camera ribbon with slack through the full pan range. It is the
   first thing to fail.
5. Power the servo supply and the Pi separately; join grounds.
6. Upload the firmware and verify with `uavtrack check` before mounting the
   camera.

## Calibrating the field of view

`horizontal_fov_deg` sets the pixel-to-angle gain for the whole control loop.
Getting it wrong is indistinguishable from mistuning every gain at once.

Measure it: place a target of known width `W` at a known distance `D`,
perpendicular to the optical axis, and note the pixel width `p` it occupies in a
full-frame capture.

$$\text{HFOV} = 2\arctan\left(\frac{W_\text{frame}}{2D}\right), \qquad W_\text{frame} = W\cdot\frac{W_\text{image px}}{p}$$

Put the result in your config. Do not trust the marketing figure — it is quoted
for the full sensor, and most capture modes crop.

---

**See also:** [control-design.md](control-design.md) ·
[hailo-deployment.md](hailo-deployment.md)
