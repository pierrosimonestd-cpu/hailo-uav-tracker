#!/usr/bin/env python3
"""Live view plus tuning controls, served over HTTP.

Tuning a tracking loop by editing YAML and restarting costs a minute per
attempt, and the thing being tuned -- a drone someone is holding up -- does not
hold still for it. This serves the annotated stream next to the parameters that
matter and applies changes to the running pipeline between frames, so the
effect of a threshold or a gain is visible immediately.

Changes are queued and applied by the worker thread at a frame boundary, never
from the HTTP thread: a detector whose confidence threshold changes midway
through its own inference is a race nobody wants to debug.

Nothing is written to disk until `Save to config` is used, so experimenting
cannot quietly corrupt a working configuration.

Usage, on the Pi:
    python tools/tune_dashboard.py --config configs/bench_rig.yaml

Then open http://<pi-address>:8080/ from any machine on the network.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import queue
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from uavtrack.config import PipelineConfig  # noqa: E402
from uavtrack.pipeline import TrackingPipeline  # noqa: E402
from uavtrack.viz import annotate  # noqa: E402

logger = logging.getLogger("tune")
BOUNDARY = "uavtrackframe"

#: Every adjustable parameter: where it lives, its range, and why it matters.
#: Keeping this one table means the UI, the API and the YAML writer cannot
#: disagree about what exists.
PARAMS = [
    (
        "detector.conf_threshold",
        0.05,
        0.9,
        0.01,
        "Detector confidence. Lower finds more, including more false positives.",
    ),
    (
        "detector.iou_threshold",
        0.1,
        0.9,
        0.05,
        "NMS overlap. Lower merges overlapping boxes more aggressively.",
    ),
    (
        "tracker.high_threshold",
        0.1,
        0.9,
        0.05,
        "Score above which a detection may start a new track.",
    ),
    (
        "tracker.low_threshold",
        0.02,
        0.5,
        0.01,
        "Second-pass score: recovers weak detections on tracks already alive.",
    ),
    ("tracker.match_threshold", 0.1, 0.9, 0.05, "IoU needed to associate on the first pass."),
    (
        "tracker.second_match_threshold",
        0.05,
        0.9,
        0.05,
        "IoU needed on the second, more forgiving pass.",
    ),
    (
        "tracker.max_age",
        1,
        60,
        1,
        "Frames a track survives unseen. Large values coast blind, then jump on re-acquisition.",
    ),
    ("tracker.min_hits", 1, 10, 1, "Detections required before a track may steer the turret."),
    (
        "control.latency_s",
        0.0,
        0.3,
        0.005,
        "Sense-to-act delay compensated for. Set it to the measured value.",
    ),
    (
        "control.feedforward_gain",
        0.0,
        1.2,
        0.05,
        "How much of the estimated target rate to feed forward.",
    ),
    ("control.pan.kp", 0.0, 8.0, 0.1, "Pan proportional gain."),
    ("control.pan.ki", 0.0, 3.0, 0.05, "Pan integral gain."),
    ("control.pan.kd", 0.0, 2.0, 0.05, "Pan derivative gain."),
    ("control.pan.deadband_deg", 0.0, 5.0, 0.1, "Pan error below which nothing is commanded."),
    ("control.pan.max_rate_deg_s", 20.0, 600.0, 10.0, "Pan slew limit."),
    ("control.tilt.kp", 0.0, 8.0, 0.1, "Tilt proportional gain."),
    ("control.tilt.ki", 0.0, 3.0, 0.05, "Tilt integral gain."),
    ("control.tilt.kd", 0.0, 2.0, 0.05, "Tilt derivative gain."),
    ("control.tilt.deadband_deg", 0.0, 5.0, 0.1, "Tilt error below which nothing is commanded."),
    ("control.tilt.max_rate_deg_s", 20.0, 600.0, 10.0, "Tilt slew limit."),
]

TOGGLES = [
    ("control.pan.invert", "Reverse pan"),
    ("control.tilt.invert", "Reverse tilt"),
    ("link.enabled", "Drive the servos"),
]

INTEGER_PARAMS = {"tracker.max_age", "tracker.min_hits"}


class Shared:
    """Newest frame, newest stats, and the queue of pending parameter edits."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._stopped = False
        self.edits: queue.Queue[tuple[str, float | bool]] = queue.Queue()
        self.lock = threading.Lock()
        self.stats: dict = {}
        self.params: dict = {}

    def publish(self, jpeg: bytes) -> None:
        with self._condition:
            self._jpeg = jpeg
            self._sequence += 1
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def wait_for_next(self, last_seen: int, timeout: float = 5.0):
        with self._condition:
            if not self._condition.wait_for(
                lambda: self._stopped or self._sequence > last_seen, timeout=timeout
            ):
                return None, last_seen
            if self._stopped:
                return None, self._sequence
            return self._jpeg, self._sequence


# ------------------------------------------------------------- live plumbing


def read_param(pipeline, config, path: str):
    """Current value, taken from the live objects rather than the config."""
    detector, tracker, controller = pipeline.detector, pipeline.tracker, pipeline.controller
    match path.split("."):
        case ["detector", name]:
            return getattr(detector, name)
        case ["tracker", name]:
            return getattr(tracker, name)
        case ["control", "latency_s"]:
            return controller.latency_s
        case ["control", "feedforward_gain"]:
            return controller.feedforward_gain
        case ["control", axis, name] if axis in ("pan", "tilt"):
            pid = controller.pan_pid if axis == "pan" else controller.tilt_pid
            limits = controller.pan_limits if axis == "pan" else controller.tilt_limits
            if name in ("kp", "ki", "kd"):
                return getattr(pid.gains, name)
            return getattr(limits, name)
        case ["link", "enabled"]:
            return pipeline.link is not None
    raise KeyError(path)


def apply_param(pipeline, config, path: str, value) -> None:
    """Write one parameter into the running objects and the in-memory config."""
    detector, tracker, controller = pipeline.detector, pipeline.tracker, pipeline.controller
    match path.split("."):
        case ["detector", name]:
            setattr(detector, name, value)
            setattr(config.detector, name, value)
        case ["tracker", name]:
            setattr(tracker, name, value)
            setattr(config.tracker, name, value)
        case ["control", "latency_s"]:
            controller.latency_s = value
            config.control.latency_s = value
        case ["control", "feedforward_gain"]:
            controller.feedforward_gain = value
            config.control.feedforward_gain = value
        case ["control", axis, name] if axis in ("pan", "tilt"):
            pid = controller.pan_pid if axis == "pan" else controller.tilt_pid
            axis_cfg = config.control.pan if axis == "pan" else config.control.tilt
            if name in ("kp", "ki", "kd"):
                setattr(pid.gains, name, value)
            else:
                # AxisLimits is frozen: replace it rather than mutate it.
                current = controller.pan_limits if axis == "pan" else controller.tilt_limits
                updated = dataclasses.replace(current, **{name: value})
                if axis == "pan":
                    controller.pan_limits = updated
                else:
                    controller.tilt_limits = updated
            setattr(axis_cfg, name, value)
        case ["link", "enabled"]:
            # Detaching the link is the stop button: the pipeline simply has
            # nothing to send to, and the firmware's own 500 ms failsafe then
            # releases the servos without anything else having to happen.
            pipeline.link = pipeline._saved_link if value else None
            config.link.enabled = bool(value)
        case _:
            raise KeyError(path)


def write_config(config, path: Path) -> None:
    """Persist the current parameters back into the YAML, comments and all.

    A line-wise rewrite rather than a yaml.dump: dumping would discard every
    comment in the file, and in this project the comments carry the reasoning
    for the numbers, which is worth more than the numbers.
    """
    import re

    text = path.read_text(encoding="utf-8")
    values: dict[str, object] = {}
    for spec in PARAMS:
        key = spec[0]
        values[key.split(".")[-1] + "@" + ".".join(key.split(".")[:-1])] = None

    def set_leaf(section_path: list[str], leaf: str, value) -> None:
        nonlocal text
        # Find the section, then the first occurrence of the leaf inside it.
        indent_of_section = 0
        pos = 0
        for depth, part in enumerate(section_path):
            pattern = re.compile(rf"^{' ' * (depth * 2)}{re.escape(part)}:\s*$", re.M)
            m = pattern.search(text, pos)
            if not m:
                return
            pos = m.end()
            indent_of_section = depth * 2 + 2
        leaf_pattern = re.compile(rf"^({' ' * indent_of_section}{re.escape(leaf)}:)([^\n]*)$", re.M)
        m = leaf_pattern.search(text, pos)
        if not m:
            return
        rendered = "true" if value is True else "false" if value is False else f"{value}"
        text = text[: m.start()] + f"{m.group(1)} {rendered}" + text[m.end() :]

    for key, *_ in PARAMS:
        parts = key.split(".")
        sections = {
            "detector": config.detector,
            "tracker": config.tracker,
            "control": config.control,
        }
        section = sections[parts[0]]
        obj = section
        for part in parts[1:-1]:
            obj = getattr(obj, part)
        set_leaf(parts[:-1], parts[-1], getattr(obj, parts[-1]))

    for key, _label in TOGGLES:
        parts = key.split(".")
        root = {"control": config.control, "link": config.link}[parts[0]]
        obj = root
        for part in parts[1:-1]:
            obj = getattr(obj, part)
        set_leaf(parts[:-1], parts[-1], getattr(obj, parts[-1]))

    path.write_text(text, encoding="utf-8")


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>uavtrack tuning</title>
<style>
  :root { color-scheme: dark; --bg:#0f1115; --panel:#171a21; --line:#272c36;
          --text:#e6e8ec; --dim:#8b93a1; --accent:#4c9aff; --warn:#ffb648; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }
  header { padding:12px 18px; border-bottom:1px solid var(--line);
           display:flex; gap:18px; align-items:baseline; flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:600; letter-spacing:.2px; }
  .stat { color:var(--dim); font-variant-numeric:tabular-nums; }
  .stat b { color:var(--text); font-weight:600; }
  main { display:grid; grid-template-columns:minmax(320px,1fr) 380px; gap:16px;
         padding:16px; align-items:start; }
  @media (max-width:900px){ main{grid-template-columns:1fr} }
  img { width:100%; border:1px solid var(--line); border-radius:8px;
        display:block; background:#000; }
  .legend { color:var(--dim); font-size:12px; margin-top:8px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px;
           padding:14px; max-height:calc(100vh - 120px); overflow:auto; }
  .group { font-size:11px; text-transform:uppercase; letter-spacing:.08em;
           color:var(--dim); margin:16px 0 8px; }
  .group:first-child { margin-top:0; }
  .row { margin-bottom:12px; }
  .row label { display:flex; justify-content:space-between; gap:8px; font-size:12px; }
  .row .val { font-variant-numeric:tabular-nums; color:var(--accent); }
  input[type=range]{ width:100%; margin:4px 0 0; accent-color:var(--accent); }
  .hint { font-size:11px; color:var(--dim); margin-top:2px; }
  .toggles { display:flex; flex-direction:column; gap:8px; }
  .toggle { display:flex; align-items:center; gap:8px; font-size:13px; }
  button { background:var(--accent); color:#04101f; border:0; border-radius:6px;
           padding:8px 12px; font-weight:600; cursor:pointer; }
  button.secondary { background:#2a303b; color:var(--text); }
  .actions { display:flex; gap:8px; margin-top:16px; flex-wrap:wrap; }
  #saved { font-size:12px; color:var(--warn); margin-top:8px; min-height:16px; }
</style></head>
<body>
<header>
  <h1>uavtrack &mdash; live tuning</h1>
  <span class="stat">fps <b id="fps">-</b></span>
  <span class="stat">det <b id="det">-</b></span>
  <span class="stat">tracks <b id="trk">-</b></span>
  <span class="stat">target <b id="tid">-</b></span>
  <span class="stat">pan <b id="pan">-</b></span>
  <span class="stat">tilt <b id="tilt">-</b></span>
  <span class="stat">link <b id="link">-</b></span>
</header>
<main>
  <div>
    <img src="/stream.mjpg" alt="live view">
    <div class="legend">grey = detections &middot; green = tracks &middot;
      orange = the one being followed</div>
  </div>
  <div class="panel" id="panel"></div>
</main>
<script>
function build(state) {
  const panel = document.getElementById('panel');
  let html = '';
  let lastGroup = '';
  for (const p of state.params) {
    const parts = p.key.split('.');
    const group = parts.length > 2 ? parts[0] + ' / ' + parts[1] : parts[0];
    if (group !== lastGroup) {
      html += '<div class="group">' + group + '</div>';
      lastGroup = group;
    }
    const name = parts[parts.length - 1];
    html += '<div class="row">' +
            '<label><span>' + name + '</span>' +
            '<span class="val" id="v-' + p.key + '">' + p.value + '</span></label>' +
            '<input type="range" id="r-' + p.key + '" min="' + p.min + '" max="' + p.max +
            '" step="' + p.step + '" value="' + p.value + '">' +
            '<div class="hint">' + p.hint + '</div></div>';
  }
  html += '<div class="group">switches</div><div class="toggles">';
  for (const t of state.toggles) {
    html += '<label class="toggle"><input type="checkbox" id="c-' + t.key + '"' +
            (t.value ? ' checked' : '') + '> ' + t.label + '</label>';
  }
  html += '</div><div class="actions">' +
          '<button id="save">Save to config</button>' +
          '<button class="secondary" id="reload">Reload from config</button>' +
          '</div><div id="saved"></div>';
  panel.innerHTML = html;

  for (const p of state.params) {
    const r = document.getElementById('r-' + p.key);
    r.addEventListener('input', function () {
      document.getElementById('v-' + p.key).textContent = r.value;
      post({key: p.key, value: parseFloat(r.value)});
    });
  }
  for (const t of state.toggles) {
    const c = document.getElementById('c-' + t.key);
    c.addEventListener('change', function () { post({key: t.key, value: c.checked}); });
  }
  document.getElementById('save').onclick = function () {
    fetch('/api/save', {method: 'POST'}).then(function (r) { return r.json(); })
      .then(function (j) { document.getElementById('saved').textContent = j.message; });
  };
  document.getElementById('reload').onclick = function () {
    fetch('/api/reload', {method: 'POST'}).then(function () { location.reload(); });
  };
}

function post(body) {
  fetch('/api/param', {method: 'POST',
                       headers: {'Content-Type': 'application/json'},
                       body: JSON.stringify(body)});
}

function poll() {
  fetch('/api/state').then(function (r) { return r.json(); }).then(function (s) {
    if (!document.getElementById('panel').innerHTML) build(s);
    const st = s.stats || {};
    const show = function (id, v) {
      document.getElementById(id).textContent = (v === null || v === undefined) ? '-' : v;
    };
    show('fps', st.fps); show('det', st.detections); show('trk', st.tracks);
    show('tid', st.target_id); show('pan', st.pan); show('tilt', st.tilt);
    show('link', st.link);
  }).catch(function () { /* between frames */ })
    .finally(function () { setTimeout(poll, 400); });
}
poll();
</script>
</body></html>
"""


def make_handler(shared: Shared, on_save, on_reload):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: ARG002
            logger.debug("%s - %s", self.address_string(), fmt % args)

        def _json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/api/state":
                with shared.lock:
                    values = dict(shared.params)
                    stats = dict(shared.stats)
                self._json(
                    {
                        "params": [
                            {
                                "key": key,
                                "min": lo,
                                "max": hi,
                                "step": step,
                                "hint": hint,
                                "value": values.get(key),
                            }
                            for key, lo, hi, step, hint in PARAMS
                        ],
                        "toggles": [
                            {"key": key, "label": label, "value": bool(values.get(key))}
                            for key, label in TOGGLES
                        ],
                        "stats": stats,
                    }
                )
                return

            if self.path != "/stream.mjpg":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.end_headers()
            seen = 0
            try:
                while True:
                    jpeg, seen = shared.wait_for_next(seen)
                    if jpeg is None:
                        break
                    self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                logger.info("viewer disconnected")

        def do_POST(self):  # noqa: N802
            if self.path == "/api/param":
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                key, value = payload.get("key"), payload.get("value")
                known = any(key == spec[0] for spec in PARAMS) or any(key == t[0] for t in TOGGLES)
                if not known:
                    self._json({"error": f"unknown parameter {key!r}"}, 400)
                    return
                if key in INTEGER_PARAMS:
                    value = int(value)
                shared.edits.put((key, value))
                self._json({"ok": True})
                return

            if self.path == "/api/save":
                self._json({"message": on_save()})
                return

            if self.path == "/api/reload":
                on_reload()
                self._json({"ok": True})
                return

            self.send_error(404)

    return Handler


def local_urls(port: int) -> list[str]:
    urls = []
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        urls.append(f"http://{probe.getsockname()[0]}:{port}/")
        probe.close()
    except OSError:
        pass
    urls.append(f"http://{socket.gethostname()}.local:{port}/")
    return urls


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/bench_rig.yaml"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    config = PipelineConfig.load(args.config)

    link = None
    if config.link.enabled:
        from uavtrack.io.serial_link import SerialLink

        link = SerialLink(
            config.link.port,
            config.link.baudrate,
            heartbeat_interval_s=config.link.heartbeat_interval_s,
        )
        logger.info("serial link open on %s", config.link.port)

    shared = Shared()
    stop = threading.Event()
    reload_requested = threading.Event()
    state: dict = {"pipeline": None}

    def on_save() -> str:
        if state["pipeline"] is None:
            return "not running"
        try:
            write_config(config, args.config)
            return f"saved to {args.config}"
        except Exception as exc:  # pragma: no cover - surfaced in the UI
            logger.exception("save failed")
            return f"save failed: {exc}"

    def on_reload() -> None:
        reload_requested.set()

    def worker() -> None:
        try:
            with TrackingPipeline(config, link=link) as pipeline:
                pipeline._saved_link = link
                state["pipeline"] = pipeline
                logger.info(
                    "detector: %s at %s", pipeline.detector.name, pipeline.detector.input_size
                )
                window_start = time.monotonic()
                window_frames = 0
                fps = 0.0

                for frames, result in enumerate(pipeline.run(), start=1):
                    if stop.is_set():
                        break

                    # Apply queued edits at a frame boundary, never mid-inference.
                    while True:
                        try:
                            key, value = shared.edits.get_nowait()
                        except queue.Empty:
                            break
                        try:
                            apply_param(pipeline, config, key, value)
                            logger.info("%s = %s", key, value)
                        except Exception:
                            logger.exception("could not apply %s", key)

                    if reload_requested.is_set():
                        reload_requested.clear()
                        fresh = PipelineConfig.load(args.config)
                        sections = {
                            "detector": fresh.detector,
                            "tracker": fresh.tracker,
                            "control": fresh.control,
                            "link": fresh.link,
                        }
                        for key in [spec[0] for spec in PARAMS] + [t[0] for t in TOGGLES]:
                            parts = key.split(".")
                            obj = sections[parts[0]]
                            for part in parts[1:-1]:
                                obj = getattr(obj, part)
                            apply_param(pipeline, config, key, getattr(obj, parts[-1]))
                        logger.info("reloaded %s", args.config)

                    window_frames += 1
                    now = time.monotonic()
                    if now - window_start >= 1.0:
                        fps = window_frames / (now - window_start)
                        window_start, window_frames = now, 0

                    annotated = annotate(result)
                    ok, buffer = cv2.imencode(
                        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, args.quality]
                    )
                    if ok:
                        shared.publish(buffer.tobytes())

                    command = result.command
                    with shared.lock:
                        shared.stats = {
                            "fps": round(fps, 1),
                            "detections": len(result.detections),
                            "tracks": len(result.tracks),
                            "target_id": result.target.track_id if result.target else None,
                            "pan": round(command.pan_deg, 1) if command else None,
                            "tilt": round(command.tilt_deg, 1) if command else None,
                            "link": "on" if pipeline.link is not None else "off",
                            "frames": frames,
                        }
                        live = {key: read_param(pipeline, config, key) for key, *_ in PARAMS}
                        live.update({key: read_param(pipeline, config, key) for key, _ in TOGGLES})
                        shared.params = live
        except Exception:
            logger.exception("pipeline stopped")
        finally:
            shared.stop()
            stop.set()

    thread = threading.Thread(target=worker, name="pipeline", daemon=True)
    thread.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(shared, on_save, on_reload))
    server.daemon_threads = True
    server.timeout = 0.5
    for url in local_urls(args.port):
        logger.info("open %s", url)

    try:
        while not stop.is_set():
            server.handle_request()
    except KeyboardInterrupt:
        logger.info("stopping")
    finally:
        stop.set()
        shared.stop()
        server.server_close()
        thread.join(timeout=5.0)
        # The pipeline owns the link and closes it on exit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
