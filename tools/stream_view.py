#!/usr/bin/env python3
"""Serve the annotated live view over HTTP, for a headless Pi.

`uavtrack run --display` needs a screen attached to the Pi. On a board that is
being driven over SSH -- which is how anyone actually works on one -- that is
the wrong shape. This runs the same pipeline with the same annotations and
serves the frames as MJPEG, so the view opens in a browser on any machine on
the network.

The pipeline runs in a worker thread and publishes the newest annotated frame;
the HTTP handler sends whatever is current when it asks. Slow or disconnected
viewers therefore drop frames instead of stalling the control loop, which is
the correct trade: the turret must not wait for a browser.

Usage, on the Pi:
    python tools/stream_view.py --config configs/bench_rig.yaml
    python tools/stream_view.py --config configs/bench_rig.yaml --no-link

Then open http://<pi-address>:8080/ from anywhere on the network.
"""

from __future__ import annotations

import argparse
import logging
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

logger = logging.getLogger("stream_view")

BOUNDARY = "uavtrackframe"


class LatestFrame:
    """One slot holding the newest JPEG, plus a counter viewers can wait on."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0
        self._stopped = False

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
        """Block until a frame newer than ``last_seen`` exists."""
        with self._condition:
            if not self._condition.wait_for(
                lambda: self._stopped or self._sequence > last_seen, timeout=timeout
            ):
                return None, last_seen
            if self._stopped:
                return None, self._sequence
            return self._jpeg, self._sequence


PAGE = b"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>uavtrack live</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#111; color:#ddd; font:14px system-ui,sans-serif;
         display:flex; flex-direction:column; align-items:center; gap:12px; padding:16px; }
  img { max-width:100%; height:auto; border:1px solid #333; border-radius:6px; }
  p { margin:0; color:#888; }
</style></head>
<body>
  <img src="/stream.mjpg" alt="live view">
  <p>grey = detections &middot; green = tracks &middot; orange = the one being followed</p>
</body></html>
"""


def make_handler(latest: LatestFrame):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: ARG002 - quieter than the default
            logger.debug("%s - %s", self.address_string(), fmt % args)

        def do_GET(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
                return

            if self.path != "/stream.mjpg":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.end_headers()

            seen = 0
            try:
                while True:
                    jpeg, seen = latest.wait_for_next(seen)
                    if jpeg is None:
                        break
                    self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                logger.info("viewer disconnected")

    return Handler


def local_addresses(port: int) -> list[str]:
    """Best-effort list of URLs this server can be reached on."""
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
    parser.add_argument("--config", type=Path, default=Path("configs/rpi5_hailo8l.yaml"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality, 1-100")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-link", action="store_true", help="never open the serial link")
    parser.add_argument("--record", type=Path, default=None, help="also write an annotated video")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    config = PipelineConfig.load(args.config)

    link = None
    if config.link.enabled and not args.no_link:
        from uavtrack.io.serial_link import SerialLink

        link = SerialLink(
            config.link.port,
            config.link.baudrate,
            heartbeat_interval_s=config.link.heartbeat_interval_s,
        )
        logger.info("serial link open on %s", config.link.port)

    latest = LatestFrame()
    stop = threading.Event()
    counters = {"frames": 0, "tracked": 0, "detections": 0}

    def worker() -> None:
        writer = None
        try:
            with TrackingPipeline(config, link=link) as pipeline:
                logger.info(
                    "detector: %s at %s", pipeline.detector.name, pipeline.detector.input_size
                )
                for result in pipeline.run(max_frames=args.max_frames):
                    if stop.is_set():
                        break
                    counters["frames"] += 1
                    counters["detections"] += len(result.detections)
                    if result.target is not None:
                        counters["tracked"] += 1

                    annotated = annotate(result)
                    if args.record is not None:
                        if writer is None:
                            height, width = annotated.shape[:2]
                            writer = cv2.VideoWriter(
                                str(args.record),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                config.camera.fps,
                                (width, height),
                            )
                        writer.write(annotated)

                    ok, buffer = cv2.imencode(
                        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, args.quality]
                    )
                    if ok:
                        latest.publish(buffer.tobytes())
        except Exception:
            logger.exception("pipeline stopped")
        finally:
            if writer is not None:
                writer.release()
            latest.stop()
            stop.set()

    thread = threading.Thread(target=worker, name="pipeline", daemon=True)
    thread.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(latest))
    server.daemon_threads = True
    for url in local_addresses(args.port):
        logger.info("open %s", url)

    reporter = time.monotonic()
    try:
        server.timeout = 0.5
        while not stop.is_set():
            server.handle_request()
            if time.monotonic() - reporter >= 5.0:
                reporter = time.monotonic()
                logger.info(
                    "%d frames, %d detections, %d frames with a target",
                    counters["frames"],
                    counters["detections"],
                    counters["tracked"],
                )
    except KeyboardInterrupt:
        logger.info("stopping")
    finally:
        stop.set()
        latest.stop()
        server.server_close()
        thread.join(timeout=5.0)
        # The link is deliberately not closed here. TrackingPipeline takes
        # ownership of whatever link it is given and closes it on __exit__, so
        # closing it again raises PortNotOpenError out of the shutdown path --
        # which it did, on the first run against the real board.

    logger.info(
        "final: %d frames, %d detections, %d frames with a target",
        counters["frames"],
        counters["detections"],
        counters["tracked"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
