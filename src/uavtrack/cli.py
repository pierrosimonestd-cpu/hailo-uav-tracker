"""Command-line entry point: ``uavtrack <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

from uavtrack import __version__
from uavtrack.config import PipelineConfig

logger = logging.getLogger("uavtrack")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uavtrack",
        description="Real-time UAV detection and pan/tilt tracking on an edge NPU.",
    )
    parser.add_argument("--version", action="version", version=f"uavtrack {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the tracking pipeline")
    run.add_argument("--config", type=Path, default=Path("configs/rpi5_hailo8l.yaml"))
    run.add_argument("--source", help="override the configured frame source")
    run.add_argument("--model", help="override the configured model path")
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument("--display", action="store_true", help="show an annotated window")
    run.add_argument("--record", type=Path, default=None, help="write an annotated video here")
    run.add_argument("--no-link", action="store_true", help="never open the serial link")
    run.add_argument("--stats", type=Path, default=None, help="write per-frame timings as JSON")

    check = sub.add_parser("check", help="validate a config and report the resolved backend")
    check.add_argument("--config", type=Path, default=Path("configs/rpi5_hailo8l.yaml"))

    simulate = sub.add_parser("simulate", help="run a closed-loop pointing simulation")
    simulate.add_argument(
        "--trajectory",
        default="circular",
        choices=["static", "step", "linear", "circular", "crossing"],
    )
    simulate.add_argument("--latency-ms", type=float, default=45.0)
    simulate.add_argument("--feedforward", type=float, default=0.9)
    simulate.add_argument("--duration", type=float, default=8.0)
    return parser


def _cmd_check(args: argparse.Namespace) -> int:
    config = PipelineConfig.load(args.config)
    from uavtrack.detect.factory import infer_backend_from_path

    backend = config.detector.backend or infer_backend_from_path(config.detector.model)
    exists = Path(config.detector.model).exists()

    print(f"config      {args.config}")
    print(f"model       {config.detector.model} ({'found' if exists else 'MISSING'})")
    print(f"backend     {backend}")
    print(f"source      {config.camera.source} @ {config.camera.width}x{config.camera.height}")
    print(f"link        {'enabled on ' + config.link.port if config.link.enabled else 'disabled'}")
    print(f"latency     {config.control.latency_s * 1000:.0f} ms compensated")
    return 0 if exists else 1


def _cmd_simulate(args: argparse.Namespace) -> int:
    from uavtrack.control import SimulationConfig, TargetTrajectory, simulate

    result = simulate(
        TargetTrajectory(kind=args.trajectory),
        SimulationConfig(
            duration_s=args.duration,
            detection_latency_s=args.latency_ms / 1000.0,
            feedforward_gain=args.feedforward,
        ),
    )
    summary = result.summary()
    width = max(len(k) for k in summary)
    for key, value in summary.items():
        formatted = "n/a" if value is None else f"{value:.4f}"
        print(f"{key:<{width}}  {formatted}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    import cv2

    from uavtrack.pipeline import TrackingPipeline
    from uavtrack.viz import annotate

    config = PipelineConfig.load(args.config)
    if args.source:
        config.camera.source = args.source
    if args.model:
        config.detector.model = args.model
        config.detector.backend = None

    link = None
    if config.link.enabled and not args.no_link:
        from uavtrack.io.serial_link import SerialLink

        link = SerialLink(
            config.link.port,
            config.link.baudrate,
            heartbeat_interval_s=config.link.heartbeat_interval_s,
        )
        logger.info("serial link open on %s", config.link.port)

    writer = None
    stats: list[dict] = []
    interrupted = False

    def _on_signal(signum, frame):  # noqa: ARG001
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _on_signal)

    with TrackingPipeline(config, link=link) as pipeline:
        logger.info("detector: %s at %s", pipeline.detector.name, pipeline.detector.input_size)
        for result in pipeline.run(max_frames=args.max_frames):
            stats.append({"index": result.index, **result.timings_ms})

            if args.display or args.record:
                annotated = annotate(result)
                if args.record and writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(args.record),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        config.camera.fps,
                        (width, height),
                    )
                if writer is not None:
                    writer.write(annotated)
                if args.display:
                    cv2.imshow("uavtrack", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if interrupted:
                logger.info("interrupted, shutting down")
                break

    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    if stats:
        total = [sum(v for k, v in s.items() if k != "index") for s in stats]
        total.sort()
        print(f"frames        {len(total)}")
        print(f"median total  {total[len(total) // 2]:.1f} ms")
        print(f"p95 total     {total[int(len(total) * 0.95)]:.1f} ms")
    if args.stats:
        args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``uavtrack`` console script."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    )

    handlers = {"run": _cmd_run, "check": _cmd_check, "simulate": _cmd_simulate}
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
