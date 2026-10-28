import argparse
import json
import sys
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(description="Local fatigue monitoring and UTA-RLDD evaluation")
    commands = root.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-model", help="Fetch the official MediaPipe task model")
    download.add_argument("--model", default="models/face_landmarker.task")
    serve = commands.add_parser("serve", help="Run the local browser dashboard")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model", default="models/face_landmarker.task")
    serve.add_argument(
        "--classifier",
        default="models/behavior.joblib",
        help="Trusted locally trained behavioural classifier",
    )
    serve.add_argument("--rules", action="store_true", help="Use the legacy rule engine")
    doctor = commands.add_parser("doctor", help="Check runtime, model, and optional CUDA training")
    doctor.add_argument("--gpu", action="store_true")
    doctor.add_argument("--model", default="models/face_landmarker.task")
    manifest = commands.add_parser("manifest", help="Index subject/0,5,10 video files")
    manifest.add_argument("root", type=Path)
    manifest.add_argument("--output", type=Path, default=Path("data/manifest.csv"))
    extract = commands.add_parser("extract", help="Extract reusable facial-feature CSV files")
    extract.add_argument("manifest", type=Path)
    extract.add_argument("--output", type=Path, default=Path("data/features"))
    extract.add_argument("--model", default="models/face_landmarker.task")
    extract.add_argument("--clahe", action="store_true")
    extract.add_argument("--fps", type=float, default=30.0)
    evaluate = commands.add_parser("evaluate", help="Evaluate rules and optional RF/GRU using LOSO")
    evaluate.add_argument("index", type=Path)
    evaluate.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    evaluate.add_argument("--protocol", choices=["enrolled", "fixed"], default="enrolled")
    evaluate.add_argument("--model", choices=["rules", "rf", "gru"], default="rf")
    evaluate.add_argument("--seconds", type=int, default=30)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--epochs", type=int, default=20)
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    analyze = commands.add_parser("analyze", help="Replay a video through the live model or rules")
    analyze.add_argument("video", type=Path)
    analyze.add_argument("--output", type=Path, default=Path("outputs/analysis.jsonl"))
    analyze.add_argument("--model", default="models/face_landmarker.task")
    analyze.add_argument("--clahe", action="store_true")
    analyze.add_argument("--fixed", action="store_true", help="Skip personal calibration")
    analyze.add_argument("--classifier", default="models/behavior.joblib")
    analyze.add_argument("--rules", action="store_true", help="Use the legacy rule engine")
    return root


def main():
    args = parser().parse_args()
    try:
        if args.command == "download-model":
            from .vision import download_model

            print(json.dumps(download_model(args.model), indent=2))
        elif args.command == "serve":
            import uvicorn

            from .server import create_app

            print(
                f"Open http://127.0.0.1:{args.port} — camera starts only when you choose Start camera"
            )
            uvicorn.run(
                create_app(args.model, None if args.rules else args.classifier),
                host="127.0.0.1",
                port=args.port,
                ws_max_size=650_000,
                log_level="info",
            )
        elif args.command == "doctor":
            import cv2
            import mediapipe as mp

            from .vision import Tracker

            report = {
                "python": sys.version.split()[0],
                "opencv": cv2.__version__,
                "mediapipe": mp.__version__,
                "model_ready": Path(args.model).is_file(),
            }
            if report["model_ready"]:
                import numpy as np

                with Tracker(args.model) as tracker:
                    frame, _ = tracker.process(np.zeros((480, 640, 3), dtype=np.uint8), 0.0)
                report["landmarker_smoke_passed"] = not frame.valid
            if args.gpu:
                from .temporal import gpu_diagnostic

                report["training"] = gpu_diagnostic()
            print(json.dumps(report, indent=2))
        elif args.command == "manifest":
            from .dataset import make_manifest

            print(json.dumps(make_manifest(args.root, args.output), indent=2))
        elif args.command == "extract":
            from .dataset import extract

            print(extract(args.manifest, args.output, args.model, args.clahe, args.fps))
        elif args.command == "evaluate":
            from .evaluation import evaluate

            report = evaluate(
                args.index,
                args.output,
                args.protocol,
                args.model,
                args.seconds,
                args.seed,
                args.epochs,
                args.device,
            )
            print(json.dumps(report["window_metrics"], indent=2))
        elif args.command == "analyze":
            from .core import Baseline, Monitor
            from .dataset import video_frames, write_json
            from .learned import LearnedMonitor, load_classifier
            from .vision import Tracker

            learned = (
                load_classifier(args.classifier)
                if not args.fixed and not args.rules and Path(args.classifier).is_file()
                else None
            )
            monitor = (
                LearnedMonitor(*learned)
                if learned
                else Monitor(baseline=Baseline() if args.fixed else None)
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            with (
                Tracker(
                    args.model, args.clahe, bool(learned and "expression_names" in learned[1])
                ) as tracker,
                args.output.open("w", encoding="utf-8") as stream,
            ):
                for image, timestamp in video_frames(args.video):
                    frame, _ = tracker.process(image, timestamp)
                    result = (
                        monitor.update(frame, tracker.last_expression)
                        if learned
                        else monitor.update(frame)
                    )
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                    count += 1
            if count == 0:
                raise ValueError("Video contains no decoded frames")
            write_json(args.output.with_suffix(".metadata.json"), monitor.metadata())
            print(f"Processed {count} frames. Results: {args.output}")
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
