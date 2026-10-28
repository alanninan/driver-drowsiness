"""Local-only dashboard service. Frames are processed in memory and discarded."""

import asyncio
import contextlib
import io
import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import __version__
from .core import Baseline, Config, Frame, Monitor
from .learned import DEFAULT_CLASSIFIER, LearnedMonitor, load_classifier
from .vision import DEFAULT_MODEL, Tracker

STATIC = Path(__file__).parent / "static"


def decode_packet(packet):
    if not 12 <= len(packet) <= 600_000:
        raise ValueError("Invalid frame size")
    timestamp = struct.unpack("<d", packet[:8])[0]
    if not np.isfinite(timestamp) or timestamp < 0:
        raise ValueError("Invalid frame timestamp")
    data = packet[8:]
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG" or image.width > 1920 or image.height > 1080:
            raise ValueError("Send JPEG frames no larger than 1920 x 1080")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Frame could not be decoded")
    return image, timestamp


def demo_frame(t):
    """Deterministic synthetic measurements, not detection or validation footage."""
    phase = t % 65
    closed = (20 < phase < 22.2) or (35 < phase < 44) or (t % 4 < 0.16)
    ear = 0.10 if closed else 0.30
    return Frame(
        t, ear, ear, 0.65 if 10 < phase < 12 else 0.05, 20 if 14 < phase < 15 else 0, 0, 0, 120
    )


def create_app(model=DEFAULT_MODEL, classifier=DEFAULT_CLASSIFIER):
    app = FastAPI(title="Driver Drowsiness", docs_url=None, redoc_url=None)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    slots = asyncio.Semaphore(2)
    learned = load_classifier(classifier) if classifier and Path(classifier).is_file() else None

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' blob: data:; media-src 'self' blob:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
        )
        return response

    @app.get("/")
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {
            "version": __version__,
            "model_ready": Path(model).is_file(),
            "landmark_device": "cpu",
            "scoring": "behavioral_classifier" if learned else "heuristic",
            "classifier_ready": learned is not None,
            "classifier": learned[1]["estimator"] if learned else None,
            "calibration_seconds": 180 if learned else 45,
            "dataset_benchmark": "Fold1 development benchmark",
        }

    @app.websocket("/ws/{mode}")
    async def session(ws: WebSocket, mode: str):
        origin = ws.headers.get("origin")
        expected = f"{'https' if ws.url.scheme == 'wss' else 'http'}://{ws.headers.get('host')}"
        if mode not in {"live", "demo"} or origin != expected:
            await ws.close(code=1008)
            return
        await ws.accept()
        if slots.locked():
            await ws.send_json({"error": "Two sessions are already active. Stop one and retry"})
            await ws.close(code=1013)
            return
        async with slots:
            tracker = None
            try:
                options = await asyncio.wait_for(ws.receive_json(), timeout=10)
                if not isinstance(options, dict):
                    raise ValueError("Expected session options")
                seconds = float(options.get("calibration_seconds", 45))
                use_learned = mode == "live" and learned is not None
                if not use_learned and not 30 <= seconds <= 60:
                    if mode == "demo" and seconds == 180:
                        seconds = 45
                    else:
                        raise ValueError("Calibration must be 30 to 60 seconds")
                if use_learned:
                    monitor = LearnedMonitor(*learned)
                else:
                    monitor = Monitor(
                        Config(calibration_seconds=seconds),
                        baseline=Baseline() if mode == "demo" else None,
                    )
                if mode == "live":
                    tracker = await asyncio.to_thread(
                        Tracker,
                        model,
                        bool(options.get("clahe")),
                        bool(use_learned and "expression_names" in learned[1]),
                    )
                await ws.send_json(
                    {
                        "ready": True,
                        "mode": mode,
                        "scoring": "behavioral_classifier" if use_learned else "heuristic",
                        "calibration_seconds": 180 if use_learned else seconds,
                    }
                )
                started = time.perf_counter()
                paused = False
                while True:
                    overlay = []
                    if mode == "live":
                        message = await asyncio.wait_for(ws.receive(), timeout=None if paused else 10)
                        if message["type"] == "websocket.disconnect":
                            break
                        if message.get("text") is not None:
                            command = json.loads(message["text"])
                            if not isinstance(command, dict) or command.get("action") not in {"pause", "resume"}:
                                raise ValueError("Expected pause or resume command")
                            if command["action"] == "pause":
                                calibrated = monitor.reference is not None if use_learned else monitor.baseline is not None
                                if not calibrated:
                                    await ws.send_json({"pause_error": "Complete calibration before pausing"})
                                    continue
                                paused = True
                            elif paused:
                                if use_learned:
                                    monitor.resume()
                                else:
                                    monitor = Monitor(config=monitor.config, baseline=monitor.baseline)
                                paused = False
                            await ws.send_json({"paused": paused})
                            continue
                        if paused:
                            continue
                        packet = message.get("bytes")
                        if packet is None:
                            raise ValueError("Expected camera frame")
                        image, timestamp = decode_packet(packet)
                        tick = time.perf_counter()
                        frame, overlay = await asyncio.to_thread(tracker.process, image, timestamp)
                        processing_ms = (time.perf_counter() - tick) * 1000
                    else:
                        # Consume disconnects even when the demo client sends no frames.
                        # This releases the session slot immediately on Stop/tab close.
                        try:
                            message = await asyncio.wait_for(ws.receive(), timeout=1 / 15)
                            if message["type"] == "websocket.disconnect":
                                break
                        except TimeoutError:
                            pass
                        frame = demo_frame(time.perf_counter() - started)
                        processing_ms = 0.0
                    result = (
                        await asyncio.to_thread(monitor.update, frame, tracker.last_expression)
                        if use_learned
                        else monitor.update(frame)
                    )
                    await ws.send_json(
                        dict(
                            result,
                            overlay=overlay,
                            mode=mode,
                            processing_ms=round(processing_ms, 1),
                            metadata=monitor.metadata(),
                        )
                    )
            except WebSocketDisconnect:
                pass
            except (
                ValueError,
                TypeError,
                KeyError,
                RuntimeError,
                FileNotFoundError,
                TimeoutError,
                OSError,
            ) as exc:
                with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                    await ws.send_json(
                        {"error": str(exc) or "Camera stream timed out. Restart camera"}
                    )
                    await ws.close(code=1011)
            finally:
                if tracker is not None:
                    await asyncio.to_thread(tracker.close)

    return app
