"""MediaPipe Tasks adapter; pixel geometry avoids aspect-ratio distortions."""

import hashlib
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .core import Frame

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL = Path("models/face_landmarker.task")
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
LEFT = [33, 160, 158, 133, 153, 144]
RIGHT = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 81, 311, 291, 402, 178]


def download_model(path=DEFAULT_MODEL):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".download")
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
                with temporary.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != MODEL_SHA256:
                raise ValueError(
                    "Downloaded model checksum does not match the pinned official model"
                )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("Existing file is not the pinned model. Choose a new --model output path")
    return {
        "path": str(path.resolve()),
        "url": MODEL_URL,
        "sha256": digest,
    }


def aspect_ratio(points):
    width = np.linalg.norm(points[0] - points[3])
    if width < 1e-6:
        return 0.0
    return float(
        (np.linalg.norm(points[1] - points[5]) + np.linalg.norm(points[2] - points[4]))
        / (2 * width)
    )


def rotation_angles(matrix):
    rotation = np.asarray(matrix, dtype=float)[:3, :3]
    # Remove numerical scale before extracting intrinsic XYZ Euler angles.
    u, _, vh = np.linalg.svd(rotation)
    rotation = u @ vh
    pitch = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    roll = math.atan2(rotation[1, 0], rotation[0, 0])
    return tuple(math.degrees(a) for a in (pitch, yaw, roll))


class Tracker:
    def __init__(self, model=DEFAULT_MODEL, clahe=False, expressions=False, max_width=960):
        import mediapipe as mp

        if not Path(model).is_file():
            raise FileNotFoundError("Face model missing. Run: uv run drowsiness download-model")
        if not isinstance(max_width, int) or max_width < 320:
            raise ValueError("max_width must be an integer of at least 320 pixels")
        self.max_width = max_width
        self.mp = mp
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None
        self.last_ms = -1
        self.expression_names = []
        self.last_expression = None
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(
            mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=2,  # Reject multiple people rather than silently switch identity.
                min_face_detection_confidence=0.6,
                min_face_presence_confidence=0.6,
                min_tracking_confidence=0.6,
                output_facial_transformation_matrixes=True,
                output_face_blendshapes=expressions,
            )
        )

    def process(self, bgr, timestamp):
        self.last_expression = None
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image")
        h, w = bgr.shape[:2]
        if w > self.max_width:
            bgr = cv2.resize(bgr, (self.max_width, round(h * self.max_width / w)))
            h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        if self.clahe is not None:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ms = max(self.last_ms + 1, int(timestamp * 1000))
        self.last_ms = ms
        detection = self.detector.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ms
        )
        faces = detection.face_landmarks
        if len(faces) != 1:
            reason = "No face detected" if not faces else "Multiple faces detected. One driver only"
            return Frame(timestamp, brightness=brightness, valid=False, reason=reason), []
        points = np.array([[p.x * w, p.y * h] for p in faces[0]])
        x0, y0 = np.maximum(points.min(axis=0).astype(int), 0)
        x1, y1 = np.minimum(points.max(axis=0).astype(int), [w - 1, h - 1])
        crop = gray[y0 : y1 + 1, x0 : x1 + 1]
        brightness = float(crop.mean()) if crop.size else 0.0
        pitch, yaw, roll = rotation_angles(detection.facial_transformation_matrixes[0])
        left, right = aspect_ratio(points[LEFT]), aspect_ratio(points[RIGHT])
        mar = aspect_ratio(points[MOUTH])
        reason = ""
        if brightness < 25:
            reason = "Face is too dark. Add light"
        elif (
            x1 - x0 < 90
            or min(
                np.linalg.norm(points[33] - points[133]), np.linalg.norm(points[362] - points[263])
            )
            < 14
        ):
            reason = "Move closer to the camera"
        elif abs(yaw) > 35 or abs(pitch) > 40 or abs(roll) > 35:
            reason = "Face the camera for reliable eye measurements"
        elif np.any(points[LEFT + RIGHT + MOUTH] < 0) or np.any(
            points[LEFT + RIGHT + MOUTH] > [w, h]
        ):
            reason = "Keep your full face in view"
        elif not (0 <= left <= 0.8 and 0 <= right <= 0.8 and 0 <= mar <= 2):
            reason = "Landmark geometry is unreliable"
        frame = Frame(timestamp, left, right, mar, pitch, yaw, roll, brightness, not reason, reason)
        if frame.valid and detection.face_blendshapes:
            categories = sorted(detection.face_blendshapes[0], key=lambda item: item.category_name)
            self.expression_names = [item.category_name for item in categories]
            self.last_expression = [item.score for item in categories]
        overlay = (points[LEFT + RIGHT + MOUTH] / [w, h]).round(4).tolist()
        return frame, overlay

    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
