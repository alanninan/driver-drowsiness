import csv
import json
import struct
from dataclasses import asdict, fields

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from drowsiness.core import Frame
from drowsiness.dataset import read_manifest, write_json
from drowsiness.evaluation import build_windows, evaluate, subject_splits
from drowsiness.server import create_app, decode_packet
from drowsiness.vision import aspect_ratio, rotation_angles


@pytest.fixture
def feature_index(tmp_path):
    videos = []
    for subject in ("01", "02", "03"):
        for label in (0, 5, 10):
            name = f"{subject}-{label}.csv"
            with (tmp_path / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[f.name for f in fields(Frame)])
                writer.writeheader()
                for t in np.arange(0, 110, 0.1):
                    ear = 0.10 if label == 10 and t % 8 < 3 else 0.30
                    writer.writerow(asdict(Frame(float(t), ear, ear, 0.05, brightness=120)))
            videos.append({"subject_id": subject, "label": label, "feature_file": name})
    path = tmp_path / "index.json"
    write_json(path, {"settings": {"synthetic": True}, "videos": videos})
    return path


def test_enrolment_prefix_excluded_and_subjects_disjoint(feature_index):
    windows, metadata = build_windows(feature_index, seconds=10)
    assert metadata["baselines"]["01"]["personalised"]
    for window in windows:
        if window["label"] == 0:
            assert window["start"] >= 71
    for train, test in subject_splits(windows):
        assert not {windows[i]["subject"] for i in train} & {windows[i]["subject"] for i in test}


def test_evaluation_outputs_complete_auditable_folds(feature_index, tmp_path):
    report = evaluate(feature_index, tmp_path / "report", seconds=10, model="rf")
    assert len(report["folds"]) == 3
    assert all(len(f["train_subjects"]) == 2 for f in report["folds"])
    assert set(report["window_metrics"]) == {"fixed_rules", "personal_rules", "rf"}
    assert (tmp_path / "report" / "predictions.json").exists()
    assert len(report["window_metrics"]["rf"]["confusion_matrix"]) == 3


def test_missing_alert_enrolment_fails_instead_of_using_drowsy_video(feature_index):
    index = json.loads(feature_index.read_text())
    index["videos"] = [v for v in index["videos"] if v["label"] != 0]
    write_json(feature_index, index)
    with pytest.raises(ValueError, match="exactly one alert"):
        build_windows(feature_index)


def test_duplicate_feature_source_rejected(feature_index):
    index = json.loads(feature_index.read_text())
    index["videos"].append(index["videos"][0])
    write_json(feature_index, index)
    with pytest.raises(ValueError, match="more than once"):
        build_windows(feature_index)


def test_manifest_resolves_relative_paths_and_keeps_zero_padded_subjects(tmp_path):
    (tmp_path / "clip.mp4").touch()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("subject_id,video_path,label\n01,clip.mp4,5\n")
    row = read_manifest(manifest)[0]
    assert row["subject_id"] == "01"
    assert row["label"] == 5
    assert row["video_path"] == str((tmp_path / "clip.mp4").resolve())
    manifest.write_text("subject_id,video_path,label\n01,clip.mp4,1\n")
    with pytest.raises(ValueError, match="original UTA"):
        read_manifest(manifest)


def test_geometry():
    points = np.array([[0, 0], [1, 1], [3, 1], [4, 0], [3, -1], [1, -1]])
    assert aspect_ratio(points) == pytest.approx(0.5)
    assert aspect_ratio(points * 200) == pytest.approx(0.5)
    assert aspect_ratio(np.zeros((6, 2))) == 0
    assert rotation_angles(np.eye(4)) == pytest.approx((0, 0, 0))


def test_frame_decode_and_limits():
    ok, data = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
    assert ok
    decoded, t = decode_packet(struct.pack("<d", 1.25) + data.tobytes())
    assert decoded.shape == (48, 64, 3) and t == 1.25
    with pytest.raises(ValueError, match="timestamp"):
        decode_packet(struct.pack("<d", float("nan")) + data.tobytes())
    with pytest.raises(ValueError, match="size"):
        decode_packet(b"bad")


def test_dashboard_assets_and_cross_origin_rejection():
    with TestClient(create_app("missing.task")) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Start camera" in response.text
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/health").json()["model_ready"] is False
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/demo", headers={"origin": "https://example.com"}):
                pass


def test_websocket_demo_uses_real_engine_without_camera():
    with TestClient(create_app("missing.task")) as client:
        with client.websocket_connect("/ws/demo", headers={"origin": "http://testserver"}) as ws:
            ws.send_json({"calibration_seconds": 45})
            assert ws.receive_json()["ready"]
            data = ws.receive_json()
            assert data["mode"] == "demo" and data["state"] in {"unknown", "warming"}
            assert data["metadata"]["baseline"]["personalised"] is False


def test_live_session_reports_missing_model():
    with TestClient(create_app("missing.task")) as client:
        with client.websocket_connect("/ws/live", headers={"origin": "http://testserver"}) as ws:
            ws.send_json({"calibration_seconds": 45})
            assert "download-model" in ws.receive_json()["error"]


def test_trained_classifier_health_and_demo_with_three_minute_setting(tmp_path):
    import hashlib

    import joblib
    from sklearn.dummy import DummyClassifier

    path = tmp_path / "behavior.joblib"
    model = DummyClassifier(strategy="most_frequent").fit([[0], [1], [2]], [0, 1, 2])
    joblib.dump(model, path)
    write_json(
        path.with_suffix(".json"),
        dict(
            schema=1,
            variant="enrolled",
            estimator="test",
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ),
    )
    with TestClient(create_app("missing.task", classifier=path)) as client:
        health = client.get("/api/health").json()
        assert health["scoring"] == "behavioral_classifier" and health["calibration_seconds"] == 180
        with client.websocket_connect("/ws/demo", headers={"origin": "http://testserver"}) as ws:
            ws.send_json({"calibration_seconds": 180})
            ready = ws.receive_json()
            assert ready["scoring"] == "heuristic"
            assert ws.receive_json()["mode"] == "demo"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        create_app("missing.task", classifier=path)
