"""Verify and extract one UTA fold without touching other fold manifests."""

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import cv2

from drowsiness.dataset import VIDEO_EXTENSIONS, write_json


def member_label(path):
    """Keep the dataset's split drowsy recordings as separate temporal streams."""
    path = Path(path)
    stems = {"0": 0, "5": 5, "10": 10}
    if path.parent.name in {"32", "49"}:
        stems.update({"10_1": 10, "10_2": 10})
    if path.suffix.lower() not in VIDEO_EXTENSIONS | {".m4v"} or path.stem not in stems:
        raise ValueError(f"Unexpected archive member: {path}")
    return stems[path.stem]


def validate_members(paths, expected_subjects):
    if {p.parent.name for p in paths} != expected_subjects:
        raise ValueError("Unexpected fold subjects")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate archive paths")
    for subject in expected_subjects:
        expected = {"0", "5", "10_1", "10_2"} if subject in {"32", "49"} else {"0", "5", "10"}
        actual = [p.stem for p in paths if p.parent.name == subject]
        if Counter(actual) != Counter(expected):
            raise ValueError(f"Missing or duplicate recordings: {subject}")
    for path in paths:
        member_label(path)


def prepare(fold):
    root = Path(__file__).resolve().parents[1]
    dest = (root / "data/UTA-RLDD").resolve()
    output = root / f"outputs/fold{fold}-preparation"
    rows, inventory = [], []
    archives = [root / f"data/raw/Fold{fold}_part{part}.zip" for part in (1, 2)]
    expected_subjects = {f"{n:02}" for n in range(12 * (fold - 1) + 1, 12 * fold + 1)}
    members = []
    for archive in archives:
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                target = (dest / entry.filename).resolve()
                if not target.is_relative_to(dest):
                    raise ValueError(f"Unsafe archive path: {entry.filename}")
                if entry.is_dir():
                    continue
                member_label(target)
                members.append((archive, entry.filename, target, entry.file_size, entry.CRC))
    validate_members([m[2] for m in members], expected_subjects)
    if (
        shutil.disk_usage(root).free
        < sum(m[3] for m in members if not m[2].exists()) + 5_000_000_000
    ):
        raise ValueError("Insufficient disk space for extraction plus 5 GB reserve")
    for archive, member, target, size, crc in members:
        expected = dict(archive=archive.name, member=member, bytes=size, crc32=crc)
        receipt = output / "receipts" / (hashlib.sha256(member.encode()).hexdigest() + ".json")
        if target.exists():
            if not receipt.exists() or target.stat().st_size != size:
                raise ValueError(f"Unverified existing file: {target}")
            saved = json.loads(receipt.read_text())
            if any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Receipt mismatch: {target}")
            digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
            if digest != saved["sha256"]:
                raise ValueError(f"Extracted file checksum mismatch: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            digest = hashlib.sha256()
            with (
                zipfile.ZipFile(archive) as source,
                source.open(member) as stream,
                temporary.open("xb") as sink,
            ):
                while chunk := stream.read(8 * 1024 * 1024):
                    sink.write(chunk)
                    digest.update(chunk)
            if temporary.stat().st_size != size:
                raise ValueError(f"Incomplete extraction: {temporary}")
            temporary.replace(target)
            saved = dict(expected, sha256=digest.hexdigest())
            write_json(receipt, saved)
        inventory.append(saved)
        capture = cv2.VideoCapture(str(target))
        try:
            ok, _ = capture.read()
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not ok or fps <= 0:
                raise ValueError(f"Undecodable video: {target}")
            rows.append(
                dict(subject_id=target.parent.name, video_path=str(target), label=member_label(target))
            )
            inventory[-1] = dict(
                saved,
                fps=fps,
                frames=capture.get(cv2.CAP_PROP_FRAME_COUNT),
                duration=capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps,
            )
        finally:
            capture.release()
        write_json(output / "inventory.json", inventory)
        print(f"Verified and extracted {len(rows)}/{len(members)}: {member}", flush=True)
    for subject in expected_subjects:
        expected = {0: 1, 5: 1, 10: 2 if subject in {"32", "49"} else 1}
        if Counter(r["label"] for r in rows if r["subject_id"] == subject) != Counter(expected):
            raise ValueError(f"Missing or duplicate labels: {subject}")
    manifest = root / f"data/fold{fold}-manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "video_path", "label"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["subject_id"], r["label"])))
    print(f"Ready: {manifest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    prepare(parser.parse_args().fold)
