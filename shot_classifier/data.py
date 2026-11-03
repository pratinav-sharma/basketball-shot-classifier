"""Read short, already-trimmed shot clips and prevent split leakage."""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights

CLASSES = ("layup", "jump_shot")
FRAMES = 16
WEIGHTS = R3D_18_Weights.KINETICS400_V1


def read_manifest(path):
    path = Path(path)
    rows, seen_paths, groups = [], set(), {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label", "split", "group"} <= set(reader.fieldnames or []):
            raise ValueError("CSV needs path,label,split,group columns")
        for line, row in enumerate(reader, 2):
            row = {key: (row.get(key) or "").strip() for key in ("path", "label", "split", "group")}
            if not all(row.values()):
                raise ValueError(f"Line {line}: all four fields must be filled")
            if row["label"] not in CLASSES or row["split"] not in ("train", "val", "test"):
                raise ValueError(f"Line {line}: unknown label or split")
            clip = (path.parent / row["path"]).resolve()
            if not clip.is_file():
                raise ValueError(f"Line {line}: clip does not exist: {clip}")
            if clip in seen_paths:
                raise ValueError(f"Duplicate clip: {clip}")
            seen_paths.add(clip)
            group, split = row["group"], row["split"]
            if group in groups and groups[group] != split:
                raise ValueError(f"Group {group!r} leaks across splits")
            groups[group] = split
            rows.append({**row, "path": clip})
    for split in ("train", "val", "test"):
        if {r["label"] for r in rows if r["split"] == split} != set(CLASSES):
            raise ValueError(f"{split} must contain both classes")
    return rows


def load_clip(path):
    """Uniformly sample 16 frames; short clips repeat frames deterministically.

    Clips must be trimmed to one action (roughly 1–3 seconds). This simple
    baseline samples across the whole clip, not at a fixed frame rate.
    """
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # Bound memory before stacking; matches the weights' resize size.
            frame = cv2.resize(frame, (171, 128))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) > 900:
                raise ValueError(f"Clip too long: trim {path} to one shot (1–3 seconds)")
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"Video has no decodable frames: {path}")
    indices = np.linspace(0, len(frames) - 1, FRAMES).astype(int)
    video = torch.from_numpy(np.stack([frames[i] for i in indices])).permute(0, 3, 1, 2)
    # uint8 T,C,H,W -> normalized float32 C,T,H,W
    return WEIGHTS.transforms()(video)
