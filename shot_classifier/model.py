"""A frozen video CNN plus a trainable two-output linear classifier."""
import os
from pathlib import Path

import torch
from torch import nn
from torchvision.models.video import r3d_18

from .data import CLASSES, WEIGHTS


def make_encoder(pretrained=True):
    if "TORCH_HOME" not in os.environ:
        torch.hub.set_dir(str(Path.cwd() / ".cache" / "torch" / "hub"))
    encoder = r3d_18(weights=WEIGHTS if pretrained else None)
    encoder.fc = nn.Identity()  # The CNN now returns 512 learned video features.
    encoder.requires_grad_(False)
    return encoder.eval()  # Also freeze BatchNorm running statistics.


def make_head():
    return nn.Linear(512, len(CLASSES))


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or a verified NVIDIA environment.")
    return torch.device(name)


def metrics(logits, targets):
    predictions = logits.argmax(1)
    matrix = torch.zeros(2, 2, dtype=torch.int64)
    for actual, predicted in zip(targets.cpu(), predictions.cpu()):
        matrix[actual, predicted] += 1
    recalls = matrix.diag().float() / matrix.sum(1).clamp(min=1)
    return {
        "accuracy": float(matrix.diag().sum() / matrix.sum()),
        "balanced_accuracy": float(recalls.mean()),
        "recall": dict(zip(CLASSES, recalls.tolist())),
        "confusion_matrix_actual_rows_predicted_columns": matrix.tolist(),
        "class_order": list(CLASSES),
    }
