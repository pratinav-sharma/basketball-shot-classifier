"""Run with python -m shot_classifier --help."""
import argparse
import json
import platform
import shutil
from pathlib import Path

import torch
import torchvision
from torch import nn

from .data import CLASSES, FRAMES, WEIGHTS, load_clip, read_manifest
from .model import choose_device, make_encoder, make_head, metrics


def extract(encoder, rows, device):
    features, targets = [], []
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            clip = load_clip(row["path"]).unsqueeze(0).to(device)
            features.append(encoder(clip).cpu())
            targets.append(CLASSES.index(row["label"]))
            print(f"Extracted {index}/{len(rows)}: {row['path'].name}", flush=True)
    return torch.cat(features).to(device), torch.tensor(targets, device=device)


def train(args):
    rows = read_manifest(args.manifest)
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    encoder, head = make_encoder().to(device), make_head().to(device)
    # Cache frozen features once in memory: CPU-friendly and no video decoding per epoch.
    train_x, train_y = extract(encoder, [r for r in rows if r["split"] == "train"], device)
    val_x, val_y = extract(encoder, [r for r in rows if r["split"] == "val"], device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_loss, history = float("inf"), []
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    for epoch in range(1, args.epochs + 1):
        head.train()
        optimizer.zero_grad()
        loss = criterion(head(train_x), train_y)
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.no_grad():
            val_logits = head(val_x)
            val_loss = float(criterion(val_logits, val_y))
            result = {"epoch": epoch, "train_loss": float(loss.detach()), "val_loss": val_loss,
                      **metrics(val_logits, val_y)}
        history.append(result)
        print(json.dumps(result), flush=True)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                        "classes": list(CLASSES), "weights": WEIGHTS.name, "frames": FRAMES,
                        "seed": args.seed, "epoch": epoch}, output / "best.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps({"manifest": str(Path(args.manifest).resolve()),
        "epochs": args.epochs, "lr": args.lr, "seed": args.seed, "device": str(device),
        "torch": torch.__version__, "torchvision": torchvision.__version__}, indent=2) + "\n")
    print(f"Saved best validation checkpoint to {output / 'best.pt'}")


def restore(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["classes"] != list(CLASSES) or checkpoint["weights"] != WEIGHTS.name
            or checkpoint["frames"] != FRAMES):
        raise ValueError("Checkpoint classes or preprocessing do not match this starter")
    head = make_head()
    head.load_state_dict(checkpoint["head"])
    return make_encoder().to(device), head.to(device).eval()


def inspect_video(clip=None, pretrained=False):
    """Trace one video through the CNN and an untrained classification head."""
    torch.manual_seed(42)
    video = load_clip(clip).unsqueeze(0) if clip else torch.randn(1, 3, FRAMES, 112, 112)
    encoder, head = make_encoder(pretrained=pretrained), make_head().eval()
    with torch.inference_mode():
        features = encoder(video)
        logits = head(features)
        scores = logits.softmax(dim=1)[0]
        predicted_index = int(logits.argmax(dim=1)[0])
    return {
        "input": {
            "source": str(clip) if clip else "synthetic random numbers",
            "shape": list(video.shape),
            "dimension_order": ["batch", "channels", "frames", "height", "width"],
            "dtype": str(video.dtype), "device": str(video.device),
        },
        "encoder": {
            "model": "R3D-18",
            "weights": WEIGHTS.name if pretrained else "random",
            "feature_shape": list(features.shape),
        },
        "classifier": {
            "weights": "random, untrained",
            "logit_shape": list(logits.shape),
            "class_order": list(CLASSES),
            "logits": logits[0].tolist(),
            "softmax_scores": dict(zip(CLASSES, scores.tolist())),
            "prediction": CLASSES[predicted_index],
        },
        "note": "Demonstration only: the head is untrained. Scores are not calibrated confidence or evidence of basketball accuracy.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads (default: 2)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check Python, PyTorch, GPU, and Nsight")
    commands.add_parser("smoke", help="Offline synthetic forward/backward check; no weights download")
    inspection = commands.add_parser("inspect", help="Trace video -> features -> scores -> demo prediction on CPU")
    inspection.add_argument("clip", nargs="?", help="Optional trimmed video; otherwise use synthetic input")
    inspection.add_argument("--pretrained", action="store_true",
                            help="Use pretrained CNN weights (downloads on first use); head stays untrained")
    check = commands.add_parser("check-data", help="Validate manifest and decode every clip")
    check.add_argument("manifest")
    training = commands.add_parser("train", help="Train only the final layer of pretrained R3D-18")
    training.add_argument("manifest")
    training.add_argument("--output", default="runs/baseline")
    training.add_argument("--epochs", type=int, default=30)
    training.add_argument("--lr", type=float, default=0.001)
    training.add_argument("--seed", type=int, default=42)
    evaluation = commands.add_parser("evaluate", help="Evaluate a checkpoint on untouched test clips")
    evaluation.add_argument("manifest")
    evaluation.add_argument("checkpoint")
    prediction = commands.add_parser("predict", help="Classify one trimmed shot clip")
    prediction.add_argument("clip")
    prediction.add_argument("checkpoint")
    for command in (training, evaluation, prediction):
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        if args.command == "doctor":
            print(json.dumps({"python": platform.python_version(), "torch": torch.__version__,
                "torchvision": torchvision.__version__, "cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "nvidia_smi": shutil.which("nvidia-smi"), "nsys": shutil.which("nsys"),
                "ncu": shutil.which("ncu")}, indent=2))
        elif args.command == "smoke":
            torch.manual_seed(42)
            encoder, head = make_encoder(pretrained=False), make_head()
            with torch.no_grad():
                video = torch.randn(1, 3, FRAMES, 112, 112)
                print("Video shape:", video.shape)
                features = encoder(video)
                print("Feature shape:", features.shape)
            before = head.weight.detach().clone()
            optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
            logits = head(features)
            loss = nn.functional.cross_entropy(logits, torch.tensor([0]))
            loss.backward()
            optimizer.step()
            assert logits.shape == (1, 2) and torch.isfinite(loss)
            assert not torch.equal(before, head.weight)
            assert all(p.grad is None for p in encoder.parameters())
            print("PASS: synthetic video -> 512 CNN features -> 2 logits -> head update (CPU).")
            print("Random weights and synthetic input: this does not measure shot accuracy.")
        elif args.command == "check-data":
            rows = read_manifest(args.manifest)
            for row in rows:
                load_clip(row["path"])
            print(f"PASS: {len(rows)} clips decoded; paths, labels, and group splits validated.")
        elif args.command == "inspect":
            print(json.dumps(inspect_video(args.clip, args.pretrained), indent=2))
        elif args.command == "train":
            if args.epochs < 1 or not 0 < args.lr < float("inf"):
                parser.error("--epochs and --lr must be positive and finite")
            if Path(args.output).exists():
                parser.error("Output directory exists; choose a new --output to preserve previous runs")
            train(args)
        else:
            device = choose_device(args.device)
            rows = read_manifest(args.manifest) if args.command == "evaluate" else None
            encoder, head = restore(args.checkpoint, device)
            with torch.no_grad():
                if rows is not None:
                    features, targets = extract(encoder, [r for r in rows if r["split"] == "test"], device)
                    print(json.dumps(metrics(head(features), targets), indent=2))
                else:
                    features = encoder(load_clip(args.clip).unsqueeze(0).to(device))
                    scores = head(features).softmax(1)[0].cpu().tolist()
                    print(json.dumps({"prediction": CLASSES[scores.index(max(scores))],
                        "scores": dict(zip(CLASSES, scores)),
                        "note": "Scores are not calibrated confidence; input must be one of the two shot types."}, indent=2))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
