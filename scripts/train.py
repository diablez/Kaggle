from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import torch
import torchvision
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from enge707.data import (  # noqa: E402
    FacialFeatureDataset,
    build_transforms,
    load_labels,
    make_split,
    seed_everything,
)
from enge707.engine import run_epoch  # noqa: E402
from enge707.model import build_resnet18  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train an eight-class ResNet18 baseline")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def save_plots(history, confusion, classes, output_dir):
    epochs = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Learning curves: loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="train")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Learning curves: accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_xticks(range(len(classes)), labels=classes, rotation=45, ha="right")
    axis.set_yticks(range(len(classes)), labels=classes)
    axis.set(xlabel="Predicted", ylabel="Actual", title="Validation confusion matrix")
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(column, row, confusion[row, column], ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = load_labels(args.data_dir / "train_labels.csv")
    classes = sorted(labels["class"].unique().tolist())
    if len(classes) != 8:
        raise ValueError(f"Expected 8 classes, found {len(classes)}: {classes}")
    class_to_index = {name: index for index, name in enumerate(classes)}
    train_frame, validation_frame = make_split(labels, args.validation_fraction, args.seed)
    if args.max_train_samples:
        per_class = max(1, args.max_train_samples // len(classes))
        train_frame = pd.concat(
            [group.sample(min(len(group), per_class), random_state=args.seed) for _, group in train_frame.groupby("class")],
            ignore_index=True,
        )
    if args.max_validation_samples:
        per_class = max(1, args.max_validation_samples // len(classes))
        validation_frame = pd.concat(
            [group.sample(min(len(group), per_class), random_state=args.seed) for _, group in validation_frame.groupby("class")],
            ignore_index=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(args.output_dir / "train_split.csv", index=False)
    validation_frame.to_csv(args.output_dir / "validation_split.csv", index=False)
    train_transform, evaluation_transform = build_transforms()
    image_dir = args.data_dir / "train" / "train"
    train_dataset = FacialFeatureDataset(train_frame, image_dir, train_transform, class_to_index)
    validation_dataset = FacialFeatureDataset(validation_frame, image_dir, evaluation_transform, class_to_index)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = build_resnet18(
        len(classes), pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=1)
    configuration = {
        **vars(args),
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "device": str(device),
        "classes": classes,
        "class_to_index": class_to_index,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "sklearn": sklearn.__version__,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "config.json").write_text(json.dumps(configuration, indent=2, default=str), encoding="utf-8")
    history, best_loss, epochs_without_improvement = [], float("inf"), 0
    checkpoint_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, loss_fn, device, optimizer)
        validation_result = run_epoch(model, validation_loader, loss_fn, device)
        scheduler.step(validation_result["loss"])
        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "validation_loss": validation_result["loss"],
            "validation_accuracy": validation_result["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
        print(json.dumps(row), flush=True)
        if validation_result["loss"] < best_loss:
            best_loss = validation_result["loss"]
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "classes": classes, "config": configuration}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"Early stopping after epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_validation = run_epoch(model, validation_loader, loss_fn, device)
    report = classification_report(
        final_validation["targets"], final_validation["predictions"],
        labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0,
    )
    confusion = confusion_matrix(final_validation["targets"], final_validation["predictions"], labels=list(range(len(classes))))
    metrics = {
        "validation_loss": final_validation["loss"],
        "validation_accuracy": final_validation["accuracy"],
        "classification_report": report,
        "confusion_matrix": confusion.tolist(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_plots(history, confusion, classes, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

