from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from enge707.data import FacialFeatureDataset, build_transforms  # noqa: E402
from enge707.model import build_resnet18  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Create and validate a Kaggle submission")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "baseline" / "best_model.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "baseline" / "submission.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    model = build_resnet18(len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    sample = pd.read_csv(args.data_dir / "sample_submission.csv", dtype={"image_id": str, "class": str})
    _, transform = build_transforms()
    dataset = FacialFeatureDataset(sample[["image_id"]], args.data_dir / "test" / "test", transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    predicted_classes = []
    with torch.inference_mode():
        for images, _ in loader:
            indices = model(images.to(device, non_blocking=True)).argmax(dim=1).cpu().tolist()
            predicted_classes.extend(classes[index] for index in indices)
    submission = sample.copy()
    submission["class"] = predicted_classes
    if submission.columns.tolist() != sample.columns.tolist():
        raise ValueError("Submission column order changed")
    if len(submission) != len(sample) or not submission["image_id"].equals(sample["image_id"]):
        raise ValueError("Submission image IDs or row order do not match sample_submission.csv")
    invalid = set(submission["class"]) - set(classes)
    if invalid:
        raise ValueError(f"Submission contains invalid classes: {sorted(invalid)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Validated submission written to {args.output} ({len(submission)} rows)")


if __name__ == "__main__":
    main()

