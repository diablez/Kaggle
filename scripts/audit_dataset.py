from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from enge707.data import image_filename, load_labels  # noqa: E402


def inspect_one(path, hash_content):
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image_format = image.format or "UNKNOWN"
            dimensions = f"{image.width}x{image.height}"
        digest = hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest() if hash_content else None
        return path.name, image_format, dimensions, digest, None
    except (OSError, UnidentifiedImageError) as error:
        return path.name, None, None, None, {"file": str(path), "error": str(error)}


def inspect_images(paths, workers, hash_content):
    formats, dimensions, corrupt = Counter(), Counter(), []
    digests = defaultdict(list)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for name, image_format, size, digest, error in executor.map(
            lambda path: inspect_one(path, hash_content), paths
        ):
            if error:
                corrupt.append(error)
                continue
            formats[image_format] += 1
            dimensions[size] += 1
            if digest:
                digests[digest].append(name)
    duplicate_groups = [names for names in digests.values() if len(names) > 1]
    return formats, dimensions, corrupt, duplicate_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "audit")
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    labels = load_labels(args.data_dir / "train_labels.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv", dtype={"image_id": str, "class": str})
    train_dir, test_dir = args.data_dir / "train" / "train", args.data_dir / "test" / "test"
    train_paths, test_paths = sorted(train_dir.glob("*")), sorted(test_dir.glob("*"))
    expected_train = {image_filename(value) for value in labels["image_id"]}
    expected_test = {image_filename(value) for value in sample["image_id"]}
    actual_train, actual_test = {p.name for p in train_paths}, {p.name for p in test_paths}

    train_formats, train_dimensions, train_corrupt, train_duplicates = inspect_images(
        train_paths, args.workers, not args.skip_hashes
    )
    test_formats, test_dimensions, test_corrupt, test_duplicates = inspect_images(
        test_paths, args.workers, not args.skip_hashes
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(labels["class"].unique().tolist())
    report = {
        "labels_file": "train_labels.csv",
        "train_rows": len(labels),
        "test_rows": len(sample),
        "classes": classes,
        "class_distribution": labels["class"].value_counts().sort_index().to_dict(),
        "train_files": len(train_paths),
        "test_files": len(test_paths),
        "missing_train_files": sorted(expected_train - actual_train),
        "unexpected_train_files": sorted(actual_train - expected_train),
        "missing_test_files": sorted(expected_test - actual_test),
        "unexpected_test_files": sorted(actual_test - expected_test),
        "duplicate_train_filenames": int(pd.Series([p.name for p in train_paths]).duplicated().sum()),
        "duplicate_test_filenames": int(pd.Series([p.name for p in test_paths]).duplicated().sum()),
        "overlapping_filenames_across_splits": len(actual_train & actual_test),
        "train_formats": dict(train_formats),
        "test_formats": dict(test_formats),
        "train_dimensions": dict(train_dimensions),
        "test_dimensions": dict(test_dimensions),
        "corrupt_train": train_corrupt,
        "corrupt_test": test_corrupt,
        "duplicate_train_content_groups": train_duplicates,
        "duplicate_test_content_groups": test_duplicates,
    }
    (args.output_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for axis, class_name in zip(axes.flat, classes):
        image_id = labels.loc[labels["class"] == class_name, "image_id"].iloc[0]
        with Image.open(train_dir / image_filename(image_id)) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(class_name)
        axis.axis("off")
    fig.suptitle("Representative training image from each class")
    fig.tight_layout()
    fig.savefig(args.output_dir / "representative_classes.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

