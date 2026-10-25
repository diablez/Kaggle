from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_filename(image_id: str) -> str:
    value = str(image_id).strip()
    stem = value[:-4] if value.lower().endswith(".jpg") else value
    return f"{int(stem):05d}.jpg" if stem.isdigit() else f"{stem}.jpg"


def load_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"image_id": str, "class": str})
    required = {"image_id", "class"}
    if set(frame.columns) != required:
        raise ValueError(f"Expected columns {sorted(required)}, got {frame.columns.tolist()}")
    if frame.isna().any().any():
        raise ValueError("Labels contain missing values")
    return frame


def make_split(frame: pd.DataFrame, validation_fraction: float, seed: int, duplicate_groups=None):
    if duplicate_groups:
        folds = round(1 / validation_fraction)
        if not np.isclose(validation_fraction, 1 / folds):
            raise ValueError("Duplicate-aware splitting requires validation_fraction to be a reciprocal integer")
        group_by_file = {}
        for group_number, filenames in enumerate(duplicate_groups):
            for filename in filenames:
                group_by_file[filename] = f"duplicate-{group_number}"
        groups = [
            group_by_file.get(image_filename(image_id), f"unique-{image_filename(image_id)}")
            for image_id in frame["image_id"]
        ]
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        train_indices, validation_indices = next(splitter.split(frame, frame["class"], groups))
        return (
            frame.iloc[train_indices].reset_index(drop=True),
            frame.iloc[validation_indices].reset_index(drop=True),
        )
    train, validation = train_test_split(
        frame,
        test_size=validation_fraction,
        random_state=seed,
        stratify=frame["class"],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def build_transforms(image_size: int = IMAGE_SIZE):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


class FacialFeatureDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: Path, transform, class_to_index=None):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.class_to_index = class_to_index

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = self.image_dir / image_filename(row["image_id"])
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        if self.class_to_index is None:
            return tensor, row["image_id"]
        return tensor, self.class_to_index[row["class"]]

