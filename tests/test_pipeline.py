import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from enge707.data import build_transforms, image_filename, make_split
from enge707.model import build_resnet18


def test_image_filename_normalizes_numeric_ids():
    assert image_filename("1") == "00001.jpg"
    assert image_filename("00001") == "00001.jpg"


def test_stratified_split_is_deterministic():
    frame = pd.DataFrame({"image_id": [str(i) for i in range(40)], "class": ["a"] * 20 + ["b"] * 20})
    first = make_split(frame, 0.2, 42)
    second = make_split(frame, 0.2, 42)
    assert first[0].equals(second[0]) and first[1].equals(second[1])
    assert first[1]["class"].value_counts().to_dict() == {"a": 4, "b": 4}


def test_transform_and_model_shapes():
    from PIL import Image

    _, transform = build_transforms()
    tensor = transform(Image.new("RGB", (300, 250)))
    model = build_resnet18(8, pretrained=False)
    output = model(tensor.unsqueeze(0))
    assert tensor.shape == (3, 224, 224)
    assert output.shape == (1, 8)

