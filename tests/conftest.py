import random
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def make_image():
    """Create a small image file. Different seeds give visually different noise images."""

    def _make(path: Path, seed: int = 0, size=(64, 64), color=None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if color is not None:
            image = Image.new("RGB", size, color)
        else:
            rng = random.Random(seed)
            data = bytes(rng.randrange(256) for _ in range(size[0] * size[1] * 3))
            image = Image.frombytes("RGB", size, data)
        image.save(path)
        return path

    return _make


@pytest.fixture
def make_dataset(make_image):
    """Build a YOLO dataset folder: items are (split, stem, seed, label_text_or_None)."""

    def _build(root: Path, items) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for split, stem, seed, label_text in items:
            image_path = make_image(root / "images" / split / f"{stem}.jpg", seed=seed)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
            if label_text is not None:
                (root / "labels" / split / f"{stem}.txt").write_text(label_text)
            paths[f"{split}/{stem}"] = image_path
        return paths

    return _build
