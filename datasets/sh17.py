"""Add SH17 images to the training split of a prepared ``core`` dataset.

SH17 (Ahmad and Rahimi, 2024) has 8,099 photos with 17 classes. We take only the classes the
``core`` profile uses (person, helmet, vest) and only photos that contain a helmet or a vest,
because those are the cases our own training set lacks (vests facing the camera, caps, side
and back views).

Rules this module enforces:
- SH17 images go into TRAIN only. Validation and test stay exactly as in the base dataset,
  so results stay comparable with every earlier run.
- SH17 images that look like near-duplicates of a base val/test image are dropped.
- Selection is seeded and sorted, so the same inputs always give the same dataset.

Licence: SH17 is CC BY-NC-SA 4.0 (non-commercial, share-alike, attribution). Nothing built here
may be committed to the repository or redistributed. See docs/DATASET_SH17.md.
"""
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from datasets.construction_ppe import profile_mapping
from datasets.hashing import dhash, hamming
from datasets.materialize import _data_yaml, remap_label_text
from datasets.validate import IMAGE_EXTENSIONS

# Class order of the Kaggle copy, checked on 2026-11-28 by pairing the YOLO label ids with the
# class names in the VOC xml files (the order differs from the list on the GitHub page).
SH17_CLASS_NAMES: list[str] = [
    "person",  # 0
    "ear",  # 1
    "ear-mufs",  # 2
    "face",  # 3
    "face-guard",  # 4
    "face-mask-medical",  # 5
    "foot",  # 6
    "tools",  # 7
    "glasses",  # 8
    "gloves",  # 9
    "helmet",  # 10
    "hands",  # 11
    "head",  # 12
    "medical-suit",  # 13
    "shoes",  # 14
    "safety-suit",  # 15
    "safety-vest",  # 16
]

# SH17 name -> core class name. Everything else is dropped.
SH17_TO_CORE: dict[str, str] = {"person": "person", "helmet": "helmet", "safety-vest": "vest"}
# Photos are chosen because they contain one of these.
TARGET_CLASSES = ("helmet", "safety-vest")

LICENCE_NOTE = (
    "SH17 images are CC BY-NC-SA 4.0 (non-commercial, share-alike). Do not commit or "
    "redistribute them or this dataset. Weights trained on it are for research/portfolio use."
)


def sh17_id_map() -> dict[int, int]:
    """SH17 class id -> core class id (person 0, helmet 1, vest 2, forklift 3)."""
    _, core_names = profile_mapping("core")
    return {
        SH17_CLASS_NAMES.index(src): core_names.index(dst) for src, dst in SH17_TO_CORE.items()
    }


def has_target(label_text: str) -> bool:
    """True if the label file contains at least one helmet or vest box."""
    wanted = {str(SH17_CLASS_NAMES.index(name)) for name in TARGET_CLASSES}
    return any(line.split()[0] in wanted for line in label_text.splitlines() if line.strip())


def find_candidates(sh17_root: Path) -> list[Path]:
    """Sorted image paths in ``sh17_root/images`` whose label has a helmet or vest."""
    candidates = []
    for image in sorted((sh17_root / "images").iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = sh17_root / "labels" / f"{image.stem}.txt"
        if label.exists() and has_target(label.read_text()):
            candidates.append(image)
    return candidates


def _shrink(source: Path, target: Path, max_side: int) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        scale = max_side / max(image.size)
        if scale < 1:
            size = (round(image.width * scale), round(image.height * scale))
            image = image.resize(size, Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, quality=90)


@dataclass(frozen=True)
class ExtendResult:
    manifest: dict
    selected: list[str]
    pool_size: int
    dropped_duplicates: list[str]


def extend_dataset(
    base_dir: Path,
    sh17_root: Path,
    out_dir: Path,
    *,
    max_images: int = 1500,
    max_side: int = 1280,
    seed: int = 42,
    hash_threshold: int = 4,
) -> ExtendResult:
    """Copy ``base_dir`` to ``out_dir`` and add selected SH17 photos to train."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; choose a new output folder")
    base_manifest = json.loads((base_dir / "manifest.json").read_text())
    if base_manifest["profile"] != "core":
        raise ValueError("the base dataset must use the 'core' profile")
    names: list[str] = base_manifest["class_names"]
    id_map = sh17_id_map()

    pool = find_candidates(sh17_root)
    rng = random.Random(seed)
    order = pool[:]
    rng.shuffle(order)

    # Hash base val/test images once; drop SH17 photos that look like any of them.
    guarded = [
        dhash(p)
        for split in ("val", "test")
        for p in sorted((base_dir / "images" / split).iterdir())
    ]
    selected: list[Path] = []
    dropped: list[Path] = []
    for image in order:
        if len(selected) >= max_images:
            break
        value = dhash(image)
        if any(hamming(value, other) <= hash_threshold for other in guarded):
            dropped.append(image)
            continue
        selected.append(image)
    selected.sort()

    shutil.copytree(base_dir / "images", out_dir / "images")
    shutil.copytree(base_dir / "labels", out_dir / "labels")

    sh17_instances = dict.fromkeys(names, 0)
    kept: list[Path] = []
    for image in selected:
        text = remap_label_text((sh17_root / "labels" / f"{image.stem}.txt").read_text(), id_map)
        if not text:
            continue
        stem = f"sh17_{image.stem}"
        _shrink(image, out_dir / "images" / "train" / f"{stem}.jpg", max_side)
        (out_dir / "labels" / "train" / f"{stem}.txt").write_text(text)
        for line in text.splitlines():
            sh17_instances[names[int(line.split()[0])]] += 1
        kept.append(image)

    counts = json.loads(json.dumps(base_manifest["counts"]))
    counts["train"]["images"] += len(kept)
    for name, number in sh17_instances.items():
        counts["train"]["instances"][name] += number

    manifest = {
        "dataset": "construction-ppe + sh17",
        "profile": "core_sh17",
        "class_names": names,
        "base_dataset_version": base_manifest["dataset_version"],
        "sh17_licence": "CC BY-NC-SA 4.0",
        "sh17_selection": {
            "rule": "photos with at least one helmet or safety-vest box",
            "seed": seed,
            "max_images": max_images,
            "max_side": max_side,
            "hash_threshold": hash_threshold,
            "pool_size": len(pool),
            "dropped_near_duplicates": len(dropped),
            "added_images": len(kept),
            "added_instances": sh17_instances,
        },
        "counts": counts,
        "sh17_files": [p.name for p in kept],
    }
    canonical = json.dumps(manifest, sort_keys=True).encode()
    manifest["dataset_version"] = hashlib.sha256(canonical).hexdigest()[:12]

    (out_dir / "data.yaml").write_text(_data_yaml(out_dir, names))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return ExtendResult(
        manifest=manifest,
        selected=[p.name for p in kept],
        pool_size=len(pool),
        dropped_duplicates=[p.name for p in dropped],
    )
