"""Leakage-aware train/val/test split.

Policy (decision D13):
- The dataset ships with train and val only. The official val images become our
  TEST set and are never used for tuning.
- A new validation set is carved out of the official train images.
- Near-duplicate images are grouped so a picture and its copy can never land in
  different splits.
- Train images that are near-duplicates of a test image are removed from train
  (default), because they would inflate the test score.
"""
import random
from dataclasses import dataclass, field
from pathlib import Path

from datasets.hashing import cluster_by_hash, dhash, hamming


@dataclass(frozen=True)
class SplitResult:
    train: list[Path]
    val: list[Path]
    test: list[Path]
    leaking_train: list[Path]  # train images that near-duplicate a test image
    excluded_train: list[Path]  # train images actually removed (leaking + cluster mates)
    unreadable: list[Path] = field(default_factory=list)
    seed: int = 42
    val_fraction: float = 0.2
    hash_threshold: int = 4


def _hash_all(paths: list[Path], unreadable: list[Path]) -> dict[Path, int]:
    hashes: dict[Path, int] = {}
    for path in paths:
        try:
            hashes[path] = dhash(path)
        except (OSError, ValueError):
            unreadable.append(path)
    return hashes


def make_split(
    train_images: list[Path],
    test_images: list[Path],
    *,
    val_fraction: float = 0.2,
    seed: int = 42,
    hash_threshold: int = 4,
    drop_leaking: bool = True,
) -> SplitResult:
    """Split official train images into train/val; official val becomes test."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1 (exclusive)")
    if hash_threshold < 0:
        raise ValueError("hash_threshold must be >= 0")

    unreadable: list[Path] = []
    train_hashes = _hash_all(sorted(train_images), unreadable)
    test_hashes = _hash_all(sorted(test_images), unreadable)

    test_values = list(test_hashes.values())
    leaking = sorted(
        path
        for path, value in train_hashes.items()
        if any(hamming(value, other) <= hash_threshold for other in test_values)
    )
    leaking_set = set(leaking)

    clusters = cluster_by_hash(train_hashes, hash_threshold)
    excluded: list[Path] = []
    if drop_leaking:
        kept = []
        for cluster in clusters:
            if leaking_set.intersection(cluster):
                excluded.extend(cluster)
            else:
                kept.append(cluster)
        clusters = kept

    ordered = sorted((sorted(c) for c in clusters), key=lambda c: str(c[0]))
    random.Random(seed).shuffle(ordered)

    remaining = sum(len(c) for c in ordered)
    target_val = round(val_fraction * remaining)
    train: list[Path] = []
    val: list[Path] = []
    for cluster in ordered:
        if len(val) < target_val:
            val.extend(cluster)
        else:
            train.extend(cluster)

    return SplitResult(
        train=sorted(train),
        val=sorted(val),
        test=sorted(test_hashes),
        leaking_train=leaking,
        excluded_train=sorted(excluded),
        unreadable=sorted(unreadable),
        seed=seed,
        val_fraction=val_fraction,
        hash_threshold=hash_threshold,
    )
