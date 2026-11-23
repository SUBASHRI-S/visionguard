"""Perceptual hashing to find near-duplicate images.

dHash compares neighbouring brightness values of a tiny 9x8 grayscale copy of the
image. Re-saved, re-compressed or slightly resized copies of the same picture get
(almost) the same hash. It does NOT detect two different pictures of the same
scene, or two generated images from the same prompt.
"""
from collections.abc import Hashable, Mapping
from pathlib import Path

from PIL import Image


def dhash(path: Path) -> int:
    """64-bit difference hash. Raises OSError if the image cannot be read."""
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    data = gray.tobytes()
    value = 0
    for row in range(8):
        for col in range(8):
            left = data[row * 9 + col]
            right = data[row * 9 + col + 1]
            value = (value << 1) | int(left > right)
    return value


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return (a ^ b).bit_count()


def cluster_by_hash(hashes: Mapping[Hashable, int], threshold: int) -> list[list]:
    """Group keys whose hashes are within ``threshold`` bits (transitively).

    Pairwise comparison is O(n^2), which is fine for a few thousand images.
    """
    keys = list(hashes)
    parent = {key: key for key in keys}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if hamming(hashes[a], hashes[b]) <= threshold:
                root_a, root_b = find(a), find(b)
                if root_a != root_b:
                    parent[root_b] = root_a

    groups: dict = {}
    for key in keys:
        groups.setdefault(find(key), []).append(key)
    return list(groups.values())
