"""Write a split to disk as a ready-to-train YOLO dataset, with a version manifest."""
import hashlib
import json
import os
import shutil
from pathlib import Path

from datasets.construction_ppe import SOURCE_CLASS_NAMES, profile_mapping
from datasets.split import SplitResult


def remap_label_text(text: str, id_map: dict[int, int]) -> str:
    """Rewrite class ids using ``id_map`` and drop lines whose class is not mapped."""
    out_lines: list[str] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        try:
            source_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"line {line_no}: class id is not an integer") from exc
        if source_id in id_map:
            out_lines.append(" ".join([str(id_map[source_id]), *parts[1:]]))
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def _place(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "link":
        try:
            os.link(source, target)
            return
        except OSError:
            pass  # different drive or unsupported: fall back to copying
    shutil.copy2(source, target)


def _data_yaml(out_dir: Path, names: list[str]) -> str:
    lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines += [f"  {index}: {name}" for index, name in enumerate(names)]
    return "\n".join(lines) + "\n"


def materialize(
    root: Path,
    split: SplitResult,
    out_dir: Path,
    profile: str,
    *,
    source_sha256: str | None = None,
    mode: str = "copy",
    source_class_names: list[str] | None = None,
) -> dict:
    """Create ``out_dir`` with images/, labels/, data.yaml and manifest.json.

    ``root`` is the extracted source dataset (images/<split>, labels/<split>).
    Returns the manifest dict. Refuses to write into a non-empty directory.
    """
    if mode not in {"copy", "link"}:
        raise ValueError("mode must be 'copy' or 'link'")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; choose a new output folder")

    id_map, names = profile_mapping(profile, source_class_names or SOURCE_CLASS_NAMES)
    image_root = root / "images"
    label_root = root / "labels"

    groups = {"train": split.train, "val": split.val, "test": split.test}
    files: dict[str, list[str]] = {}
    counts: dict[str, dict] = {}

    for split_name, images in groups.items():
        files[split_name] = []
        instances = dict.fromkeys(names, 0)
        without_objects = 0
        for image_path in images:
            relative = image_path.relative_to(image_root)
            files[split_name].append(relative.as_posix())

            _place(image_path, out_dir / "images" / split_name / image_path.name, mode)

            label_source = label_root / relative.parent / f"{image_path.stem}.txt"
            text = label_source.read_text() if label_source.exists() else ""
            remapped = remap_label_text(text, id_map)
            label_target = out_dir / "labels" / split_name / f"{image_path.stem}.txt"
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_text(remapped)

            kept_lines = [ln for ln in remapped.splitlines() if ln.strip()]
            if not kept_lines:
                without_objects += 1
            for line in kept_lines:
                instances[names[int(line.split()[0])]] += 1
        counts[split_name] = {
            "images": len(images),
            "images_without_objects": without_objects,
            "instances": instances,
        }

    manifest = {
        "dataset": "ultralytics-construction-ppe",
        "source_release": "November 2026",
        "source_licence": "CC BY 4.0",
        "source_zip_sha256": source_sha256,
        "profile": profile,
        "class_names": names,
        "seed": split.seed,
        "val_fraction": split.val_fraction,
        "hash_threshold": split.hash_threshold,
        "counts": counts,
        "leaking_train_found": len(split.leaking_train),
        "train_images_excluded": len(split.excluded_train),
        "unreadable_images": [p.name for p in split.unreadable],
        "files": files,
    }
    canonical = json.dumps(manifest, sort_keys=True).encode()
    manifest["dataset_version"] = hashlib.sha256(canonical).hexdigest()[:12]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.yaml").write_text(_data_yaml(out_dir, names))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
