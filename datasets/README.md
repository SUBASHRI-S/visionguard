# Dataset card: Ultralytics Construction-PPE

## Source

- Name: Ultralytics Construction-PPE (November 2026 release)
- Documentation: https://docs.ultralytics.com/datasets/detect/construction-ppe
- Download: https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip
- Documentation page read on: 2026-11-23
- **SHA256 of the zip we downloaded:** `E5817AA9C73F33E525D76CA6BC6AF4E4EAC9DCEA9BCFEF1CFD1A71C09FBC1232`

## What the source says about itself

- 1,000 images (800 train, 200 validation), 12 classes, 14,863 bounding boxes
- Combines human-collected and generated images
- Annotated by humans plus AI auto-annotation
- No separate test split. The documentation recommends final evaluation on independent images from the intended deployment environment.
- The documentation notes that an occluded or out-of-frame body part does not by itself establish missing equipment.

Classes (source ids): 0 helmet, 1 gloves, 2 vest, 3 boots, 4 goggles, 5 Person, 6 no_helmet, 7 no_goggle, 8 no_gloves, 9 no_boots, 10 forklift, 11 no_vest.

## Licence and attribution

- Images and annotations: **CC BY 4.0** (sharing and adaptation, including commercial use, with attribution).
- Attribution text: *Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0.* Link to the dataset page and to https://creativecommons.org/licenses/by/4.0/.
- **Changes we make** (CC BY requires saying so): new train/val/test splits, removal of near-duplicate images from train, optional class remapping and dropping of unused classes.
- Software licence is separate: the Ultralytics code is AGPL-3.0. Fine for this portfolio; a commercial product needs a licence review.

## How we use it

| Our name | Built from | Classes |
|---|---|---|
| `test` | official **val** images, never used for tuning | per profile |
| `val` | about 20% of official train, chosen by clusters of near-duplicates | per profile |
| `train` | rest of official train, minus images that duplicate a test image | per profile |

Class profiles (`datasets/construction_ppe.py`):

- `core`: person, helmet, vest, forklift. Violations are inferred from absence (decision D1).
- `core_missing`: core plus no_helmet and no_vest, for the ablation "detect missing PPE directly vs infer it".
- `all`: all 12 classes with the original ids.

## Reproduce

Run from the project root with the virtual environment active.

```
New-Item -ItemType Directory -Force datasets\raw
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip" -OutFile datasets\raw\construction-ppe.zip
Get-FileHash datasets\raw\construction-ppe.zip -Algorithm SHA256
Expand-Archive datasets\raw\construction-ppe.zip -DestinationPath datasets\raw
python -m scripts.run_validation
python -m scripts.analyze_dataset
python -m scripts.make_split --profile core --source-sha256 <hash>
python -m scripts.make_split --profile core_missing --source-sha256 <hash>
```

Each `make_split` run prints a `dataset_version` and saves a small manifest in `datasets/versions/`. Commit the manifest; the images stay out of git. Record the version in every MLflow run.

## Known issues and limitations

- **Small.** About 800 training images. Expect noisy per-class metrics, especially for rare classes (see `docs/dataset_analysis.md`).
- **Partly generated.** Generated and real images cannot be separated, so scores here may overstate real-world performance. Real-footage replay evaluation is the check (`docs/REAL_FOOTAGE_PROTOCOL.md`).
- **Near-duplicate detection is limited.** The hash catches re-saved, re-compressed and resized copies, not different pictures of the same scene or generated variants of one prompt.
- **New dataset.** Public discussion on the documentation page mostly describes an earlier version. We measure class balance ourselves instead of relying on it.
- **Forklifts are the only vehicle class.**
