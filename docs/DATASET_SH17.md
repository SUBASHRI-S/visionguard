# Dataset card: SH17 (extra training photos)

## Why we add it
Real-footage tests showed two weaknesses of the model trained on Construction-PPE alone: an orange hi-vis vest worn facing the camera was read as "no vest", and a cap was read as a helmet. The base training set has only 638 images, so it has few examples of these cases. SH17 has 8,099 varied photos of people at work.

## Source and licence
- SH17: Ahmad, H. M. and Rahimi, A., "SH17: A dataset for human safety and personal protective equipment detection in manufacturing industry", Journal of Safety Science and Resilience, 2024. DOI 10.1016/j.jnlssr.2024.09.002. Repository: https://github.com/ahmadmughees/SH17dataset. Kaggle copy: `mugheesahmad/sh17-dataset-for-ppe-detection`.
- Licence: **CC BY-NC-SA 4.0**: attribution, non-commercial, share-alike. The authors say it is intended for educational and research use. The photos come from Pexels.
- What this means here:
  - The SH17 images, labels and any dataset built from them are **never committed** to this repository and are never redistributed.
  - Weights trained on SH17 are for research and portfolio use only. A commercial product would need different data.
  - Only the small manifest (file names, counts, version id) is committed.

## Class ids (checked 2026-11-28)
The Kaggle copy's label ids are 0-based and do not follow the order on the GitHub page. We confirmed them by pairing the YOLO label ids with the class names in the VOC xml files: 0 person, 1 ear, 2 ear-mufs, 3 face, 4 face-guard, 5 face-mask-medical, 6 foot, 7 tools, 8 glasses, 9 gloves, 10 helmet, 11 hands, 12 head, 13 medical-suit, 14 shoes, 15 safety-suit, 16 safety-vest. Instance counts in the Kaggle copy: person 13,802, helmet 927, safety-vest 530. So SH17 adds variety, not a large number of vests.

## How we use it (`datasets/sh17.py`, `scripts/extend_with_sh17.py`)
1. Start from the prepared `core` dataset (version `d1dc2ab4d1a9`), unchanged.
2. Take SH17 photos that contain at least one helmet or safety-vest box.
3. Drop any that look like near-duplicates of a base validation or test image (same hash test as the base split).
4. Keep person, helmet and vest boxes (SH17 id 0, 10, 16 become core ids 0, 1, 2). Everything else, including head and safety-suit, is dropped. SH17 has no forklift class, so forklifts in these photos are unlabelled.
5. Shrink to at most 1280 px on the long side. Add to **train only**, at most 1,500 photos, chosen with a fixed seed.
6. Write a new dataset `core_sh17` with its own manifest and version id.

Validation and test are identical to the base dataset, so every score stays comparable with earlier runs.

## Known risks
- Unlabelled objects: SH17 does not label forklifts, so the model sees some forklifts as background in these photos.
- Different label style: SH17 was labelled by different people with different rules, which can shift box sizes or what counts as a person.
- Only 530 vests in the whole dataset: do not expect a large gain on vests. Measure it.
- Static photos from stock images, not CCTV.
