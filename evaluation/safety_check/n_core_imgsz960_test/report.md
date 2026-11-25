# Safety check: n_core_imgsz960 on the test split

- Dataset version `d1dc2ab4d1a9`, detector confidence 0.35, inference size 960, person match IoU 0.5
- 200 images. Persons: 442 matched, 39 missed by the detector, 16 detected but not in the labels
- Safety thresholds are the untuned placeholder defaults unless a config was given.
- 'Truth' = the same association rules applied to the ground-truth labels. This measures what detector errors do to the safety conclusions, not whether the rules are right.

## Helmet: what the system concluded, by what the labels say

| Labels say | People | Said yes | Said no | Said unknown | Person not detected |
|---|---|---|---|---|---|
| no (a real violation) | 151 | 6 (4%) | 127 (84%) | 10 (7%) | 8 (5%) |
| yes (compliant) | 275 | 230 (84%) | 11 (4%) | 24 (9%) | 10 (4%) |
| unknown (unclear in the labels) | 55 | 1 (2%) | 3 (5%) | 30 (55%) | 21 (38%) |

## Vest: what the system concluded, by what the labels say

| Labels say | People | Said yes | Said no | Said unknown | Person not detected |
|---|---|---|---|---|---|
| no (a real violation) | 98 | 8 (8%) | 76 (78%) | 4 (4%) | 10 (10%) |
| yes (compliant) | 334 | 310 (93%) | 5 (1%) | 12 (4%) | 7 (2%) |
| unknown (unclear in the labels) | 49 | 1 (2%) | 3 (6%) | 23 (47%) | 22 (45%) |

## Why compliant people were called violators

| Item | Detector never found one | Found, but the rules rejected it |
|---|---|---|
| helmet | 10 | 1 |
| vest | 4 | 1 |

## Confidence of the matched item (min / median / max)

| Item | Wrongly accepted (n) | Correctly accepted (n) |
|---|---|---|
| helmet | 0.45 / 0.64 / 0.89 (9) | 0.35 / 0.91 / 0.96 (248) |
| vest | 0.37 / 0.61 / 0.91 (8) | 0.36 / 0.89 / 0.95 (310) |

## Events the engine raised on the model's detections

| Rule | Violations | Uncertain |
|---|---|---|
| forklift_proximity | 20 | 25 |
| no_helmet | 148 | 71 |
| no_vest | 89 | 46 |
