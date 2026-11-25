# Safety check: n_core_imgsz960 on the val split

- Dataset version `d1dc2ab4d1a9`, detector confidence 0.35, inference size 960, person match IoU 0.5
- 160 images. Persons: 324 matched, 14 missed by the detector, 7 detected but not in the labels
- Safety thresholds are the untuned placeholder defaults unless a config was given.
- 'Truth' = the same association rules applied to the ground-truth labels. This measures what detector errors do to the safety conclusions, not whether the rules are right.

## Helmet: what the system concluded, by what the labels say

| Labels say | People | Said yes | Said no | Said unknown | Person not detected |
|---|---|---|---|---|---|
| no (a real violation) | 79 | 8 (10%) | 68 (86%) | 0 (0%) | 3 (4%) |
| yes (compliant) | 228 | 210 (92%) | 7 (3%) | 11 (5%) | 0 (0%) |
| unknown (unclear in the labels) | 31 | 0 (0%) | 1 (3%) | 19 (61%) | 11 (35%) |

## Vest: what the system concluded, by what the labels say

| Labels say | People | Said yes | Said no | Said unknown | Person not detected |
|---|---|---|---|---|---|
| no (a real violation) | 57 | 2 (4%) | 50 (88%) | 2 (4%) | 3 (5%) |
| yes (compliant) | 251 | 234 (93%) | 9 (4%) | 6 (2%) | 2 (1%) |
| unknown (unclear in the labels) | 30 | 3 (10%) | 1 (3%) | 17 (57%) | 9 (30%) |

## Events the engine raised on the model's detections

| Rule | Violations | Uncertain |
|---|---|---|
| forklift_proximity | 18 | 24 |
| no_helmet | 79 | 34 |
| no_vest | 61 | 29 |
