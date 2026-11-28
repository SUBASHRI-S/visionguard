# Error analysis: s_core_imgsz960_s43 on the val split

- Dataset version `d1dc2ab4d1a9`, confidence threshold 0.25, IoU threshold 0.5
- 160 images, 47 with at least one error
- Precision and recall here are at one fixed confidence threshold, so they differ from the mAP-based numbers (which sweep the threshold).

## Per class

| Class | Found | Missed or wrong class | False alarms | Recall | Precision |
|---|---|---|---|---|---|
| person | 324 | 14 | 10 | 0.959 | 0.970 |
| helmet | 208 | 28 | 20 | 0.881 | 0.912 |
| vest | 241 | 13 | 4 | 0.949 | 0.984 |
| forklift | 53 | 6 | 7 | 0.898 | 0.883 |

## Recall by object size

Size is the box area as a share of the image: small <1%, medium 1-5%, large >5%.

| Class | Small (n) | Medium (n) | Large (n) |
|---|---|---|---|
| person | 0.903 (103) | 0.957 (47) | 0.989 (188) |
| helmet | 0.820 (150) | 0.988 (85) | 1.000 (1) |
| vest | 0.925 (107) | 0.953 (43) | 0.971 (104) |
| forklift | 0.800 (5) | 1.000 (8) | 0.891 (46) |

## What each class was mistaken for

- Real **person** objects not found as person: background: 14
- Real **helmet** objects not found as helmet: background: 28
- Real **vest** objects not found as vest: background: 13
- Real **forklift** objects not found as forklift: background: 6

(`background` means the object was missed entirely.)

## Images with the most errors

- `source-2dcc1cb94093fa4285e9392019a06378.jpg`: 13 error(s)
- `source-f2af2a3bc9f3db238c5e968c87b7d7ca.jpg`: 9 error(s)
- `source-c804b98bcd60ea646255dad2e4b4ebec.jpg`: 8 error(s)
- `source-29a54f6e5e66462bbcbec90273e92a51.jpg`: 6 error(s)
- `source-b7365710e3f2d090a30fae7678bef407.jpg`: 4 error(s)
- `source-c3101bfd2bb97558b6fd8081d3bb8785.jpg`: 4 error(s)
- `source-263d7b7ac85694a4522bfcb8c63666cd.jpg`: 3 error(s)
- `source-fdb6b7a03a5d9723c3ea161ddfd7b0bb.jpg`: 3 error(s)
- `ultralytics-ppe-nov2026-311.jpg`: 3 error(s)
- `source-0e2a4d4463396c949ccdf3a3253e1552.jpg`: 2 error(s)
- `source-5bf9ef3b2de6cb1e7af8e9301cdeaa73.jpg`: 2 error(s)
- `source-77d2d535123acf7562f7e64c2e2803fb.jpg`: 2 error(s)
- `source-9a1a7af602890ceff8a7513dddfe1e10.jpg`: 2 error(s)
- `source-bb1a432046e3dc54d4343db11452eef0.jpg`: 2 error(s)
- `source-d7438871aefee49d413dcede6cd5d626.jpg`: 2 error(s)
- `source-d9f9c995d964d8a06df134b5637ad554.jpg`: 2 error(s)
- `ultralytics-ppe-nov2026-001.jpg`: 2 error(s)
- `ultralytics-ppe-nov2026-030.jpg`: 2 error(s)
- `ultralytics-ppe-nov2026-207.jpg`: 2 error(s)
- `ultralytics-ppe-nov2026-435.jpg`: 2 error(s)
