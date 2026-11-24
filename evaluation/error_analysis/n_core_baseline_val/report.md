# Error analysis: n_core_baseline on the val split

- Dataset version `d1dc2ab4d1a9`, confidence threshold 0.25, IoU threshold 0.5
- 160 images, 56 with at least one error
- Precision and recall here are at one fixed confidence threshold, so they differ from the mAP-based numbers (which sweep the threshold).

## Per class

| Class | Found | Missed or wrong class | False alarms | Recall | Precision |
|---|---|---|---|---|---|
| person | 320 | 18 | 23 | 0.947 | 0.933 |
| helmet | 200 | 36 | 16 | 0.847 | 0.926 |
| vest | 229 | 25 | 12 | 0.902 | 0.950 |
| forklift | 53 | 6 | 14 | 0.898 | 0.791 |

## Recall by object size

Size is the box area as a share of the image: small <1%, medium 1-5%, large >5%.

| Class | Small (n) | Medium (n) | Large (n) |
|---|---|---|---|
| person | 0.864 (103) | 0.957 (47) | 0.989 (188) |
| helmet | 0.760 (150) | 1.000 (85) | 1.000 (1) |
| vest | 0.813 (107) | 0.953 (43) | 0.971 (104) |
| forklift | 0.800 (5) | 1.000 (8) | 0.891 (46) |

## What each class was mistaken for

- Real **person** objects not found as person: background: 18
- Real **helmet** objects not found as helmet: background: 36
- Real **vest** objects not found as vest: background: 25
- Real **forklift** objects not found as forklift: background: 6

(`background` means the object was missed entirely.)

## Images with the most errors

- `source-2dcc1cb94093fa4285e9392019a06378.jpg`: 17 error(s)
- `source-f2af2a3bc9f3db238c5e968c87b7d7ca.jpg`: 11 error(s)
- `source-08e4ee526a5ea46b79ae4b6da87e91a1.jpg`: 8 error(s)
- `ultralytics-ppe-nov2026-001.jpg`: 8 error(s)
- `source-263d7b7ac85694a4522bfcb8c63666cd.jpg`: 7 error(s)
- `source-bb1a432046e3dc54d4343db11452eef0.jpg`: 7 error(s)
- `source-c804b98bcd60ea646255dad2e4b4ebec.jpg`: 7 error(s)
- `source-29a54f6e5e66462bbcbec90273e92a51.jpg`: 5 error(s)
- `source-1cf7cb3866d943a954bc8e591591e519.jpg`: 4 error(s)
- `source-5bf9ef3b2de6cb1e7af8e9301cdeaa73.jpg`: 4 error(s)
- `source-8d0f06f4f83594e0a66dc5567587f7f6.jpg`: 4 error(s)
- `source-b7365710e3f2d090a30fae7678bef407.jpg`: 4 error(s)
- `source-c3101bfd2bb97558b6fd8081d3bb8785.jpg`: 4 error(s)
- `source-d9f9c995d964d8a06df134b5637ad554.jpg`: 3 error(s)
- `ultralytics-ppe-nov2026-311.jpg`: 3 error(s)
- `ultralytics-ppe-nov2026-392.jpg`: 3 error(s)
- `source-0e2a4d4463396c949ccdf3a3253e1552.jpg`: 2 error(s)
- `source-1b1d16e594d2940d1ce4dcc46b896830.jpg`: 2 error(s)
- `source-3f2f6690bf1180b1505b49e7da6fb80a.jpg`: 2 error(s)
- `source-5245a69eda05ca8afade22f08bfbdce7.jpg`: 2 error(s)
