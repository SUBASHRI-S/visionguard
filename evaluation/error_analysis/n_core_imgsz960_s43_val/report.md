# Error analysis: n_core_imgsz960_s43 on the val split

- Dataset version `d1dc2ab4d1a9`, confidence threshold 0.25, IoU threshold 0.5
- 160 images, 53 with at least one error
- Precision and recall here are at one fixed confidence threshold, so they differ from the mAP-based numbers (which sweep the threshold).

## Per class

| Class | Found | Missed or wrong class | False alarms | Recall | Precision |
|---|---|---|---|---|---|
| person | 325 | 13 | 14 | 0.962 | 0.959 |
| helmet | 206 | 30 | 25 | 0.873 | 0.892 |
| vest | 240 | 14 | 14 | 0.945 | 0.945 |
| forklift | 55 | 4 | 10 | 0.932 | 0.846 |

## Recall by object size

Size is the box area as a share of the image: small <1%, medium 1-5%, large >5%.

| Class | Small (n) | Medium (n) | Large (n) |
|---|---|---|---|
| person | 0.903 (103) | 0.979 (47) | 0.989 (188) |
| helmet | 0.807 (150) | 0.988 (85) | 1.000 (1) |
| vest | 0.916 (107) | 0.953 (43) | 0.971 (104) |
| forklift | 1.000 (5) | 1.000 (8) | 0.913 (46) |

## What each class was mistaken for

- Real **person** objects not found as person: background: 13
- Real **helmet** objects not found as helmet: background: 30
- Real **vest** objects not found as vest: background: 14
- Real **forklift** objects not found as forklift: background: 4

(`background` means the object was missed entirely.)

## Images with the most errors

- `source-2dcc1cb94093fa4285e9392019a06378.jpg`: 14 error(s)
- `source-f2af2a3bc9f3db238c5e968c87b7d7ca.jpg`: 12 error(s)
- `source-c3101bfd2bb97558b6fd8081d3bb8785.jpg`: 7 error(s)
- `source-c804b98bcd60ea646255dad2e4b4ebec.jpg`: 7 error(s)
- `source-29a54f6e5e66462bbcbec90273e92a51.jpg`: 6 error(s)
- `source-d7438871aefee49d413dcede6cd5d626.jpg`: 5 error(s)
- `source-3f2f6690bf1180b1505b49e7da6fb80a.jpg`: 4 error(s)
- `source-8d0f06f4f83594e0a66dc5567587f7f6.jpg`: 4 error(s)
- `ultralytics-ppe-nov2026-001.jpg`: 4 error(s)
- `source-263d7b7ac85694a4522bfcb8c63666cd.jpg`: 3 error(s)
- `source-5bf9ef3b2de6cb1e7af8e9301cdeaa73.jpg`: 3 error(s)
- `ultralytics-ppe-nov2026-311.jpg`: 3 error(s)
- `source-0e2a4d4463396c949ccdf3a3253e1552.jpg`: 2 error(s)
- `source-1c60612624732b80742dd50ae6225c57.jpg`: 2 error(s)
- `source-4e950f5884ceede2ad9753aaa490ecae.jpg`: 2 error(s)
- `source-5245a69eda05ca8afade22f08bfbdce7.jpg`: 2 error(s)
- `source-9ec80cefaef867295b0c9b5b6c7757c8.jpg`: 2 error(s)
- `source-b7365710e3f2d090a30fae7678bef407.jpg`: 2 error(s)
- `source-bb1a432046e3dc54d4343db11452eef0.jpg`: 2 error(s)
- `source-fdb6b7a03a5d9723c3ea161ddfd7b0bb.jpg`: 2 error(s)
