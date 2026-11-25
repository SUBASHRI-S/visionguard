# Evaluation: n_core_imgsz960 on the test split

- Dataset version: `d1dc2ab4d1a9` (profile `core`)
- Images: 200
- Weights SHA256: `571a2118e6e32faf74ba6495a701de60c6f755bfc48d4309fa24c2197a466704`
- Settings: {"imgsz": 960, "batch": 16, "conf": 0.001, "iou": 0.6}
- Evaluated at: 2026-11-25T04:02:53+00:00

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| **all** | 1188 | 0.953 | 0.862 | 0.912 | 0.704 |
| person | 481 | 0.978 | 0.892 | 0.959 | 0.789 |
| helmet | 294 | 0.965 | 0.840 | 0.911 | 0.728 |
| vest | 340 | 0.971 | 0.924 | 0.948 | 0.766 |
| forklift | 73 | 0.900 | 0.795 | 0.827 | 0.531 |

Speed per image on `cpu` (not a deployment benchmark): preprocess 5.0 ms, inference 82.4 ms, postprocess 0.5 ms.

Small classes have few instances, so their numbers are noisy.
