# Evaluation: n_core_imgsz960 on the val split

- Dataset version: `d1dc2ab4d1a9` (profile `core`)
- Images: 160
- Weights SHA256: `571a2118e6e32faf74ba6495a701de60c6f755bfc48d4309fa24c2197a466704`
- Settings: {"imgsz": 960, "batch": 16, "conf": 0.001, "iou": 0.6}
- Evaluated at: 2026-11-25T03:58:11+00:00

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| **all** | 887 | 0.957 | 0.924 | 0.953 | 0.754 |
| person | 338 | 0.976 | 0.962 | 0.981 | 0.853 |
| helmet | 236 | 0.920 | 0.872 | 0.906 | 0.749 |
| vest | 254 | 0.985 | 0.929 | 0.966 | 0.807 |
| forklift | 59 | 0.947 | 0.932 | 0.957 | 0.605 |

Speed per image on `cpu` (not a deployment benchmark): preprocess 4.9 ms, inference 89.2 ms, postprocess 0.5 ms.

Small classes have few instances, so their numbers are noisy.
