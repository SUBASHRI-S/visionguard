# Evaluation: n_core_baseline on the test split

- Dataset version: `d1dc2ab4d1a9` (profile `core`)
- Images: 200
- Weights SHA256: `cce0167847927ef35c83ddd08be755badb32188dd731d5b8f0173a8c36294304`
- Settings: {"imgsz": 640, "batch": 16, "conf": 0.001, "iou": 0.6}
- Evaluated at: 2026-11-24T09:05:13+00:00

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| **all** | 1188 | 0.922 | 0.839 | 0.888 | 0.676 |
| person | 481 | 0.966 | 0.889 | 0.943 | 0.758 |
| helmet | 294 | 0.947 | 0.786 | 0.848 | 0.671 |
| vest | 340 | 0.968 | 0.874 | 0.935 | 0.734 |
| forklift | 73 | 0.808 | 0.809 | 0.825 | 0.541 |

Speed per image on `cpu` (not a deployment benchmark): preprocess 2.7 ms, inference 33.5 ms, postprocess 0.6 ms.

Small classes have few instances, so their numbers are noisy.
