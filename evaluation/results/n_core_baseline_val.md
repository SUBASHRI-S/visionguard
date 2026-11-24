# Evaluation: n_core_baseline on the val split

- Dataset version: `d1dc2ab4d1a9` (profile `core`)
- Images: 160
- Weights SHA256: `cce0167847927ef35c83ddd08be755badb32188dd731d5b8f0173a8c36294304`
- Settings: {"imgsz": 640, "batch": 16, "conf": 0.001, "iou": 0.6}
- Evaluated at: 2026-11-24T09:04:44+00:00

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| **all** | 887 | 0.965 | 0.871 | 0.921 | 0.728 |
| person | 338 | 0.981 | 0.921 | 0.967 | 0.822 |
| helmet | 236 | 0.975 | 0.797 | 0.870 | 0.699 |
| vest | 254 | 0.974 | 0.885 | 0.936 | 0.770 |
| forklift | 59 | 0.930 | 0.881 | 0.911 | 0.623 |

Speed per image on `cpu` (not a deployment benchmark): preprocess 2.5 ms, inference 35.3 ms, postprocess 0.6 ms.

Small classes have few instances, so their numbers are noisy.
