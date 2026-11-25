# Experiments

Every number here comes from a file in `evaluation/` or a run folder. Models are chosen on the
**validation** split. The **test** split is evaluated once per chosen run.

Dataset: `core` profile, version `d1dc2ab4d1a9` (train 638, val 160, test 200). Model: YOLO11n.
Training: Kaggle Tesla T4, seed 42, 100 epochs, identical settings except where noted.

## Experiment 1 vs 2: input size 640 vs 960

Question: does a larger input size help find small objects (helmets, distant people)?
Hypothesis came from the baseline error analysis: nearly all missed helmets were small boxes.

| Run | imgsz | Train time | Val mAP50 | Val mAP50-95 | Val recall | Test mAP50-95 |
|---|---|---|---|---|---|---|
| n_core_baseline | 640 | 0.279 h | 0.921 | 0.730 | 0.875 | 0.676 |
| n_core_imgsz960 | 960 | 0.438 h | 0.953 | 0.755 | 0.922 | 0.704 |

Test per class for the 960 run (mAP50-95): person 0.789, helmet 0.728, vest 0.766, forklift 0.531
(forklift recall 0.795 on 73 objects).

### Error analysis on val (confidence 0.25, IoU 0.5, each run inferred at its own training size)

| Small-object recall | 640 | 960 |
|---|---|---|
| helmet (150) | 0.760 | 0.827 |
| vest (107) | 0.813 | 0.897 |
| person (103) | 0.864 | 0.893 |

- Objects found: 802 -> 827 of 887. Images with at least one error: 56 -> 49.
- False alarms fell for person (23 -> 12) and forklift (14 -> 8) but **rose for helmet (16 -> 25)**.
- All 26 remaining missed helmets are small ones.

### Conclusion

960 is adopted as the working model. The gain held on the untouched test split (+0.028 mAP50-95).

### Limits (read before quoting these numbers)

- One run per setting, one seed. Run-to-run variation across seeds is not measured. Repeating
  the 960 run with the same seed on the same GPU type reproduced the validation metrics exactly.
- best.pt was selected on validation, so validation numbers are slightly optimistic. Quote test.
- Forklift did not improve and has few examples (59 val, 73 test).
- Inference at 960 is slower. A CPU validation pass on a laptop (i5-1135G7) showed about 82-89 ms
  per image at 960. Not a benchmark: no matching 640 measurement under the same conditions.
  Proper CPU benchmarks come with the ONNX / OpenVINO milestone.

## Open items from these experiments

- `inference/detector.py` has no image-size setting. Serving must use the training size (960).
- Test whether a model trained with the `core_missing` profile helps (config exists:
  `experiments/configs/n_core_missing.yaml`).
- Consider an error category for badly aligned boxes (a miss paired with a nearby false alarm).
- Helmet false alarms increased at 960; inspect `evaluation/error_analysis/n_core_imgsz960_val`.
