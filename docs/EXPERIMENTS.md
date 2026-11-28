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

## Experiment 3: model size YOLO11n vs YOLO11s (both at 960) - provisional, one seed

Question: is the detector limited by model capacity? One change from `n_core_imgsz960`: `yolo11n.pt` -> `yolo11s.pt`. Same dataset (`d1dc2ab4d1a9`), seed 42, 100 epochs, batch 16, Kaggle Tesla T4. Training log: all 100 epochs ran (no early stop).

| Run | Model | Train time | Val mAP50 | Val mAP50-95 | Val recall |
|---|---|---|---|---|---|
| n_core_imgsz960 | YOLO11n (about 2.6 M parameters, from the Ultralytics model table) | 0.438 h | 0.953 | 0.755 | 0.922 |
| s_core_imgsz960 | YOLO11s (9.4 M parameters, 21.7 GFLOPs in the training log) | 0.563 h | 0.946 | 0.769 | 0.923 |

YOLO11s per class on validation (mAP50-95): person 0.860, helmet 0.767, vest 0.821, forklift 0.627.

**Reading:** mAP50 is slightly lower (-0.007), mAP50-95 slightly higher (+0.014), recall the same. In the last 40 epochs the validation mAP50-95 of the 11s run moved between 0.74 and 0.77 from epoch to epoch, which is about the size of the difference. **This is not evidence that the larger model is better.** The test split has not been used for this run.

**CPU cost (measured, one clip, one run):** on the development laptop (Windows 11, Intel CPU with 8 logical cores, torch 2.15.0+cpu with 7 threads), `YoloDetector.detect` at input 960 took a median of **66.2 ms/frame for YOLO11n** (15.1 frames/s, 95th percentile 74.6 ms) and **150.3 ms/frame for YOLO11s** (6.65 frames/s, 95th percentile 153.6 ms): YOLO11s is 2.3 times slower. My earlier estimate from arithmetic alone was about 3 times, so the estimate was too high. Source: `scripts/benchmark_models.py` on `154950-809834548_medium.mp4`, 60 frames, 5 warm-up calls dropped. Under about 10 percent difference would be noise; this difference is far larger.

### Seed repeat (seed 43), validation split

| Run | Seed | Val mAP50 | Val mAP50-95 | Val recall | Where the number comes from |
|---|---|---|---|---|---|
| n_core_imgsz960 | 42 | 0.953 | 0.755 | 0.922 | best weights, final validation table |
| n_core_imgsz960_s43 | 43 | 0.939 | 0.753 | 0.898 | **last epoch** (`results.csv`); the best-weights table was not read |
| s_core_imgsz960 | 42 | 0.946 | 0.769 | 0.923 | best weights (last epoch: 0.945 / 0.763 / 0.923) |
| s_core_imgsz960_s43 | 43 | 0.950 | 0.766 | 0.895 | best weights (last epoch: 0.946 / 0.766 / 0.910) |

The rows are not all measured the same way (best weights against last epoch), so differences of about 0.01 should not be over-read. Training time: 0.563 h and 0.625 h for 11s, 0.438 h for 11n (seed 42).

**Reading, written after both seeds:**
- mAP50-95: YOLO11s is ahead in both seeds, by about 0.011 to 0.014 (0.766 to 0.769 against 0.753 to 0.755). The two 11s seeds differ by only 0.003.
- mAP50 and recall: no consistent difference (11n seed 42 has the highest mAP50, 0.953). Recall moved by up to 0.028 between seeds of the same model.
- So the larger model draws tighter boxes (mAP50-95 rewards that) but does not find more objects. Violations depend on finding the helmet, the vest and the person, not on how tight the box is.
- Cost: about 3 times the arithmetic per frame (see above) and 28 to 43 percent longer training. Not benchmarked on CPU.

### Error analysis of the four runs (validation, confidence 0.25, IoU 0.5, input 960)

| Measure (160 images) | n seed 42 | n seed 43 | s seed 42 | s seed 43 |
|---|---|---|---|---|
| Images with at least one error | 49 | 53 | 40 | 47 |
| Helmet recall, all sizes | 0.890 | 0.873 | 0.907 | 0.881 |
| Helmet recall, small (150 objects) | 0.827 | 0.807 | 0.853 | 0.820 |
| Vest recall, small (107) | 0.897 | 0.916 | 0.907 | 0.925 |
| Person recall, small (103) | 0.893 | 0.903 | 0.913 | 0.903 |
| False alarms: helmet | 25 | 25 | 16 | 20 |
| False alarms: vest | 10 | 14 | 5 | 4 |
| False alarms: person | 12 | 14 | 10 | 10 |
| False alarms: forklift | 8 | 10 | 6 | 7 |
| False alarms, all four classes | 55 | 63 | 37 | 41 |

(The n seed-42 column is `evaluation/error_analysis/n_core_imgsz960_val`; the others were added with this experiment.)

**Reading:**
- Small-object recall: YOLO11s is ahead in most cells, but by 1 to 4 objects out of 100 to 150, and the two seeds of the same model differ by about as much (small helmet: n 0.827 and 0.807, s 0.853 and 0.820). I cannot call this a gain.
- False alarms: YOLO11s has fewer in every class and in both seeds (all classes 37 and 41 against 55 and 63). The counts are small, but the direction never flips. This is the clearest difference between the models.
- Why it matters here: a false helmet detection on a bare head or a cap hides a no-helmet violation, and a false vest detection hides a no-vest violation. That is the failure seen on real clips (a cap read as a helmet). Fewer false helmet and vest boxes is the effect that would help the product, if it holds on real footage.

**Decision:** settled by the protocol in `docs/HELDOUT_V2.md`, see "Fresh held-out clips v2" at the end of this file. In short: YOLO11n stays as the working model.

## Open items from these experiments

- `inference/detector.py` has no image-size setting. Serving must use the training size (960).
- Test whether a model trained with the `core_missing` profile helps (config exists:
  `experiments/configs/n_core_missing.yaml`).
- Consider an error category for badly aligned boxes (a miss paired with a nearby false alarm).
- Helmet false alarms increased at 960; inspect `evaluation/error_analysis/n_core_imgsz960_val`.

## Safety stage: engine on real detections (n_core_imgsz960, validation split)

Question: how often does the safety engine reach the same PPE conclusion from the model's
detections as it does from the ground-truth labels? (`scripts/safety_check.py`.)
Limit: "truth" is the labels run through the same association rules, so this measures what
**detector errors** cost the safety logic, not whether the rules themselves are right.
All safety thresholds other than the ones below are untuned placeholders.

At detector confidence 0.35 with the placeholder rules (160 images, 324 matched people):

| Item | Real violations flagged | Silently called compliant | Compliant people falsely flagged |
|---|---|---|---|
| helmet (79 / 228 people) | 68 (86%) | 8 (10%) | 7 (3%) |
| vest (57 / 251 people) | 50 (88%) | 2 (4%) | 9 (4%) |

About 90% of false violations came from the detector never finding the helmet or vest
(helmet 6 of 7, vest 8 of 9); the association rules rejected a found item only once each.

### Detector confidence sweep (helmet, same rules)

| Confidence | Violations flagged | Silent misses | Compliant falsely flagged |
|---|---|---|---|
| 0.35 | 68 | 8 | 7 |
| 0.25 | 66 | 10 | 5 |
| 0.20 | 64 | 12 | 5 |

Lowering the threshold removed about two false alarms per step but added about two silent
misses, which is the worse error. 0.35 was kept.

### Weak-helmet cutoff

Wrongly accepted helmets had lower confidence (median 0.59, n=8) than correctly accepted ones
(median 0.91, n=210). A matched helmet weaker than a cutoff now gives `unknown` instead of `yes`.

| helmet_min_confidence | Silent misses | Compliant made unknown | Compliant still yes |
|---|---|---|---|
| off | 8 | 11 | 210 |
| 0.50 | 6 | 15 | 206 |
| **0.60** | **3** | 21 | 200 |
| 0.70 | 3 | 27 | 194 |
| 0.80 | 2 | 48 | 173 |

0.60 chosen (`experiments/safety/chosen.yaml`). The cutoff was selected on 8 silent-miss cases, so
it may be partly luck; the highest wrongly accepted helmet had confidence 0.86, so no cutoff
removes all of them. The vest had only 2 wrong acceptances, so no vest cutoff was set.

### Test split, evaluated once (`experiments/safety/chosen.yaml`, detector confidence 0.35)

200 images. Run once after the settings above were fixed on validation.

| Item | Real violations flagged | Silently called compliant | Made unknown | Person not detected | Compliant falsely flagged |
|---|---|---|---|---|---|
| helmet (151 / 275 people) | 127 (84%) | 6 (4%) | 10 (7%) | 8 (5%) | 11 (4%) |
| vest (98 / 334 people) | 76 (78%) | 8 (8%) | 4 (4%) | 10 (10%) | 5 (1%) |

- The helmet results match validation closely (84% vs 86% flagged, 4% silent misses), so the
  weak-helmet cutoff chosen on validation did not just fit validation.
- Vest is worse than on validation (78% vs 88% flagged): more whole people were not detected
  (10% of vest violators) and silent misses doubled (8% vs 4%).
- Wrongly accepted vests on test were weaker (median confidence 0.61, n=8) than correct ones (0.89),
  like helmets. On validation only 2 wrong vests existed and they were confident (0.88). This is a
  **hypothesis, not adopted**: tuning a vest cutoff on the test split would spend the clean check.
  To be checked on the real-footage set (M11).
- Counts are per person per image, not incidents. Temporal logic (M6/M8) is not applied yet.

## Video pipeline: first real clip (informal, one clip)

**What this is.** The full chain (detector at 960, ByteTrack, safety rules, incident logic) run on
one stock clip: `67460-522170651_medium.mp4` (Pixabay file name; 10 s, 25 fps, shrunk to 640x360).
One worker in an orange vest and **no helmet** shovels sand; passers-by walk in the background.
I looked at frames of the clip myself; the labels below are my reading of them, not an annotation.
Settings: detector confidence 0.35, `experiments/safety/chosen.yaml`, tracker and incident timing
at their placeholder defaults (open after 8 frames, close after 30, track lost after 3 s).

| | Run 1: whole frame checked | Run 2: work area (bottom 40% of the frame) |
|---|---|---|
| Incidents opened | 5 | 1 |
| On the worker (no helmet, whole clip, track 1) | 1 | 1 |
| On people who appear to be passers-by or far background | 4 | 0 |
| Track IDs in the clip | 10 | 2 (one is a single frame at the very end) |
| Worker helmet frames yes / no / unknown | 0 / 246 / 1 | 0 / 246 / 1 |
| Worker vest frames yes / no / unknown | 242 / 5 / 0 | 242 / 5 / 0 |
| Mean detect time per frame (CPU, this laptop) | 99 ms | 87 ms |

**What it shows**
- The worker was followed with one stable ID for the whole clip and the missing helmet was reported
  as one incident. The vest was correctly seen on 242 of 247 frames; the 5 wrong frames did not
  open an incident because an incident needs 8 in a row.
- Small, far-away people were marked "unknown" for the vest instead of "no" and raised nothing.
- Run 1's extra incidents were people nobody asked to wear PPE. The system has no notion of "worker"
  versus "passer-by"; the work-area polygon is the mitigation.

**What it does NOT show**
- The work area was drawn by eye **after** seeing run 1 on this same clip, so run 2 is not
  independent evidence. It shows the mechanism works, not that a polygon will be right on another camera.
- One clip with one violating person. There is no recall or false-alarm rate here, and nothing about
  the forklift and zone rules on video.
- The 100 ms detect time is one run on a 640x360 copy; it is not a benchmark (M5 is still to do).
- Pixel thresholds (e.g. minimum person height 80 px) depend on video size, so this result may not
  carry over to the full-resolution original.
- Evidence frames from run 1 were not inspected one by one; "passers-by" is inferred from the video.

## Video pipeline: three tuning clips, before and after equipment memory and timing in seconds

All three clips are from the **tuning** subset (see docs/REAL_FOOTAGE_PROTOCOL.md). Same model
(n_core_imgsz960, size 960, confidence 0.35), same safety settings (`chosen.yaml`).

- **Before:** an incident opened after 8 processed frames (0.3 s), closed after 30 clear frames, no memory.
- **After:** opens after 3 s, closes after 2 s clear, and an item seen on a tracked person in the last
  5 s is never reported missing (only "unknown"). These values come from reasoning about how equipment
  behaves, not from a search over values.

| Clip | What is in it | Incidents before | Incidents after |
|---|---|---|---|
| `67460` (work area applied) | one worker, vest, **no helmet** (true violation) | 1 (correct) | 1 (same, correct) |
| `19832492` | two workers, helmets and vests (compliant) | 3, all false `no_vest` | 0 |
| `13752015` (stride 2) | forklift operator in helmet and vest (compliant) | 5, all false | 0 |

**Notes on how independent this is.** The new settings were designed after looking at `19832492`,
so that improvement (3 to 0) is not independent evidence. They were fixed before I saw the forklift
report, so 5 to 0 on `13752015` is a weak independent check. `67460` shows the settings do not
lose a real violation, on one person.

**What still looks wrong (the improvement hides some of it)**
- On `13752015` the operator was recognised as an operator on only 24 of 121 frames, so the
  proximity rule judged the driver against his own forklift for 52 frames. No incident opened only
  because those frames never ran 3 s in a row. The fix is not in place; the symptom is masked.
- The same operator is "unknown" for helmet on 99 and for vest on 104 of 121 frames (the forklift
  mast crosses his body), so for most of the clip the system cannot confirm PPE either way. "Unknown"
  is not "safe": a real violator seated behind a mast could be missed, because memory only helps after
  the item has been seen once.
- The operator's track id changed once (track 1 to track 7, near 3.3 s) when the mast passed in
  front, and a second track (6) overlaps track 1 in time; I have not looked at it, so I do not know
  whether it is a duplicate box or a real person.
- Memory means a real vest or helmet removal is noticed up to 5 s late.

**Not yet done:** the two held-out clips (`45686`, `13751987`) have not been run. They are to be run
once with these settings, and the result recorded whatever it is.

## Video pipeline: two held-out clips (run once, settings not changed afterwards)

Same model and settings as the three tuning clips (open after 3 s, close after 2 s, memory 5 s, no
work area, `chosen.yaml`). I predicted before running: `13751987` no incident, `45686` at least one
real incident. **Both predictions were wrong.**

| Clip | What is in it (my reading of frames) | Result |
|---|---|---|
| `13751987` (stride 2) | worker in blue helmet and **red hi-vis vest** with reflective stripes, side view | **1 false incident:** `no_vest` on the whole clip (193 violating frames). Helmet 198 yes of 202. Vest: 0 yes, 193 no, 9 unknown |
| `45686` | man in orange shirt and olive cap sweeping a drain; passers-by in motorbike helmets; a blurred man in white in the foreground | **0 incidents.** Probably a missed violation (see below) |

**Why, as far as I can tell**
- `13751987`: the detector never found the red vest in any frame. Equipment memory cannot help
  when the item has never been seen. This is a detector gap (vest colour or view angle), not a
  threshold problem.
- `45686`: three tracks had vest "no" on 36, 70 and 46 frames, but none reached the 72 frames
  (3 s) needed, because people were in view for under 3 s or were hidden by the foreground person
  (track 4 was "unknown" on 81 of 151 frames). One track had helmet "yes" on 79 frames although I
  see a cap, which suggests the helmet detector accepts caps and motorbike helmets. I did not
  inspect the evidence images, so this is a hypothesis.
- The 3 s persistence requirement that removed false alarms on the tuning clips can also hide real
  violations in short clips of moving people. On a long continuous camera this matters less; I have
  no data on that.

**What this means**
- The gains on the tuning clips did **not** carry over: 1 false alarm and 1 probable miss on 2
  held-out clips. The numbers were not tuned further to fix them.
- These two clips have now been examined, so they are no longer clean held-out data. A new set of
  clips is needed for any later claim.
- The limiting factor looks like detector coverage (hi-vis colours, caps versus helmets, side
  and back views, 638 training images that are partly generated), not the rule logic.
- Next experiments to try, each judged on the image test split and on new clips, not on these two:
  a larger model (`yolo11s`) at 960, and more varied training data.

## Fresh held-out clips v2: YOLO11n vs YOLO11s (each run once, settings fixed in advance)

Protocol and decision rule: `docs/HELDOUT_V2.md`, committed before these runs. Five clips, `scripts/run_video.py` defaults (confidence 0.35, `experiments/safety/chosen.yaml`, open after 3 s, close after 2 s, memory 5 s, every frame, no work area). Both models at input 960.

**How the judging was done and how far to trust it:** I looked at the annotated videos (640 px wide, a handful of frames per clip, zoomed in once for one clip) and the reports. All five clips were treated as sites that require a helmet and a vest, as the protocol says. Identity of some people after track-id changes, and whether a distant head wears a cap or a helmet, could not always be settled; I counted only clear-cut cases and left the rest out. Treat the totals as a rough count, not a measurement. Nothing was changed after seeing a result.

### Incidents raised

| Clip | YOLO11n | YOLO11s |
|---|---|---|
| 154950 (forklift, operator, person on foot) | 4: no_helmet on both people, no_vest on the operator, forklift_proximity on the person on foot (all true) | 4: no_helmet and no_vest on the person on foot, no_helmet on the operator in two episodes because the track id changed (all true). **No forklift_proximity incident, although the report counts 478 proximity violation frames; cause not found** |
| 12098511 (bridge pier, orange vests, yellow helmets) | 1: no_vest on person #3 (**false alarm**: zoomed frames show an orange vest with reflective strips) | 1: the same false no_vest on #3 |
| 13715751 (brick workers, orange T-shirt) | 2: no_helmet and no_vest on one worker (true) | 3: the same two plus no_vest on the worker in the orange T-shirt (true; the shirt was not accepted as a vest) |
| 14493704 (portrait, workers with hats, no vests) | 1: no_vest (true) | 2: no_vest on two people (true) |
| 18689366 (stone wall, caps and scarves) | 2: no_helmet and no_vest on the worker in the white shirt (true) | 2: no_vest on two people (true); the white-shirt worker's missing helmet was not raised (helmet mostly "unknown") |
| **Total** | 10 incidents, 9 true, 1 false | 12 incidents, 11 true, 1 false |

### Clear-cut errors (false alarms plus visible violations not raised)

| Clip | YOLO11n | YOLO11s |
|---|---|---|
| 154950 | 1 missed: no_vest on the person on foot | 1 missed: no_vest on the operator; plus the unexplained missing proximity incident |
| 12098511 | 1 false alarm (orange vest) | 1 false alarm (orange vest) |
| 13715751 | 2 missed: both violations of the worker in the orange shirt | 1 missed: that worker's missing helmet |
| 14493704 | 1 missed: no_helmet on the man in the hat (a hat, not a helmet), not raised | 1 missed: the same man in the hat |
| 18689366 | 2 missed: caps read as helmets on two people | 3 missed: the same two caps, plus the white-shirt worker's missing helmet |
| **Total** | **7** | **7 (8 if the proximity gap counts)** |

**Result against the rule:** YOLO11s would need at least 2 fewer errors; it has the same number or one more. **YOLO11n stays as the working model**, and YOLO11s is not evaluated on the test split.

### What both models got wrong (more important than the comparison)

- **Orange hi-vis vest with reflective strips, worn by a person facing the camera, read as "no vest"** for the whole clip (12098511), in both models. This is the red/orange vest weakness already seen on the tuning clips, now on a clip neither model was tuned on.
- **Caps read as helmets** on people in 18689366, so their missing helmets were never raised, in both models.
- **The orange T-shirt (13715751) was briefly read as a vest** (9 frames for YOLO11n, 2 for YOLO11s) but did not produce a false "vest present" overall.
- **Equipment on the forklift operator stays "unknown"** most of the time (154950), and **track ids change** (the same operator became several tracks; the same worker in the orange shirt too), which splits one long violation into pieces shorter than the 3 s needed to open an incident.
- Neither model's gaps look like a rule-logic problem. They are detector coverage and track continuity, as concluded before.

### Speed in the pipeline

Mean detect time per frame in the pipeline reports: YOLO11n 75 to 88 ms, YOLO11s 176 to 394 ms. The isolated benchmark gave 66 ms and 150 ms. The YOLO11s runs were slower than the benchmark; other programs may have been running on the laptop, which I could not check.

### Status of these clips

The five clips are now used for these two model versions. Any later claim needs new clips.



## Experiment 4 (prepared, not run): add SH17 photos to training

- **Question:** do varied photos with helmets and vests fix the orange-vest and cap errors, without hurting the rest?
- **One change:** training data only (`core` to `core_sh17`: same base data plus SH17 photos with a helmet or vest, train only). Model YOLO11n, 960, 100 epochs, batch 16, seed 42, as in `n_core_imgsz960`.
- **Fixed before running:** val and test are unchanged. The decision rule uses new clips only, because the heldout_v2 clips are used up for earlier model versions. Image-level scores are reported but are not enough on their own to call it a win.
- **Data notes:** `docs/DATASET_SH17.md`. Licence is non-commercial, so nothing from SH17 goes into the repository.
- **Results:** not yet available.
