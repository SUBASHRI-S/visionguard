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
