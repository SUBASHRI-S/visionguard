# Fresh held-out clip set (v2)

Purpose: measure the pipeline and the next model on footage that has **never been used for tuning or looked at through the pipeline**. The first held-out clips (45686, 13751987) were run once and are now considered examined.

## Rules

1. These clips are never used to choose thresholds, timing values or training data.
2. Each clip is run through the pipeline once per model version, with settings fixed beforehand. Every run is recorded in `docs/EXPERIMENTS.md`, including bad results.
3. What a clip should produce (below) was written from looking at a few frames, before any run. It is my reading, not verified ground truth. Frame-level labels come before any accuracy claim.
4. Files stay in `data/clips/heldout_v2/` (not committed). The first 16 characters of each file's SHA-256 are listed so the set can be shown not to have changed.

## The five clips

All were shrunk to 10 s at 640 px wide with `scripts/shrink_clip.py`. File names carry the stock-site video ids; the licence was to be confirmed on each download page and is not yet recorded here.

| Clip (shrunk file) | SHA-256 (first 16) | What it tests | What I expect from looking at frames |
|---|---|---|---|
| `12098511-hd_1920_1080_50fps` | e993a4c39d192994 | Many small, distant people on a bridge pier; orange vests, yellow helmets | Workers look compliant, so ideally 0 incidents; the risk is missed vests and helmets at small size causing false alarms |
| `154950-809834548_medium` | dee788d9b63bbc10 | Forklift with operator and a person on foot; caps, no helmets or vests visible | PPE rules should fire if a helmet and a vest are required here; proximity between the person and the forklift |
| `14493704_1080_1920_30fps` | 273d483ab3659203 | Back views, hats (not helmets), portrait video, people partly cut off | No helmets visible on most people, so helmet incidents are plausible; tests hat-versus-helmet and the portrait frame |
| `18689366-uhd_3840_2160_30fps` | 0827477283169a82 | Upper bodies behind a wall; caps, face scarves | Caps must not count as helmets; vests mostly not visible, so many "unknown" outcomes are correct |
| `13715751_1920_1080_30fps` | 033044a58bf1533f | Brick workers, orange T-shirt (not a vest), no helmets, low camera | An orange shirt must not count as a vest; people partly out of frame |

## Reserve clips (not part of the test set)

| Clip | Why held back |
|---|---|
| `42924-434300920_medium` (aerial, tiny people) | Very small people; kept as a stress test, not a fair measure |
| `209883_medium` (forklift far down a warehouse aisle) | Tiny forklift and operator; kept as a stress test |
| `40814-426920963_medium` (close-up rebar work) | Mostly partial bodies |
| `19832481-hd_1920_1080_25fps` | Its id is next to the tuning clip `19832492` and it looks like the same site, so it would not be independent |
| `12981246`, `17145555` | Indoor factory and workshop; PPE requirements unclear |
