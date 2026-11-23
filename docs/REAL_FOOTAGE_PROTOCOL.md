# Real-footage protocol

Why this exists: part of the training data is generated, so dataset metrics alone cannot show how VisionGuard behaves on a real camera. A small set of real clips, labelled by hand, is the honest check. It is small, and every result will say how small.

## 1. Where clips may come from

Only use footage you are allowed to use.

- **Your own recordings:** you or someone you have permission from filmed it. Everyone clearly visible has agreed to be filmed for this project. Record `consent_obtained: true` and `licence: "own work"`.
- **Third-party clips:** only with a licence that allows use and, where required, attribution (for example CC BY). Record the exact `url` and `licence`.
- Do not scrape platforms, and do not use footage with unclear terms.
- Do not publish clips (repo, demo video) if identifiable people did not agree to that. Blur faces in anything you publish.
- Staged scenes are fine and often easier: a friend with and without a helmet or vest in a safe place. Never put anyone in a real hazard.

## 2. What to collect

- Target: 10 to 20 clips, 20 to 60 seconds each, from a fixed camera position.
- Include compliant footage (no incidents), not only violations.
- Vary: lighting (bright, shade, evening), distance (close and far), number of people, partial occlusion.
- Keep the original files in `datasets/raw/real_clips/` (ignored by git).

## 3. How to label

Write the manifest at `evaluation/replay_manifest.json` (copy `evaluation/replay_manifest.example.json`).

- One entry per clip. `incidents` is a list of intervals in seconds, one per rule: `no_helmet`, `no_vest`, `zone_intrusion`, `proximity`. Use `[]` for a compliant clip.
- An interval starts when the condition is clearly visible and ends when it clearly stops. Intervals of the same rule must not overlap in one clip.
- If you are unsure, still label it and set `"ambiguous": true`. Results are reported with and without ambiguous intervals.
- Label before you run the system on the clip, so the model's output cannot influence you.
- Mark each clip `"subset": "tune"` or `"final"`. Choose thresholds on `tune`; report results on `final` only.

## 4. Validate

```
python -c "from evaluation.clip_labels import load_manifest; print(len(load_manifest('evaluation/replay_manifest.json')), 'clips valid')"
```

A problem is reported with the clip and field, and all problems are listed together.

## 5. What we will and will not claim

- Counts of clips, incidents and seconds are always reported next to any metric.
- With a set this small, a difference between two settings can be noise. We say so instead of declaring a winner.
- Nothing from this set is used for training.
