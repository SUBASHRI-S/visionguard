# VisionGuard roadmap

Last updated 2026-11-27. "Done" means built **and** tested as described in `docs/ARCHITECTURE.md`; it does not mean accurate on real footage. Accuracy claims are in `docs/EXPERIMENTS.md`.

## Done

| Area | What exists | How it was checked |
|---|---|---|
| Dataset | Versioned PPE dataset (person, helmet, vest), licence and provenance notes, validation and leakage checks | Unit tests, manifest hashes |
| Training | Config-driven training, MLflow logging, Kaggle workflow, two runs (`n_core_baseline`, `n_core_imgsz960`) | Test split evaluated once per run (mAP50-95 0.676, then 0.704) |
| Error analysis | Per-class and per-size breakdown on the validation split | Real validation data |
| Safety rules | PPE association, helmet / vest / zone / forklift-proximity rules, unknown handling | Many unit tests with deliberate-bug checks, on fabricated detections |
| Incident logic | Open after N frames, close after M, uncertain frames hold state, timing in seconds, equipment memory | Unit tests; behaviour seen on real clips |
| Tracking | ByteTrack adapter with person-index to track-id map | Unit tests on fabricated motion; real clips |
| Video pipeline | Detect, track, work-area filter, rules, incidents; annotated video, evidence, report | Real clips: tuning on three, two held-out clips run once |
| Incident store | SQLAlchemy store with status history and evidence | Unit tests, database-level checks |
| API | Incidents list/detail/review/evidence/stats, optional key, CORS allow-list | Unit tests and a real server run |
| Dashboard | Review queue, incident page with duration bar, pictures, history, review actions, summary | 26 Vitest tests, headless browser run, your own run on real data |
| Docker and CI | Two images, compose file, GitHub Actions (Python, dashboard, image build) | CI run #1 green |

## Not done yet, in the order I would do it

1. **Detection quality on real footage (the main weakness).** Findings so far: red high-visibility vests are never detected from the side, a cap is read as a helmet, vests are missed from some angles, a forklift mast hides the operator, and gains from tuning did not carry over to the held-out clips.
   - Collect 4 or 5 new clips (licence checked) as a fresh held-out set, since the old held-out clips have been looked at.
   - Experiment `v2`: YOLO11s at image size 960, and more varied training data (side and back views, red and orange vests, caps), with the licence check first.
   - Report results in `docs/EXPERIMENTS.md`, including when it does not help.
2. **Replay evaluation (M11).** Label a few clips and measure incidents found, missed and false, per clip. This is the number that matters for the product.
3. **CPU speed benchmarks (M5).** PyTorch against ONNX and OpenVINO on CPU, frames per second at 640 and 960, on a stated machine.
4. **Make the pipeline robust to video size.** Replace pixel thresholds such as `min_person_height_px=80` with sizes relative to the frame.
5. **Operator and proximity handling.** The operator was recognised on only 24 of 121 frames in the forklift clip, and the 3 s wait can hide short real violations. Track ids also split after occlusion; an appearance-based re-identification step is one option.
6. **Reconcile `inference/detector.py` with `pipeline/detector.py`.** Two detector wrappers exist; keep one. `tests/test_detector.py` uses the old one.
7. **API completion.** Image and video detection endpoints with size and type limits, cameras / zones / rules endpoints, model information, live updates (WebSocket), per-user login (JWT) instead of a shared key, rate limiting, PostgreSQL with Alembic migrations.
8. **Dashboard completion.** Live view, zone drawing, camera and rules pages, a login screen.
9. **Alerts.** Notification when an incident opens (e-mail or webhook), with a quiet-hours and repeat limit.
10. **Operations.** Evidence retention with automatic deletion, optional face blurring on saved pictures, structured logs, a deployment guide, publishing the images.
11. **Project polish.** Repository description, topics and licence file, a short demo video or GIF, a README section with the honest results table, `docs/KAGGLE_TRAINING.md` brought up to date.

## Known limits (also in the README)

- Results on a few clips illustrate behaviour; they are not an accuracy measurement.
- Thresholds were tuned on three clips and did not transfer to two others.
- Equipment memory can delay a real removal being reported by up to 5 s.
- CPU only; inference takes roughly 70 to 100 ms per frame on the development laptop (not a benchmark).
- The reviewer name is typed by the person; there is no login yet.
