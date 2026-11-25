# VisionGuard: Architecture and Engineering Plan

Status: v1.1 (dataset decisions added). This document is the source of truth for design decisions.
Rule: nothing in this repo claims a metric that was not measured. Measured results live in `experiments/` and `docs/benchmarks.md`.

## 1. Goal and scope

VisionGuard watches workplace camera feeds, detects people, helmets, vests and vehicles, tracks people over time, applies configurable safety rules, and records reviewable incidents with evidence.

### Scope tiers (so the project is finishable)

| Tier | Contents | Rule |
|---|---|---|
| **Core (must ship)** | Fine-tuned PPE model, evaluation, MLflow, tracking, PPE rules, zone intrusion, incident store with dedup, FastAPI, dashboard (live, incidents, zones), Docker, CI, tests | Project is not "done" without all of it |
| **Extended (should ship)** | Proximity rule, alert webhooks, replay evaluation on labelled clips, ONNX/OpenVINO benchmarks, auth | Ship if Core is solid |
| **Stretch (only if time remains)** | Face blurring, `/metrics` endpoint, retention jobs, robustness tests with synthetic degradation | Optional |
| **Excluded on purpose** | Face recognition, worker identification, physical distance or collision prediction | Privacy/ethics risk or needs calibration we do not have |

## 2. Current status (update this table as work lands)

| Component | Status |
|---|---|
| Repo skeleton, venv, pytest, git hygiene | Done |
| `Detector` wrapper (pretrained YOLO) + 5 tests | Done |
| Dataset validator + 8 tests | Done (13 tests passing in the repo) |
| PPE dataset selected and licence checked (Ultralytics Construction-PPE, CC BY 4.0) | Done, see `datasets/README.md` |
| Dataset analysis, leakage-safe split, class profiles, version manifest, replay-clip schema (+ 41 tests) | Done. Run on the real dataset: `core` = `d1dc2ab4d1a9`, `core_missing` = `b1e02ace5049` |
| Real-footage protocol and replay-clip labels | Protocol written; clips not yet collected |
| Training pipeline: validated YAML config, dataset-version check, dataset rebuild from manifest, train script, MLflow logging (+ tests) | Written. Pipeline verified end to end on a **tiny synthetic dataset on CPU** (plumbing only, metrics meaningless). **No real training run yet** |
| Baseline training `n_core_baseline` (YOLO11n, core, dataset `d1dc2ab4d1a9`) on a Kaggle T4, logged to MLflow | Done. Validation split: mAP50 0.921, mAP50-95 0.730 (validation was used to pick `best.pt`, so these are slightly optimistic) |
| Evaluation script with one-time test-split guard (`scripts/evaluate.py`, + tests) | Done. Test split evaluated once for the baseline (mAP50-95 0.676) and once for `n_core_imgsz960` (0.704) |
| Error analysis on the validation split (`scripts/error_analysis.py`, + tests) | Done on the real validation split for both runs; inference uses each run's training image size |
| Experiment `n_core_imgsz960` (imgsz 960 vs 640) | Done; adopted as the working model. See `docs/EXPERIMENTS.md` for numbers and limits (single seed, slower inference not yet benchmarked) |
| PPE association, zones, proximity, four rules and engine (`safety/`, + 82 tests, with deliberate-bug checks) | Done on fabricated detections. **Thresholds are untuned placeholders**; not yet run on real model output or real clips |
| Tracking, incidents | Not started |
| API, frontend, Docker, CI/CD | Not started |

## 3. Key decisions (decision log)

| # | Decision | Reason | Trade-off |
|---|---|---|---|
| D1 | Detect `person`, `helmet`, `vest`; infer violations. No "no-helmet" class. | Learning the absence of an object is unreliable; positives are easier to detect and explain | Needs a careful association step |
| D2 | (revised) The vehicle class is `forklift`, detected by the same fine-tuned model as people and PPE. A COCO model for cars and trucks is optional and only considered if time allows | The chosen dataset already contains forklifts, so proximity needs no second model | Only forklifts are covered; other vehicles are a documented limitation |
| D3 | Violations have three outcomes: `compliant`, `violation`, `uncertain` | Avoids unjustified conclusions | Some real violations are not flagged until evidence improves |
| D4 | Rules are pure functions over a `FrameContext` | Unit-testable without models, video or DB | Needs clean data contracts |
| D5 | Incident opens only after N consecutive violating frames on one track; closes after M clear frames | Suppresses flicker and duplicate alerts | Adds detection delay (N frames) |
| D6 | Zone test uses the person's foot point | Feet show ground position | Fails when feet are occluded or cropped, so mark `uncertain` |
| D7 | Proximity = foot-point distance normalised by person height | Rough scale-invariance without calibration | Approximation only; documented as such |
| D8 | Webcam frames come from the browser over WebSocket | A server or container cannot open the user's camera | Network latency |
| D9 | DB-backed job table + worker process for video; Celery/Redis is a documented upgrade | Simplest thing that is reliable | Less throughput than a full queue |
| D10 | SQLAlchemy + Alembic; SQLite locally, PostgreSQL in Docker | One codebase, two backends | Avoid SQLite-only features |
| D11 | Train on a free cloud GPU; develop and run inference on CPU | Local machine has Intel integrated graphics only | Training is not fully local |
| D12 | All thresholds in validated YAML (Pydantic) | No hardcoded settings; per-camera overrides | Config schema must be maintained |
| D13 | Split policy: official val images become our **test** set; a new val set is carved from official train; near-duplicates are clustered so they never straddle splits; train images duplicating a test image are removed | The dataset has no test split, and leakage would inflate scores | Roughly 640 of the 800 official train images remain for training (exact numbers come from the manifest); perceptual hashing misses near-duplicates that are not re-saved copies |
| D14 | Dataset: Ultralytics Construction-PPE (Nov 2026, CC BY 4.0, 1,000 images, 12 classes). Train with class profiles: `core` (person, helmet, vest, forklift) and `core_missing` (adds no_helmet, no_vest) | Clean licence, YOLO format, has forklift; the profiles give a clean ablation of "detect missing PPE directly" vs "infer from absence" (D1) | Small, partly generated images; scores may overstate real-world performance |
| D15 | Every training run records a `dataset_version` (hash of the manifest: source zip hash, seed, thresholds, profile, file lists) | Reproducibility and MLflow tracking | The source zip hash must be recorded by hand at download time |
| D16 | A small hand-labelled set of real clips (`tune` and `final` subsets) is the honest check of the system | Part of the training data is generated, so dataset metrics alone are not enough | Small sample; results are reported with the sample size |

## 4. System architecture

```
                         +--------------------------+
 Camera / video / image  |   React dashboard        |
        |                |  live | incidents | zones|
        v                +-----+----------+---------+
 +--------------+              | REST     | WebSocket
 | inference/   |              v          v
 | Detector     |        +---------------------------+
 | (PyTorch /   |------->|  FastAPI (backend/)       |
 |  ONNX/OpenVINO)       |  routes -> services       |
 +------+-------+        +-----+-------------+-------+
        | detections           |             |
        v                      v             v
 +--------------+        +-----------+  +-------------+
 | tracking/    |        | Job queue |  | SQL DB      |
 | ByteTrack    |        | + workers |  | + evidence  |
 +------+-------+        +-----------+  |   files     |
        | tracks                        +------^------+
        v                                      |
 +--------------+   events    +----------------+--------+
 | safety/      |------------>| incidents/               |
 | association  |             | dedup, lifecycle,        |
 | zones, rules |             | evidence, alerts         |
 +--------------+             +--------------------------+

 Offline: datasets/ -> training/ (MLflow) -> evaluation/ -> models/ (versioned, ONNX)
```

Dependency rule: arrows only point downward or rightward. `safety/` never imports `backend/`. `inference/` knows nothing about HTTP. `incidents/` is the only module that writes incident rows.

## 5. Module layout

```
datasets/     YAML configs, validate.py, licence and provenance docs
training/     config-driven training, MLflow logging, seeds
evaluation/   detector metrics, error analysis, replay evaluation
models/       registry metadata, export scripts (ONNX, OpenVINO)
inference/    Detector interface, image/video/stream pipelines
tracking/     tracker wrapper, track state
safety/       association.py, zones.py, rules/{ppe,zone,proximity}.py, engine.py, config.py
incidents/    manager.py (lifecycle + dedup), evidence.py, alerts.py
backend/      api/, services/, db/, workers/, core/ (config, logging, auth)
frontend/     React dashboard
experiments/  experiment configs and reports
tests/        unit, integration, API
docker/       Dockerfiles, compose
docs/         this file, benchmarks, evaluation, limitations
```

## 6. Data contracts

```
Detection   class_name, confidence, box(x1,y1,x2,y2)
Track       track_id, class_name, box, confidence, age, last_seen_frame
PPEStatus   person_track_id, helmet: {yes|no|unknown}, vest: {yes|no|unknown}, reasons
FrameContext camera_id, frame_index, timestamp, tracks, ppe_statuses, zones, config
Event       rule_id, camera_id, track_id|None, frame_index, severity, details
Incident    id, rule_id, camera_id, track_id, status, opened_at, closed_at, evidence
```

Rules take a `FrameContext` and return a list of `Event`. They never touch the DB.

## 7. Safety logic

### 7.1 PPE association (safety/association.py)

For each person box:
1. Head region: top portion of the box (configurable fraction, e.g. top 25%). Torso region: middle portion.
2. A helmet matches if its centre is inside the head region and its area is plausible relative to the person (size sanity check).
3. A vest matches if its overlap with the torso region (intersection over vest area) exceeds a threshold.
4. If several people compete for one PPE box, assign it to the best-scoring person only.
5. Output per item: `yes`, `no`, or `unknown`.

`unknown` is returned (never `no`) when: the person box is smaller than a minimum height, the box touches the frame edge so the head or torso may be cut off, person confidence is below a threshold, or the person overlaps heavily with another person. Only `no` can trigger a violation.

All thresholds above come from config, and their values are chosen from validation data, not guessed.

### 7.2 Temporal logic

```
per (rule, track):   idle --N consecutive violating frames--> open
                     open --M consecutive clear frames-->   closed
                     open --track lost > T seconds-->       closed (reason: track_lost)
```

Deduplication key: `(camera_id, rule_id, track_id)`. While an incident is open, new events for that key only update its last-seen time. A track ID switch (tracker failure) can create a duplicate; this is a known limitation and is measured in replay evaluation.

### 7.3 Zones and proximity

- Zones are polygons per camera, stored in normalised coordinates (0-1) so they survive resolution changes. Point-in-polygon on the foot point.
- Proximity applies to person-vehicle pairs. Distance between foot points divided by the person's box height; warn below a configurable ratio.
- Documented limitation: image-space proximity is not physical distance. Perspective, camera angle and height all distort it. It is a screening heuristic, not collision prediction.

### 7.4 Rule interface

```python
class Rule(Protocol):
    rule_id: str
    def evaluate(self, ctx: FrameContext) -> list[Event]: ...
```

Rules: `NoHelmetRule`, `NoVestRule`, `ZoneIntrusionRule`, `ProximityRule`. Each is enabled and tuned per camera.

## 8. Incident management

Lifecycle: `open -> acknowledged -> resolved`, or `false_positive`. Every transition is stored with who and when. Evidence: a full-frame image with the box drawn, plus a cropped region, saved when the incident opens. Alerts (log, webhook, email) go through an async queue with rate limiting and delivery status, so a slow channel never blocks the video loop.

## 9. Database schema

- `cameras` (id, name, source, enabled, config_json)
- `zones` (id, camera_id, name, polygon_json, rule_ids)
- `rule_configs` (id, camera_id, rule_id, params_json, enabled)
- `incidents` (id, camera_id, rule_id, track_id, status, opened_at, closed_at, severity)
- `incident_evidence` (id, incident_id, path, kind)
- `incident_events` (id, incident_id, from_status, to_status, note, at)
- `alert_deliveries` (id, incident_id, channel, status, attempts, last_error)
- `feedback_labels` (id, incident_id, label, at)
- `model_versions` (id, name, architecture, format, mlflow_run_id, dataset_version, metrics_json, is_active)
- `detection_runs`, `jobs` (as in the original design)

## 10. API (v1)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and active model |
| POST | `/api/v1/detect/image` | Image detection |
| POST | `/api/v1/detect/video` | Create video job |
| GET | `/api/v1/jobs/{id}` | Job status |
| WS | `/api/v1/ws/camera/{camera_id}` | Frames in; detections, tracks, incidents out |
| GET/POST | `/api/v1/cameras` | Camera registry |
| GET/PUT | `/api/v1/cameras/{id}/zones` | Zone configuration |
| GET/PUT | `/api/v1/cameras/{id}/rules` | Rule thresholds |
| GET | `/api/v1/incidents` | Filter by camera, rule, status, date |
| GET | `/api/v1/incidents/{id}` | Detail with evidence |
| PATCH | `/api/v1/incidents/{id}` | Change status or mark false positive |
| GET | `/api/v1/stats` | Violation counts and trends |
| GET | `/api/v1/models` | Versions and metrics |
| POST | `/api/v1/models/{id}/activate` | Switch active model |
| GET | `/api/v1/experiments` | MLflow comparison data |

## 11. Configuration (example)

```yaml
camera_defaults:
  conf_threshold: 0.35
  ppe:
    head_region_fraction: 0.25
    vest_overlap_threshold: 0.5
    min_person_height_px: 80
    edge_margin_px: 5
  temporal:
    open_after_frames: 8
    close_after_frames: 30
    track_lost_seconds: 3
  proximity:
    warn_ratio: 1.0
```

The values above are placeholders. Real values come from validation data and are recorded with the experiment that chose them.

## 12. Evaluation plan

**Data splits:** see D13. Tune only on our carved validation set. The test set (official val images, leakage-checked) is used once per finished model. Real clips are split into `tune` and `final` subsets; thresholds are chosen on `tune` and reported on `final`.

**Detector (offline, held-out test split):** precision, recall, mAP@50, mAP@50-95, F1, confusion matrix, per-class AP, latency (p50/p95), FPS, model size. Compare YOLO n/s/m with fixed seed and dataset version. Ablations: image size, augmentation, frozen vs unfrozen backbone.

**Error analysis:** small objects (by box area), low light (by image brightness), occlusion (proxy: overlap between ground-truth boxes, labelled as a proxy), false positives and negatives with example images.

**System-level (replay evaluation):** a set of short clips labelled by hand with true incident intervals. Run the full pipeline and score at incident level.
- True positive: a predicted incident overlaps a true interval.
- False positive: a predicted incident with no true interval.
- False negative: a true interval with no predicted incident.
- Also report: duplicate incidents per true incident, detection delay.
The labelled set is small and will be described as small. No claim is made beyond what it supports.

**Human feedback:** incidents marked `false_positive` in the UI feed the false positive count, with the caveat that unreviewed incidents are not assumed correct.

## 13. Performance plan

Measure image latency, video throughput, webcam FPS, CPU use, memory, and model load time. Compare PyTorch, ONNX Runtime and OpenVINO on the same machine with warm-up runs, fixed inputs, and p50/p95 over N runs. Report the hardware beside every number. No improvement is claimed without a before/after table.

## 14. Security and privacy

API-key or JWT auth on the API and dashboard; upload size and type limits; evidence retention period with automatic deletion; optional face blurring on evidence (stretch); no face recognition or identity inference. Camera and incident data are treated as sensitive in logs.

## 15. Testing strategy

- Unit: each rule, association, zone geometry, temporal state machine, config validation, dataset validator.
- Component: detector loading and inference, tracker wrapper on a short clip.
- API: endpoints, invalid image, corrupted video, unsupported format, oversized upload.
- Integration: video through detector, tracker, rules and incidents on a fixture clip.
- Rule tests use fabricated `FrameContext` objects, so they need no model.

## 16. Milestones and done criteria

| # | Milestone | Done when |
|---|---|---|
| M0 | Skeleton, Detector, git hygiene | Done |
| M1 | Dataset validator, PPE dataset chosen and licence checked, data validated, analysed and split | Validator passes on the real dataset; `docs/dataset_analysis.md` generated from it; split built with a recorded `dataset_version`; licence in `datasets/README.md`; source zip hash recorded |
| M2 | Training pipeline + MLflow (cloud GPU) | Baseline run reproducible from a config file |
| M3 | Evaluation and n/s/m comparison | Table and plots generated from real runs |
| M4 | Error analysis | Report with examples per failure type |
| M5 | Inference pipelines, ONNX/OpenVINO export | Outputs match PyTorch within tolerance |
| M6 | Tracking and temporal state | Persistent IDs on a test clip; state machine tests pass |
| M7 | Safety engine (association, rules) | All rule tests pass; `uncertain` cases covered |
| M8 | Incident management, evidence, alerts | Dedup verified; lifecycle tested |
| M9 | FastAPI and workers | API tests pass including bad-input cases |
| M10 | React dashboards | All pages work against the real API |
| M11 | Replay evaluation | Incident-level metrics on labelled clips |
| M12 | Benchmarks, Docker, CI, docs, demo video | `docker compose up` works from a clean clone; CI green |

M7 can proceed in parallel with M2 because rules run on fabricated detections.

## 17. Risks

| Risk | Mitigation |
|---|---|
| No local GPU | Cloud GPU for training only; CPU for everything else |
| PPE dataset quality or licence problems | Validate and check licence before training; document provenance |
| Slow CPU inference | Start with the nano model; measure; ONNX Runtime and OpenVINO in M5 |
| Training data is small and partly generated | Report this plainly; use real-clip replay evaluation; do not over-claim |
| Near-duplicate or leaked images inflate scores | Hash-based clustering and leakage removal (D13); re-check after any new data source |
| New pretrained model names (for example `yolo26n.pt`) may not load with the installed Ultralytics version | `scripts/check_models.py` before M2; compare only what loads |
| Tracker ID switches cause duplicate incidents | Measure in replay evaluation; document |
| Small evaluation set | State size and limits plainly |
| Scope too large | Follow scope tiers; cut Stretch first, then Extended |

## 18. Known limitations (to repeat in the README)

- Image-space proximity is not physical distance and does not predict collisions.
- Violations are inferred from absent PPE detections; occlusion and small people can cause misses.
- Performance depends on camera angle, lighting and the training data; results do not transfer automatically to new sites.
- The only vehicle class in the training data is `forklift`; cars, trucks and other vehicles are not detected by the fine-tuned model.
- Part of the training data is generated imagery, and generated images cannot be separated from real ones, so dataset metrics may overstate real-world performance.
- Dataset software licence: the dataset is CC BY 4.0, but the Ultralytics code is AGPL-3.0. That is fine for a portfolio; commercial use needs a licence review.
- Person-in-forklift (operator) detection is image-space: someone standing in front of a forklift can be mistaken for its operator and skipped by the proximity rule.
- Safety thresholds (head region, minimum person size, warn ratio) are placeholders until tuned on validation detections and real clips.
- The system supports human review. It is not a replacement for safety procedures.
