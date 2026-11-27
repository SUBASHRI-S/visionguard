# VisionGuard

[![CI](https://github.com/SUBASHRI-S/visionguard/actions/workflows/ci.yml/badge.svg)](https://github.com/SUBASHRI-S/visionguard/actions/workflows/ci.yml)

VisionGuard is a computer vision system for workplace safety. It detects people, helmets, vests and vehicles in camera feeds, tracks people over time, and raises reviewable incidents when someone is missing protective equipment, enters a restricted zone, or comes close to a vehicle. Every incident keeps evidence, and the system's accuracy is measured and reported honestly.

> **Project status: in development.** See the status table in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). This README only describes what is actually built and tested.

## What works today

- Pretrained YOLO detector wrapper with input validation (`inference/detector.py`), covered by automated tests
- Dataset validator for YOLO-format data (`datasets/validate.py`)
- Dataset tooling: analysis, leakage-safe train/val/test split, class profiles and a versioned manifest (`datasets/`), tested on synthetic data
- Schema and validation for hand-labelled real-footage clips (`evaluation/clip_labels.py`)
- Training pipeline: YAML configs, dataset-version check, MLflow logging, dataset rebuild for cloud notebooks (`training/`, `scripts/train.py`). Plumbing verified on a tiny synthetic dataset; real baseline results are not available yet

## Planned capabilities

- Fine-tuned PPE detection model with tracked experiments (MLflow)
- Person tracking with persistent IDs (ByteTrack)
- Rule engine: no helmet, no vest, restricted-zone intrusion, person-vehicle proximity
- Incident history with evidence images, status management and alerts
- FastAPI backend and React dashboard
- Docker deployment and CI/CD

## Setup (Windows, PowerShell)

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-lock.txt
python -m pip install -r requirements-train.txt
python -m pip install -r requirements-tracking.txt
python -m pytest -q
```

## Preparing the dataset

```
python -m scripts.analyze_dataset
python -m scripts.make_split --profile core --source-sha256 $hash
```

Full steps, licence and attribution: [datasets/README.md](datasets/README.md).

## Run on a video

```powershell
python -m scripts.run_video data/clips/small/my_clip.mp4 --run runs/n_core_imgsz960
```

Add `--db sqlite:///outputs/visionguard.db` to also save every incident (with its status history and evidence paths) to a database; this needs `python -m pip install -r requirements-backend.txt`. Outputs go to `outputs/video_runs/<clip name>/`: an annotated video, `incidents.jsonl`, evidence frames, `per_frame.csv`, `summary.json` and a `report.md` that states what the run is and is not. Inference runs at the size the model was trained at (960). Timing is in seconds (`--open-after-seconds 3`, `--close-after-seconds 2`), and `--memory-seconds 5` stops a vest or helmet that was seen on a person moments ago from being reported missing when the detector briefly misses it. Add `--zones file.yaml` to say where PPE is required (`work_area`) and to define restricted zones; see `experiments/video/work_area_67460.yaml`. Results on one clip illustrate behaviour; they are not an accuracy measurement.

## Run the API

```powershell
python -m pip install -r requirements-backend.txt
python -m uvicorn backend.main:create_app --factory --port 8000
```

Open http://127.0.0.1:8000/docs for the interactive API page. Settings are environment variables:
`VG_DB_URL` (default `sqlite:///outputs/visionguard.db`), `VG_EVIDENCE_ROOT` (default `outputs`),
`VG_API_KEY` (optional; if set, send it as the `X-API-Key` header on `/api` calls) and
`VG_CORS_ORIGINS` (browser addresses allowed to call the API).

## Run the dashboard

Needs Node 20.19+ or 22.12+. With the API running (previous section), in a second terminal:

```powershell
cd frontend
npm install
npm test
npm run dev
```

Open http://localhost:5173. The list on the left is the review queue; pick an incident to see where it sits on the video timeline, its saved pictures and its history, and to acknowledge it, resolve it or mark it as a false alarm (your name is recorded). **Connection** sets the API address and key. Start the API from the repository root: saved picture paths are relative to where the video run was started.

## Run everything with Docker

```powershell
docker compose up --build
```

Dashboard: http://localhost:8080. API docs: http://localhost:8000/docs. The database and pictures are kept in `./outputs` on your machine. To process a video and see its incidents in the dashboard, run it as before with `--db sqlite:///outputs/visionguard.db`; the API container reads the same folder. The API image does not include PyTorch (the API only serves stored incidents), so it is small. To require a key, uncomment `VG_API_KEY` in `docker-compose.yml` and enter the same key under Connection in the dashboard.

## Continuous integration

`.github/workflows/ci.yml` runs on every push: the Python test suite (Python 3.13, CPU-only PyTorch, versions pinned in `requirements-ci.txt`), the dashboard's type check, tests and build, and a build of both Docker images.

## Documentation

- [Architecture and engineering plan](docs/ARCHITECTURE.md)
- [Roadmap: what is done, what is next](docs/ROADMAP.md)
- [Dataset card](datasets/README.md)
- [Real-footage protocol](docs/REAL_FOOTAGE_PROTOCOL.md)

## Limitations

- Image-space proximity is an approximation, not physical distance or collision prediction.
- No performance numbers are published yet. They will appear only after they are measured.
- The training dataset is small and partly generated, so dataset scores may overstate real-world performance. A hand-labelled set of real clips is used as the honest check.
- Only forklifts are covered as vehicles.

## Dataset and licensing

- Dataset: Ultralytics Construction-PPE (November 2026 release), licensed CC BY 4.0.
- Attribution: Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0. https://docs.ultralytics.com/datasets/detect/construction-ppe
- Changes made here: new train/val/test splits, removal of near-duplicate images from train, and optional class remapping (see the dataset card).
- Software: the Ultralytics library is AGPL-3.0. That is fine for this portfolio project; commercial use would need a licence review.
