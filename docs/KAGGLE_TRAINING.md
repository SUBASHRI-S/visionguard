# Training on a cloud GPU (Kaggle first, Colab as backup)

Your laptop has no NVIDIA GPU, so real training runs on a free cloud GPU. The same config file and the same `dataset_version` are used everywhere, so results are reproducible.

> Status: these steps were used for the two completed Kaggle runs (`n_core_baseline`, `n_core_imgsz960`, Tesla T4). If something differs on your next run, fix this page and note it.

## Why this is safe

1. The raw dataset is rebuilt on Kaggle from the zip with `scripts.rebuild_dataset`. It replays the split and refuses to continue unless the result has exactly the committed `dataset_version`.
2. `scripts.train` refuses to start if the prepared dataset is not the version named in the config.
3. Training writes plain files. Logging to MLflow is a separate step, so a finished run is never lost.

## One-time Kaggle setup

1. Create a Kaggle account and verify your phone number (needed to enable GPU and internet).
2. Download the dataset zip on your PC (see `datasets/README.md`).
3. Kaggle > Datasets > New Dataset. Upload `construction-ppe.zip`. Keep it **Private**. Name it `construction-ppe`. (Licence CC BY 4.0 allows this; keep the attribution in the dataset description.)
4. Kaggle > Code > New Notebook. In the right panel: Accelerator = a GPU, Internet = On, and Add Input = your `construction-ppe` dataset.
5. Look at the weekly GPU quota Kaggle shows and write it down. Do not rely on a number from memory.

## Notebook cells

Cell 1, get the code and install:

```
!git clone https://github.com/SUBASHRI-S/visionguard.git
%cd visionguard
!pip install -q ultralytics==8.4.216 mlflow==3.18.0
```

Cell 2, unzip the dataset (check the real path with `!ls /kaggle/input` if it differs):

```
!mkdir -p datasets/raw
!unzip -q /kaggle/input/construction-ppe/construction-ppe.zip -d datasets/raw
!ls datasets/raw
```

The training code expects `datasets/raw/construction-ppe/images` and `labels`. If unzip created a different folder name, rename it to `construction-ppe`.

Cell 3, rebuild the exact dataset version and check it:

```
!python -m scripts.rebuild_dataset --version d1dc2ab4d1a9
```

It must print `rebuilt d1dc2ab4d1a9 (core)`. Any error means the data is not the original; stop and investigate.

Cell 4, train (`--no-mlflow` because the notebook is temporary; you import the run later):

```
!python -m scripts.train --config experiments/configs/n_core_baseline.yaml --no-mlflow
```

Cell 5, package the result so it survives the session:

```
!cd runs && zip -qr /kaggle/working/n_core_baseline.zip n_core_baseline
```

Download `n_core_baseline.zip` from the notebook Output tab.

## Import the run on your PC

Unzip into `runs\n_core_baseline` inside your project, then:

```
python -m scripts.train --log-only runs\n_core_baseline
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000. Run the `smoke` config first on your PC to confirm everything works before touching Kaggle.

## Rules for honest results

- Metrics logged here are on the **validation** split. The test split is untouched until M3.
- Weights (`*.pt`) are not committed. Keep them in a GitHub Release or locally.
- Change one thing per run, give each run a new name, never overwrite a run.

## Experiment 3: `s_core_imgsz960` (YOLO11s at 960)

One change from `n_core_imgsz960`: `yolo11n.pt` becomes `yolo11s.pt`. Config: `experiments/configs/s_core_imgsz960.yaml`. A ready notebook is `notebooks/kaggle_s_core_imgsz960.ipynb` (Kaggle > New Notebook > File > Import Notebook), or paste the same cells by hand: replace `n_core_baseline` with `s_core_imgsz960` in cells 4 and 5 above.

Things to know before reading the result:
- The training set is only 638 images, so a bigger model may overfit instead of helping. That is a valid result and goes in `docs/EXPERIMENTS.md` either way.
- One seed per run. Validation and test differences of about 0.02 mAP are within what a seed change can do. If the gain is that small, repeat both the 11n and 11s runs with `seed: 43` before claiming anything.
- Choose the run on **validation**. The test split is evaluated once (`scripts/evaluate.py` has a guard). Then run the new weights on the five clips in `docs/HELDOUT_V2.md`, once, with settings fixed beforehand.
- Out-of-memory: do not change the config. Copy it to a new name with `batch: 8` and record that as a separate run.
