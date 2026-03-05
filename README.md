# resum_xenon_clean_template

Clean, reproducible Xenon pipeline template for preprocessing, CNP, and MF-GP.
This template does **not** require `resum` helper functions for the clean CNP/MFGP paths.

## 1) Folder Structure

```text
resum_xenon_clean_template/
├── process_xenon2.py
├── split_data.py
├── convert_csv_to_h5_xenon2.py
├── requirements.txt
├── src/
│   ├── xenon/
│   │   └── settings2.yaml
│   ├── run_cnp/
│   │   ├── cnp_clean_pipeline.py
│   │   ├── cnp_clean_workflow.ipynb
│   │   ├── cnp_clean_predict_only.ipynb
│   │   ├── cnp_predict_per_signal.py
│   │   └── preprocess_mixup_xenon2.py
│   └── run_mfgp/
│       ├── mfgp_clean_pipeline.py
│       └── mfgp_clean_workflow.ipynb
└── data/
    ├── raw/
    │   ├── ScintillatorLF/
    │   ├── TPCLF/
    │   ├── TPCHF/
    │   └── ScintillatorHF/
    ├── processed/
    │   ├── temp_new_data/
    │   │   ├── lf/
    │   │   └── hf/
    │   └── new_both/
    │       ├── training/
    │       │   ├── lf/
    │       │   └── hf/
    │       └── validation/
    │           ├── lf/
    │           └── hf/
    └── out/
        ├── cnp/
        ├── mfgp/
        └── pce/
```

Notes:
- Leaf data folders contain `.gitignore` so folder structure stays in Git while data/artifacts remain untracked.
- No data is committed in this template.

## 2) Install

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Path Config

Main config file:
- `src/xenon/settings2.yaml`

Defaults are already repository-relative and point into `data/...`.
If your data is outside this repo, replace those paths with your absolute paths.

## 4) Preprocessing Run Order

Run from repo root:

```bash
python process_xenon2.py
python split_data.py
python convert_csv_to_h5_xenon2.py
```

Default behavior:
- reads raw CSVs from `data/raw/...`
- writes merged temp CSVs to `data/processed/temp_new_data/{lf,hf}`
- writes split train/val CSVs to `data/processed/new_both/...`
- writes `.h5` files alongside split CSV files

Optional mixup preprocessing:

```bash
python src/run_cnp/preprocess_mixup_xenon2.py --config src/xenon/settings2.yaml
```

## 5) CNP

### Train + Predict via script

```bash
python src/run_cnp/cnp_clean_pipeline.py --config src/xenon/settings2.yaml full
```

### Notebooks
- `src/run_cnp/cnp_clean_workflow.ipynb` (train + predict)
- `src/run_cnp/cnp_clean_predict_only.ipynb` (predict with existing model)

### Per-event (per-signal row) export

```bash
python src/run_cnp/cnp_predict_per_signal.py --config src/xenon/settings2.yaml
```

## 6) MF-GP

### Script

```bash
python src/run_mfgp/mfgp_clean_pipeline.py --config src/xenon/settings2.yaml
```

### Notebook
- `src/run_mfgp/mfgp_clean_workflow.ipynb`

Outputs go to:
- `data/out/mfgp`

## 7) Recommended Workflow

1. Preprocess raw -> split -> H5 conversion
2. (Optional) mixup preprocessing
3. CNP train + prediction export
4. MF-GP training/inference on CNP outputs

## 8) Validation vs Training Data for MF-GP

Typical usage:
- Fit MF-GP with CNP output containing **training LF + training HF**
- Run MF-GP inference/evaluation on **validation LF** (and validation HF if available)
