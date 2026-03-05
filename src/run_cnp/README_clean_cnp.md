# Clean CNP Pipeline (No `resum` Dependency)

Files added:
- `src/run_cnp/cnp_clean_pipeline.py`
- `src/run_cnp/cnp_clean_workflow.ipynb`
- `src/run_cnp/preprocess_mixup_xenon2.py` (clean rewrite)

## What it does
- Reads `settings2.yaml` paths and headers
- Trains a self-contained deterministic CNP on H5 data
- Saves model + training history CSV + training plots
- Runs prediction on configured folders
- Exports CSV compatible with MFGP usage (`y_cnp`, `y_cnp_err`, `y_raw`, `fidelity`, `iteration`)
- Saves prediction heatmaps

## CLI usage
Run mixup preprocessing first:
```bash
python src/run_cnp/preprocess_mixup_xenon2.py --config src/xenon/settings2.yaml
```

Then train and predict:
```bash
python src/run_cnp/cnp_clean_pipeline.py --config src/xenon/settings2.yaml train --steps-per-epoch 5000 --monitor-every 1000
python src/run_cnp/cnp_clean_pipeline.py --config src/xenon/settings2.yaml predict --model-path src/xenon/out/cnp/cnp_v101.0_model_15epochs.pth
```

or end-to-end:
```bash
python src/run_cnp/cnp_clean_pipeline.py --config src/xenon/settings2.yaml full --steps-per-epoch 5000 --monitor-every 1000
```

## Notebook usage
Open `src/run_cnp/cnp_clean_workflow.ipynb` and run cells in order.

## Notes
- This pipeline deliberately avoids importing from `resum`.
- If `use_data_augmentation: mixup` is set in `settings2.yaml`, training reads `phi_mixedup` / `target_mixedup` (with fallback to base datasets per-file if mixup arrays are empty).
