#!/usr/bin/env python3
"""Convert Xenon2 split CSV files to HDF5 files expected by the CNP pipeline.

Creates datasets:
- theta, theta_headers
- phi, phi_labels
- target, target_headers
- weights, weights_labels
- fidelity
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd

THETA_COLS = ["scint_x", "scint_y"]
PHI_COLS = ["initial_m_x", "initial_m_y", "initial_m_z"]
TARGET_COLS = ["tag_final"]
WEIGHTS_COL = "weights"


def to_bytes_array(items: List[str]) -> np.ndarray:
    return np.array(items, dtype="S")


def read_csv_checked(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = THETA_COLS + PHI_COLS + TARGET_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing required columns {missing}")
    if df.empty:
        raise ValueError(f"{csv_path.name}: CSV is empty")
    return df


def build_arrays(df: pd.DataFrame, fidelity_value: float) -> Tuple[np.ndarray, ...]:
    theta_values = df[THETA_COLS].iloc[0].to_numpy(dtype=np.float64)

    phi_values = df[PHI_COLS].to_numpy(dtype=np.float32)
    target_values = df[TARGET_COLS].to_numpy(dtype=np.int8)

    if WEIGHTS_COL in df.columns:
        weights_values = df[[WEIGHTS_COL]].to_numpy(dtype=np.float32)
    else:
        weights_values = np.ones((len(df), 1), dtype=np.float32)

    fidelity_values = np.full((len(df), 1), float(fidelity_value), dtype=np.float32)

    return theta_values, phi_values, target_values, weights_values, fidelity_values


def write_h5(
    h5_path: Path,
    theta: np.ndarray,
    phi: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    fidelity: np.ndarray,
) -> None:
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("theta", data=theta, compression="gzip")
        f.create_dataset("theta_headers", data=to_bytes_array(THETA_COLS), compression="gzip")

        f.create_dataset("phi", data=phi, compression="gzip")
        f.create_dataset("phi_labels", data=to_bytes_array(PHI_COLS), compression="gzip")

        f.create_dataset("target", data=target, compression="gzip")
        f.create_dataset("target_headers", data=to_bytes_array(TARGET_COLS), compression="gzip")

        f.create_dataset("weights", data=weights, compression="gzip")
        f.create_dataset("weights_labels", data=to_bytes_array([WEIGHTS_COL]), compression="gzip")

        f.create_dataset("fidelity", data=fidelity, compression="gzip")


def convert_one(csv_path: Path, h5_path: Path, fidelity_value: float, force: bool) -> str:
    if h5_path.exists() and not force and h5_path.stat().st_mtime >= csv_path.stat().st_mtime:
        return f"skip {csv_path.name} (up-to-date)"

    df = read_csv_checked(csv_path)
    theta, phi, target, weights, fidelity = build_arrays(df, fidelity_value)
    write_h5(h5_path, theta, phi, target, weights, fidelity)

    return f"ok   {csv_path.name} -> {h5_path.name}"


def convert_directory(
    directory: Path,
    fidelity_value: float,
    workers: int,
    force: bool,
) -> Tuple[int, int]:
    if not directory.exists():
        print(f"[warn] missing directory: {directory}")
        return 0, 0

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        print(f"[warn] no CSV files found: {directory}")
        return 0, 0

    print(f"[{directory}] converting {len(csv_files)} files (fidelity={fidelity_value})")

    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_map = {
            ex.submit(convert_one, csv_file, csv_file.with_suffix(".h5"), fidelity_value, force): csv_file
            for csv_file in csv_files
        }
        for idx, fut in enumerate(as_completed(future_map), start=1):
            csv_file = future_map[fut]
            try:
                msg = fut.result()
                success += 1
                if idx <= 5 or idx % 100 == 0:
                    print(f"  {msg}")
            except Exception as exc:
                failed += 1
                print(f"  fail {csv_file.name}: {exc}")

    return success, failed


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Convert split Xenon2 CSV files to HDF5")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "data" / "processed" / "new_both",
        help="Dataset root containing training/ and validation/ folders",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild H5 even when output appears up-to-date",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root

    layout: List[Tuple[Path, float]] = [
        (root / "training" / "lf", 0.0),
        (root / "training" / "hf", 1.0),
        (root / "validation" / "lf", 0.0),
        (root / "validation" / "hf", 1.0),
    ]

    print("=" * 72)
    print("CSV -> HDF5 conversion (Xenon2)")
    print(f"dataset_root: {root}")
    print(f"workers     : {args.workers}")
    print(f"force       : {args.force}")
    print("=" * 72)

    total_success = 0
    total_failed = 0
    for folder, fidelity in layout:
        ok, fail = convert_directory(folder, fidelity, workers=args.workers, force=args.force)
        total_success += ok
        total_failed += fail

    print("=" * 72)
    print(f"finished: success={total_success}, failed={total_failed}")
    if total_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
