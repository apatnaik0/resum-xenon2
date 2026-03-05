#!/usr/bin/env python3
"""Split processed Xenon CSV data into training/validation sets.

Expected input from process_xenon2.py:
- <processed_root>/lf/sim_X{X}_Y{Y}_task0.csv
- <processed_root>/hf/sim_X{X}_Y{Y}_ALL.csv

Output layout:
- <dataset_root>/training/{lf,hf}
- <dataset_root>/validation/{lf,hf}
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


XY_PATTERN = re.compile(r"X(\d+)_Y(\d+)")


def parse_xy(name: str) -> Optional[Tuple[int, int]]:
    m = XY_PATTERN.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def evenly_spaced_indices(total: int, count: int) -> List[int]:
    """Return deterministic evenly-spaced indices in [0, total)."""
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))

    indices = sorted({int(i * total / count) for i in range(count)})

    # Ensure exact count by filling any gaps from left to right.
    if len(indices) < count:
        existing = set(indices)
        for idx in range(total):
            if idx not in existing:
                indices.append(idx)
                existing.add(idx)
                if len(indices) == count:
                    break
        indices.sort()
    return indices


def collect_sorted_csvs(folder: Path) -> List[Path]:
    files = [p for p in folder.glob("*.csv") if p.is_file()]
    return sorted(files, key=lambda p: (parse_xy(p.name) is None, parse_xy(p.name), p.name))


def clear_csv_h5(folder: Path) -> None:
    if not folder.exists():
        return
    for pattern in ("*.csv", "*.h5"):
        for file in folder.glob(pattern):
            if file.is_file():
                file.unlink()


def copy_selected(files: Sequence[Path], selected_indices: set[int], selected_dir: Path, other_dir: Path) -> Tuple[int, int]:
    selected_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)

    selected_count = 0
    other_count = 0

    for i, file in enumerate(files):
        if i in selected_indices:
            shutil.copy2(file, selected_dir / file.name)
            selected_count += 1
        else:
            shutil.copy2(file, other_dir / file.name)
            other_count += 1

    return selected_count, other_count


def split_hf(
    src_hf_dir: Path,
    train_hf_dir: Path,
    val_hf_dir: Path,
    hf_train_count: int,
) -> None:
    files = collect_sorted_csvs(src_hf_dir)
    total = len(files)
    train_indices = set(evenly_spaced_indices(total, hf_train_count))

    train_count, val_count = copy_selected(files, train_indices, train_hf_dir, val_hf_dir)

    print("[HF]")
    print(f"  source files : {total}")
    print(f"  train files  : {train_count}")
    print(f"  val files    : {val_count}")


def split_lf(
    src_lf_dir: Path,
    train_lf_dir: Path,
    val_lf_dir: Path,
    lf_val_ratio: float,
) -> None:
    files = collect_sorted_csvs(src_lf_dir)
    total = len(files)

    raw_val = int(round(total * lf_val_ratio))
    val_count_target = min(total, max(1 if total > 0 else 0, raw_val))
    val_indices = set(evenly_spaced_indices(total, val_count_target))

    val_count, train_count = copy_selected(files, val_indices, val_lf_dir, train_lf_dir)

    print("[LF]")
    print(f"  source files : {total}")
    print(f"  train files  : {train_count}")
    print(f"  val files    : {val_count}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Split processed Xenon CSV data")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=repo_root / "data" / "processed" / "temp_new_data",
        help="Processed CSV root containing lf/ and hf/",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "data" / "processed" / "new_both",
        help="Output dataset root for training/validation folders",
    )
    parser.add_argument(
        "--hf-train-count",
        type=int,
        default=10,
        help="Number of HF files placed in training (evenly spaced)",
    )
    parser.add_argument(
        "--lf-val-ratio",
        type=float,
        default=0.10,
        help="Fraction of LF files placed in validation (evenly spaced)",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove existing CSV/H5 files from output train/val dirs before copying",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    src_lf = args.processed_root / "lf"
    src_hf = args.processed_root / "hf"

    if not src_lf.exists():
        raise FileNotFoundError(f"LF source directory not found: {src_lf}")
    if not src_hf.exists():
        raise FileNotFoundError(f"HF source directory not found: {src_hf}")

    train_lf = args.dataset_root / "training" / "lf"
    train_hf = args.dataset_root / "training" / "hf"
    val_lf = args.dataset_root / "validation" / "lf"
    val_hf = args.dataset_root / "validation" / "hf"

    if args.clean_output:
        for folder in (train_lf, train_hf, val_lf, val_hf):
            clear_csv_h5(folder)

    print("=" * 72)
    print("Split processed data")
    print(f"processed_root: {args.processed_root}")
    print(f"dataset_root  : {args.dataset_root}")
    print(f"hf_train_count: {args.hf_train_count}")
    print(f"lf_val_ratio  : {args.lf_val_ratio}")
    print(f"clean_output  : {args.clean_output}")
    print("=" * 72)

    split_hf(src_hf, train_hf, val_hf, args.hf_train_count)
    split_lf(src_lf, train_lf, val_lf, args.lf_val_ratio)

    print("Done.")


if __name__ == "__main__":
    main()
