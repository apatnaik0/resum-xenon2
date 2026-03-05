#!/usr/bin/env python3
"""Process ReSUM2 Xenon raw CSV data into unified LF/HF processed CSV files.

Pipeline overview:
1) Read raw files from ReSUM2 directories.
2) Convert each event into columns:
   eventid, scint_x, scint_y, initial_m_x, initial_m_y, initial_m_z, tag_final
3) Merge LF inputs (ScintillatorLF + TPCLF) by (x, y).
4) Merge HF inputs (ScintillatorHF + TPCHF) by (x, y).

Output:
- <output_root>/lf/sim_X{X}_Y{Y}_task0.csv
- <output_root>/hf/sim_X{X}_Y{Y}_ALL.csv
"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


Coordinate = Tuple[int, int]

# Input filename/folder coordinate parsers.
SIM_TASK_PATTERN = re.compile(r"sim_X(\d+)_Y(\d+)_task\d+\.csv$")
HF_FILE_PATTERN = re.compile(r"HFX(\d+)_?Y(\d+)\.csv$")
HF_DIR_NEW_PATTERN = re.compile(r"X(\d+)_Y(\d+)$")
HF_DIR_OLD_PATTERN = re.compile(r"ScintorHFX(\d+)Y(\d+)$")

REQUIRED_COLUMNS = [
    "eventid",
    "initial_m_x",
    "initial_m_y",
    "initial_m_z",
    "second_m_x",
    "third_m_x",
]


def parse_sim_task_xy(name: str) -> Optional[Coordinate]:
    m = SIM_TASK_PATTERN.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_hf_file_xy(name: str) -> Optional[Coordinate]:
    m = HF_FILE_PATTERN.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_hf_dir_xy(name: str) -> Optional[Coordinate]:
    m = HF_DIR_NEW_PATTERN.search(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = HF_DIR_OLD_PATTERN.search(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def read_needed_columns(path: Path) -> pd.DataFrame:
    """Read only needed columns with a safe fallback if parser options fail."""
    dtype_map = {
        "eventid": "int64",
        "initial_m_x": "float32",
        "initial_m_y": "float32",
        "initial_m_z": "float32",
        "second_m_x": "float32",
        "third_m_x": "float32",
    }
    usecols = lambda c: c in REQUIRED_COLUMNS

    try:
        df = pd.read_csv(path, usecols=usecols, dtype=dtype_map, engine="pyarrow")
    except Exception:
        df = pd.read_csv(path, usecols=usecols, dtype=dtype_map)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")
    return df


def event_level_transform(df: pd.DataFrame, xy: Coordinate) -> pd.DataFrame:
    """Create event-level rows and `tag_final`.

    `tag_final = 1` when an event has any row where both second_m_x and third_m_x
    are non-null, otherwise 0.
    """
    scint_x, scint_y = xy

    if df.empty:
        return pd.DataFrame(
            columns=[
                "eventid",
                "scint_x",
                "scint_y",
                "initial_m_x",
                "initial_m_y",
                "initial_m_z",
                "tag_final",
            ]
        )

    both_hit = (df["second_m_x"].notna() & df["third_m_x"].notna()).astype("int8")
    work = df[["eventid", "initial_m_x", "initial_m_y", "initial_m_z"]].copy()
    work["_both_hit"] = both_hit

    grouped = (
        work.groupby("eventid", as_index=False)
        .agg(
            {
                "initial_m_x": "first",
                "initial_m_y": "first",
                "initial_m_z": "first",
                "_both_hit": "max",
            }
        )
        .rename(columns={"_both_hit": "tag_final"})
    )

    grouped.insert(1, "scint_x", int(scint_x))
    grouped.insert(2, "scint_y", int(scint_y))
    grouped["tag_final"] = grouped["tag_final"].astype("int8")

    return grouped[
        [
            "eventid",
            "scint_x",
            "scint_y",
            "initial_m_x",
            "initial_m_y",
            "initial_m_z",
            "tag_final",
        ]
    ]


def process_one_file(path: Path, xy: Coordinate) -> pd.DataFrame:
    df = read_needed_columns(path)
    return event_level_transform(df, xy)


def merge_frames(frames: List[pd.DataFrame], shuffle: bool, seed: Optional[int]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if shuffle and not out.empty:
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def _worker_process(args: Tuple[str, int, int]) -> Tuple[Coordinate, pd.DataFrame, str]:
    path_str, x, y = args
    path = Path(path_str)
    out = process_one_file(path, (x, y))
    return (x, y), out, path.name


def process_grouped_jobs(
    jobs: Iterable[Tuple[Path, Coordinate]],
    max_workers: int,
    label: str,
) -> Dict[Coordinate, List[pd.DataFrame]]:
    grouped: Dict[Coordinate, List[pd.DataFrame]] = {}
    job_args = [(str(path), xy[0], xy[1]) for path, xy in jobs]

    if not job_args:
        print(f"[{label}] no jobs found")
        return grouped

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_worker_process, arg) for arg in job_args]
        for idx, fut in enumerate(as_completed(futures), start=1):
            xy, frame, name = fut.result()
            grouped.setdefault(xy, []).append(frame)
            if idx <= 5 or idx % 50 == 0:
                print(f"[{label}] {idx}/{len(job_args)} {name} -> {len(frame)} rows")

    return grouped


def collect_lf_jobs(raw_root: Path) -> List[Tuple[Path, Coordinate]]:
    jobs: List[Tuple[Path, Coordinate]] = []
    for subdir in ("ScintillatorLF", "TPCLF"):
        folder = raw_root / subdir
        if not folder.exists():
            print(f"[LF] missing folder: {folder}")
            continue
        for path in sorted(folder.glob("*.csv")):
            xy = parse_sim_task_xy(path.name)
            if xy is None:
                print(f"[LF] skip unmatched filename: {path.name}")
                continue
            jobs.append((path, xy))
    return jobs


def collect_hf_jobs(raw_root: Path) -> List[Tuple[Path, Coordinate]]:
    jobs: List[Tuple[Path, Coordinate]] = []

    # TPCHF files (flat directory).
    tpchf_dir = raw_root / "TPCHF"
    if tpchf_dir.exists():
        for path in sorted(tpchf_dir.glob("*.csv")):
            xy = parse_hf_file_xy(path.name)
            if xy is None:
                print(f"[HF] skip unmatched TPCHF filename: {path.name}")
                continue
            jobs.append((path, xy))
    else:
        print(f"[HF] missing folder: {tpchf_dir}")

    # ScintillatorHF files (subdirectories).
    scint_hf_root = raw_root / "ScintillatorHF"
    if scint_hf_root.exists():
        for sub in sorted(scint_hf_root.iterdir()):
            if not sub.is_dir():
                continue
            xy = parse_hf_dir_xy(sub.name)
            if xy is None:
                print(f"[HF] skip unmatched ScintillatorHF folder: {sub.name}")
                continue
            for path in sorted(sub.glob("*.csv")):
                jobs.append((path, xy))
    else:
        print(f"[HF] missing folder: {scint_hf_root}")

    return jobs


def write_grouped_outputs(
    grouped: Dict[Coordinate, List[pd.DataFrame]],
    out_dir: Path,
    name_template: str,
    shuffle: bool,
    seed: Optional[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (x, y) in sorted(grouped):
        merged = merge_frames(grouped[(x, y)], shuffle=shuffle, seed=seed)
        out_name = name_template.format(x=x, y=y)
        out_path = out_dir / out_name
        merged.to_csv(out_path, index=False)

        signal_count = int(merged["tag_final"].sum()) if "tag_final" in merged.columns else 0
        print(f"[write] {out_path.name}: {len(merged)} rows, signals={signal_count}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Process ReSUM2 raw data into temp_new_data")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=repo_root / "data" / "raw",
        help="Root folder containing ScintillatorLF/TPCLF/TPCHF/ScintillatorHF",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "data" / "processed" / "temp_new_data",
        help="Output root for processed files (lf/ and hf/ will be created)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle rows inside each combined output CSV",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed used only when --shuffle is enabled",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root: Path = args.raw_root
    out_root: Path = args.output_root

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    print("=" * 72)
    print("ReSUM2 processing")
    print(f"raw_root   : {raw_root}")
    print(f"output_root: {out_root}")
    print(f"workers    : {args.workers}")
    print(f"shuffle    : {args.shuffle}")
    print("=" * 72)

    lf_jobs = collect_lf_jobs(raw_root)
    hf_jobs = collect_hf_jobs(raw_root)

    print(f"[LF] files discovered: {len(lf_jobs)}")
    print(f"[HF] files discovered: {len(hf_jobs)}")

    lf_grouped = process_grouped_jobs(lf_jobs, max_workers=args.workers, label="LF")
    hf_grouped = process_grouped_jobs(hf_jobs, max_workers=args.workers, label="HF")

    write_grouped_outputs(
        grouped=lf_grouped,
        out_dir=out_root / "lf",
        name_template="sim_X{x}_Y{y}_task0.csv",
        shuffle=args.shuffle,
        seed=args.seed,
    )
    write_grouped_outputs(
        grouped=hf_grouped,
        out_dir=out_root / "hf",
        name_template="sim_X{x}_Y{y}_ALL.csv",
        shuffle=args.shuffle,
        seed=args.seed,
    )

    print("Done.")


if __name__ == "__main__":
    main()
