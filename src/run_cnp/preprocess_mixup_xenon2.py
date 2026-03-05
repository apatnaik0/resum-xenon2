#!/usr/bin/env python3
"""Apply mixup augmentation to Xenon2 HDF5 training files.

This script is standalone and does not import `resum`.
It reads base datasets from each H5 file:
- phi
- target
- (optional) weights

And writes mixup datasets back into the same file:
- phi_mixedup
- target_mixedup
- (optional) weights_mixedup
- signal_condition

Recommended usage:
1) Run this script on training H5 files.
2) Set `use_data_augmentation: mixup` in settings2.yaml.
3) Run `cnp_clean_workflow.ipynb` training cells.
"""

from __future__ import annotations

import argparse
import operator
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import h5py
import numpy as np
import yaml
from tqdm import tqdm


OPS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
}
OP_ORDER = ["==", "!=", ">=", "<=", ">", "<"]


def decode_labels(values: np.ndarray) -> List[str]:
    labels: List[str] = []
    for v in values:
        labels.append(v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v))
    return labels


def parse_condition(condition: str, target_labels: Sequence[str]) -> Tuple[int, str, float]:
    for op in OP_ORDER:
        if op in condition:
            left, right = condition.split(op, 1)
            col = left.strip()
            value_text = right.strip()
            if col not in target_labels:
                raise ValueError(f"Condition column '{col}' not in target headers {list(target_labels)}")
            try:
                value = float(value_text)
            except ValueError as exc:
                raise ValueError(f"Condition value must be numeric for mixup: {condition}") from exc
            return target_labels.index(col), op, value
    raise ValueError(f"No valid operator found in condition: {condition}")


def build_signal_mask(target: np.ndarray, target_headers: Sequence[str], conditions: Sequence[str]) -> np.ndarray:
    if target.ndim == 1:
        target_2d = target.reshape(-1, 1)
    else:
        target_2d = target

    mask = np.ones(target_2d.shape[0], dtype=bool)
    for condition in conditions:
        col_idx, op, value = parse_condition(condition, target_headers)
        fn = OPS[op]
        mask &= fn(target_2d[:, col_idx], value)
    return mask


def write_dataset(h5: h5py.File, name: str, data: np.ndarray) -> None:
    if name in h5:
        del h5[name]
    h5.create_dataset(name, data=data, compression="gzip")


def create_empty_mixup_datasets(h5: h5py.File, phi: np.ndarray, target: np.ndarray, weights: np.ndarray | None) -> None:
    phi_empty = np.empty((0,) + phi.shape[1:], dtype=phi.dtype)
    write_dataset(h5, "phi_mixedup", phi_empty)

    target_empty = np.empty((0,) + (() if target.ndim == 1 else target.shape[1:]), dtype=target.dtype)
    write_dataset(h5, "target_mixedup", target_empty)

    if weights is not None:
        weights_empty = np.empty((0,) + (() if weights.ndim == 1 else weights.shape[1:]), dtype=weights.dtype)
        write_dataset(h5, "weights_mixedup", weights_empty)


def mixup_one_file(
    file_path: Path,
    use_beta: Sequence[float] | None,
    signal_conditions: Sequence[str],
    seed: int,
    force: bool,
) -> str:
    rng = np.random.default_rng(seed)

    with h5py.File(file_path, "a") as h5:
        required = ["phi", "target"]
        missing = [k for k in required if k not in h5]
        if missing:
            raise ValueError(f"{file_path.name}: missing required datasets {missing}")

        existing_conditions = None
        if "signal_condition" in h5:
            existing_conditions = decode_labels(np.asarray(h5["signal_condition"]))

        if (
            not force
            and "phi_mixedup" in h5
            and "target_mixedup" in h5
            and existing_conditions == list(signal_conditions)
        ):
            return f"skip {file_path.name} (already mixed with same conditions)"

        phi = np.asarray(h5["phi"])
        target = np.asarray(h5["target"])
        weights = np.asarray(h5["weights"]) if "weights" in h5 else None

        target_headers = decode_labels(np.asarray(h5["target_headers"])) if "target_headers" in h5 else ["tag_final"]
        signal_mask = build_signal_mask(target, target_headers, signal_conditions)

        signal_idx = np.where(signal_mask)[0]
        all_idx = np.arange(len(target))
        background_idx = np.setdiff1d(all_idx, signal_idx)

        if len(signal_idx) == 0 or len(background_idx) == 0:
            create_empty_mixup_datasets(h5, phi, target, weights)
            write_dataset(h5, "signal_condition", np.array(signal_conditions, dtype="S"))
            return f"skip {file_path.name} (signals={len(signal_idx)}, background={len(background_idx)})"

        sampled_signal = rng.choice(signal_idx, size=len(background_idx), replace=True)

        if use_beta and len(use_beta) == 2:
            alpha = rng.beta(float(use_beta[0]), float(use_beta[1]), size=(len(background_idx), 1))
        else:
            alpha = rng.uniform(0.0, 1.0, size=(len(background_idx), 1))

        phi_mix = alpha * phi[sampled_signal] + (1.0 - alpha) * phi[background_idx]

        target_signal = target[sampled_signal]
        target_bg = target[background_idx]
        target_signal_2d = target_signal.reshape(-1, 1) if target_signal.ndim == 1 else target_signal
        target_bg_2d = target_bg.reshape(-1, 1) if target_bg.ndim == 1 else target_bg
        target_mix = alpha * target_signal_2d + (1.0 - alpha) * target_bg_2d
        if target.ndim == 1:
            target_mix = target_mix.reshape(-1)

        write_dataset(h5, "phi_mixedup", phi_mix.astype(phi.dtype, copy=False))
        write_dataset(h5, "target_mixedup", target_mix.astype(target.dtype, copy=False))

        if weights is not None:
            w_signal = weights[sampled_signal]
            w_bg = weights[background_idx]
            w_signal_2d = w_signal.reshape(-1, 1) if w_signal.ndim == 1 else w_signal
            w_bg_2d = w_bg.reshape(-1, 1) if w_bg.ndim == 1 else w_bg
            w_mix = alpha * w_signal_2d + (1.0 - alpha) * w_bg_2d
            if weights.ndim == 1:
                w_mix = w_mix.reshape(-1)
            write_dataset(h5, "weights_mixedup", w_mix.astype(weights.dtype, copy=False))

        write_dataset(h5, "signal_condition", np.array(signal_conditions, dtype="S"))

        return f"ok   {file_path.name}: mixed_samples={len(background_idx)}"


def load_settings(config_path: Path) -> Tuple[Path, List[float], List[str]]:
    cfg = yaml.safe_load(config_path.read_text())

    train_path = Path(cfg["path_settings"]["path_to_files_train"])
    if not train_path.is_absolute():
        train_path = (config_path.parent / train_path).resolve()

    beta = cfg.get("cnp_settings", {}).get("use_beta", [0.1, 0.1])
    if beta is None:
        beta = []
    beta = [float(x) for x in beta]

    signal_condition = cfg.get("simulation_settings", {}).get("signal_condition", ["tag_final==1"])
    signal_condition = [str(x) for x in signal_condition]

    return train_path, beta, signal_condition


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean mixup preprocessing for Xenon2 H5 training files")
    default_config = Path(__file__).resolve().parents[1] / "xenon" / "settings2.yaml"
    p.add_argument("--config", type=Path, default=default_config, help="Path to settings2.yaml")
    p.add_argument("--path", type=Path, default=None, help="Override training H5 folder")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--force", action="store_true", help="Recompute even if same signal_condition already exists")
    return p


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.resolve()

    train_path, beta, signal_conditions = load_settings(config_path)
    if args.path is not None:
        train_path = args.path.resolve()

    if not train_path.exists():
        raise FileNotFoundError(f"Training path not found: {train_path}")

    h5_files = sorted([p for p in train_path.glob("*.h5") if p.is_file()])
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in: {train_path}")

    print("=" * 80)
    print("MIXUP PREPROCESSING (CLEAN)")
    print("=" * 80)
    print(f"config           : {config_path}")
    print(f"training_h5_path : {train_path}")
    print(f"num_files        : {len(h5_files)}")
    print(f"use_beta         : {beta}")
    print(f"signal_condition : {signal_conditions}")
    print(f"force            : {args.force}")

    ok = 0
    skipped = 0
    failed = 0

    for i, file_path in enumerate(tqdm(h5_files, desc="Applying mixup"), start=1):
        try:
            msg = mixup_one_file(
                file_path=file_path,
                use_beta=beta,
                signal_conditions=signal_conditions,
                seed=args.seed + i,
                force=args.force,
            )
            if msg.startswith("ok"):
                ok += 1
            else:
                skipped += 1
            if i <= 5 or i % 100 == 0:
                tqdm.write(msg)
        except Exception as exc:
            failed += 1
            tqdm.write(f"fail {file_path.name}: {exc}")

    print("\n" + "=" * 80)
    print("MIXUP COMPLETE")
    print("=" * 80)
    print(f"ok      : {ok}")
    print(f"skipped : {skipped}")
    print(f"failed  : {failed}")

    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
