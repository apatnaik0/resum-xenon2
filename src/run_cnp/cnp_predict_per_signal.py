#!/usr/bin/env python3
"""Per-event CNP prediction export for Xenon H5 data.

This script writes one prediction row per event/signal (not per-file averages).
It is intentionally independent from any `resum` package code and uses only the
clean pipeline utilities from `cnp_clean_pipeline.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from cnp_clean_pipeline import H5EventPool, load_model_checkpoint, load_runtime_config, set_seed


def _build_output_columns(theta_headers: List[str], phi_headers: List[str], target_headers: List[str]) -> List[str]:
    cols = [
        "iteration",
        "fidelity",
        "source_file",
        "event_index",
        "n_samples_file",
        "is_context",
    ]
    cols.extend(theta_headers)
    cols.extend(phi_headers)

    if len(target_headers) == 1:
        cols.extend(["y_raw", "y_cnp", "y_cnp_err"])
    else:
        for h in target_headers:
            cols.append(f"y_raw_{h}")
            cols.append(f"y_cnp_{h}")
            cols.append(f"y_cnp_err_{h}")
    return cols


def _default_config_path() -> Path:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        cwd / "src/xenon/settings2.yaml",
        cwd / "xenon/settings2.yaml",
        script_dir / "../xenon/settings2.yaml",
    ]
    for c in candidates:
        c = c.resolve()
        if c.exists():
            return c
    # Fallback to canonical repo-relative location.
    return (script_dir / "../xenon/settings2.yaml").resolve()


def _resolve_model_path(model_path: Optional[str], runtime) -> Path:
    if model_path:
        return Path(model_path).expanduser().resolve()
    return (runtime.out_dir / f"cnp_{runtime.version}_model_{runtime.epochs}epochs.pth").resolve()


def _resolve_output_path(out_csv: Optional[str], runtime) -> Path:
    if out_csv:
        return Path(out_csv).expanduser().resolve()
    return (runtime.out_dir / f"cnp_{runtime.version}_output_per_signal_{runtime.epochs}epochs.csv").resolve()


def run_per_signal_prediction(
    *,
    config_path: Path,
    model_path: Optional[str],
    out_csv: Optional[str],
    mc_samples: int,
    chunk_size: int,
    context_ratio_override: Optional[float],
    only_signal: bool,
    signal_threshold: float,
    signal_target_index: int,
    max_files_per_dir: Optional[int],
    seed: int,
    device: Optional[str],
    overwrite: bool,
) -> Path:
    runtime = load_runtime_config(config_path, seed=seed)
    if context_ratio_override is not None:
        runtime.context_ratio = float(context_ratio_override)

    model_ckpt = _resolve_model_path(model_path, runtime)
    if not model_ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt}")

    out_path = _resolve_output_path(out_csv, runtime)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output CSV already exists: {out_path}\n"
            "Use --overwrite to replace it."
        )
    if out_path.exists() and overwrite:
        out_path.unlink()

    if not runtime.predict_dirs:
        raise ValueError("No prediction directories found in config path_to_files_predict.")

    set_seed(seed)
    model = load_model_checkpoint(model_ckpt, device=device)
    dev = next(model.parameters()).device

    y_dim = len(runtime.target_headers)
    if signal_target_index < 0 or signal_target_index >= y_dim:
        raise ValueError(
            f"signal_target_index={signal_target_index} is out of range for y_dim={y_dim}"
        )

    output_columns = _build_output_columns(runtime.theta_headers, runtime.phi_headers, runtime.target_headers)
    header_written = False
    total_rows = 0
    total_files = 0

    for dir_idx, pred_dir in enumerate(runtime.predict_dirs):
        fidelity = int(runtime.predict_fidelities[dir_idx]) if dir_idx < len(runtime.predict_fidelities) else 0
        iteration = int(runtime.predict_iterations[dir_idx]) if dir_idx < len(runtime.predict_iterations) else 0

        pool = H5EventPool(
            pred_dir,
            theta_headers=runtime.theta_headers,
            phi_headers=runtime.phi_headers,
            target_headers=runtime.target_headers,
            use_mixedup=False,
            seed=seed + dir_idx,
            cache_files=True,
        )

        print(f"[dir {dir_idx}] {pred_dir} | files={len(pool.files)} | fidelity={fidelity} iteration={iteration}")

        for file_idx, (file_path, x_np, y_np, _theta_np) in enumerate(pool.iter_file_data()):
            if max_files_per_dir is not None and file_idx >= max_files_per_dir:
                break

            n = len(y_np)
            if n <= 2:
                continue

            total_files += 1
            rng = np.random.default_rng(seed + dir_idx * 1_000_003 + file_idx)
            n_context = max(2, int(runtime.context_ratio * n))
            n_context = min(n_context, n - 1)
            c_idx = rng.choice(n, size=n_context, replace=False)
            context_mask = np.zeros(n, dtype=np.uint8)
            context_mask[c_idx] = 1

            context_x = torch.from_numpy(x_np[c_idx]).to(dev)
            context_y = torch.from_numpy(y_np[c_idx]).to(dev)

            written_this_file = 0
            with torch.no_grad():
                for start in range(0, n, chunk_size):
                    end = min(n, start + chunk_size)
                    tx_np = x_np[start:end]
                    ty_np = y_np[start:end]
                    tx = torch.from_numpy(tx_np).to(dev)

                    mu_t, std_t = model.predict_proba_mc(context_x, context_y, tx, mc_samples=mc_samples)
                    mu_np = mu_t.cpu().numpy()
                    std_np = std_t.cpu().numpy()

                    keep_mask = np.ones(end - start, dtype=bool)
                    if only_signal:
                        keep_mask = ty_np[:, signal_target_index] > signal_threshold
                    if not np.any(keep_mask):
                        continue

                    tx_keep = tx_np[keep_mask]
                    ty_keep = ty_np[keep_mask]
                    mu_keep = mu_np[keep_mask]
                    std_keep = std_np[keep_mask]
                    idx_keep = np.arange(start, end)[keep_mask]
                    ctx_keep = context_mask[start:end][keep_mask]

                    theta_dim = len(runtime.theta_headers)
                    phi_start = theta_dim
                    phi_end = phi_start + len(runtime.phi_headers)

                    data: Dict[str, np.ndarray | List[str]] = {
                        "iteration": np.full(len(idx_keep), iteration, dtype=np.int32),
                        "fidelity": np.full(len(idx_keep), fidelity, dtype=np.int32),
                        "source_file": [file_path.name] * len(idx_keep),
                        "event_index": idx_keep.astype(np.int64),
                        "n_samples_file": np.full(len(idx_keep), n, dtype=np.int32),
                        "is_context": ctx_keep.astype(np.uint8),
                    }

                    for j, name in enumerate(runtime.theta_headers):
                        data[name] = tx_keep[:, j]
                    for j, name in enumerate(runtime.phi_headers):
                        data[name] = tx_keep[:, phi_start + j]

                    if y_dim == 1:
                        data["y_raw"] = ty_keep[:, 0]
                        data["y_cnp"] = mu_keep[:, 0]
                        data["y_cnp_err"] = std_keep[:, 0]
                    else:
                        for j, h in enumerate(runtime.target_headers):
                            data[f"y_raw_{h}"] = ty_keep[:, j]
                            data[f"y_cnp_{h}"] = mu_keep[:, j]
                            data[f"y_cnp_err_{h}"] = std_keep[:, j]

                    df_chunk = pd.DataFrame(data)
                    df_chunk = df_chunk[output_columns]
                    df_chunk.to_csv(out_path, mode="a", index=False, header=not header_written)
                    header_written = True

                    rows = len(df_chunk)
                    total_rows += rows
                    written_this_file += rows

            print(
                f"  file {file_idx + 1}: {file_path.name} | n={n} | context={n_context} | "
                f"rows_written={written_this_file}"
            )

    print(f"Done. files_processed={total_files}, rows_written={total_rows}")
    print(f"Output CSV: {out_path}")
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Per-event/per-signal CNP prediction export")
    p.add_argument("--config", type=str, default=str(_default_config_path()), help="Path to settings2.yaml")
    p.add_argument("--model-path", type=str, default=None, help="Model checkpoint .pth (defaults from config)")
    p.add_argument("--out-csv", type=str, default=None, help="Output CSV path (defaults in path_out_cnp)")
    p.add_argument("--mc-samples", type=int, default=30, help="MC dropout samples per chunk")
    p.add_argument("--chunk-size", type=int, default=20000, help="Prediction chunk size")
    p.add_argument("--context-ratio", type=float, default=None, help="Override context ratio")
    p.add_argument("--only-signal", action="store_true", help="Keep only rows with y_raw > threshold")
    p.add_argument("--signal-threshold", type=float, default=0.5, help="Threshold used by --only-signal")
    p.add_argument(
        "--signal-target-index",
        type=int,
        default=0,
        help="Target index used by --only-signal when multiple targets exist",
    )
    p.add_argument("--max-files-per-dir", type=int, default=None, help="Limit files per predict directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", type=str, default=None, help="Force device, e.g. cpu or cuda")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output CSV if it already exists")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_per_signal_prediction(
        config_path=Path(args.config).expanduser().resolve(),
        model_path=args.model_path,
        out_csv=args.out_csv,
        mc_samples=args.mc_samples,
        chunk_size=args.chunk_size,
        context_ratio_override=args.context_ratio,
        only_signal=args.only_signal,
        signal_threshold=args.signal_threshold,
        signal_target_index=args.signal_target_index,
        max_files_per_dir=args.max_files_per_dir,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
