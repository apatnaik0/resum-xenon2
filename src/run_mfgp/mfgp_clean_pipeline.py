#!/usr/bin/env python3
"""Clean multi-fidelity GP pipeline for Xenon CNP outputs.

This module avoids any dependency on `resum` helper code.
Model: two-level autoregressive MF-GP (Kennedy-O'Hagan style):
- LF GP on (theta -> y_cnp) for fidelity=0
- rho estimated on HF points
- Discrepancy GP on (theta -> y_raw - rho * LF(theta)) for fidelity=1

Outputs:
- trained model artifacts
- metrics JSON
- prediction CSV (observed points + grid)
- plots similar to old notebook style
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


@dataclass
class MFGPRuntimeConfig:
    config_path: Path
    version: str
    theta_headers: List[str]
    theta_min: List[float]
    theta_max: List[float]
    out_dir_cnp: Path
    out_dir_mfgp: Path


def _default_config_path() -> Path:
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        cwd / "src/xenon/settings2.yaml",
        cwd / "xenon/settings2.yaml",
        here / "../xenon/settings2.yaml",
    ]
    for c in candidates:
        c = c.resolve()
        if c.exists():
            return c
    return (here / "../xenon/settings2.yaml").resolve()


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (base / p).resolve()


def load_runtime_config(config_path: str | Path) -> MFGPRuntimeConfig:
    cp = Path(config_path).resolve()
    raw = yaml.safe_load(cp.read_text())
    sim = raw.get("simulation_settings", {})
    paths = raw.get("path_settings", {})
    base = cp.parent

    return MFGPRuntimeConfig(
        config_path=cp,
        version=str(paths.get("version", "v_clean")),
        theta_headers=list(sim.get("theta_headers", ["scint_x", "scint_y"])),
        theta_min=[float(x) for x in sim.get("theta_min", [0.0, 0.0])],
        theta_max=[float(x) for x in sim.get("theta_max", [1.0, 1.0])],
        out_dir_cnp=_resolve_path(paths.get("path_out_cnp", "../xenon/out/cnp"), base),
        out_dir_mfgp=_resolve_path(paths.get("path_out_mfgp", "../xenon/out/mfgp"), base),
    )


def discover_cnp_output_csv(out_dir_cnp: Path, version: str, prefer_validation: bool = False) -> Path:
    out_dir_cnp = out_dir_cnp.resolve()
    if not out_dir_cnp.exists():
        raise FileNotFoundError(f"CNP output directory does not exist: {out_dir_cnp}")

    if prefer_validation:
        pats = [f"cnp_{version}_output_validation_*epochs.csv", f"cnp_{version}_output_*epochs.csv"]
    else:
        pats = [f"cnp_{version}_output_*epochs.csv", f"cnp_{version}_output_validation_*epochs.csv"]

    candidates: List[Path] = []
    for pat in pats:
        candidates.extend(sorted(out_dir_cnp.glob(pat)))

    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No CNP CSV found in {out_dir_cnp} for version={version}. "
            f"Expected pattern like cnp_{version}_output_*epochs.csv"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _aggregate_rows(df: pd.DataFrame, x_cols: Sequence[str]) -> pd.DataFrame:
    # Multiple rows can share the same theta/fidelity/iteration; collapse to stable means.
    keys = [*x_cols, "fidelity", "iteration"]
    for c in ["y_cnp", "y_cnp_err", "y_raw"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in CNP CSV")
    agg_map: Dict[str, str] = {
        "y_cnp": "mean",
        "y_cnp_err": "mean",
        "y_raw": "mean",
    }
    if "n_samples" in df.columns:
        agg_map["n_samples"] = "mean"
    out = df.groupby(keys, dropna=False, as_index=False).agg(agg_map)
    if "n_samples" in out.columns:
        out = out.rename(columns={"n_samples": "n_samples_agg"})
    else:
        # Fall back to grouped row count when input has no n_samples column.
        group_sizes = df.groupby(keys, dropna=False).size().reset_index(name="n_samples_agg")
        out = out.merge(group_sizes, on=keys, how="left")
    return out


def load_mfgp_training_data(
    csv_path: str | Path,
    x_cols: Sequence[str],
    iteration: int = 0,
    lf_fidelity: int = 0,
    hf_fidelity: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    required = set([*x_cols, "iteration", "fidelity", "y_cnp", "y_cnp_err", "y_raw"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in CNP CSV: {missing}")

    df = _aggregate_rows(df, x_cols)
    df = df[df["iteration"].astype(int) == int(iteration)].copy()
    if df.empty:
        raise ValueError(f"No rows found for iteration={iteration}")

    lf = df[df["fidelity"].astype(int) == int(lf_fidelity)].copy()
    hf = df[df["fidelity"].astype(int) == int(hf_fidelity)].copy()

    if lf.empty or hf.empty:
        raise ValueError(
            f"Need both LF and HF rows at iteration={iteration}. "
            f"Found lf={len(lf)}, hf={len(hf)}"
        )

    for c in x_cols:
        lf[c] = lf[c].astype(float)
        hf[c] = hf[c].astype(float)
    lf["y_cnp"] = lf["y_cnp"].astype(float)
    lf["y_cnp_err"] = lf["y_cnp_err"].astype(float)
    hf["y_raw"] = hf["y_raw"].astype(float)

    return df, lf, hf


class CleanAutoregressiveMFGP:
    """Two-level autoregressive MF-GP using sklearn GPs."""

    def __init__(self, random_state: int = 42, alpha_lf: float = 1e-8, alpha_hf: float = 1e-8) -> None:
        self.random_state = int(random_state)
        self.alpha_lf = float(alpha_lf)
        self.alpha_hf = float(alpha_hf)

        self.x_scaler: Optional[StandardScaler] = None
        self.y_lf_scaler: Optional[StandardScaler] = None
        self.y_d_scaler: Optional[StandardScaler] = None

        self.gp_lf: Optional[GaussianProcessRegressor] = None
        self.gp_d: Optional[GaussianProcessRegressor] = None
        self.rho: Optional[float] = None
        self.x_dim: Optional[int] = None

    def _kernel(self, input_dim: int) -> ConstantKernel:
        return (
            ConstantKernel(1.0, (1e-4, 1e4))
            * Matern(length_scale=np.ones(input_dim), length_scale_bounds=(1e-3, 1e3), nu=1.5)
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
        )

    def fit(
        self,
        x_lf: np.ndarray,
        y_lf: np.ndarray,
        x_hf: np.ndarray,
        y_hf: np.ndarray,
        verbose: bool = False,
    ) -> "CleanAutoregressiveMFGP":
        x_lf = np.asarray(x_lf, dtype=float)
        y_lf = np.asarray(y_lf, dtype=float).reshape(-1, 1)
        x_hf = np.asarray(x_hf, dtype=float)
        y_hf = np.asarray(y_hf, dtype=float).reshape(-1, 1)

        if x_lf.ndim != 2 or x_hf.ndim != 2:
            raise ValueError("x_lf and x_hf must be 2D")
        if x_lf.shape[1] != x_hf.shape[1]:
            raise ValueError("x_lf and x_hf must have same feature dimension")
        if len(x_lf) < 3 or len(x_hf) < 3:
            raise ValueError("Need at least 3 LF and 3 HF points")

        self.x_dim = x_lf.shape[1]

        self.x_scaler = StandardScaler().fit(np.vstack([x_lf, x_hf]))
        x_lf_s = self.x_scaler.transform(x_lf)
        x_hf_s = self.x_scaler.transform(x_hf)

        self.y_lf_scaler = StandardScaler().fit(y_lf)
        y_lf_s = self.y_lf_scaler.transform(y_lf).ravel()

        self.gp_lf = GaussianProcessRegressor(
            kernel=self._kernel(self.x_dim),
            alpha=self.alpha_lf,
            normalize_y=False,
            n_restarts_optimizer=2,
            random_state=self.random_state,
        )
        if verbose:
            print("[fit] Training LF GP...")
        self.gp_lf.fit(x_lf_s, y_lf_s)

        mu_lf_hf_s, _ = self.gp_lf.predict(x_hf_s, return_std=True)
        mu_lf_hf = self.y_lf_scaler.inverse_transform(mu_lf_hf_s.reshape(-1, 1)).ravel()

        y_hf_vec = y_hf.ravel()
        denom = float(np.dot(mu_lf_hf, mu_lf_hf)) + 1e-12
        self.rho = float(np.dot(mu_lf_hf, y_hf_vec) / denom)
        if verbose:
            print(f"[fit] Estimated rho={self.rho:.6f}")

        y_d = y_hf_vec - self.rho * mu_lf_hf
        self.y_d_scaler = StandardScaler().fit(y_d.reshape(-1, 1))
        y_d_s = self.y_d_scaler.transform(y_d.reshape(-1, 1)).ravel()

        self.gp_d = GaussianProcessRegressor(
            kernel=self._kernel(self.x_dim),
            alpha=self.alpha_hf,
            normalize_y=False,
            n_restarts_optimizer=2,
            random_state=self.random_state,
        )
        if verbose:
            print("[fit] Training HF discrepancy GP...")
        self.gp_d.fit(x_hf_s, y_d_s)
        if verbose:
            print("[fit] GP training complete.")

        return self

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.gp_lf is None or self.gp_d is None or self.x_scaler is None:
            raise RuntimeError("Model not fitted")
        if self.rho is None or self.y_lf_scaler is None or self.y_d_scaler is None:
            raise RuntimeError("Model not fitted")

        x = np.asarray(x, dtype=float)
        x_s = self.x_scaler.transform(x)

        mu_lf_s, std_lf_s = self.gp_lf.predict(x_s, return_std=True)
        mu_d_s, std_d_s = self.gp_d.predict(x_s, return_std=True)

        y_lf_scale = float(self.y_lf_scaler.scale_[0])
        y_d_scale = float(self.y_d_scaler.scale_[0])

        mu_lf = self.y_lf_scaler.inverse_transform(mu_lf_s.reshape(-1, 1)).ravel()
        mu_d = self.y_d_scaler.inverse_transform(mu_d_s.reshape(-1, 1)).ravel()

        std_lf = np.maximum(std_lf_s * y_lf_scale, 1e-12)
        std_d = np.maximum(std_d_s * y_d_scale, 1e-12)

        mu_hf = self.rho * mu_lf + mu_d
        var_hf = (self.rho ** 2) * (std_lf ** 2) + (std_d ** 2)
        std_hf = np.sqrt(np.maximum(var_hf, 1e-12))

        return mu_hf, std_hf, mu_lf, std_lf


@dataclass
class MFGPResult:
    cnp_csv: Path
    model_json: Path
    metrics_json: Path
    prediction_csv: Path
    data_plot: Path
    parity_plot: Path
    mean_std_plot: Path
    residual_plot: Path


def _grid_from_bounds(theta_min: Sequence[float], theta_max: Sequence[float], n: int = 120) -> np.ndarray:
    x = np.linspace(float(theta_min[0]), float(theta_max[0]), n)
    y = np.linspace(float(theta_min[1]), float(theta_max[1]), n)
    gx, gy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack([gx.ravel(), gy.ravel()])


def _scatter_grid(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, n: int = 120) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    z = df[z_col].to_numpy(dtype=float)
    gx = np.linspace(x.min(), x.max(), n)
    gy = np.linspace(y.min(), y.max(), n)
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    try:
        from scipy.interpolate import griddata

        Z = griddata((x, y), z, (X, Y), method="cubic")
        if Z is None or np.isnan(Z).all():
            Z = griddata((x, y), z, (X, Y), method="linear")
    except Exception:
        Z = None

    if Z is None:
        Z = np.full_like(X, np.nan, dtype=float)
    return X, Y, Z


def _plot_data_maps(lf: pd.DataFrame, hf: pd.DataFrame, x_cols: Sequence[str], out_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    s1 = ax[0].scatter(lf[x_cols[0]], lf[x_cols[1]], c=lf["y_cnp"], cmap="viridis", s=35, edgecolor="none")
    ax[0].set_title("LF observations (y_cnp)")
    ax[0].set_xlabel(x_cols[0])
    ax[0].set_ylabel(x_cols[1])
    plt.colorbar(s1, ax=ax[0], label="y_cnp")

    s2 = ax[1].scatter(hf[x_cols[0]], hf[x_cols[1]], c=hf["y_raw"], cmap="viridis", s=35, edgecolor="none")
    ax[1].set_title("HF observations (y_raw)")
    ax[1].set_xlabel(x_cols[0])
    ax[1].set_ylabel(x_cols[1])
    plt.colorbar(s2, ax=ax[1], label="y_raw")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_parity(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.errorbar(y_true, y_pred, yerr=y_std, fmt="o", ms=4, alpha=0.6, capsize=2, color="tab:blue")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("HF truth (y_raw)")
    ax.set_ylabel("MF-GP prediction")
    ax.set_title("HF Parity Plot")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_mean_std_heatmaps(
    grid_xy: np.ndarray,
    mean_hf: np.ndarray,
    std_hf: np.ndarray,
    x_cols: Sequence[str],
    out_path: Path,
) -> None:
    gx = np.unique(grid_xy[:, 0])
    gy = np.unique(grid_xy[:, 1])
    nx, ny = len(gx), len(gy)
    Zm = mean_hf.reshape(ny, nx)
    Zs = std_hf.reshape(ny, nx)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax[0].contourf(gx, gy, Zm, levels=24, cmap="viridis")
    ax[0].set_title("MF-GP Mean (HF)")
    ax[0].set_xlabel(x_cols[0])
    ax[0].set_ylabel(x_cols[1])
    plt.colorbar(im1, ax=ax[0], label="mean")

    im2 = ax[1].contourf(gx, gy, Zs, levels=24, cmap="Reds")
    ax[1].set_title("MF-GP Std (HF)")
    ax[1].set_xlabel(x_cols[0])
    ax[1].set_ylabel(x_cols[1])
    plt.colorbar(im2, ax=ax[1], label="std")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_residual_heatmap(
    df_hf_pred: pd.DataFrame,
    x_cols: Sequence[str],
    out_path: Path,
) -> None:
    tmp = df_hf_pred.copy()
    tmp["residual"] = tmp["mf_mean"] - tmp["y_raw"]
    X, Y, Z = _scatter_grid(tmp, x_cols[0], x_cols[1], "residual", n=120)

    vmax = float(np.nanmax(np.abs(tmp["residual"].to_numpy()))) if len(tmp) else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    if np.isfinite(Z).any():
        im = ax.contourf(X, Y, Z, levels=24, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label="mf_mean - y_raw")
    sc = ax.scatter(
        tmp[x_cols[0]],
        tmp[x_cols[1]],
        c=tmp["residual"],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        s=25,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_title("HF Residual Map")
    ax.set_xlabel(x_cols[0])
    ax.set_ylabel(x_cols[1])
    if not np.isfinite(Z).any():
        plt.colorbar(sc, ax=ax, label="mf_mean - y_raw")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run_clean_mfgp(
    config_path: str | Path,
    cnp_csv: Optional[str | Path] = None,
    iteration: int = 0,
    lf_fidelity: int = 0,
    hf_fidelity: int = 1,
    grid_points_per_axis: int = 120,
    random_state: int = 42,
    prefer_validation_csv: bool = False,
    predict_chunk_size: int = 20000,
    verbose: bool = True,
) -> MFGPResult:
    t0 = time.time()
    if verbose:
        print("[stage] Loading runtime config...")
    runtime = load_runtime_config(config_path)
    runtime.out_dir_mfgp.mkdir(parents=True, exist_ok=True)

    cnp_csv_path = Path(cnp_csv).resolve() if cnp_csv else discover_cnp_output_csv(
        runtime.out_dir_cnp,
        runtime.version,
        prefer_validation=prefer_validation_csv,
    )
    if not cnp_csv_path.exists():
        raise FileNotFoundError(f"CNP CSV not found: {cnp_csv_path}")
    if verbose:
        print(f"[stage] Using CNP CSV: {cnp_csv_path}")

    if verbose:
        print("[stage] Loading LF/HF training rows...")
    _, lf, hf = load_mfgp_training_data(
        cnp_csv_path,
        x_cols=runtime.theta_headers,
        iteration=iteration,
        lf_fidelity=lf_fidelity,
        hf_fidelity=hf_fidelity,
    )

    x_lf = lf[runtime.theta_headers].to_numpy(dtype=float)
    y_lf = lf["y_cnp"].to_numpy(dtype=float)
    x_hf = hf[runtime.theta_headers].to_numpy(dtype=float)
    y_hf = hf["y_raw"].to_numpy(dtype=float)
    if verbose:
        print(f"[data] n_lf={len(lf)} n_hf={len(hf)} x_dim={x_lf.shape[1]}")

    # Noise levels from data statistics; keep small floor for numeric stability.
    alpha_lf = float(np.nanmean(lf["y_cnp_err"].to_numpy(dtype=float)) ** 2)
    alpha_hf = float(np.nanstd(y_hf) ** 2 * 1e-4)
    alpha_lf = max(alpha_lf, 1e-10)
    alpha_hf = max(alpha_hf, 1e-10)
    if verbose:
        print(f"[data] alpha_lf={alpha_lf:.3e} alpha_hf={alpha_hf:.3e}")

    model = CleanAutoregressiveMFGP(random_state=random_state, alpha_lf=alpha_lf, alpha_hf=alpha_hf)
    model.fit(x_lf=x_lf, y_lf=y_lf, x_hf=x_hf, y_hf=y_hf, verbose=verbose)

    if verbose:
        print("[stage] Predicting on HF observations...")
    hf_pred_mean_parts: List[np.ndarray] = []
    hf_pred_std_parts: List[np.ndarray] = []
    n_hf = len(x_hf)
    for i in range(0, n_hf, max(1, int(predict_chunk_size))):
        j = min(n_hf, i + max(1, int(predict_chunk_size)))
        mu_hf, std_hf, _, _ = model.predict(x_hf[i:j])
        hf_pred_mean_parts.append(mu_hf)
        hf_pred_std_parts.append(std_hf)
        if verbose:
            print(f"[progress] HF predict {j}/{n_hf}")
    hf_pred_mean = np.concatenate(hf_pred_mean_parts, axis=0)
    hf_pred_std = np.concatenate(hf_pred_std_parts, axis=0)

    rmse = float(np.sqrt(mean_squared_error(y_hf, hf_pred_mean)))
    mae = float(mean_absolute_error(y_hf, hf_pred_mean))
    r2 = float(r2_score(y_hf, hf_pred_mean))

    # Grid prediction
    if len(runtime.theta_min) >= 2 and len(runtime.theta_max) >= 2:
        grid_xy = _grid_from_bounds(runtime.theta_min, runtime.theta_max, n=grid_points_per_axis)
    else:
        x0, x1 = runtime.theta_headers
        theta_min = [float(min(lf[x0].min(), hf[x0].min())), float(min(lf[x1].min(), hf[x1].min()))]
        theta_max = [float(max(lf[x0].max(), hf[x0].max())), float(max(lf[x1].max(), hf[x1].max()))]
        grid_xy = _grid_from_bounds(theta_min, theta_max, n=grid_points_per_axis)

    if verbose:
        print(f"[stage] Predicting on grid ({len(grid_xy)} points)...")
    gm_parts: List[np.ndarray] = []
    gs_parts: List[np.ndarray] = []
    glm_parts: List[np.ndarray] = []
    gls_parts: List[np.ndarray] = []
    n_grid = len(grid_xy)
    for i in range(0, n_grid, max(1, int(predict_chunk_size))):
        j = min(n_grid, i + max(1, int(predict_chunk_size)))
        gm, gs, glm, gls = model.predict(grid_xy[i:j])
        gm_parts.append(gm)
        gs_parts.append(gs)
        glm_parts.append(glm)
        gls_parts.append(gls)
        if verbose:
            print(f"[progress] Grid predict {j}/{n_grid}")
    grid_mean = np.concatenate(gm_parts, axis=0)
    grid_std = np.concatenate(gs_parts, axis=0)
    grid_lf_mean = np.concatenate(glm_parts, axis=0)
    grid_lf_std = np.concatenate(gls_parts, axis=0)

    df_hf = hf.copy()
    df_hf["mf_mean"] = hf_pred_mean
    df_hf["mf_std"] = hf_pred_std

    df_grid = pd.DataFrame(grid_xy, columns=runtime.theta_headers)
    df_grid["mf_mean"] = grid_mean
    df_grid["mf_std"] = grid_std
    df_grid["lf_mean"] = grid_lf_mean
    df_grid["lf_std"] = grid_lf_std

    pred_csv = runtime.out_dir_mfgp / f"mfgp_{runtime.version}_predictions_iter{iteration}.csv"
    grid_csv = runtime.out_dir_mfgp / f"mfgp_{runtime.version}_grid_iter{iteration}.csv"
    df_hf.to_csv(pred_csv, index=False)
    df_grid.to_csv(grid_csv, index=False)

    model_json = runtime.out_dir_mfgp / f"mfgp_{runtime.version}_model_iter{iteration}.json"
    model_json.write_text(
        json.dumps(
            {
                "version": runtime.version,
                "config_path": str(runtime.config_path),
                "cnp_csv": str(cnp_csv_path),
                "iteration": int(iteration),
                "lf_fidelity": int(lf_fidelity),
                "hf_fidelity": int(hf_fidelity),
                "theta_headers": runtime.theta_headers,
                "rho": float(model.rho),
                "alpha_lf": float(alpha_lf),
                "alpha_hf": float(alpha_hf),
                "n_lf": int(len(lf)),
                "n_hf": int(len(hf)),
            },
            indent=2,
        )
    )

    metrics_json = runtime.out_dir_mfgp / f"mfgp_{runtime.version}_metrics_iter{iteration}.json"
    metrics_json.write_text(
        json.dumps(
            {
                "rmse_hf": rmse,
                "mae_hf": mae,
                "r2_hf": r2,
                "n_lf": int(len(lf)),
                "n_hf": int(len(hf)),
                "rho": float(model.rho),
            },
            indent=2,
        )
    )

    data_plot = runtime.out_dir_mfgp / f"mfgp_{runtime.version}_data_maps_iter{iteration}.png"
    parity_plot = runtime.out_dir_mfgp / f"mfgp_{runtime.version}_parity_iter{iteration}.png"
    mean_std_plot = runtime.out_dir_mfgp / f"mfgp_{runtime.version}_mean_std_iter{iteration}.png"
    residual_plot = runtime.out_dir_mfgp / f"mfgp_{runtime.version}_residual_iter{iteration}.png"

    _plot_data_maps(lf, hf, runtime.theta_headers, data_plot)
    _plot_parity(y_hf, hf_pred_mean, hf_pred_std, parity_plot)
    _plot_mean_std_heatmaps(grid_xy, grid_mean, grid_std, runtime.theta_headers, mean_std_plot)
    _plot_residual_heatmap(df_hf, runtime.theta_headers, residual_plot)

    elapsed = time.time() - t0
    if verbose:
        print(f"[done] MFGP complete | version={runtime.version} | n_lf={len(lf)} n_hf={len(hf)}")
        print(f"[done] HF metrics: RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.6f}, rho={float(model.rho):.6f}")
        print(f"[done] Outputs -> {runtime.out_dir_mfgp}")
        print(f"[done] Elapsed: {elapsed:.1f}s")

    return MFGPResult(
        cnp_csv=cnp_csv_path,
        model_json=model_json,
        metrics_json=metrics_json,
        prediction_csv=pred_csv,
        data_plot=data_plot,
        parity_plot=parity_plot,
        mean_std_plot=mean_std_plot,
        residual_plot=residual_plot,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean multi-fidelity GP pipeline (no resum helpers)")
    p.add_argument("--config", type=str, default=str(_default_config_path()), help="Path to settings2.yaml")
    p.add_argument("--cnp-csv", type=str, default=None, help="Optional explicit CNP output CSV")
    p.add_argument("--iteration", type=int, default=0, help="Iteration value to filter")
    p.add_argument("--lf-fidelity", type=int, default=0, help="LF fidelity id")
    p.add_argument("--hf-fidelity", type=int, default=1, help="HF fidelity id")
    p.add_argument("--grid-points", type=int, default=120, help="Grid points per axis")
    p.add_argument("--predict-chunk-size", type=int, default=20000, help="Chunk size for prediction progress")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    p.add_argument(
        "--prefer-validation-csv",
        action="store_true",
        help="When auto-discovering CNP CSV, prefer output_validation files first",
    )
    p.add_argument("--quiet", action="store_true", help="Reduce progress logs")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_clean_mfgp(
        config_path=args.config,
        cnp_csv=args.cnp_csv,
        iteration=args.iteration,
        lf_fidelity=args.lf_fidelity,
        hf_fidelity=args.hf_fidelity,
        grid_points_per_axis=args.grid_points,
        random_state=args.random_state,
        prefer_validation_csv=args.prefer_validation_csv,
        predict_chunk_size=args.predict_chunk_size,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
