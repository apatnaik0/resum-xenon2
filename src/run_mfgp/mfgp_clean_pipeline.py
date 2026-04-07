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
from matplotlib.lines import Line2D
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

    # Prefer regular aggregated output for MF-GP fitting; exclude per-signal exports.
    regular = sorted(
        [
            p
            for p in out_dir_cnp.glob(f"cnp_{version}_output_*epochs.csv")
            if ("validation" not in p.name and "per_signal" not in p.name)
        ]
    )
    validation = sorted(
        [
            p
            for p in out_dir_cnp.glob(f"cnp_{version}_output_validation_*epochs.csv")
            if "per_signal" not in p.name
        ]
    )
    if prefer_validation:
        candidates = validation + regular
    else:
        candidates = regular + validation

    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"No CNP CSV found in {out_dir_cnp} for version={version}. "
            f"Expected pattern like cnp_{version}_output_*epochs.csv"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _missing_csv_message(csv_path: Path, out_dir_cnp: Path, version: str) -> str:
    nearby = sorted(out_dir_cnp.glob(f"cnp_{version}_*epochs.csv")) if out_dir_cnp.exists() else []
    nearby_lines = "\n".join([f"  - {p}" for p in nearby[:20]]) if nearby else "  (none found)"
    return (
        f"CSV not found: {csv_path}\n"
        f"Searched under: {out_dir_cnp}\n"
        f"Available matching files:\n{nearby_lines}"
    )


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
    grid_csv: Path
    data_plot: Optional[Path]
    parity_plot: Optional[Path]
    mean_std_plot: Path
    residual_plot: Optional[Path]
    theta_group_plot_dir: Optional[Path]
    across_theta_plot: Optional[Path]
    across_theta_zoom_plot: Optional[Path]
    coverage_plot: Optional[Path]
    validation_parity_plot: Optional[Path]


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


def _plot_hf_observation_map(hf: pd.DataFrame, x_cols: Sequence[str], out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    s = ax.scatter(hf[x_cols[0]], hf[x_cols[1]], c=hf["y_raw"], cmap="viridis", s=40, edgecolor="none")
    ax.set_title("HF observations (y_raw)")
    ax.set_xlabel(x_cols[0])
    ax.set_ylabel(x_cols[1])
    plt.colorbar(s, ax=ax, label="y_raw")
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
    hf_points: Optional[np.ndarray],
    out_path: Path,
) -> None:
    gx = np.unique(grid_xy[:, 0])
    gy = np.unique(grid_xy[:, 1])
    nx, ny = len(gx), len(gy)
    Zm = mean_hf.reshape(ny, nx)
    Zs = std_hf.reshape(ny, nx)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax[0].contourf(gx, gy, Zm, levels=24, cmap="viridis")
    if hf_points is not None and len(hf_points):
        ax[0].scatter(
            hf_points[:, 0],
            hf_points[:, 1],
            s=32,
            c="white",
            edgecolor="black",
            linewidth=0.7,
            alpha=0.95,
            label="HF points",
        )
    ax[0].set_title("MF-GP Mean (HF)")
    ax[0].set_xlabel(x_cols[0])
    ax[0].set_ylabel(x_cols[1])
    plt.colorbar(im1, ax=ax[0], label="mean")
    if hf_points is not None and len(hf_points):
        ax[0].legend(loc="best", fontsize=8, frameon=True)

    im2 = ax[1].contourf(gx, gy, Zs, levels=24, cmap="Reds")
    if hf_points is not None and len(hf_points):
        ax[1].scatter(
            hf_points[:, 0],
            hf_points[:, 1],
            s=32,
            c="white",
            edgecolor="black",
            linewidth=0.7,
            alpha=0.95,
            label="HF points",
        )
    ax[1].set_title("MF-GP Std (HF)")
    ax[1].set_xlabel(x_cols[0])
    ax[1].set_ylabel(x_cols[1])
    plt.colorbar(im2, ax=ax[1], label="std")
    if hf_points is not None and len(hf_points):
        ax[1].legend(loc="best", fontsize=8, frameon=True)

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


def _resolve_optional_validation_csv(runtime: MFGPRuntimeConfig) -> Optional[Path]:
    try:
        raw = yaml.safe_load(runtime.config_path.read_text())
        p = raw.get("path_settings", {}).get("path_to_files_validation")
        if not p:
            return None
        cand = Path(p)
        if not cand.is_absolute():
            cand = (runtime.config_path.parent / cand).resolve()
        return cand if cand.exists() else None
    except Exception:
        return None


def _plot_theta_group_uncertainty_bands(
    df_val: pd.DataFrame,
    theta_headers: Sequence[str],
    model: "CleanAutoregressiveMFGP",
    out_dir: Path,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    thx, thy = theta_headers[0], theta_headers[1]
    groups = list(df_val.groupby([thx, thy], dropna=False))
    count = 0

    for (xv, yv), g in groups:
        y_true = g["y_raw"].to_numpy(dtype=float)
        if len(y_true) == 0:
            continue

        x_pred = np.array([[float(xv), float(yv)]], dtype=float)
        mu, std, _, _ = model.predict(x_pred)
        mu = float(mu[0])
        std = float(std[0])
        if not np.isfinite(mu):
            mu = 0.0
        if not np.isfinite(std) or std < 0:
            std = 0.0
        idx = np.arange(len(y_true), dtype=int)

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(idx, y_true, "o", ms=3.5, alpha=0.7, label="validation y_raw")
        ax.axhline(mu, color="tab:red", lw=1.5, label="MF-GP mean")
        # Use true model std for both statistics and rendered bands.
        std_true = max(std, 1e-12)
        err = np.abs(y_true - mu)
        n = int(len(y_true))
        c1 = int(np.sum(err <= 1.0 * std_true))
        c2 = int(np.sum(err <= 2.0 * std_true))
        c3 = int(np.sum(err <= 3.0 * std_true))
        y_raw_mean = float(np.mean(y_true))
        y_raw_std = float(np.std(y_true))

        # Render bands using axhspan to guarantee drawing even for tiny/degenerate x ranges.
        ax.axhspan(mu - 3 * std_true, mu + 3 * std_true, color="tab:purple", alpha=0.10, label="3σ band")
        ax.axhspan(mu - 2 * std_true, mu + 2 * std_true, color="tab:orange", alpha=0.18, label="2σ band")
        ax.axhspan(mu - 1 * std_true, mu + 1 * std_true, color="tab:red", alpha=0.30, label="1σ band")
        # Draw boundary lines explicitly so band edges are always visible.
        for k, color in [(1, "tab:red"), (2, "tab:orange"), (3, "tab:purple")]:
            ax.axhline(mu + k * std_true, color=color, lw=0.8, alpha=0.9)
            ax.axhline(mu - k * std_true, color=color, lw=0.8, alpha=0.9)
        ax.set_title(f"Theta ({thx}={xv}, {thy}={yv})")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("y_raw")
        # Mild zoom-in around the main mass while keeping uncertainty region in view.
        if n >= 5:
            q_lo, q_hi = np.quantile(y_true, [0.02, 0.98])
            y_lo = float(min(q_lo, mu - 3 * std_true))
            y_hi = float(max(q_hi, mu + 3 * std_true))
        else:
            y_lo = float(min(np.min(y_true), mu - 3 * std_true))
            y_hi = float(max(np.max(y_true), mu + 3 * std_true))
        y_span = max(y_hi - y_lo, 1e-9)
        ax.set_ylim(y_lo - 0.10 * y_span, y_hi + 0.10 * y_span)
        ax.grid(True, alpha=0.3)
        # Add numeric diagnostics directly in the legend.
        extra = [
            Line2D([0], [0], color="none", label=f"n points: {n}"),
            Line2D([0], [0], color="none", label=f"y_raw mean: {y_raw_mean:.6g}"),
            Line2D([0], [0], color="none", label=f"y_raw std: {y_raw_std:.6g}"),
            Line2D([0], [0], color="none", label=f"model mean: {mu:.6g}"),
            Line2D([0], [0], color="none", label=f"model std: {std_true:.6g}"),
            Line2D([0], [0], color="none", label=f"within 1σ: {c1}/{n}"),
            Line2D([0], [0], color="none", label=f"within 2σ: {c2}/{n}"),
            Line2D([0], [0], color="none", label=f"within 3σ: {c3}/{n}"),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + extra, labels + [h.get_label() for h in extra], loc="best", fontsize=8, frameon=True)
        fig.tight_layout()
        f = out_dir / f"uncertainty_bands_theta_{int(round(float(xv)))}_{int(round(float(yv)))}.png"
        fig.savefig(f, dpi=170)
        plt.close(fig)
        count += 1
    return count


def _plot_across_thetas(
    df_val: pd.DataFrame,
    theta_headers: Sequence[str],
    model: "CleanAutoregressiveMFGP",
    out_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    thx, thy = theta_headers[0], theta_headers[1]
    agg = (
        df_val.groupby([thx, thy], as_index=False)
        .agg(y_raw_mean=("y_raw", "mean"))
        .sort_values([thx, thy], kind="mergesort")
        .reset_index(drop=True)
    )
    x_pred = agg[[thx, thy]].to_numpy(dtype=float)
    mu, std, _, _ = model.predict(x_pred)

    idx = np.arange(len(agg), dtype=int)
    y_true = agg["y_raw_mean"].to_numpy(dtype=float)
    y_pred = mu.astype(float)
    y_std = std.astype(float)
    y_std_true = np.maximum(y_std, 1e-12)
    err = np.abs(y_true - y_pred)
    n = int(len(y_true))
    c1 = int(np.sum(err <= 1.0 * y_std_true))
    c2 = int(np.sum(err <= 2.0 * y_std_true))
    c3 = int(np.sum(err <= 3.0 * y_std_true))

    fig, ax = plt.subplots(1, 1, figsize=(max(10, 0.18 * len(idx)), 5))
    ax.plot(idx, y_true, "o", ms=4, color="black", alpha=0.75, label="Validation y_raw mean")
    ax.plot(idx, y_pred, "-", lw=1.5, color="tab:blue", label="MF-GP mean")
    ax.fill_between(idx, y_pred - y_std_true, y_pred + y_std_true, color="tab:blue", alpha=0.24, label="1σ band")
    ax.fill_between(idx, y_pred - 2 * y_std_true, y_pred + 2 * y_std_true, color="tab:orange", alpha=0.16, label="2σ band")
    ax.fill_between(idx, y_pred - 3 * y_std_true, y_pred + 3 * y_std_true, color="tab:purple", alpha=0.10, label="3σ band")
    ax.set_xlabel("Theta index (sorted by scint_x, scint_y)")
    ax.set_ylabel("y")
    ax.set_title("Validation Thetas: Mean y_raw vs MF-GP Prediction")
    ax.grid(True, alpha=0.3)
    extra = [
        Line2D([0], [0], color="none", label=f"n points: {n}"),
        Line2D([0], [0], color="none", label=f"within 1σ: {c1}/{n}"),
        Line2D([0], [0], color="none", label=f"within 2σ: {c2}/{n}"),
        Line2D([0], [0], color="none", label=f"within 3σ: {c3}/{n}"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + extra, labels + [h.get_label() for h in extra], loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return y_true, y_pred, y_std


def _plot_coverage_summary(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, out_path: Path) -> None:
    if len(y_true) == 0:
        return
    err = np.abs(y_true - y_pred)
    s = np.maximum(y_std, 1e-12)
    cov1 = float(np.mean(err <= 1.0 * s))
    cov2 = float(np.mean(err <= 2.0 * s))
    cov3 = float(np.mean(err <= 3.0 * s))
    mae = float(np.mean(err))

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    labels = ["1σ coverage", "2σ coverage", "3σ coverage"]
    vals = [cov1, cov2, cov3]
    ax.bar(labels, vals, color=["tab:blue", "tab:green", "tab:purple"], alpha=0.85)
    ax.axhline(0.68, ls="--", lw=1, color="tab:blue", alpha=0.8, label="Ideal 1σ ~ 0.68")
    ax.axhline(0.95, ls="--", lw=1, color="tab:green", alpha=0.8, label="Ideal 2σ ~ 0.95")
    ax.axhline(0.997, ls="--", lw=1, color="tab:purple", alpha=0.8, label="Ideal 3σ ~ 0.997")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction")
    ax.set_title(f"Coverage Summary (MAE={mae:.4g})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run_clean_mfgp(
    config_path: str | Path,
    cnp_csv: Optional[str | Path] = None,
    validation_csv: Optional[str | Path] = None,
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

    cnp_csv_path = Path(cnp_csv).expanduser().resolve() if cnp_csv else discover_cnp_output_csv(
        runtime.out_dir_cnp,
        runtime.version,
        prefer_validation=prefer_validation_csv,
    )
    if not cnp_csv_path.exists():
        raise FileNotFoundError(_missing_csv_message(cnp_csv_path, runtime.out_dir_cnp, runtime.version))
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

    # Noise levels reverted to old-notebook style.
    # LF noise = mean(y_cnp_err)
    # HF noise = std(y_raw) * 1e-8
    lf_cnp_noise = float(np.nanmean(lf["y_cnp_err"].to_numpy(dtype=float)))
    hf_sim_noise = float(np.nanstd(y_hf))
    alpha_lf = lf_cnp_noise
    alpha_hf = hf_sim_noise * 1e-8

    alpha_lf = max(alpha_lf, 1e-10)
    alpha_hf = max(alpha_hf, 1e-10)
    if verbose:
        print(
            f"[data] lf_cnp_noise={lf_cnp_noise:.3e} hf_sim_noise={hf_sim_noise:.3e} "
            f"alpha_lf={alpha_lf:.3e} alpha_hf={alpha_hf:.3e}"
        )

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

    data_plot = runtime.out_dir_mfgp / f"mfgp_{runtime.version}_hf_observations_iter{iteration}.png"
    parity_plot = None
    mean_std_plot = runtime.out_dir_mfgp / f"mfgp_{runtime.version}_mean_std_iter{iteration}.png"
    residual_plot = None

    _plot_hf_observation_map(hf, runtime.theta_headers, data_plot)
    # Training-side parity plot intentionally omitted to reduce redundant outputs.
    _plot_mean_std_heatmaps(
        grid_xy,
        grid_mean,
        grid_std,
        runtime.theta_headers,
        hf_points=x_hf,
        out_path=mean_std_plot,
    )

    theta_group_plot_dir: Optional[Path] = None
    across_theta_plot: Optional[Path] = None
    across_theta_zoom_plot: Optional[Path] = None
    coverage_plot: Optional[Path] = None
    validation_parity_plot: Optional[Path] = None

    # Additional old-notebook-style validation plots.
    val_csv = Path(validation_csv).expanduser().resolve() if validation_csv is not None else _resolve_optional_validation_csv(runtime)
    if val_csv is not None:
        if not val_csv.exists():
            if verbose:
                print(_missing_csv_message(val_csv, runtime.out_dir_cnp, runtime.version))
                print("[warn] Explicit validation_csv does not exist; skipping extra validation plots.")
            val_csv = None
    if val_csv is not None:
        try:
            if verbose:
                print(f"[stage] Loading validation CSV for extra plots: {val_csv}")
            df_val = pd.read_csv(val_csv)
            need = set([*runtime.theta_headers, "iteration", "y_raw"])
            if need.issubset(df_val.columns):
                df_val = df_val[df_val["iteration"].astype(int) == int(iteration)].copy()
                if len(df_val):
                    theta_group_plot_dir = runtime.out_dir_mfgp / f"mfgp_{runtime.version}_theta_group_plots_iter{iteration}"
                    n_groups = _plot_theta_group_uncertainty_bands(
                        df_val=df_val,
                        theta_headers=runtime.theta_headers,
                        model=model,
                        out_dir=theta_group_plot_dir,
                    )
                    if verbose:
                        print(f"[done] Generated theta-group uncertainty plots: {n_groups}")

                    across_theta_plot = runtime.out_dir_mfgp / f"mfgp_{runtime.version}_across_thetas_iter{iteration}.png"
                    across_theta_zoom_plot = None
                    y_true_v, y_pred_v, y_std_v = _plot_across_thetas(
                        df_val=df_val,
                        theta_headers=runtime.theta_headers,
                        model=model,
                        out_path=across_theta_plot,
                    )

                    coverage_plot = runtime.out_dir_mfgp / f"mfgp_{runtime.version}_coverage_summary_iter{iteration}.png"
                    _plot_coverage_summary(y_true_v, y_pred_v, y_std_v, coverage_plot)

                    # Validation parity (per-theta means).
                    validation_parity_plot = runtime.out_dir_mfgp / f"mfgp_{runtime.version}_validation_parity_iter{iteration}.png"
                    _plot_parity(y_true_v, y_pred_v, y_std_v, validation_parity_plot)
            elif verbose:
                print("[warn] Validation CSV missing required columns for extra plots; skipping.")
        except Exception as exc:
            if verbose:
                print(f"[warn] Could not generate validation extra plots: {exc}")
    elif verbose:
        print("[info] path_to_files_validation not found; skipping extra validation plots.")

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
        grid_csv=grid_csv,
        data_plot=data_plot,
        parity_plot=parity_plot,
        mean_std_plot=mean_std_plot,
        residual_plot=residual_plot,
        theta_group_plot_dir=theta_group_plot_dir,
        across_theta_plot=across_theta_plot,
        across_theta_zoom_plot=across_theta_zoom_plot,
        coverage_plot=coverage_plot,
        validation_parity_plot=validation_parity_plot,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean multi-fidelity GP pipeline (no resum helpers)")
    p.add_argument("--config", type=str, default=str(_default_config_path()), help="Path to settings2.yaml")
    p.add_argument("--cnp-csv", type=str, default=None, help="Optional explicit CNP output CSV")
    p.add_argument("--validation-csv", type=str, default=None, help="Optional explicit validation CSV for extra plots")
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
        validation_csv=args.validation_csv,
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
