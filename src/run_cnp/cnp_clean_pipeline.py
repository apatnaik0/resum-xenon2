#!/usr/bin/env python3
"""Self-contained CNP pipeline for Xenon HDF5 data.

This module intentionally avoids any dependency on the `resum` package.
It provides:
- HDF5 event sampling
- Deterministic Conditional Neural Process model
- Training loop with history/plots/checkpoint
- Prediction/export pipeline compatible with downstream MFGP CSV usage
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# -----------------------------
# Configuration and utilities
# -----------------------------


@dataclass
class CNPRuntimeConfig:
    config_path: Path
    version: str
    train_dir: Path
    predict_dirs: List[Path]
    predict_iterations: List[int]
    predict_fidelities: List[int]
    out_dir: Path
    theta_headers: List[str]
    theta_min: List[float]
    theta_max: List[float]
    phi_headers: List[str]
    target_headers: List[str]
    target_range: List[float]
    context_ratio: float
    epochs: int
    batch_size_train: int
    files_per_batch_train: int
    batch_size_predict: List[int]
    files_per_batch_predict: int
    ratio_testing_vs_training: float
    plot_after: int
    use_data_augmentation: str | bool
    use_beta: List[float]
    seed: int


def _as_float_fraction(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if "/" in text:
            a, b = text.split("/", 1)
            return float(a) / float(b)
        return float(text)
    return default


def _resolve_path(path_value: str | Path, base: Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (base / p).resolve()


def load_runtime_config(config_path: str | Path, seed: int = 42) -> CNPRuntimeConfig:
    config_path = Path(config_path).resolve()
    raw = yaml.safe_load(config_path.read_text())

    cnp = raw.get("cnp_settings", {})
    sim = raw.get("simulation_settings", {})
    paths = raw.get("path_settings", {})
    base = config_path.parent

    predict_dirs = [_resolve_path(p, base) for p in paths.get("path_to_files_predict", [])]
    predict_iterations = [int(x) for x in paths.get("iteration", [0] * len(predict_dirs))]
    predict_fidelities = [int(x) for x in paths.get("fidelity", [0] * len(predict_dirs))]

    if len(predict_iterations) < len(predict_dirs):
        predict_iterations.extend([0] * (len(predict_dirs) - len(predict_iterations)))
    if len(predict_fidelities) < len(predict_dirs):
        predict_fidelities.extend([0] * (len(predict_dirs) - len(predict_fidelities)))

    return CNPRuntimeConfig(
        config_path=config_path,
        version=str(paths.get("version", "v_clean")),
        train_dir=_resolve_path(paths["path_to_files_train"], base),
        predict_dirs=predict_dirs,
        predict_iterations=predict_iterations,
        predict_fidelities=predict_fidelities,
        out_dir=_resolve_path(paths.get("path_out_cnp", "../xenon/out/cnp"), base),
        theta_headers=list(sim.get("theta_headers", ["scint_x", "scint_y"])),
        theta_min=[float(x) for x in sim.get("theta_min", [0.0, 0.0])],
        theta_max=[float(x) for x in sim.get("theta_max", [1.0, 1.0])],
        phi_headers=list(sim.get("phi_labels", ["initial_m_x", "initial_m_y", "initial_m_z"])),
        target_headers=list(sim.get("target_headers", ["tag_final"])),
        target_range=[float(x) for x in sim.get("target_range", [0, 1])],
        context_ratio=float(cnp.get("context_ratio", 1 / 3)),
        epochs=int(cnp.get("training_epochs", 15)),
        batch_size_train=int(cnp.get("batch_size_train", 4096)),
        files_per_batch_train=int(cnp.get("files_per_batch_train", 32)),
        batch_size_predict=[int(x) for x in cnp.get("batch_size_predict", [30000, 100000])],
        files_per_batch_predict=int(cnp.get("files_per_batch_predict", 1)),
        ratio_testing_vs_training=_as_float_fraction(cnp.get("ratio_testing_vs_training", "1/40"), default=1 / 40),
        plot_after=int(cnp.get("plot_after", 1000)),
        use_data_augmentation=cnp.get("use_data_augmentation", False),
        use_beta=list(cnp.get("use_beta", [])) if cnp.get("use_beta") is not None else [],
        seed=seed,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# HDF5 data access
# -----------------------------


@dataclass
class EventBatch:
    x: torch.Tensor
    y: torch.Tensor


class H5EventPool:
    """Event sampler across multiple H5 files.

    Features x = [theta, phi], target y = target headers.
    """

    def __init__(
        self,
        directory: str | Path,
        theta_headers: Sequence[str],
        phi_headers: Sequence[str],
        target_headers: Sequence[str],
        use_mixedup: bool = False,
        seed: int = 42,
        cache_files: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.theta_headers = list(theta_headers)
        self.phi_headers = list(phi_headers)
        self.target_headers = list(target_headers)
        self.use_mixedup = bool(use_mixedup)
        self.rng = np.random.default_rng(seed)
        self.cache_files = cache_files
        self._cache: Dict[Path, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        if not self.directory.exists():
            raise FileNotFoundError(f"H5 directory does not exist: {self.directory}")
        self.files = sorted([p for p in self.directory.glob("*.h5") if p.is_file()])
        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in {self.directory}")

    def _decode_labels(self, arr: np.ndarray) -> List[str]:
        out = []
        for item in arr:
            out.append(item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item))
        return out

    def _load_one(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.cache_files and file_path in self._cache:
            return self._cache[file_path]

        with h5py.File(file_path, "r") as f:
            theta = np.asarray(f["theta"], dtype=np.float32)
            phi_key = "phi_mixedup" if self.use_mixedup and "phi_mixedup" in f else "phi"
            target_key = "target_mixedup" if self.use_mixedup and "target_mixedup" in f else "target"

            phi = np.asarray(f[phi_key], dtype=np.float32)
            target = np.asarray(f[target_key], dtype=np.float32)
            if self.use_mixedup and (len(phi) == 0 or len(target) == 0):
                # Fallback to original datasets when mixup is unavailable/empty for this file.
                phi = np.asarray(f["phi"], dtype=np.float32)
                target = np.asarray(f["target"], dtype=np.float32)

            if theta.ndim != 1:
                theta = np.asarray(theta[0], dtype=np.float32)
            if target.ndim == 1:
                target = target.reshape(-1, 1)

            # Optional label validation when available.
            if "theta_headers" in f:
                labels = self._decode_labels(np.asarray(f["theta_headers"]))
                if labels[: len(self.theta_headers)] != self.theta_headers:
                    raise ValueError(
                        f"Theta headers mismatch in {file_path.name}. "
                        f"Expected {self.theta_headers}, got {labels}"
                    )
            if "phi_labels" in f:
                labels = self._decode_labels(np.asarray(f["phi_labels"]))
                if labels[: len(self.phi_headers)] != self.phi_headers:
                    raise ValueError(
                        f"Phi labels mismatch in {file_path.name}. "
                        f"Expected {self.phi_headers}, got {labels}"
                    )
            if "target_headers" in f:
                labels = self._decode_labels(np.asarray(f["target_headers"]))
                if labels[: len(self.target_headers)] != self.target_headers:
                    raise ValueError(
                        f"Target headers mismatch in {file_path.name}. "
                        f"Expected {self.target_headers}, got {labels}"
                    )

        if self.cache_files:
            self._cache[file_path] = (theta, phi, target)
        return theta, phi, target

    def sample_batch(self, batch_size: int, files_per_batch: int) -> EventBatch:
        chosen = self._choose_files(files_per_batch)
        per_file = max(1, batch_size // len(chosen))

        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []

        for f in chosen:
            theta, phi, target = self._load_one(f)
            n = len(target)
            if n == 0:
                continue
            idx = self.rng.integers(0, n, size=per_file)
            theta_rep = np.repeat(theta.reshape(1, -1), repeats=per_file, axis=0)
            x = np.hstack([theta_rep, phi[idx]])
            y = target[idx]
            xs.append(x)
            ys.append(y)

        if not xs:
            raise RuntimeError("Could not sample non-empty batch from H5 files")

        x_arr = np.vstack(xs).astype(np.float32)
        y_arr = np.vstack(ys).astype(np.float32)
        return EventBatch(x=torch.from_numpy(x_arr), y=torch.from_numpy(y_arr))

    def iter_file_data(self) -> Iterable[Tuple[Path, np.ndarray, np.ndarray, np.ndarray]]:
        for f in self.files:
            theta, phi, target = self._load_one(f)
            theta_rep = np.repeat(theta.reshape(1, -1), repeats=len(target), axis=0)
            x = np.hstack([theta_rep, phi]).astype(np.float32)
            y = target.astype(np.float32)
            yield f, x, y, theta.astype(np.float32)

    def _choose_files(self, files_per_batch: int) -> List[Path]:
        k = min(files_per_batch, len(self.files))
        if k == len(self.files):
            return self.files
        idx = self.rng.choice(len(self.files), size=k, replace=False)
        return [self.files[i] for i in idx]


# -----------------------------
# CNP model
# -----------------------------


class MLP(nn.Module):
    def __init__(self, sizes: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeterministicCNP(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        repr_dim: int = 32,
        hidden: int = 128,
        dropout: float = 0.1,
        encoder_sizes: Optional[Sequence[int]] = None,
        decoder_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        # Legacy-compatible deep architecture (same shape style as old notebooks):
        # encoder: [d_in, 32, 64, 128, 128, 128, 64, 48, representation_size]
        # decoder: [representation_size + d_x, 32, 64, 128, 128, 128, 64, 48, d_out]
        # where d_out = y_dim * 2 (prediction mean/logit + uncertainty head)
        if encoder_sizes is None:
            encoder_sizes = [x_dim + y_dim, 32, 64, 128, 128, 128, 64, 48, repr_dim]
        if decoder_sizes is None:
            decoder_sizes = [x_dim + repr_dim, 32, 64, 128, 128, 128, 64, 48, y_dim * 2]

        self.encoder = MLP(encoder_sizes, dropout=dropout)
        self.decoder = MLP(decoder_sizes, dropout=dropout)

    def encode(self, context_x: torch.Tensor, context_y: torch.Tensor) -> torch.Tensor:
        h = torch.cat([context_x, context_y], dim=-1)
        r_i = self.encoder(h)
        return r_i.mean(dim=0, keepdim=True)  # [1, repr_dim]

    def forward(self, context_x: torch.Tensor, context_y: torch.Tensor, target_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.encode(context_x, context_y)
        r_rep = r.expand(target_x.shape[0], -1)
        out = self.decoder(torch.cat([target_x, r_rep], dim=-1))
        logits = out[:, : self.y_dim]
        raw_sigma = out[:, self.y_dim :]
        sigma = F.softplus(raw_sigma) + 1e-6
        return logits, sigma

    @torch.no_grad()
    def predict_proba_mc(
        self,
        context_x: torch.Tensor,
        context_y: torch.Tensor,
        target_x: torch.Tensor,
        mc_samples: int = 30,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.train()  # Enable dropout for MC uncertainty.

        preds = []
        sigmas = []
        for _ in range(mc_samples):
            logits, sigma = self.forward(context_x, context_y, target_x)
            preds.append(torch.sigmoid(logits))
            sigmas.append(sigma)
        pred = torch.stack(preds, dim=0)
        sigma_stack = torch.stack(sigmas, dim=0)
        mean = pred.mean(dim=0)
        # Blend epistemic (MC variance of probabilities) + aleatoric (model sigma head).
        epistemic = pred.std(dim=0, unbiased=False)
        aleatoric = sigma_stack.mean(dim=0)
        std = torch.sqrt(epistemic.pow(2) + aleatoric.pow(2))

        if not was_training:
            self.eval()
        return mean, std


# -----------------------------
# Training and prediction
# -----------------------------


@dataclass
class TrainResult:
    model_path: Path
    history_csv: Path
    history_plot: Path
    sample_plot: Path


@dataclass
class PredictResult:
    csv_path: Path
    heatmap_path: Path
    error_heatmap_path: Path


def split_context_target(
    x: torch.Tensor,
    y: torch.Tensor,
    context_ratio: float,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    if n < 4:
        raise ValueError("Batch too small; need at least 4 samples for context-target split")

    min_context = max(2, int(0.1 * n))
    max_context = max(min_context + 1, int(context_ratio * n))
    num_context = int(rng.integers(min_context, min(max_context, n - 1) + 1))

    perm = rng.permutation(n)
    context_idx = perm[:num_context]

    context_x = x[context_idx]
    context_y = y[context_idx]

    target_x = x
    target_y = y
    return context_x, context_y, target_x, target_y


def _plot_training_history(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(df["step"], df["train_bce"], label="train BCE")
    ax.plot(df["step"], df["val_bce"], label="val BCE")
    ax.set_xlabel("step")
    ax.set_ylabel("BCE")
    ax.set_title("CNP training history")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_sample_predictions(pred: np.ndarray, truth: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    order = np.argsort(pred)
    ax.plot(pred[order], label="pred", lw=1.5)
    ax.plot(truth[order], label="truth", lw=1.0)
    ax.set_xlabel("sample index (sorted by prediction)")
    ax.set_ylabel("target")
    ax.set_title("Sample batch prediction vs truth")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _split_signal_background(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    sig_idx = np.where(y_true > threshold)[0]
    bkg_idx = np.where(y_true <= threshold)[0]
    return y_true[sig_idx], y_pred[sig_idx], y_true[bkg_idx], y_pred[bkg_idx]


def _plot_train_val_snapshot(
    train_pred: np.ndarray,
    train_truth: np.ndarray,
    val_pred: np.ndarray,
    val_truth: np.ndarray,
    out_path: Path,
    step: int,
    train_loss: float,
    val_loss: float,
    target_range: Sequence[float] = (0.0, 1.0),
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Training Iteration {step}", fontsize=10)

    tr_sig_true, tr_sig_pred, tr_bkg_true, tr_bkg_pred = _split_signal_background(train_truth, train_pred)
    va_sig_true, va_sig_pred, va_bkg_true, va_bkg_pred = _split_signal_background(val_truth, val_pred)

    bins = 100
    label_bkg = (3 / 255, 37 / 255, 46 / 255)
    pred_bkg = (113 / 255, 150 / 255, 159 / 255)
    label_sig = "orangered"
    pred_sig = "coral"

    if len(tr_sig_true) > 0:
        axes[0].hist(tr_sig_true, range=target_range, bins=bins, color=label_sig, alpha=1.0, label="label (signal)")
    axes[0].hist(tr_bkg_true, range=target_range, bins=bins, color=label_bkg, alpha=0.8, label="label (bkg)")
    axes[0].hist(tr_bkg_pred, range=target_range, bins=bins, color=pred_bkg, alpha=0.8, label="network (bkg)")
    if len(tr_sig_pred) > 0:
        axes[0].hist(tr_sig_pred, range=target_range, bins=bins, color=pred_sig, alpha=0.8, label="network (signal)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel(r"$y_{CNP}$")
    axes[0].set_title(f"Training (loss {train_loss:.4f})", fontsize=10)

    if len(va_sig_true) > 0:
        axes[1].hist(va_sig_true, range=target_range, bins=bins, color=label_sig, alpha=1.0, label="label (signal)")
    axes[1].hist(va_bkg_true, range=target_range, bins=bins, color=label_bkg, alpha=0.8, label="label (bkg)")
    axes[1].hist(va_bkg_pred, range=target_range, bins=bins, color=pred_bkg, alpha=0.8, label="network (bkg)")
    if len(va_sig_pred) > 0:
        axes[1].hist(va_sig_pred, range=target_range, bins=bins, color=pred_sig, alpha=0.8, label="network (signal)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel(r"$y_{CNP}$")
    axes[1].set_title(f"Testing (loss {val_loss:.4f})", fontsize=10)

    handles0, labels0 = axes[0].get_legend_handles_labels()
    if handles0:
        axes[0].legend(loc="upper right", fontsize=8, frameon=True)
    handles1, labels1 = axes[1].get_legend_handles_labels()
    if handles1:
        axes[1].legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def train_cnp(
    runtime: CNPRuntimeConfig,
    steps_per_epoch: int = 5000,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    repr_dim: int = 32,
    hidden: int = 128,
    dropout: float = 0.1,
    monitor_every: Optional[int] = None,
    show_monitor_plots: bool = False,
    device: Optional[str] = None,
) -> TrainResult:
    runtime.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(runtime.seed)
    use_mixup = str(runtime.use_data_augmentation).strip().lower() == "mixup"
    if use_mixup:
        print("Training with mixup datasets: phi_mixedup / target_mixedup (if present).")

    pool = H5EventPool(
        runtime.train_dir,
        theta_headers=runtime.theta_headers,
        phi_headers=runtime.phi_headers,
        target_headers=runtime.target_headers,
        use_mixedup=use_mixup,
        seed=runtime.seed,
        cache_files=True,
    )

    x_dim = len(runtime.theta_headers) + len(runtime.phi_headers)
    y_dim = len(runtime.target_headers)

    model = DeterministicCNP(x_dim=x_dim, y_dim=y_dim, repr_dim=repr_dim, hidden=hidden, dropout=dropout)

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(runtime.seed)

    val_batch_size = max(128, int(runtime.batch_size_train * runtime.ratio_testing_vs_training))
    monitor_every = int(monitor_every if monitor_every is not None else runtime.plot_after)
    history_rows: List[Dict[str, float]] = []
    global_step = 0

    for epoch in range(runtime.epochs):
        model.train()
        for _ in range(steps_per_epoch):
            batch = pool.sample_batch(runtime.batch_size_train, runtime.files_per_batch_train)
            x = batch.x.to(dev)
            y = batch.y.to(dev)

            cx, cy, tx, ty = split_context_target(x, y, runtime.context_ratio, rng)
            logits, sigma = model(cx, cy, tx)
            train_bce = F.binary_cross_entropy_with_logits(logits, ty)

            optimizer.zero_grad()
            train_bce.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Validation step from an independent sampled mini-batch.
            with torch.no_grad():
                val_batch = pool.sample_batch(val_batch_size, max(1, runtime.files_per_batch_train // 2))
                vx = val_batch.x.to(dev)
                vy = val_batch.y.to(dev)
                vcx, vcy, vtx, vty = split_context_target(vx, vy, runtime.context_ratio, rng)
                vlogits, vsigma = model(vcx, vcy, vtx)
                val_bce = F.binary_cross_entropy_with_logits(vlogits, vty)

            history_rows.append(
                {
                    "epoch": float(epoch),
                    "step": float(global_step),
                    "train_bce": float(train_bce.item()),
                    "val_bce": float(val_bce.item()),
                }
            )

            if monitor_every > 0 and global_step % monitor_every == 0:
                ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"{ts} Iteration: {epoch}/{global_step}, "
                    f"train BCE: {train_bce.item():.4f}, val BCE: {val_bce.item():.4f}"
                )

                with torch.no_grad():
                    train_prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                    train_true = ty.detach().cpu().numpy().reshape(-1)
                    val_prob = torch.sigmoid(vlogits).detach().cpu().numpy().reshape(-1)
                    val_true = vty.detach().cpu().numpy().reshape(-1)

                monitor_plot = runtime.out_dir / f"cnp_{runtime.version}_monitor_step_{global_step}.png"
                _plot_train_val_snapshot(
                    train_pred=train_prob,
                    train_truth=train_true,
                    val_pred=val_prob,
                    val_truth=val_true,
                    out_path=monitor_plot,
                    step=global_step,
                    train_loss=float(train_bce.item()),
                    val_loss=float(val_bce.item()),
                    target_range=runtime.target_range,
                )
                latest_plot = runtime.out_dir / f"cnp_{runtime.version}_monitor_latest.png"
                _plot_train_val_snapshot(
                    train_pred=train_prob,
                    train_truth=train_true,
                    val_pred=val_prob,
                    val_truth=val_true,
                    out_path=latest_plot,
                    step=global_step,
                    train_loss=float(train_bce.item()),
                    val_loss=float(val_bce.item()),
                    target_range=runtime.target_range,
                )
                if show_monitor_plots:
                    try:
                        from IPython.display import Image, display
                        display(Image(filename=str(monitor_plot)))
                    except Exception:
                        pass

                hist_df_live = pd.DataFrame(history_rows)
                history_csv_live = runtime.out_dir / f"cnp_{runtime.version}_history_{runtime.epochs}epochs.csv"
                hist_df_live.to_csv(history_csv_live, index=False)
                history_plot_live = runtime.out_dir / f"cnp_{runtime.version}_training_curve_{runtime.epochs}epochs.png"
                _plot_training_history(hist_df_live, history_plot_live)

            global_step += 1

        latest = history_rows[-1]
        print(
            f"Epoch {epoch + 1}/{runtime.epochs} | "
            f"step={int(latest['step'])} | "
            f"train_bce={latest['train_bce']:.5f} | "
            f"val_bce={latest['val_bce']:.5f}"
        )

    # Save model and artifacts.
    model_path = runtime.out_dir / f"cnp_{runtime.version}_model_{runtime.epochs}epochs.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_dim": x_dim,
            "y_dim": y_dim,
            "repr_dim": repr_dim,
            "hidden": hidden,
            "dropout": dropout,
            "encoder_sizes": [x_dim + y_dim, 32, 64, 128, 128, 128, 64, 48, repr_dim],
            "decoder_sizes": [x_dim + repr_dim, 32, 64, 128, 128, 128, 64, 48, y_dim * 2],
            "theta_headers": runtime.theta_headers,
            "phi_headers": runtime.phi_headers,
            "target_headers": runtime.target_headers,
            "epochs": runtime.epochs,
            "version": runtime.version,
        },
        model_path,
    )

    hist_df = pd.DataFrame(history_rows)
    history_csv = runtime.out_dir / f"cnp_{runtime.version}_history_{runtime.epochs}epochs.csv"
    hist_df.to_csv(history_csv, index=False)

    history_plot = runtime.out_dir / f"cnp_{runtime.version}_training_curve_{runtime.epochs}epochs.png"
    _plot_training_history(hist_df, history_plot)

    # One sample-batch qualitative prediction plot.
    model.eval()
    with torch.no_grad():
        sample = pool.sample_batch(min(4096, runtime.batch_size_train), runtime.files_per_batch_train)
        sx = sample.x.to(dev)
        sy = sample.y.to(dev)
        scx, scy, stx, sty = split_context_target(sx, sy, runtime.context_ratio, rng)
        slogits, ssigma = model(scx, scy, stx)
        probs = torch.sigmoid(slogits).cpu().numpy().reshape(-1)
        truth = sty.cpu().numpy().reshape(-1)

    sample_plot = runtime.out_dir / f"cnp_{runtime.version}_sample_predictions_{runtime.epochs}epochs.png"
    _plot_sample_predictions(probs, truth, sample_plot)

    return TrainResult(
        model_path=model_path,
        history_csv=history_csv,
        history_plot=history_plot,
        sample_plot=sample_plot,
    )


def load_model_checkpoint(model_path: str | Path, device: Optional[str] = None) -> DeterministicCNP:
    model_path = Path(model_path)
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(model_path, map_location=dev)

    model = DeterministicCNP(
        x_dim=int(ckpt["x_dim"]),
        y_dim=int(ckpt["y_dim"]),
        repr_dim=int(ckpt["repr_dim"]),
        hidden=int(ckpt["hidden"]),
        dropout=float(ckpt["dropout"]),
        encoder_sizes=ckpt.get("encoder_sizes"),
        decoder_sizes=ckpt.get("decoder_sizes"),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()
    return model


def _safe_log_bernoulli(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(y_prob, eps, 1 - eps)
    y = np.clip(y_true, 0, 1)
    ll = y * np.log(p) + (1 - y) * np.log(1 - p)
    return float(ll.mean())


def _bce_numpy(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-6) -> float:
    return float(-_safe_log_bernoulli(y_true, y_prob, eps=eps))


def _plot_prediction_heatmaps(
    df: pd.DataFrame,
    out_path: Path,
    err_out_path: Path,
) -> None:
    x = df["scint_x"].to_numpy(dtype=float)
    y = df["scint_y"].to_numpy(dtype=float)
    y_raw = df["y_raw"].to_numpy(dtype=float)
    y_cnp = df["y_cnp"].to_numpy(dtype=float)

    # Filter non-finite rows first to prevent interpolation failures.
    finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_raw) & np.isfinite(y_cnp)
    x = x[finite_mask]
    y = y[finite_mask]
    y_raw = y_raw[finite_mask]
    y_cnp = y_cnp[finite_mask]

    # Robust bounds: focus on the main cluster and ignore extreme outliers.
    if len(x) >= 20:
        q_lo, q_hi = 0.02, 0.98
        x_lo_q, x_hi_q = np.quantile(x, [q_lo, q_hi])
        y_lo_q, y_hi_q = np.quantile(y, [q_lo, q_hi])
        inlier_mask = (x >= x_lo_q) & (x <= x_hi_q) & (y >= y_lo_q) & (y <= y_hi_q)
        if int(inlier_mask.sum()) >= 8:
            x = x[inlier_mask]
            y = y[inlier_mask]
            y_raw = y_raw[inlier_mask]
            y_cnp = y_cnp[inlier_mask]

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    # Small padding to avoid clipping markers on edges.
    pad_frac = 0.01
    x_lo, x_hi = x_min - pad_frac * x_span, x_max + pad_frac * x_span
    y_lo, y_hi = y_min - pad_frac * y_span, y_max + pad_frac * y_span

    def _interp_grid(px: np.ndarray, py: np.ndarray, pv: np.ndarray, n: int = 100) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        gx, gy = np.mgrid[x_lo:x_hi:complex(0, n), y_lo:y_hi:complex(0, n)]
        try:
            from scipy.interpolate import griddata  # type: ignore

            gz = griddata((px, py), pv, (gx, gy), method="cubic")
            if gz is None:
                return gx, gy, None
            if np.isnan(gz).all():
                gz = griddata((px, py), pv, (gx, gy), method="linear")
            return gx, gy, gz
        except Exception:
            return gx, gy, None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    used_contour = len(x) >= 4
    if used_contour:
        grid_x, grid_y, grid_y_raw = _interp_grid(x, y, y_raw, n=100)
        _, _, grid_y_cnp = _interp_grid(x, y, y_cnp, n=100)
        if grid_y_raw is None or grid_y_cnp is None:
            used_contour = False
        else:
            im1 = axes[0].contourf(grid_x, grid_y, grid_y_raw, levels=20, cmap="viridis")
            axes[0].scatter(x, y, c=y_raw, s=10, cmap="viridis", edgecolor="white", linewidth=0.5, alpha=0.6)
            axes[0].set_xlabel("scint_x (mm)", fontsize=12)
            axes[0].set_ylabel("scint_y (mm)", fontsize=12)
            axes[0].set_title("Ground Truth (y_raw)", fontsize=14, fontweight="bold")
            cb1 = plt.colorbar(im1, ax=axes[0])
            cb1.set_label("Detection Rate", fontsize=11)

            im2 = axes[1].contourf(grid_x, grid_y, grid_y_cnp, levels=20, cmap="viridis")
            axes[1].scatter(x, y, c=y_cnp, s=10, cmap="viridis", edgecolor="white", linewidth=0.5, alpha=0.6)
            axes[1].set_xlabel("scint_x (mm)", fontsize=12)
            axes[1].set_ylabel("scint_y (mm)", fontsize=12)
            axes[1].set_title("CNP Prediction (y_cnp)", fontsize=14, fontweight="bold")
            cb2 = plt.colorbar(im2, ax=axes[1])
            cb2.set_label("Detection Rate", fontsize=11)

    if not used_contour:
        s1 = axes[0].scatter(x, y, c=y_raw, cmap="viridis", s=45, edgecolor="k", linewidth=0.2)
        axes[0].set_title("Ground Truth (y_raw)", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("scint_x")
        axes[0].set_ylabel("scint_y")
        cb1 = plt.colorbar(s1, ax=axes[0])
        cb1.set_label("rate")

        s2 = axes[1].scatter(x, y, c=y_cnp, cmap="viridis", s=45, edgecolor="k", linewidth=0.2)
        axes[1].set_title("CNP Prediction (y_cnp)", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("scint_x")
        axes[1].set_ylabel("scint_y")
        cb2 = plt.colorbar(s2, ax=axes[1])
        cb2.set_label("rate")

    axes[0].set_xlim(x_lo, x_hi)
    axes[1].set_xlim(x_lo, x_hi)
    axes[0].set_ylim(y_lo, y_hi)
    axes[1].set_ylim(y_lo, y_hi)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    err = y_cnp - y_raw
    vmax = float(np.max(np.abs(err))) if len(err) else 1.0
    vmax = vmax if vmax > 1e-12 else 1.0

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    if used_contour:
        grid_x, grid_y, grid_diff = _interp_grid(x, y, err, n=100)
        if grid_diff is not None:
            im_diff = ax2.contourf(grid_x, grid_y, grid_diff, levels=20, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax2.scatter(x, y, c=err, s=10, cmap="RdBu_r", vmin=-vmax, vmax=vmax, edgecolor="white", linewidth=0.5, alpha=0.6)
            cbe = plt.colorbar(im_diff, ax=ax2)
        else:
            se = ax2.scatter(x, y, c=err, cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=50, edgecolor="none")
            cbe = plt.colorbar(se, ax=ax2)
    else:
        se = ax2.scatter(x, y, c=err, cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=50, edgecolor="none")
        cbe = plt.colorbar(se, ax=ax2)

    ax2.set_title("Prediction Error (y_cnp - y_raw)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("scint_x", fontsize=12)
    ax2.set_ylabel("scint_y", fontsize=12)
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(y_lo, y_hi)
    cbe.set_label("Error", fontsize=11)

    fig2.tight_layout()
    fig2.savefig(err_out_path, dpi=180)
    plt.close(fig2)


def predict_cnp(
    runtime: CNPRuntimeConfig,
    model_path: str | Path,
    mc_samples: int = 30,
    output_suffix: Optional[str] = None,
    output_epochs: Optional[int] = None,
    chunk_size: int = 20000,
    device: Optional[str] = None,
) -> PredictResult:
    runtime.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(runtime.seed)

    model_path = Path(model_path)
    model = load_model_checkpoint(model_path, device=device)
    dev = next(model.parameters()).device

    if output_suffix is None:
        first = str(runtime.predict_dirs[0]).lower() if runtime.predict_dirs else ""
        output_suffix = "output_validation" if "validation" in first else "output"
    if output_epochs is None:
        output_epochs = runtime.epochs

    rows: List[Dict[str, float]] = []

    for i, pred_dir in enumerate(runtime.predict_dirs):
        fidelity = runtime.predict_fidelities[i]
        iteration = runtime.predict_iterations[i]

        pool = H5EventPool(
            pred_dir,
            theta_headers=runtime.theta_headers,
            phi_headers=runtime.phi_headers,
            target_headers=runtime.target_headers,
            seed=runtime.seed + i,
            cache_files=True,
        )

        for file_path, x_np, y_np, theta_np in pool.iter_file_data():
            n = len(y_np)
            if n == 0:
                continue

            rng = np.random.default_rng(runtime.seed + i + n)
            n_context = max(2, int(runtime.context_ratio * n))
            n_context = min(n_context, n - 1)
            c_idx = rng.choice(n, size=n_context, replace=False)

            context_x = torch.from_numpy(x_np[c_idx]).to(dev)
            context_y = torch.from_numpy(y_np[c_idx]).to(dev)

            # Predict in chunks to control memory.
            mu_parts: List[np.ndarray] = []
            std_parts: List[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, n, chunk_size):
                    end = min(n, start + chunk_size)
                    tx = torch.from_numpy(x_np[start:end]).to(dev)
                    mu_t, std_t = model.predict_proba_mc(context_x, context_y, tx, mc_samples=mc_samples)
                    mu_parts.append(mu_t.cpu().numpy())
                    std_parts.append(std_t.cpu().numpy())

            mu = np.vstack(mu_parts).reshape(-1, 1)
            std = np.vstack(std_parts).reshape(-1, 1)

            y_raw = y_np.reshape(-1, 1)
            row = {
                "iteration": float(iteration),
                "fidelity": float(fidelity),
                "n_samples": float(n),
                runtime.theta_headers[0]: float(theta_np[0]),
                runtime.theta_headers[1]: float(theta_np[1]),
                "y_cnp": float(mu.mean()),
                "y_cnp_err": float(np.sqrt(np.mean(np.square(std)))),
                "y_raw": float(y_raw.mean()),
                "log_prop": _safe_log_bernoulli(y_raw, mu),
                "bce": _bce_numpy(y_raw, mu),
                "source_file": file_path.name,
            }
            rows.append(row)

    out_csv = runtime.out_dir / f"cnp_{runtime.version}_{output_suffix}_{output_epochs}epochs.csv"
    df = pd.DataFrame(rows)

    required_cols = [
        "iteration",
        "fidelity",
        "n_samples",
        *runtime.theta_headers,
        "y_cnp",
        "y_cnp_err",
        "y_raw",
        "log_prop",
        "bce",
        "source_file",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[required_cols]
    df.to_csv(out_csv, index=False)

    heatmap = runtime.out_dir / f"cnp_{runtime.version}_{output_suffix}_{output_epochs}epochs_heatmaps.png"
    error_heatmap = runtime.out_dir / f"cnp_{runtime.version}_{output_suffix}_{output_epochs}epochs_error_heatmap.png"
    if len(df) > 0:
        _plot_prediction_heatmaps(
            df,
            heatmap,
            error_heatmap,
        )
    else:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No prediction rows generated", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(heatmap, dpi=120)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No prediction rows generated", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(error_heatmap, dpi=120)
        plt.close()

    if len(df):
        mae = float(np.mean(np.abs(df["y_cnp"].to_numpy() - df["y_raw"].to_numpy())))
        rmse = float(np.sqrt(np.mean((df["y_cnp"].to_numpy() - df["y_raw"].to_numpy()) ** 2)))
        print(f"Prediction summary: rows={len(df)}, MAE={mae:.6f}, RMSE={rmse:.6f}")

    return PredictResult(csv_path=out_csv, heatmap_path=heatmap, error_heatmap_path=error_heatmap)


# -----------------------------
# CLI entrypoint
# -----------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean CNP train/predict pipeline (no resum dependency)")
    default_config = Path(__file__).resolve().parents[1] / "xenon" / "settings2.yaml"
    p.add_argument("--config", type=Path, default=default_config, help="Path to settings YAML")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", type=str, default=None, help="Torch device (e.g., cpu, cuda)")

    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train CNP")
    tr.add_argument("--steps-per-epoch", type=int, default=5000)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--weight-decay", type=float, default=0.0)
    tr.add_argument("--repr-dim", type=int, default=32)
    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--dropout", type=float, default=0.1)
    tr.add_argument("--monitor-every", type=int, default=None, help="Log and save monitor plots every N steps (default: settings2 plot_after)")
    tr.add_argument("--show-monitor-plots", action="store_true", help="Display monitor plots inline when running in notebooks/IPython")

    pr = sub.add_parser("predict", help="Run prediction/export")
    pr.add_argument("--model-path", type=Path, required=True)
    pr.add_argument("--mc-samples", type=int, default=30)
    pr.add_argument("--chunk-size", type=int, default=20000)
    pr.add_argument("--output-suffix", type=str, default=None)
    pr.add_argument("--output-epochs", type=int, default=None)

    fu = sub.add_parser("full", help="Train then predict")
    fu.add_argument("--steps-per-epoch", type=int, default=5000)
    fu.add_argument("--lr", type=float, default=1e-4)
    fu.add_argument("--weight-decay", type=float, default=0.0)
    fu.add_argument("--repr-dim", type=int, default=32)
    fu.add_argument("--hidden", type=int, default=128)
    fu.add_argument("--dropout", type=float, default=0.1)
    fu.add_argument("--monitor-every", type=int, default=None, help="Log and save monitor plots every N steps (default: settings2 plot_after)")
    fu.add_argument("--show-monitor-plots", action="store_true", help="Display monitor plots inline when running in notebooks/IPython")
    fu.add_argument("--mc-samples", type=int, default=30)
    fu.add_argument("--chunk-size", type=int, default=20000)
    fu.add_argument("--output-suffix", type=str, default=None)
    fu.add_argument("--output-epochs", type=int, default=None)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    runtime = load_runtime_config(args.config, seed=args.seed)

    if args.cmd == "train":
        result = train_cnp(
            runtime,
            steps_per_epoch=args.steps_per_epoch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            repr_dim=args.repr_dim,
            hidden=args.hidden,
            dropout=args.dropout,
            monitor_every=args.monitor_every,
            show_monitor_plots=args.show_monitor_plots,
            device=args.device,
        )
        print(json.dumps({
            "model_path": str(result.model_path),
            "history_csv": str(result.history_csv),
            "history_plot": str(result.history_plot),
            "sample_plot": str(result.sample_plot),
        }, indent=2))
        return

    if args.cmd == "predict":
        result = predict_cnp(
            runtime,
            model_path=args.model_path,
            mc_samples=args.mc_samples,
            output_suffix=args.output_suffix,
            output_epochs=args.output_epochs,
            chunk_size=args.chunk_size,
            device=args.device,
        )
        print(json.dumps({
            "csv_path": str(result.csv_path),
            "heatmap_path": str(result.heatmap_path),
            "error_heatmap_path": str(result.error_heatmap_path),
        }, indent=2))
        return

    if args.cmd == "full":
        train_result = train_cnp(
            runtime,
            steps_per_epoch=args.steps_per_epoch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            repr_dim=args.repr_dim,
            hidden=args.hidden,
            dropout=args.dropout,
            monitor_every=args.monitor_every,
            show_monitor_plots=args.show_monitor_plots,
            device=args.device,
        )
        predict_result = predict_cnp(
            runtime,
            model_path=train_result.model_path,
            mc_samples=args.mc_samples,
            output_suffix=args.output_suffix,
            output_epochs=args.output_epochs,
            chunk_size=args.chunk_size,
            device=args.device,
        )
        print(json.dumps({
            "model_path": str(train_result.model_path),
            "history_csv": str(train_result.history_csv),
            "history_plot": str(train_result.history_plot),
            "sample_plot": str(train_result.sample_plot),
            "csv_path": str(predict_result.csv_path),
            "heatmap_path": str(predict_result.heatmap_path),
            "error_heatmap_path": str(predict_result.error_heatmap_path),
        }, indent=2))


if __name__ == "__main__":
    main()
