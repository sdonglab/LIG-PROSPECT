from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.signal import find_peaks  # type: ignore
except Exception:  # pragma: no cover
    find_peaks = None

# Expect the user to include LogIR in-package (copied from PCA_setup.py).
# Suggested location: spectra_bootstrap_v2/logir.py with class LogIR
try:
    from .logir import LogIR  # type: ignore
except Exception:  # pragma: no cover
    LogIR = None  # type: ignore


def ensure_outdir(outdir: Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_predictions_tables(
    outdir: Path,
    prefix: str,
    set_name: str,
    y_set: np.ndarray,
    pred_set: np.ndarray,
    max_wavelength: float = 900.0,
    pad_value: float = -1.0,
) -> Path:
    """Save actual/predicted wavelength table (ALL wavelengths)."""
    outdir = ensure_outdir(outdir)

    pred = pred_set.copy()
    pred[pred < 0] = 0

    w_actual = y_set[:, ::2].astype(float).ravel()
    w_pred = pred[:, ::2].astype(float).ravel()

    def _clean(w: np.ndarray) -> np.ndarray:
        w = w[np.isfinite(w)]
        w = w[w != pad_value]
        w = w[w > 0]
        w = w[w <= max_wavelength]
        return w

    w_actual = _clean(w_actual)
    w_pred = _clean(w_pred)

    n = max(len(w_actual), len(w_pred))
    df = pd.DataFrame({
        f"{set_name}_Actual_Wavelength": np.pad(w_actual, (0, n - len(w_actual)), constant_values=np.nan),
        f"{set_name}_Predicted_Wavelength": np.pad(w_pred, (0, n - len(w_pred)), constant_values=np.nan),
    })
    path = outdir / f"{prefix}-{set_name}_wavelength_data.csv"
    df.to_csv(path, index=False)
    return path


def save_wavelength_histogram(
    outdir: Path,
    prefix: str,
    set_name: str,
    y_set: np.ndarray,
    pred_set: np.ndarray,
    max_wavelength: float = 900.0,
    bins: int = 25,
    pad_value: float = -1.0,
    dpi: int = 300,
) -> tuple[Optional[Path], Optional[Path]]:
    """Save histogram counts CSV + PNG for ALL wavelengths."""
    outdir = ensure_outdir(outdir)

    pred = pred_set.copy()
    pred[pred < 0] = 0

    w_actual = y_set[:, ::2].astype(float).ravel()
    w_pred = pred[:, ::2].astype(float).ravel()

    def _clean(w: np.ndarray) -> np.ndarray:
        w = w[np.isfinite(w)]
        w = w[w != pad_value]
        w = w[w > 0]
        w = w[w <= max_wavelength]
        return w

    w_actual = _clean(w_actual)
    w_pred = _clean(w_pred)

    all_w = np.concatenate([w_actual, w_pred]) if (len(w_actual) and len(w_pred)) else (w_actual if len(w_actual) else w_pred)
    if len(all_w) == 0:
        return None, None

    bin_edges = np.histogram_bin_edges(all_w, bins=int(bins))
    act_counts, _ = np.histogram(w_actual, bins=bin_edges)
    pred_counts, _ = np.histogram(w_pred, bins=bin_edges)

    hist_df = pd.DataFrame({
        "bin_left": bin_edges[:-1],
        "bin_right": bin_edges[1:],
        "actual_count": act_counts,
        "predicted_count": pred_counts,
    })
    counts_path = outdir / f"{prefix}-{set_name}_wavelength_hist_counts.csv"
    hist_df.to_csv(counts_path, index=False)

    plt.figure(figsize=(6, 4), dpi=dpi)
    plt.hist(w_actual, bins=bin_edges, alpha=0.6, label="Actual", edgecolor="black")
    plt.hist(w_pred, bins=bin_edges, alpha=0.6, label="Predicted", edgecolor="black")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Frequency")
    plt.title(f"{prefix}-{set_name}: Actual vs Predicted Wavelengths (Test Set)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    png_path = outdir / f"{prefix}-{set_name}_wavelength_hist.png"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return counts_path, png_path

def analyze_and_save_peaks(
    outdir: Path,
    prefix: str,
    set_name: str,
    y_set: np.ndarray,
    pred_set: np.ndarray,
    sigma: float = 0.2,
    peak_height: float = 0.0,
    dpi: int = 300,
    # notebook-like histogram controls
    hist_bins: int | np.ndarray = 25,
    hist_xlim: tuple[float, float] | None = None,
    hist_ylim: tuple[float, float] | None = None,
    save_first5_overlay: bool = False,
    debug_index: int | None = None,
) -> dict:
    """Peak analysis + CSV/PNG outputs (requires LogIR + scipy)."""
    outdir = ensure_outdir(outdir)

    if LogIR is None:
        raise ImportError("LogIR not found. Add it to the package (e.g., spectra_bootstrap_v2/logir.py).")
    if find_peaks is None:
        raise ImportError("scipy is required for peak analysis (pip install scipy).")

    # match notebook behavior: clip negative predictions to 0
    pred = pred_set.copy()
    pred[pred < 0] = 0

    identified_actual, identified_pred = [], []

    # notebook-like counters
    zero_peaks = one_peak = two_peaks = more_than_two_peaks = 0

    # -----------------------------
    # 1) Find top-3 peaks per sample
    # -----------------------------
    for i in range(len(pred)):

        # optional debug prints (only for one index)
        if (debug_index is not None) and (i == debug_index):
            w = y_set[i, ::2]
            f = y_set[i, 1::2]
            print("DEBUG SAMPLE:", i)
            print("min/max wavelength:", np.nanmin(w), np.nanmax(w))
            print("any pad -1 in wavelength?", np.any(w == -1))
            print("any pad -1 in f?", np.any(f == -1))
            print("any negative f?", np.any(f < 0))
            print("first 10 (w,f) pairs:", list(zip(w[:10], f[:10])))
            print("sigma passed:", sigma)

        # ALWAYS construct LogIR for every i
        logir_pred = LogIR("predictionplot", list(pred[i, ::2]), list(pred[i, 1::2]))
        logir_act  = LogIR("actualplot",     list(y_set[i, ::2]), list(y_set[i, 1::2]))

        xs, ys = logir_act.get_data(sigma=float(sigma))
        px, py = logir_pred.get_data(sigma=float(sigma))

        # Force numpy arrays (prevents dtype/list quirks)
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        px = np.asarray(px, dtype=float)
        py = np.asarray(py, dtype=float)

        # Optional overlays for first 5
        if save_first5_overlay and i < 5:
            plt.figure(figsize=(6, 4), dpi=dpi)
            plt.plot(xs, ys, label="Actual")
            plt.plot(px, py, label="Predicted")
            plt.title(f"{prefix}-{set_name}: Sample {i}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                outdir / f"{prefix}-{set_name.lower()}_sample_{i}_overlay.png",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close()

        peaks_act, _  = find_peaks(ys, height=float(peak_height))
        peaks_pred, _ = find_peaks(py, height=float(peak_height))

        if (debug_index is not None) and (i == debug_index):
            print("peaks_act xs:", xs[peaks_act])

        # peak count stats (matches notebook intent)
        n_act = len(peaks_act)
        if n_act == 0:
            zero_peaks += 1
        elif n_act == 1:
            one_peak += 1
        elif n_act == 2:
            two_peaks += 1
        else:
            more_than_two_peaks += 1

        act_sorted  = np.sort(xs[peaks_act])[:3]  if len(peaks_act)  else np.array([])
        pred_sorted = np.sort(px[peaks_pred])[:3] if len(peaks_pred) else np.array([])

        identified_actual.append([i] + list(act_sorted)  + [np.nan] * (3 - len(act_sorted)))
        identified_pred.append([i] + list(pred_sorted) + [np.nan] * (3 - len(pred_sorted)))

    # -----------------------------
    # 2) Save structured peak tables
    # -----------------------------
    df_actual = pd.DataFrame(identified_actual, columns=["Sample Index", "Peak 1", "Peak 2", "Peak 3"])
    df_pred   = pd.DataFrame(identified_pred,   columns=["Sample Index", "Peak 1", "Peak 2", "Peak 3"])

    actual_path = outdir / f"{prefix}-{set_name.lower()}_actual_peaks.csv"
    pred_path   = outdir / f"{prefix}-{set_name.lower()}_predicted_peaks.csv"
    df_actual.to_csv(actual_path, index=False)
    df_pred.to_csv(pred_path, index=False)

    # -----------------------------
    # 3) Histogram data (notebook-like: treat as independent dists)
    # -----------------------------
    all_actual = df_actual[["Peak 1", "Peak 2", "Peak 3"]].to_numpy().flatten()
    all_pred   = df_pred[["Peak 1", "Peak 2", "Peak 3"]].to_numpy().flatten()
    all_actual = all_actual[~np.isnan(all_actual)]
    all_pred   = all_pred[~np.isnan(all_pred)]

    # pad columns ONLY for csv alignment (no truncation)
    max_len = max(len(all_actual), len(all_pred))
    actual_pad = np.full(max_len, np.nan); actual_pad[:len(all_actual)] = all_actual
    pred_pad   = np.full(max_len, np.nan); pred_pad[:len(all_pred)]     = all_pred

    hist_df = pd.DataFrame({"Actual_Peak": actual_pad, "Predicted_Peak": pred_pad})
    hist_path = outdir / f"{prefix}-{set_name.lower()}_histogram_peaks_data.csv"
    hist_df.to_csv(hist_path, index=False)

    # -----------------------------
    # 4) Histogram PNG (notebook-like limits)
    # -----------------------------
    peak_hist_png = None
    if len(all_actual) and len(all_pred):
        plt.figure(figsize=(6, 4), dpi=dpi)
        plt.hist(all_actual, bins=hist_bins, alpha=0.6, label="Actual")
        plt.hist(all_pred,   bins=hist_bins, alpha=0.6, label="Predicted")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Frequency")
        plt.title(f"{prefix}-{set_name}: Peak Histogram")
        plt.legend()
        if hist_ylim is not None:
            plt.ylim(hist_ylim[0], hist_ylim[1])
        if hist_xlim is not None:
            plt.xlim(hist_xlim[0], hist_xlim[1])
        plt.tight_layout()
        peak_hist_png = outdir / f"{prefix}-{set_name.lower()}_histogram_peaks.png"
        plt.savefig(peak_hist_png, dpi=dpi, bbox_inches="tight")
        plt.close()

    # -----------------------------
    # 5) Sorted scatter table + PNG
    # -----------------------------
    df_merge = df_actual.merge(df_pred, on="Sample Index", suffixes=("_actual", "_predicted"))
    df_merge = df_merge.dropna(subset=["Peak 1_actual", "Peak 1_predicted"])
    df_sorted = df_merge.sort_values("Peak 1_actual").reset_index(drop=True)

    sorted_scatter_path = outdir / f"{prefix}-{set_name.lower()}_sorted_scatter_peaks_data.csv"
    df_sorted.to_csv(sorted_scatter_path, index=False)

    x = np.arange(len(df_sorted))
    plt.figure(figsize=(8, 4), dpi=dpi)
    for k in [1, 2, 3]:
        plt.scatter(x, df_sorted[f"Peak {k}_actual"],    s=18, label=f"Peak{k} Actual")
        plt.scatter(x, df_sorted[f"Peak {k}_predicted"], s=18, marker="x", label=f"Peak{k} Pred")
    plt.axhline(y=550, color="gray", linestyle="--", linewidth=1.0)
    plt.xlabel("Sorted sample index (by Peak1 actual)")
    plt.ylabel("Wavelength (nm)")
    plt.title(f"{prefix}-{set_name}: Peaks")
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize=8)
    plt.tight_layout()
    scatter_png = outdir / f"{prefix}-{set_name.lower()}_scatter_peaks.png"
    plt.savefig(scatter_png, dpi=dpi, bbox_inches="tight")
    plt.close()

    return {
        "actual_peaks_csv": actual_path,
        "pred_peaks_csv": pred_path,
        "peaks_hist_csv": hist_path,
        "peaks_hist_png": peak_hist_png,
        "sorted_scatter_csv": sorted_scatter_path,
        "scatter_png": scatter_png,
        "peak_count_stats": {
            "0_peaks": int(zero_peaks),
            "1_peak": int(one_peak),
            "2_peaks": int(two_peaks),
            ">2_peaks": int(more_than_two_peaks),
        },
    }
