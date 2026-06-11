from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from .splits import SplitBundle, get_or_create_splits

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def _get_test_splits_notebook_exact(n_samples: int, seed_value: int, n_iter: int, test_size: int) -> List[np.ndarray]:
    """
    Notebook-exact test split generation:

      test_set_indices = []
      for it in range(n_iter):
          np.random.seed(seed_value + it)
          test = np.random.choice(np.arange(n_samples), size=test_size, replace=False)
          test_set_indices.append(test)

    NOTE:
    This also sets the global RNG state that the notebook uses for subsequent train sampling.
    """
    all_idx = np.arange(n_samples, dtype=int)
    out: List[np.ndarray] = []
    for it in range(int(n_iter)):
        np.random.seed(int(seed_value) + it)
        test_idx = np.random.choice(all_idx, size=int(test_size), replace=False)
        out.append(np.asarray(test_idx, dtype=int))
    return out


def _load_or_create_bundle(
    cfg: "SingleCfg",
    filenames: Optional[List[str]],
) -> Optional[SplitBundle]:
    """
    If cfg.splits_path is set, load/create the SplitBundle (test+train indices).
    """
    if cfg.splits_path is None:
        return None
    if filenames is None:
        raise ValueError("filenames must be provided when using splits_path (strict matching across descriptors).")

    bundle = get_or_create_splits(
        split_path=Path(cfg.splits_path),
        filenames=filenames,
        seed=int(cfg.seed_value),
        n_iter=int(cfg.n_bootstrap_iterations),
        test_size=int(cfg.test_set_size),
        training_sizes=list(np.asarray(cfg.training_sizes, dtype=int)),
        reuse_if_exists=bool(cfg.reuse_splits_if_exists),
        strict_filename_match=True,
    )
    return bundle
@dataclass
class ModelBundle:
    """Everything needed to reproduce inference for one trained model."""
    method: str
    scaler: StandardScaler
    model: LinearRegression
    reducer: Optional[Any] = None
    best_k: Optional[int] = None
    rmse_by_k: Optional[List[float]] = None
def _fit_predict_one_iter(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cfg: "SingleCfg",
) -> Tuple[np.ndarray, np.ndarray, ModelBundle]:
    """
    Match notebook branches for 'none'/'pca'/'umap'.
    Returns (pred_train, pred_test, bundle)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if cfg.method == "pca":
        reducer = PCA(n_components=min(int(cfg.max_components), X_train_scaled.shape[1]))
        X_train_red = reducer.fit_transform(X_train_scaled)
        X_test_red = reducer.transform(X_test_scaled)

        cv = KFold(n_splits=5, shuffle=True, random_state=1)
        rmse_by_k: List[float] = []
        for k in range(1, X_train_red.shape[1] + 1):
            s = cross_val_score(
                LinearRegression(),
                X_train_red[:, :k],
                y_train,
                cv=cv,
                scoring="neg_root_mean_squared_error",
            ).mean()
            rmse_by_k.append(float(-s))
        best_k = int(np.argmin(rmse_by_k) + 1)

        model = LinearRegression().fit(X_train_red[:, :best_k], y_train)
        pred_train = model.predict(X_train_red[:, :best_k])
        pred_test = model.predict(X_test_red[:, :best_k])
        
        bundle = ModelBundle(
            method="pca",
            scaler=scaler,
            reducer=reducer,
            model=model,
            best_k=best_k,
            rmse_by_k=rmse_by_k,
        )
        return pred_train, pred_test, bundle

    if cfg.method == "umap":
        if umap is None:
            raise ImportError("umap-learn is required for method='umap' (pip install umap-learn).")
        params = dict(n_components=min(int(cfg.max_components), X_train_scaled.shape[1]))
        if cfg.umap_params:
            params.update(cfg.umap_params)
        reducer = umap.UMAP(**params)  # type: ignore
        X_train_red = reducer.fit_transform(X_train_scaled)
        X_test_red = reducer.transform(X_test_scaled)

        cv = KFold(n_splits=5, shuffle=True, random_state=1)
        rmse_by_k: List[float] = []
        for k in range(1, X_train_red.shape[1] + 1):
            s = cross_val_score(
                LinearRegression(),
                X_train_red[:, :k],
                y_train,
                cv=cv,
                scoring="neg_root_mean_squared_error",
            ).mean()
            rmse_by_k.append(float(-s))
        best_k = int(np.argmin(rmse_by_k) + 1)

        model = LinearRegression().fit(X_train_red[:, :best_k], y_train)
        pred_train = model.predict(X_train_red[:, :best_k])
        pred_test = model.predict(X_test_red[:, :best_k])

        bundle = ModelBundle(
            method="umap",
            scaler=scaler,
            reducer=reducer,
            model=model,
            best_k=best_k,
            rmse_by_k=rmse_by_k,
        )
        return pred_train, pred_test, bundle

    # "none"
    model = LinearRegression().fit(X_train_scaled, y_train)
    pred_train = model.predict(X_train_scaled)
    pred_test = model.predict(X_test_scaled)

    bundle = ModelBundle(
        method="none",
        scaler=scaler,
        reducer=None,
        model=model,
        best_k=None,
        rmse_by_k=None,
    )
    return pred_train, pred_test, bundle


def predict_with_model_bundle(X: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    """
    Apply a saved ModelBundle to new features.

    This is the function your public inference repo should use after loading
    model_bundle.joblib with joblib.load().
    """
    X_scaled = bundle.scaler.transform(X)

    if bundle.method in {"pca", "umap"}:
        if bundle.reducer is None or bundle.best_k is None:
            raise ValueError(f"Saved bundle for method={bundle.method!r} is missing reducer/best_k.")
        X_red = bundle.reducer.transform(X_scaled)
        return bundle.model.predict(X_red[:, : int(bundle.best_k)])

    return bundle.model.predict(X_scaled)


@dataclass
class SingleCfg:
    # Must align with bootstrap defaults
    seed_value: int = 123
    n_bootstrap_iterations: int = 100
    training_sizes: Optional[List[int]] = None
    test_set_size: int = 54

    # Target point you want to reproduce
    iteration: int = 8
    train_size: int = 190

    pad_value: float = -1.0

    # Match bootstrap method options (so PCA/UMAP single run matches notebook)
    method: str = "none"  # "none" | "pca" | "umap"
    max_components: int = 30
    umap_params: Optional[dict] = None

    # Shared splits across descriptors (Option A)
    splits_path: Optional[Path] = None
    reuse_splits_if_exists: bool = True

    def __post_init__(self) -> None:
        if self.training_sizes is None:
            self.training_sizes = list(range(60, 200, 10))
        self.method = str(self.method).lower()
        if self.method == "umap" and umap is None:
            raise ImportError("umap-learn is required for method='umap' (pip install umap-learn).")


def run_single_iteration_notebook_exact(
    X: np.ndarray,
    y: np.ndarray,
    cfg: SingleCfg,
    filenames: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, ModelBundle]:
    """
    Reproduce ONE point from the notebook bootstrap grid:
      (train_size=cfg.train_size, iteration=cfg.iteration)

    If cfg.splits_path is provided, this uses the stored indices (Option A):
      test_idx = bundle.test_set_indices[it]
      train_idx = bundle.train_set_indices[train_size][it]

    Otherwise, it falls back to RNG replay.
    """
    n_boot = int(cfg.n_bootstrap_iterations)
    it_target = int(cfg.iteration)
    if it_target < 0 or it_target >= n_boot:
        raise ValueError(f"iteration must be in [0, {n_boot - 1}], got {it_target}")

    train_size = int(cfg.train_size)

    # --- Preferred path: use stored splits (Option A) ---
    bundle = _load_or_create_bundle(cfg, filenames)
    if bundle is not None:
        if train_size not in bundle.train_set_indices:
            raise ValueError(
                f"train_size={train_size} not present in splits file. "
                f"Available: {sorted(bundle.train_set_indices.keys())}"
            )
        test_idx = np.asarray(bundle.test_set_indices[it_target], dtype=int)
        train_idx = np.asarray(bundle.train_set_indices[train_size][it_target], dtype=int)

    else:
        # --- Fallback: RNG replay (not recommended once you have split files) ---
        np.random.seed(int(cfg.seed_value))

        # recreate notebook test splits
        test_set_indices = _get_test_splits_notebook_exact(
            n_samples=len(X),
            seed_value=int(cfg.seed_value),
            n_iter=n_boot,
            test_size=int(cfg.test_set_size),
        )

        all_idx_full = np.arange(len(X), dtype=int)

        # IMPORTANT: bootstrap consumes RNG in SIZE-MAJOR order:
        # for size in training_sizes:
        #   for it in range(n_boot):
        #     draw train
        training_sizes = list(np.asarray(cfg.training_sizes, dtype=int))
        if train_size not in set(training_sizes):
            raise ValueError(f"train_size={train_size} not in cfg.training_sizes={training_sizes}")

        for size in training_sizes:
            for it in range(n_boot):
                pool = np.setdiff1d(all_idx_full, test_set_indices[it])
                if int(size) == train_size and it == it_target:
                    break
                _ = np.random.choice(pool, size=int(size), replace=False)
            if int(size) == train_size:
                break

        test_idx = np.asarray(test_set_indices[it_target], dtype=int)
        train_pool = np.setdiff1d(all_idx_full, test_idx)
        train_idx = np.random.choice(train_pool, size=train_size, replace=False).astype(int)

    # ---- Train/predict (match notebook branches) ----
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    pred_train, pred_test, model_bundle = _fit_predict_one_iter(X_train, y_train, X_test, cfg)

    metrics = {
        "seed_value": int(cfg.seed_value),
        "iteration": int(cfg.iteration),
        "train_size": int(train_size),
        "test_size": int(cfg.test_set_size),
        "method": cfg.method,
        "train_rmse": _rmse(y_train, pred_train),
        "test_rmse": _rmse(y_test, pred_test),
    }
    metrics_df = pd.DataFrame([metrics])

    return test_idx, train_idx, y_test, pred_test, metrics_df, model_bundle


def write_single_csvs(
    out_dir: Path,
    tag: str,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    metrics_df: pd.DataFrame,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(out_dir / f"{tag}_single_metrics.csv", index=False)

    # Flatten wavelengths exactly like notebook (NaN-only filtering)
    actual_wl = y_test[:, ::2].flatten()
    pred_wl = pred_test[:, ::2].flatten()
    actual_wl = actual_wl[~np.isnan(actual_wl)]
    pred_wl = pred_wl[~np.isnan(pred_wl)]

    pd.DataFrame(
        {
            "Actual Wavelengths": pd.Series(actual_wl),
            "Predicted Wavelengths": pd.Series(pred_wl),
        }
    ).to_csv(out_dir / f"{tag}-Single_actual_vs_predicted_wavelengths.csv", index=False)

