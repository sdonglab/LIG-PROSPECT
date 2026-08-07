from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

from .models import HYPERPARAM_DISTRIBUTIONS, build_regressor
from .splits import get_or_create_splits

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def _grid_size(search_space: Dict[str, list]) -> int:
    """Total number of combinations in a param_distributions dict of lists."""
    size = 1
    for values in search_space.values():
        size *= max(1, len(values))
    return size


def _select_best_k(X_train_red: np.ndarray, y_train: np.ndarray, max_k: int) -> int:
    """
    Same k-selection logic as single_core.py/bootstrap_core.py: pick the PCA/UMAP
    component count via 5-fold CV RMSE, using LinearRegression as the fixed probe.
    Using linear regression here (regardless of which model is later compared)
    keeps the feature representation identical across all 5 models, so the
    comparison isolates model expressiveness rather than confounding it with a
    different feature space per model.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    rmse_by_k: List[float] = []
    for k in range(1, max_k + 1):
        s = cross_val_score(
            LinearRegression(),
            X_train_red[:, :k],
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        ).mean()
        rmse_by_k.append(float(-s))
    return int(np.argmin(rmse_by_k) + 1)


@dataclass
class BaselineCfg:
    # Must match the values the splits.npz file was created/validated with.
    seed_value: int = 123
    n_bootstrap_iterations: int = 100
    training_sizes: Optional[List[int]] = None
    test_set_size: int = 54

    # Which point in the split grid to compare at.
    iteration: int = 8
    train_size: int = 190

    splits_path: Optional[Path] = None

    # Feature representation (kept identical across all 5 models).
    method: str = "none"  # "none" | "pca" | "umap"
    max_components: int = 30
    umap_params: Optional[dict] = None

    models: List[str] = field(default_factory=lambda: [
        "linear", "kernel_ridge", "random_forest", "gaussian_process", "gradient_boosting"
    ])
    model_params: Dict[str, dict] = field(default_factory=dict)

    # Fair-comparison tuning: search each nonlinear model's hyperparameters via
    # CV on the TRAINING fold only (test set is never touched by the search).
    # Without this, models like random_forest/gradient_boosting badly overfit
    # small training sets with sklearn's default settings, making the
    # comparison misleading rather than informative.
    tune: bool = True
    search_iterations: int = 20
    tune_cv_folds: int = 5

    def __post_init__(self) -> None:
        if self.training_sizes is None:
            self.training_sizes = list(range(60, 200, 10))
        self.method = str(self.method).lower()


def run_baseline_comparison(
    X: np.ndarray,
    y: np.ndarray,
    filenames: List[str],
    cfg: BaselineCfg,
) -> pd.DataFrame:
    """
    Fit linear regression + the 4 nonlinear baselines on the SAME
    (train_size, iteration) split loaded from cfg.splits_path, and the SAME
    scaled / dimensionality-reduced feature representation. Only the final
    regressor changes between rows -- this isolates model expressiveness from
    feature engineering, dataset size, and train/test composition.
    """
    if cfg.splits_path is None:
        raise ValueError("cfg.splits_path is required (point it at your existing splits.npz).")

    bundle = get_or_create_splits(
        split_path=Path(cfg.splits_path),
        filenames=filenames,
        seed=int(cfg.seed_value),
        n_iter=int(cfg.n_bootstrap_iterations),
        test_size=int(cfg.test_set_size),
        training_sizes=list(cfg.training_sizes),
        reuse_if_exists=True,
        strict_filename_match=True,
    )

    if cfg.train_size not in bundle.train_set_indices:
        raise ValueError(
            f"train_size={cfg.train_size} not present in splits file. "
            f"Available: {sorted(bundle.train_set_indices.keys())}"
        )

    test_idx = np.asarray(bundle.test_set_indices[cfg.iteration], dtype=int)
    train_idx = np.asarray(bundle.train_set_indices[cfg.train_size][cfg.iteration], dtype=int)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    best_k: Optional[int] = None

    if cfg.method == "pca":
        reducer = PCA(n_components=min(int(cfg.max_components), X_train_s.shape[1]))
        X_train_red = reducer.fit_transform(X_train_s)
        X_test_red = reducer.transform(X_test_s)
        best_k = _select_best_k(X_train_red, y_train, X_train_red.shape[1])
        X_train_final = X_train_red[:, :best_k]
        X_test_final = X_test_red[:, :best_k]

    elif cfg.method == "umap":
        if umap is None:
            raise ImportError("umap-learn is required for method='umap' (pip install umap-learn).")
        params = dict(n_components=min(int(cfg.max_components), X_train_s.shape[1]))
        if cfg.umap_params:
            params.update(cfg.umap_params)
        reducer = umap.UMAP(**params)  # type: ignore
        X_train_red = reducer.fit_transform(X_train_s)
        X_test_red = reducer.transform(X_test_s)
        best_k = _select_best_k(X_train_red, y_train, X_train_red.shape[1])
        X_train_final = X_train_red[:, :best_k]
        X_test_final = X_test_red[:, :best_k]

    else:
        X_train_final, X_test_final = X_train_s, X_test_s

    rows = []
    for model_name in cfg.models:
        params = cfg.model_params.get(model_name, {})
        base_model = build_regressor(model_name, params, random_state=int(cfg.seed_value))

        search_space = HYPERPARAM_DISTRIBUTIONS.get(model_name)
        best_params_str = None

        if cfg.tune and search_space:
            inner_cv = KFold(n_splits=int(cfg.tune_cv_folds), shuffle=True, random_state=1)
            n_iter = min(int(cfg.search_iterations), _grid_size(search_space))
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=search_space,
                n_iter=n_iter,
                cv=inner_cv,
                scoring="neg_root_mean_squared_error",
                random_state=int(cfg.seed_value),
                n_jobs=-1,
            )
            # IMPORTANT: fit only on X_train_final/y_train. The held-out test_idx
            # set is never seen during this search -- no leakage.
            search.fit(X_train_final, y_train)
            model = search.best_estimator_
            best_params_str = str(search.best_params_)
        else:
            model = base_model
            model.fit(X_train_final, y_train)

        pred_train = model.predict(X_train_final)
        pred_test = model.predict(X_test_final)

        rows.append({
            "model": model_name,
            "train_rmse": _rmse(y_train, pred_train),
            "test_rmse": _rmse(y_test, pred_test),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_features_used": int(X_train_final.shape[1]),
            "pca_umap_best_k": best_k,
            "tuned": bool(cfg.tune and search_space is not None),
            "best_params": best_params_str,
        })

    return pd.DataFrame(rows)
