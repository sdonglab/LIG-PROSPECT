from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hashlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None

import hashlib, numpy as np

def md5(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a, dtype=np.float64)
    return hashlib.md5(a.tobytes()).hexdigest()

def summarize(a: np.ndarray, name="A"):
    a64 = np.ascontiguousarray(a, dtype=np.float64)
    print(name, "shape", a64.shape, "dtype", a64.dtype)
    print(name, "md5", md5(a64))
    print(name, "nan count", np.isnan(a64).sum())
    print(name, "min/max", np.nanmin(a64), np.nanmax(a64))
    print(name, "sum", np.nansum(a64))

def infer_mutant_type_from_name(name: str) -> Optional[str]:
    """Infer mutant label from a filename stem (notebook-style)."""
    s = str(name)
    if "A36E" in s:
        return "A36E"
    if "K283G" in s:
        return "K283G"
    if "WT" in s:
        return "WT"
    return None


def split_train_test_by_mutant(
    X: np.ndarray,
    y: np.ndarray,
    filenames: List[str],
    test_mutant: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Train on mutants != test_mutant; test on test_mutant only."""
    mutants = [infer_mutant_type_from_name(Path(f).stem) for f in filenames]
    test_mask = np.array([m == test_mutant for m in mutants], dtype=bool)
    train_mask = ~test_mask

    if not np.any(test_mask):
        raise ValueError(f"No samples found for test_mutant={test_mutant}. Check filenames and mutant tags.")

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    train_files = [filenames[i] for i in np.where(train_mask)[0].tolist()]
    test_files = [filenames[i] for i in np.where(test_mask)[0].tolist()]
    return X_train, y_train, X_test, y_test, train_files, test_files


def _rmse_multi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    return float(np.sqrt(np.mean(diff * diff)))


def choose_best_k_by_cv_rmse(
    X_train_red: np.ndarray,
    y_train: np.ndarray,
    cv: KFold,
    y_mode: str = "wavelength_only",
) -> Tuple[int, List[float]]:
    """Select best k using CV RMSE (matches notebook logic)."""
    y_mode_l = str(y_mode).lower()
    if y_mode_l == "wavelength_only":
        y_target = y_train[:, ::2]
    elif y_mode_l == "full":
        y_target = y_train
    else:
        raise ValueError(f"Unknown y_mode={y_mode}. Use 'wavelength_only' or 'full'.")

    rmse_list: List[float] = []
    for k in range(1, X_train_red.shape[1] + 1):
        s = cross_val_score(
            LinearRegression(),
            X_train_red[:, :k],
            y_target,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        ).mean()
        rmse_list.append(float(-s))

    best_k = int(np.argmin(rmse_list) + 1)
    return best_k, rmse_list


@dataclass
class MutantCfg:
    test_mutant: str = "K283G"

    cv_splits: int = 5
    cv_shuffle: bool = True
    cv_random_state: int = 100

    y_mode_for_k: str = "wavelength_only"

    method: str = "none"  # "none" | "pca" | "umap"
    max_components: int = 30

    # Scaling:
    # - DD/ADD: True (usually)
    # - PCA-CC: True (then PCA)
    # - UMAP-IC: often False in notebooks (internal coords)
    do_scale: bool = True

    umap_params: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.method = str(self.method).lower()
        if self.method == "umap" and umap is None:
            raise ImportError("umap-learn is required for method='umap' (pip install umap-learn).")
@dataclass
class ModelBundle:
    """Everything needed to reproduce inference for one trained mutant-held-out model."""
    method: str
    do_scale: bool
    scaler: Optional[StandardScaler]
    model: LinearRegression
    reducer: Optional[Any] = None
    best_k: Optional[int] = None
    rmse_by_k: Optional[List[float]] = None


def predict_with_model_bundle(X: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    """
    Apply a saved mutant ModelBundle to new features.
    """
    if bundle.do_scale and bundle.scaler is not None:
        X_s = bundle.scaler.transform(X)
    else:
        X_s = X

    if bundle.method in {"pca", "umap"}:
        if bundle.reducer is None or bundle.best_k is None:
            raise ValueError(f"Saved bundle for method={bundle.method!r} is missing reducer/best_k.")
        X_red = bundle.reducer.transform(X_s)
        return bundle.model.predict(X_red[:, : int(bundle.best_k)])

    return bundle.model.predict(X_s)


@dataclass
class MutantOut:
    test_mutant: str
    method: str
    best_k: Optional[int]
    train_rmse: float
    test_rmse: float

    y_test: np.ndarray
    pred_test: np.ndarray

    train_files: List[str]
    test_files: List[str]

    rmse_by_k: Optional[List[float]] = None
    model_bundle: Optional[Dict[str, Any]] = None


def run_mutant_prediction(
    X: np.ndarray,
    y: np.ndarray,
    filenames: List[str],
    cfg: MutantCfg,
) -> MutantOut:
    """Unified held-out mutant prediction (notebook-aligned)."""
    X_train, y_train, X_test, y_test, train_files, test_files = split_train_test_by_mutant(
        X, y, filenames, test_mutant=str(cfg.test_mutant)
    )

    cv = KFold(
        n_splits=int(cfg.cv_splits),
        shuffle=bool(cfg.cv_shuffle),
        random_state=int(cfg.cv_random_state),
    )

    scaler: Optional[StandardScaler] = None
    if cfg.method == "umap":
        X_train_s, X_test_s = X_train, X_test
    else:
       if cfg.do_scale:
           scaler = StandardScaler()
           X_train_s = scaler.fit_transform(X_train)
           X_test_s = scaler.transform(X_test)
       else:
           X_train_s, X_test_s = X_train, X_test

    best_k: Optional[int] = None
    rmse_by_k: Optional[List[float]] = None
    reducer: Optional[Any] = None

    if cfg.method == "pca":
        reducer = PCA(n_components=min(int(cfg.max_components), X_train_s.shape[1]))
        X_train_red = reducer.fit_transform(X_train_s)
        X_test_red = reducer.transform(X_test_s)

        best_k, rmse_by_k = choose_best_k_by_cv_rmse(X_train_red, y_train, cv=cv, y_mode=cfg.y_mode_for_k)

        model = LinearRegression().fit(X_train_red[:, :best_k], y_train)
        pred_train = model.predict(X_train_red[:, :best_k])
        pred_test = model.predict(X_test_red[:, :best_k])

    elif cfg.method == "umap":
        params: Dict[str, Any] = dict(
            n_components=min(int(cfg.max_components), X_train_s.shape[1]),
            n_neighbors=15, min_dist=0.1, metric="euclidean", n_jobs=-1,
        )
        if cfg.umap_params:
            params.update(cfg.umap_params)
        # FORCE reproducibility (prefer YAML, fall back to cfg/global seed)
        rs = int(params.get("random_state", 456))

        params["random_state"] = rs
        params.pop("transform_seed", None)
        #summarize(X_train_s, "X_train_s")
        #summarize(X_test_s,  "X_test_s")
        reducer = umap.UMAP(**params)  # type: ignore
        np.random.seed(rs)
        X_train_red = reducer.fit_transform(X_train_s)
        np.random.seed(rs)
        X_test_red = reducer.transform(X_test_s)

        best_k, rmse_by_k = choose_best_k_by_cv_rmse(X_train_red, y_train, cv=cv, y_mode=cfg.y_mode_for_k)

        model = LinearRegression().fit(X_train_red[:, :best_k], y_train)
        pred_train = model.predict(X_train_red[:, :best_k])
        pred_test = model.predict(X_test_red[:, :best_k])

    else:
        model = LinearRegression().fit(X_train_s, y_train)
        pred_train = model.predict(X_train_s)
        pred_test = model.predict(X_test_s)
    model_bundle = {
           "scaler": scaler,
           "reducer": reducer,
           "model": model,
           "method": str(cfg.method),
           "best_k": best_k,
           "rmse_by_k": rmse_by_k,
           "do_scale": bool(cfg.do_scale),
           "test_mutant": str(cfg.test_mutant),
           "y_mode_for_k": str(cfg.y_mode_for_k),
    }
    return MutantOut(
        test_mutant=str(cfg.test_mutant),
        method=str(cfg.method),
        best_k=best_k,
        train_rmse=_rmse_multi(y_train, pred_train),
        test_rmse=_rmse_multi(y_test, pred_test),
        y_test=np.asarray(y_test, dtype=float),
        pred_test=np.asarray(pred_test, dtype=float),
        train_files=train_files,
        test_files=test_files,
        rmse_by_k=rmse_by_k,
        model_bundle=model_bundle,
     )
