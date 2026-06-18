from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.random import default_rng
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


@dataclass
class BootstrapCfg:
    seed_value: int = 123
    n_bootstrap_iterations: int = 100
    training_sizes: Optional[List[int]] = None
    test_set_size: int = 54

    # method controls dimensionality reduction (matches notebooks)
    method: str = "none"  # "none" | "pca" | "umap"
    max_components: int = 30
    umap_params: Optional[dict] = None

    # if True, report train metric using CV RMSE on the (scaled) training features
    # NOTE: notebooks compute train_rmse on in-sample preds, so default False matches notebooks.
    use_cross_val: bool = False

    # shared splits file (stores BOTH test + train indices)
    splits_path: Optional[Path] = None
    reuse_splits_if_exists: bool = True

    def __post_init__(self) -> None:
        if self.training_sizes is None:
            self.training_sizes = list(range(60, 200, 10))
        self.method = str(self.method).lower()
        if self.method == "umap" and umap is None:
            raise ImportError("umap-learn is required for method='umap' (pip install umap-learn).")


@dataclass
class BootstrapOut:
    training_sizes: np.ndarray
    train_rmse_mean: np.ndarray
    train_rmse_std: np.ndarray
    test_rmse_mean: np.ndarray
    test_rmse_std: np.ndarray

    random_iteration: int
    random_y_test: np.ndarray
    random_pred_test: np.ndarray

    all_y_test: np.ndarray
    all_pred_test: np.ndarray

    # kept for backwards compatibility (and debugging); if splits_path is set you can ignore
    test_set_indices: List[np.ndarray]


def _get_test_splits_notebook_exact(n_samples: int, cfg: BootstrapCfg) -> List[np.ndarray]:
    """
    Notebook-exact test split generation:

        test_set_indices = []
        for iteration in range(n_boot):
            np.random.seed(seed_value + iteration)
            test_idx = np.random.choice(np.arange(n_samples), size=test_size, replace=False)
            test_set_indices.append(test_idx)

    IMPORTANT:
    In the notebook, this reseeding loop also determines the RNG state that is then
    used for subsequent training-subset sampling (which does NOT reseed).
    """
    all_idx = np.arange(n_samples, dtype=int)
    out: List[np.ndarray] = []
    for it in range(int(cfg.n_bootstrap_iterations)):
        np.random.seed(int(cfg.seed_value) + it)
        test_idx = np.random.choice(all_idx, size=int(cfg.test_set_size), replace=False)
        out.append(np.asarray(test_idx, dtype=int))
    return out


def _load_or_create_bundle(
    cfg: BootstrapCfg,
    filenames: Optional[List[str]],
    n_samples: int,
    notebook_test_splits: List[np.ndarray],
) -> Optional[SplitBundle]:
    """
    If cfg.splits_path is set, load-or-create SplitBundle (test+train indices) and
    verify that its test splits match notebook-generated test splits.

    Returns:
        SplitBundle if using splits_path else None
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

    # sanity check: expected dataset size
    if len(filenames) != n_samples:
        raise ValueError(f"n_samples mismatch: len(filenames)={len(filenames)} but n_samples={n_samples}")

    # verify test splits match notebook-exact generation
    for it in range(int(cfg.n_bootstrap_iterations)):
        if not np.array_equal(np.asarray(notebook_test_splits[it], dtype=int), np.asarray(bundle.test_set_indices[it], dtype=int)):
            raise ValueError(
                f"Split mismatch at iteration {it} between notebook-generated test splits "
                f"and saved split file {cfg.splits_path}. "
                f"This usually means dataset ordering differs between runs/descriptors."
            )

    return bundle


def _fit_predict_one_iter(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cfg: BootstrapCfg,
    cv: KFold,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit + predict for one bootstrap iteration according to cfg.method.
    Returns: (pred_train, pred_test)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if cfg.method == "pca":
        reducer = PCA(n_components=min(int(cfg.max_components), X_train_scaled.shape[1]))
        X_train_red = reducer.fit_transform(X_train_scaled)
        X_test_red = reducer.transform(X_test_scaled)

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
        return pred_train, pred_test

    if cfg.method == "umap":
        if umap is None:
            raise ImportError("umap-learn is required for method='umap' (pip install umap-learn).")

        params = dict(n_components=min(int(cfg.max_components), X_train_scaled.shape[1]))
        if cfg.umap_params:
            params.update(cfg.umap_params)
        reducer = umap.UMAP(**params)  # type: ignore
        X_train_red = reducer.fit_transform(X_train_scaled)
        X_test_red = reducer.transform(X_test_scaled)

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
        return pred_train, pred_test

    # "none"
    model = LinearRegression().fit(X_train_scaled, y_train)
    pred_train = model.predict(X_train_scaled)
    pred_test = model.predict(X_test_scaled)
    return pred_train, pred_test


def run_bootstrap_notebook_exact(
    X: np.ndarray,
    y: np.ndarray,
    cfg: BootstrapCfg,
    filenames: Optional[List[str]] = None,
) -> BootstrapOut:
    """
    Replicates notebook bootstrapping exactly:
    - np.random.seed(seed_value) once at top (harmless; notebook does it)
    - test_set_indices are generated via reseeding (seed_value + iteration)
    - training subset drawn with np.random.choice WITHOUT reseeding
    - random_iteration chosen with default_rng(seed=42)

    If cfg.splits_path is set, uses stored BOTH test+train indices from SplitBundle.
    """
    # Notebook does this; it gets overridden by per-iteration reseeds during test split generation.
    np.random.seed(int(cfg.seed_value))

    training_sizes = np.asarray(cfg.training_sizes, dtype=int)
    n_boot = int(cfg.n_bootstrap_iterations)
    cv = KFold(n_splits=5, shuffle=True, random_state=1)

    n_samples = int(len(X))
    all_idx_full = np.arange(n_samples, dtype=int)

    # 1) generate notebook-exact test splits (also sets RNG state for notebook-train draws)
    test_set_indices = _get_test_splits_notebook_exact(n_samples, cfg)

    # 2) if splits_path is set, load/create bundle and validate tests match
    bundle = _load_or_create_bundle(cfg, filenames, n_samples, test_set_indices)

    # notebook random_iteration
    rng = default_rng(seed=42)
    random_iteration = int(rng.integers(0, n_boot))

    train_rmse_mean: List[float] = []
    test_rmse_mean: List[float] = []
    train_rmse_std: List[float] = []
    test_rmse_std: List[float] = []

    all_actual_peaks: List[np.ndarray] = []
    all_predicted_peaks: List[np.ndarray] = []
    random_actual_y: Optional[np.ndarray] = None
    random_predicted_y: Optional[np.ndarray] = None

    for size in training_sizes:
        tr_iter: List[float] = []
        te_iter: List[float] = []

        for it in range(n_boot):
            # Choose indices
            if bundle is not None:
                test_idx = np.asarray(bundle.test_set_indices[it], dtype=int)
                train_idx = np.asarray(bundle.train_set_indices[int(size)][it], dtype=int)
            else:
                test_idx = np.asarray(test_set_indices[it], dtype=int)
                train_pool = np.setdiff1d(all_idx_full, test_idx)
                train_idx = np.random.choice(train_pool, size=int(size), replace=False).astype(int)

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            pred_train, pred_test = _fit_predict_one_iter(X_train, y_train, X_test, cfg, cv)

            # train metric
            if cfg.use_cross_val:
                # CV is done on scaled training features; for PCA/UMAP, notebooks do not report CV train rmse
                # but we keep this flag for your optional workflow.
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                cv_rmse = -cross_val_score(
                    LinearRegression(),
                    X_train_scaled,
                    y_train,
                    cv=cv,
                    scoring="neg_root_mean_squared_error",
                ).mean()
                tr_iter.append(float(cv_rmse))
            else:
                tr_iter.append(_rmse(y_train, pred_train))

            te_iter.append(_rmse(y_test, pred_test))

            # store largest size predictions (matches notebooks)
            if size == training_sizes[-1]:
                all_actual_peaks.append(y_test)
                all_predicted_peaks.append(pred_test)
                if it == random_iteration:
                    random_actual_y = y_test
                    random_predicted_y = pred_test

        train_rmse_mean.append(float(np.mean(tr_iter)))
        test_rmse_mean.append(float(np.mean(te_iter)))
        train_rmse_std.append(float(np.std(tr_iter)))
        test_rmse_std.append(float(np.std(te_iter)))

    if random_actual_y is None or random_predicted_y is None:
        raise RuntimeError("Random iteration sample was not captured (unexpected).")

    # In notebooks, for the last training size they "extend" arrays; here we stack blocks.
    # If y_test/pred_test are 2D, this preserves that 2D structure block-wise.
    try:
        all_y_test = np.vstack(all_actual_peaks)
        all_pred_test = np.vstack(all_predicted_peaks)
    except ValueError:
        # if y is 1D, vstack will still work (turns into column); keep consistent 2D output
        all_y_test = np.vstack([np.atleast_2d(a) for a in all_actual_peaks])
        all_pred_test = np.vstack([np.atleast_2d(a) for a in all_predicted_peaks])

    return BootstrapOut(
        training_sizes=training_sizes,
        train_rmse_mean=np.asarray(train_rmse_mean, dtype=float),
        train_rmse_std=np.asarray(train_rmse_std, dtype=float),
        test_rmse_mean=np.asarray(test_rmse_mean, dtype=float),
        test_rmse_std=np.asarray(test_rmse_std, dtype=float),
        random_iteration=int(random_iteration),
        random_y_test=np.asarray(random_actual_y, dtype=float),
        random_pred_test=np.asarray(random_predicted_y, dtype=float),
        all_y_test=np.asarray(all_y_test, dtype=float),
        all_pred_test=np.asarray(all_pred_test, dtype=float),
        test_set_indices=test_set_indices,
    )


def run_single_iteration_exact(
    X: np.ndarray,
    y: np.ndarray,
    cfg: BootstrapCfg,
    iteration: int,
    train_size: int,
    filenames: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single-run that matches bootstrap iteration K:
      - same test set as test_set_indices[K]
      - same train set as bootstrap would use for (train_size, K)

    If cfg.splits_path is set, pulls indices directly from SplitBundle.
    Otherwise, replays notebook RNG stream to reproduce the training subset.
    Returns: (test_idx, train_idx, y_test, pred_test)
    """
    n_boot = int(cfg.n_bootstrap_iterations)
    if iteration < 0 or iteration >= n_boot:
        raise ValueError(f"iteration must be in [0, {n_boot-1}]")

    # If using stored splits, we can look up directly (no RNG replay needed)
    if cfg.splits_path is not None:
        if filenames is None:
            raise ValueError("filenames must be provided when using splits_path (strict matching).")

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
        if int(train_size) not in bundle.train_set_indices:
            raise ValueError(
                f"train_size={train_size} not present in split file. "
                f"Available sizes: {sorted(bundle.train_set_indices.keys())}"
            )

        test_idx = np.asarray(bundle.test_set_indices[iteration], dtype=int)
        train_idx = np.asarray(bundle.train_set_indices[int(train_size)][iteration], dtype=int)

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        cv = KFold(n_splits=5, shuffle=True, random_state=1)
        _, pred_test = _fit_predict_one_iter(X_train, y_train, X_test, cfg, cv)
        return test_idx, train_idx, y_test, pred_test

    # --- Fallback: replay notebook RNG stream (original implementation, fixed) ---
    np.random.seed(int(cfg.seed_value))

    test_set_indices = _get_test_splits_notebook_exact(len(X), cfg)

    all_idx_full = np.arange(len(X), dtype=int)

    # In the notebook, training draws are done in this order:
    # for size in training_sizes:
    #   for it in range(n_boot):
    #     train = choice(...)
    #
    # So for a given train_size, we must advance RNG through:
    #   all sizes BEFORE train_size (all iterations)
    #   then this size up to (iteration-1)
    training_sizes = list(np.asarray(cfg.training_sizes, dtype=int))
    if int(train_size) not in set(training_sizes):
        raise ValueError(f"train_size={train_size} not in cfg.training_sizes={training_sizes}")

    for size in training_sizes:
        for it in range(n_boot):
            pool = np.setdiff1d(all_idx_full, test_set_indices[it])
            if int(size) == int(train_size) and it == int(iteration):
                # stop before drawing the target train subset
                break
            _ = np.random.choice(pool, size=int(size), replace=False)
        if int(size) == int(train_size):
            break

    test_idx = np.asarray(test_set_indices[iteration], dtype=int)
    train_pool = np.setdiff1d(all_idx_full, test_idx)
    train_idx = np.random.choice(train_pool, size=int(train_size), replace=False).astype(int)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    _, pred_test = _fit_predict_one_iter(X_train, y_train, X_test, cfg, cv)
    return test_idx, train_idx, y_test, pred_test


def write_bootstrap_csvs(out_dir: Path, tag: str, out: BootstrapOut) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # rmse curve
    pd.DataFrame(
        {
            "Training Set Size": out.training_sizes,
            "Train RMSE Mean": out.train_rmse_mean,
            "Train RMSE Std": out.train_rmse_std,
            "Test RMSE Mean": out.test_rmse_mean,
            "Test RMSE Std": out.test_rmse_std,
        }
    ).to_csv(out_dir / f"{tag}_rmse_vs_training_size.csv", index=False)

    # large peaks (NaN-only filtering like notebooks)
    actual_wl = out.all_y_test[:, ::2].flatten()
    pred_wl = out.all_pred_test[:, ::2].flatten()
    actual_wl = actual_wl[~np.isnan(actual_wl)]
    pred_wl = pred_wl[~np.isnan(pred_wl)]
    pd.DataFrame(
        {
            "Actual Wavelengths": pd.Series(actual_wl),
            "Predicted Wavelengths": pd.Series(pred_wl),
        }
    ).to_csv(out_dir / f"{tag}-Large_actual_vs_predicted_wavelengths.csv", index=False)

    # random iteration
    a_rand = out.random_y_test[:, ::2].flatten()
    p_rand = out.random_pred_test[:, ::2].flatten()
    a_rand = a_rand[~np.isnan(a_rand)]
    p_rand = p_rand[~np.isnan(p_rand)]
    pd.DataFrame(
        {
            "Actual Wavelengths": pd.Series(a_rand),
            "Predicted Wavelengths": pd.Series(p_rand),
        }
    ).to_csv(out_dir / f"{tag}-Random_actual_vs_predicted_wavelengths.csv", index=False)

    meta = {
        "random_iteration": int(out.random_iteration),
        "stacked_test_blocks": int(out.all_y_test.shape[0]),
    }
    (out_dir / "run_metadata.json").write_text(pd.Series(meta).to_json(indent=2), encoding="utf-8")

