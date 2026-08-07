from __future__ import annotations

from typing import Any, Dict, List, Optional

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

# Canonical order used everywhere for plots/tables.
BASELINE_MODELS: List[str] = [
    "linear",
    "kernel_ridge",
    "random_forest",
    "gaussian_process",
    "gradient_boosting",
]

DISPLAY_NAMES: Dict[str, str] = {
    "linear": "Linear Regression",
    "kernel_ridge": "Kernel Ridge",
    "random_forest": "Random Forest",
    "gaussian_process": "Gaussian Process",
    "gradient_boosting": "Gradient Boosting",
}

# Hyperparameter search spaces used by compare_baselines_core.py's inner CV tuning.
# Deliberately modest (small dataset -> small search budget is appropriate, and
# avoids overfitting the hyperparameters themselves). "linear" has no entry:
# LinearRegression has nothing meaningful to tune here.
HYPERPARAM_DISTRIBUTIONS: Dict[str, Dict[str, Any]] = {
    "kernel_ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "gamma": [1e-3, 1e-2, 1e-1, 1.0, None],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [1, 2, 4, 8],
    },
    "gaussian_process": {
        "alpha": [1e-8, 1e-5, 1e-2, 1e-1, 1.0],
    },
    "gradient_boosting": {
        # GBR is wrapped in MultiOutputRegressor, so params need the "estimator__" prefix.
        "estimator__n_estimators": [100, 200, 300],
        "estimator__max_depth": [2, 3, 4],
        "estimator__learning_rate": [0.01, 0.05, 0.1],
        "estimator__subsample": [0.6, 0.8, 1.0],
    },
}


def build_regressor(name: str, params: Optional[Dict[str, Any]] = None, random_state: int = 123) -> Any:
    """
    Build one of the 5 comparison regressors by name. `params` overrides the
    defaults below (pass whatever the sklearn estimator accepts).

    Multi-output note: y has 8 columns (4 wavelength/oscillator pairs).
    LinearRegression, KernelRidge, RandomForestRegressor, and
    GaussianProcessRegressor all handle 2D y natively. GradientBoostingRegressor
    does NOT, so it's wrapped in MultiOutputRegressor (fits one GBR per output
    column independently).
    """
    name_l = str(name).lower().strip()
    params = dict(params or {})

    if name_l in ("linear", "linear_regression", "ols"):
        return LinearRegression(**params)

    if name_l in ("kernel_ridge", "krr"):
        params.setdefault("kernel", "rbf")
        params.setdefault("alpha", 1.0)
        return KernelRidge(**params)

    if name_l in ("random_forest", "rf"):
        params.setdefault("n_estimators", 300)
        params.setdefault("random_state", random_state)
        params.setdefault("n_jobs", -1)
        return RandomForestRegressor(**params)

    if name_l in ("gaussian_process", "gpr"):
        kernel = params.pop("kernel", None)
        if kernel is None:
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        params.setdefault("normalize_y", True)
        params.setdefault("random_state", random_state)
        return GaussianProcessRegressor(kernel=kernel, **params)

    if name_l in ("gradient_boosting", "gbr"):
        params.setdefault("n_estimators", 300)
        params.setdefault("random_state", random_state)
        base = GradientBoostingRegressor(**params)
        return MultiOutputRegressor(base)

    raise ValueError(
        f"Unknown model name: {name!r}. Use one of: {BASELINE_MODELS}"
    )
