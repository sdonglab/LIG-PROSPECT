from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Any

import joblib  # type: ignore
import numpy as np
import pandas as pd
import yaml  # type: ignore

from .io import (
    _read_xyz_tokens_as_numeric,
    _internal_coords_from_xyz,
    _read_descriptor_floats,
)

# Import these modules so joblib can unpickle ModelBundle objects saved from either workflow.
# The loaded object is used by duck-typing, so it can come from single_core.ModelBundle
# or mutant_core.ModelBundle.
from . import single_core as _single_core  # noqa: F401
from . import mutant_core as _mutant_core  # noqa: F401


DESCRIPTOR_TAGS = {
    "pca_cc": "PCA-CC",
    "umap_ic": "UMAP-IC",
    "dd": "DD",
    "add": "ADD",
}


def _clean_desc_key(k: str) -> str:
    return str(k).lower().replace("-", "_").strip()


def _load_yaml(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _bundle_get(bundle: Any, key: str, default: Any = None) -> Any:
    """Read from either a dict bundle or a dataclass/object bundle."""
    if isinstance(bundle, dict):
        return bundle.get(key, default)
    return getattr(bundle, key, default)


def infer_expected_dim(bundle: Any) -> Optional[int]:
    """Infer the number of raw input features expected before scaling."""
    scaler = _bundle_get(bundle, "scaler", None)
    if scaler is not None and hasattr(scaler, "n_features_in_"):
        return int(scaler.n_features_in_)

    model = _bundle_get(bundle, "model", None)
    if model is not None and hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)

    return None

def _read_feature(path: Path, descriptor: str) -> np.ndarray:
    descriptor = _clean_desc_key(descriptor)
    if descriptor == "pca_cc":
        return _read_xyz_tokens_as_numeric(path)
    if descriptor == "umap_ic":
        return _internal_coords_from_xyz(path)
    if descriptor in {"dd", "add"}:
        return _read_descriptor_floats(path)
    raise ValueError("descriptor must be one of: pca_cc, umap_ic, dd, add")


def load_features_only(
    *,
    descriptor: str,
    features_dir: Path,
    file_glob: str,
    expected_dim: Optional[int],
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load X features only. Unlike load_dataset(), this does not require excitation CSV files.

    For DD/ADD, features_dir should contain descriptor text files.
    For PCA-CC/UMAP-IC, features_dir should contain XYZ files.
    """
    features_dir = Path(features_dir)
    feature_paths = sorted(features_dir.glob(file_glob))
    if not feature_paths:
        raise FileNotFoundError(
            f"No feature files matched file_glob={file_glob!r} under features_dir={features_dir}"
        )

    rows: List[np.ndarray] = []
    names: List[str] = []

    for fp in feature_paths:
        x = _read_feature(fp, descriptor=descriptor)
        rows.append(np.asarray(x, dtype=float))
        names.append(fp.stem)

    if expected_dim is None:
        target_dim = max(len(r) for r in rows)
        print(
            "WARNING: Could not infer expected feature dimension from the saved bundle. "
            f"Padding this batch to max length {target_dim}."
        )
    else:
        target_dim = int(expected_dim)

    fixed_rows: List[np.ndarray] = []
    for name, row in zip(names, rows):
        n = len(row)
        if n > target_dim:
            raise ValueError(
                f"Feature vector for {name!r} has length {n}, but the saved model expects "
                f"{target_dim}. Check descriptor type, file_glob, or feature-generation format."
            )
        if n < target_dim:
            row = np.pad(row, (0, target_dim - n), mode="constant", constant_values=pad_value)
        fixed_rows.append(row)

    return np.vstack(fixed_rows), names

def predict_with_bundle(bundle: Any, X: np.ndarray) -> np.ndarray:
    """Apply saved scaler, optional reducer, best_k, and model to new X."""
    scaler = _bundle_get(bundle, "scaler", None)
    reducer = _bundle_get(bundle, "reducer", None)
    model = _bundle_get(bundle, "model", None)
    method = str(_bundle_get(bundle, "method", "none")).lower()
    best_k = _bundle_get(bundle, "best_k", None)

    if model is None:
        raise AttributeError(
            "Saved object does not contain a model. Expected a saved bundle from "
            "ligprospect-train-single-split or ligprospect-evaluate-heldout-mutant."
        )

    X_work = np.asarray(X, dtype=float)

    if scaler is not None:
        X_work = scaler.transform(X_work)

    if method in {"pca", "umap"}:
        if reducer is None:
            raise ValueError(f"Saved bundle method={method!r} requires a reducer, but reducer is missing.")

        X_work = reducer.transform(X_work)

        if best_k is not None:
            X_work = X_work[:, : int(best_k)]

    return np.asarray(model.predict(X_work), dtype=float)

def save_predictions(
    output_path: Path,
    sample_names: list[str],
    predictions: np.ndarray,
) -> None:
    """
    Save predictions as:
        sample,pred_1,osc_1,pred_2,osc_2,pred_3,osc_3,pred_4,osc_4
    """
    pred_2d = np.asarray(predictions, dtype=float)

    if pred_2d.ndim == 1:
        pred_2d = pred_2d.reshape(1, -1)

    if pred_2d.shape[1] != 8:
        raise ValueError(
            f"Expected 8 prediction outputs per sample "
            f"(4 wavelength/oscillator pairs), got {pred_2d.shape[1]}"
        )

    rows = []

    for name, yhat in zip(sample_names, pred_2d):
        row = {
            "sample": name,
            "pred_1": float(yhat[0]),
            "osc_1": float(yhat[1]),
            "pred_2": float(yhat[2]),
            "osc_2": float(yhat[3]),
            "pred_3": float(yhat[4]),
            "osc_3": float(yhat[5]),
            "pred_4": float(yhat[6]),
            "osc_4": float(yhat[7]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved prediction CSV to: {output_path}")
def _descriptor_folder_name(descriptor: str) -> str:
    mapping = {
        "dd": "DD",
        "add": "ADD",
        "pca_cc": "PCA-CC",
        "umap_ic": "UMAP-IC",
    }
    descriptor = _clean_desc_key(descriptor)
    if descriptor not in mapping:
        raise ValueError(f"Unknown descriptor: {descriptor}")
    return mapping[descriptor]


def _sanitize_name(s: str) -> str:
    return (
        str(s)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def _build_model_paths_from_config(cfg: dict) -> tuple[Path, Path, str, Optional[str], str, str]:
    pred_cfg = cfg.get("prediction", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    workflow = str(pred_cfg.get("workflow", "")).strip()
    descriptor = _clean_desc_key(pred_cfg.get("descriptor", ""))
    holdout_mutant = pred_cfg.get("holdout_mutant", None)

    if workflow not in {"single_run_random_split", "hold_out_mutant"}:
        raise ValueError(
            "prediction.workflow must be one of: "
            "'single_run_random_split' or 'hold_out_mutant'"
        )

    if descriptor not in {"dd", "add", "pca_cc", "umap_ic"}:
        raise ValueError("prediction.descriptor must be one of: dd, add, pca_cc, umap_ic")

    model_root = Path(model_cfg.get("root", "saved_models"))
    descriptor_folder = _descriptor_folder_name(descriptor)

    if workflow == "single_run_random_split":
        model_dir = model_root / workflow / descriptor_folder
        model_name = f"ligprospect_{descriptor}_single_iter008_best_model.joblib"
        metadata_name = f"ligprospect_{descriptor}_single_iter008_best_metadata.yaml"
    else:
        if not holdout_mutant:
            raise ValueError("prediction.holdout_mutant is required for workflow='hold_out_mutant'")

        holdout_mutant = str(holdout_mutant)
        model_dir = model_root / workflow / descriptor_folder
        model_name = f"ligprospect_{descriptor}_mutant_holdout_{holdout_mutant}_model.joblib"
        metadata_name = f"ligprospect_{descriptor}_mutant_holdout_{holdout_mutant}_metadata.yaml"

    model_path = model_dir / model_name
    metadata_path = model_dir / metadata_name

    return model_path, metadata_path, workflow, holdout_mutant, descriptor_folder, descriptor


def _build_output_path_from_config(
    cfg: dict,
    workflow: str,
    descriptor: str,
    descriptor_folder: str,
    holdout_mutant: Optional[str],
) -> Path:
    output_cfg = cfg.get("output", {}) or {}
    data_cfg = cfg.get("data", {}) or {}

    output_root = Path(output_cfg.get("root", "outputs/prediction"))
    filename_suffix = output_cfg.get("filename_suffix", "predictions")

    dataset_name = data_cfg.get("dataset_name", None)
    if dataset_name is None:
        features_dir = Path(data_cfg.get("features_dir", "new_data"))
        dataset_name = str(features_dir).replace("/", "_")

    dataset_name = _sanitize_name(dataset_name)
    filename_suffix = _sanitize_name(filename_suffix)
    descriptor = _clean_desc_key(descriptor)

    if workflow == "single_run_random_split":
        return (
            output_root
            / workflow
            / "iter008_best"
            / descriptor_folder
            / f"{dataset_name}_{filename_suffix}.csv"
        )

    return (
        output_root
        / workflow
        / str(holdout_mutant)
        / descriptor_folder
        / f"{dataset_name}_{filename_suffix}.csv"
    )
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ligprospect-predict",
        description="Use a saved LIG-PROSPECT model to predict spectra for new, unlabeled structures.",
    )

    parser.add_argument("--config", type=Path, help="Optional prediction YAML config file")
    parser.add_argument("--model", type=Path, default=None, help="Path to saved model .joblib")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional model_metadata.yaml")
    parser.add_argument("--descriptor", choices=["add", "dd", "pca_cc", "umap_ic"], default=None)
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--file-glob", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pad-value", type=float, default=None)

    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    data_cfg = cfg.get("data", {}) or {}
    output_cfg = cfg.get("output", {}) or {}

    # A manual model path takes precedence over the model selected by the config.
    if args.model is not None:
        model_path = args.model
        metadata_path = args.metadata
        descriptor = args.descriptor
        workflow = "manual"
        holdout_mutant = None
        if descriptor is None:
            raise ValueError("--descriptor is required when using --model manually")
        descriptor = _clean_desc_key(descriptor)
        descriptor_folder = _descriptor_folder_name(descriptor)
    else:
        (
            model_path,
            metadata_path,
            workflow,
            holdout_mutant,
            descriptor_folder,
            descriptor,
        ) = _build_model_paths_from_config(cfg)

    features_dir = args.features_dir or (
        Path(data_cfg["features_dir"]) if data_cfg.get("features_dir") else None
    )
    file_glob = args.file_glob or data_cfg.get("file_glob", "*")

    if args.output is not None:
        output_path = args.output
    else:
        output_path = _build_output_path_from_config(
            cfg=cfg,
            workflow=workflow,
            descriptor=descriptor,
            descriptor_folder=descriptor_folder,
            holdout_mutant=holdout_mutant,
        )

    pad_value = args.pad_value if args.pad_value is not None else float(output_cfg.get("pad_value", 0.0))

    missing: list[str] = []
    if not model_path.exists():
        missing.append(f"model file not found: {model_path}")
    if metadata_path is not None and not metadata_path.exists():
        missing.append(f"metadata file not found: {metadata_path}")
    if features_dir is None:
        missing.append("data.features_dir or --features-dir")
    elif not features_dir.exists():
        missing.append(f"features_dir not found: {features_dir}")

    if missing:
        raise ValueError("Missing required prediction inputs:\n  - " + "\n  - ".join(missing))

    metadata = _load_yaml(metadata_path)

    print(f"Loading model: {model_path}")
    bundle = joblib.load(model_path)

    expected_dim = infer_expected_dim(bundle)
    if expected_dim is None:
        print("WARNING: expected raw input dimension could not be inferred from model bundle.")
    else:
        print(f"Expected raw input feature dimension: {expected_dim}")

    if metadata:
        print("Loaded metadata:")
        for k in [
            "descriptor",
            "descriptor_key",
            "method",
            "workflow",
            "iteration",
            "train_size",
            "test_mutant",
            "model_file",
        ]:
            if k in metadata:
                print(f"  {k}: {metadata[k]}")

    X_new, sample_names = load_features_only(
        descriptor=descriptor,
        features_dir=features_dir,
        file_glob=file_glob,
        expected_dim=expected_dim,
        pad_value=float(pad_value),
    )

    print(f"Loaded {len(sample_names)} samples. X_new shape = {X_new.shape}")

    pred = predict_with_bundle(bundle, X_new)

    print(f"Prediction shape = {pred.shape}")

    save_predictions(output_path, sample_names, pred)

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
