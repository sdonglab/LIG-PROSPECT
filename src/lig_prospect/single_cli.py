from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore
import joblib  # type: ignore

from .io import load_dataset
from .single_core import SingleCfg, run_single_iteration_notebook_exact, write_single_csvs
from .mutant_outputs import save_predictions_tables, save_wavelength_histogram, analyze_and_save_peaks

# Pretty names for printing only
DESCRIPTOR_TAGS = {
    "pca_cc": "PCA-CC",
    "umap_ic": "UMAP-IC",
    "dd": "DD",
    "add": "ADD",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spectra-single", description="Notebook-matching single-iteration runner.")
    p.add_argument("--config", required=True, type=str, help="YAML config file")
    return p


def _normalize_optional_path(val: Any) -> Path | None:
    """Treat None/'none'/''/'null' as None; otherwise Path(val)."""
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() in {"", "none", "null"}:
        return None
    return Path(str(val))


def _clean_desc_key(k: str) -> str:
    """Filesystem-safe descriptor key."""
    return str(k).lower().replace("-", "_").strip()


def main() -> None:
    args = build_parser().parse_args()
    cfg_all = _load_yaml(Path(args.config))

    # ---------------------------
    # Output root: outputs/single
    # ---------------------------
    run_cfg = cfg_all.get("run", {}) or {}
    out_root = Path(run_cfg.get("out_root", "outputs")) / "single"
    out_root.mkdir(parents=True, exist_ok=True)
    descriptor_sel = str(run_cfg.get("descriptor", "all")).lower()

    # Common dataset IO
    common = dict(
        file_glob=str(cfg_all.get("file_glob", "*cluster*")),
        wavelength_max_nm=float(cfg_all.get("wavelength_max_nm", 900)),
        pad_value=float(cfg_all.get("pad_value", -1)),
        excitations_dir=Path(cfg_all["excitations_dir"]),
    )

    # Bootstrap-aligned defaults (keep these consistent with bootstrap_cli.py)
    seed_value = int(cfg_all.get("seed", 123))
    n_boot = int(cfg_all.get("n_bootstrap_iterations", 100))
    test_set_size = int(cfg_all.get("test_set_size", 54))
    training_sizes = [int(x) for x in cfg_all.get("training_sizes", list(range(60, 200, 10)))]

    # Single config block
    single_cfg = cfg_all.get("single", {}) or {}
    iteration = int(single_cfg.get("iteration", 8))
    train_size = int(single_cfg.get("train_size", 190))

    # Peak analysis settings (optional; mirrors mutant defaults)
    sigma = float(single_cfg.get("sigma", 0.2))
    peak_height = float(single_cfg.get("peak_height", 0.0))
    bins = int(single_cfg.get("bins", 25))
    do_peaks = bool(single_cfg.get("do_peak_analysis", True))
    max_wl = float(single_cfg.get("max_wavelength_nm", common["wavelength_max_nm"]))

    # Splits behavior (Option A)
    use_splits = bool(single_cfg.get("use_splits", True))
    reuse_splits_if_exists = bool(single_cfg.get("reuse_splits_if_exists", True))

    splits_path: Path | None
    if not use_splits:
        splits_path = None
    else:
        splits_path_cfg = _normalize_optional_path(single_cfg.get("splits_path", None))
        if splits_path_cfg is not None:
            splits_path = splits_path_cfg
        else:
            # Default: keep consistent with bootstrap (same file name is ideal)
            splits_path = out_root / "shared_splits.npz"

    # Stable run tag for folder/files
    run_tag = f"iter_{iteration:03d}_train_{train_size}"

    descriptors = cfg_all.get("descriptors", {}) or {}
    for key, desc_cfg in descriptors.items():
        desc_key = _clean_desc_key(key)
        if desc_key not in DESCRIPTOR_TAGS:
            raise ValueError(f"Unknown descriptor key: {desc_key}. Use one of {sorted(DESCRIPTOR_TAGS)}")
        if descriptor_sel != "all" and desc_key != descriptor_sel:
            continue

        pretty = DESCRIPTOR_TAGS[desc_key]

        ds = load_dataset(
            descriptor=desc_key,
            features_dir=Path(desc_cfg["features_dir"]),
            excitations_dir=common["excitations_dir"],
            file_glob=common["file_glob"],
            wavelength_max_nm=common["wavelength_max_nm"],
            pad_value=common["pad_value"],
        )

        method = str(desc_cfg.get("method", "none")).lower()
        max_components = int(desc_cfg.get("max_components", 30))
        umap_params = desc_cfg.get("umap", None)

        scfg = SingleCfg(
            seed_value=seed_value,
            n_bootstrap_iterations=n_boot,
            training_sizes=training_sizes,  # IMPORTANT for splits file compatibility
            test_set_size=test_set_size,
            iteration=iteration,
            train_size=train_size,
            pad_value=common["pad_value"],
            method=method,
            max_components=max_components,
            umap_params=umap_params,
            splits_path=splits_path,
            reuse_splits_if_exists=reuse_splits_if_exists,
        )

        print(f"🧪 Single run | {pretty} | {run_tag}")

        test_idx, train_idx, y_test, pred_test, metrics_df, model_bundle = run_single_iteration_notebook_exact(
            ds.X, ds.y, scfg, filenames=ds.filenames
        )

        # Consistent output structure:
        # outputs/single/<descriptor_key>/<iter_xxx_train_yyy>/
        out_dir = out_root / desc_key / run_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        # Keep your existing single CSV outputs
        write_single_csvs(out_dir, tag=desc_key, y_test=y_test, pred_test=pred_test, metrics_df=metrics_df)

        # Add mutant-style wavelength tables + histograms + peak analysis
        save_predictions_tables(
            outdir=out_dir,
            prefix=desc_key,
            set_name=run_tag,
            y_set=y_test,
            pred_set=pred_test,
            max_wavelength=max_wl,
            pad_value=common["pad_value"],
        )

        save_wavelength_histogram(
            outdir=out_dir,
            prefix=desc_key,
            set_name=run_tag,
            y_set=y_test,
            pred_set=pred_test,
            max_wavelength=max_wl,
            bins=bins,
            pad_value=common["pad_value"],
        )

        if do_peaks:
            analyze_and_save_peaks(
                outdir=out_dir,
                prefix=desc_key,
                set_name=run_tag,
                y_set=y_test,
                pred_set=pred_test,
                sigma=sigma,
                peak_height=peak_height,
            )

        # Optional: save which files were in train/test (super helpful for debugging)
        test_files = [ds.filenames[i] for i in test_idx.tolist()]
        train_files = [ds.filenames[i] for i in train_idx.tolist()]
        (out_dir / "test_files.txt").write_text("\n".join(test_files) + "\n", encoding="utf-8")
        (out_dir / "train_files.txt").write_text("\n".join(train_files) + "\n", encoding="utf-8")

        # Optional: small YAML summary (like mutant)
        summary = {
            "mode": "single",
            "descriptor": desc_key,
            "descriptor_pretty": pretty,
            "iteration": iteration,
            "train_size": train_size,
            "test_size": test_set_size,
            "seed": seed_value,
            "method": method,
            "max_components": max_components,
            "splits_path": str(splits_path) if splits_path is not None else None,
            "reuse_splits_if_exists": bool(reuse_splits_if_exists),
            "train_rmse": float(metrics_df["train_rmse"].iloc[0]),
            "test_rmse": float(metrics_df["test_rmse"].iloc[0]),
        }
        (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
        model_dir = out_dir / "saved_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_name = f"ligprospect_{desc_key}_single_iter{iteration:03d}_best_model.joblib"
        metadata_name = f"ligprospect_{desc_key}_single_iter{iteration:03d}_best_metadata.yaml"
        joblib.dump(model_bundle, model_dir / model_name)
        model_metadata = {
            **summary,
            "n_features_in": int(ds.X.shape[1]),
            "n_outputs": int(ds.y.shape[1]) if ds.y.ndim == 2 else 1,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "model_file": model_name,
            "metadata_file": metadata_name,
            "contains_training_data": False,
            "contains_private_raw_data": False,
            "public_inference_note": (
                "Use this model with spectra-predict. Provide new input features"
                "generated using the same descriptor type and feature ordering."
            ),
        }
        (model_dir / metadata_name).write_text(
            yaml.safe_dump(model_metadata, sort_keys=False),
            encoding="utf-8",
        )

        print(f"💾 Saved model bundle to: {model_dir / model_name}")
        print(f"Saved metadata to: {model_dir / metadata_name}")


    print(f"📁 Saving single outputs to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
