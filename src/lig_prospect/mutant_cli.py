from __future__ import annotations

import argparse
from pathlib import Path
import time

import joblib  # type: ignore
import yaml

from .config import DESCRIPTOR_TAGS, add_config_argument, data_settings, load_config, selected_descriptors
from .io import load_dataset
from .mutant_core import MutantCfg, run_mutant_prediction
from .mutant_outputs import save_predictions_tables, save_wavelength_histogram, analyze_and_save_peaks

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ligprospect-evaluate-heldout-mutant",
        description="Evaluate transfer to one mutant held out from training.",
    )
    add_config_argument(p)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg_all = load_config(args.config)
    run_cfg = cfg_all.get("run", {}) or {}
    out_root = Path(run_cfg.get("out_root", "outputs")) / "mutant"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"📁 Saving mutant outputs to: {out_root.resolve()}")
    common = data_settings(cfg_all)

    mp_cfg = cfg_all.get("mutant_prediction", {}) or {}
    test_mutant = str(mp_cfg.get("test_mutant", "K283G"))
    y_mode_for_k = str(mp_cfg.get("y_mode_for_k", "wavelength_only"))
    sigma = float(mp_cfg.get("sigma", 0.2))
    peak_height = float(mp_cfg.get("peak_height", 0.0))
    bins = int(mp_cfg.get("bins", 25))
    if common["filter_wavelengths"]:
        max_wl = float(mp_cfg.get("max_wavelength_nm", common["wavelength_max_nm"]))
    else:
        max_wl = float("inf")
    do_peaks = bool(mp_cfg.get("do_peak_analysis", True))

    for key_l, desc_cfg in selected_descriptors(cfg_all).items():
        tag = DESCRIPTOR_TAGS[key_l]

        ds = load_dataset(
            descriptor=key_l,
            features_dir=Path(desc_cfg["features_dir"]),
            excitations_dir=common["excitations_dir"],
            file_glob=common["file_glob"],
            wavelength_max_nm=common["wavelength_max_nm"],
            filter_wavelengths=common["filter_wavelengths"],
            pad_value=common["pad_value"],
        )
        print("descriptor:", key_l)
        print("X shape:", ds.X.shape, "y shape:", ds.y.shape, "n files:", len(ds.filenames))

        method = str(desc_cfg.get("method", "none")).lower()
        max_components = int(desc_cfg.get("max_components", 30))
        umap_params = desc_cfg.get("umap", None)

        if method == "umap":
            do_scale = bool(desc_cfg.get("do_scale", False))
        else:
            do_scale = bool(desc_cfg.get("do_scale", True))

        cfg = MutantCfg(
            test_mutant=test_mutant,
            y_mode_for_k=y_mode_for_k,
            method=method,
            max_components=max_components,
            do_scale=do_scale,
            umap_params=umap_params,
        )

        out_dir = out_root / tag / test_mutant
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🧪 Mutant held-out prediction | {tag} | test_mutant={test_mutant}")
        t0 = time.perf_counter()

        out = run_mutant_prediction(ds.X, ds.y, ds.filenames, cfg)

        summary = {
            "descriptor": tag,
            "test_mutant": test_mutant,
            "method": out.method,
            "best_k": out.best_k,
            "train_rmse": out.train_rmse,
            "test_rmse": out.test_rmse,
            "n_train": len(out.train_files),
            "n_test": len(out.test_files),
        }
        (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
        (out_dir / "test_files.txt").write_text("\n".join(out.test_files) + "\n", encoding="utf-8")
        (out_dir / "train_files.txt").write_text("\n".join(out.train_files) + "\n", encoding="utf-8")
        if out.model_bundle is not None:
            model_dir = out_dir / "saved_model"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_name = f"ligprospect_{key_l}_mutant_holdout_{test_mutant}_model.joblib"
            metadata_name = f"ligprospect_{key_l}_mutant_holdout_{test_mutant}_metadata.yaml"
            joblib.dump(out.model_bundle, model_dir / model_name)

            model_metadata = {
                **summary,
                "n_features_in": int(ds.X.shape[1]),
                "n_outputs": int(ds.y.shape[1]) if ds.y.ndim == 2 else 1,
                "model_file": model_name,
                "contains_training_data": False,
                "contains_private_raw_data": False,
                "public_inference_note": (
                    "Load model_bundle.joblib and apply the same descriptor-generation "
                    "steps before calling predict_with_model_bundle()."
                ),
            }
            (model_dir / metadata_name).write_text(
                yaml.safe_dump(model_metadata, sort_keys=False),
                encoding="utf-8",
            )
            print(f"💾 Saved model bundle to: {model_dir / model_name}")

        save_predictions_tables(
            outdir=out_dir,
            prefix=tag,
            set_name=test_mutant,
            y_set=out.y_test,
            pred_set=out.pred_test,
            max_wavelength=max_wl,
            pad_value=common["pad_value"],
        )

        save_wavelength_histogram(
            outdir=out_dir,
            prefix=tag,
            set_name=test_mutant,
            y_set=out.y_test,
            pred_set=out.pred_test,
            max_wavelength=max_wl,
            bins=bins,
            pad_value=common["pad_value"],
        )

        if do_peaks:
            analyze_and_save_peaks(
                outdir=out_dir,
                prefix=tag,
                set_name=test_mutant,
                y_set=out.y_test,
                pred_set=out.pred_test,
                sigma=sigma,
                peak_height=peak_height,
            )

        runtime = time.perf_counter() - t0
        print(f"✅ {tag} finished in {runtime:.2f} seconds | test RMSE={out.test_rmse:.4f}")
