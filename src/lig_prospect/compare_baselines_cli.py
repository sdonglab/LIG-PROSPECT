from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import DESCRIPTOR_TAGS, add_config_argument, data_settings, load_config, selected_descriptors
from .io import load_dataset
from .compare_baselines_core import BaselineCfg, run_baseline_comparison
from .models import BASELINE_MODELS, DISPLAY_NAMES
from .sizing import resolve_split_sizes

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ligprospect-compare-models",
        description="Compare linear regression vs. nonlinear baselines (kernel ridge, "
                     "random forest, Gaussian process, gradient boosting) on a fixed, "
                     "existing train/test split.",
    )
    add_config_argument(p)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg_all = load_config(args.config)

    run_cfg = cfg_all.get("run", {}) or {}
    out_root = Path(run_cfg.get("out_root", "outputs")) / "baselines"
    out_root.mkdir(parents=True, exist_ok=True)
    common = data_settings(cfg_all)

    seed_value = int(cfg_all.get("seed", 123))
    n_boot = int(cfg_all.get("n_bootstrap_iterations", 100))
    selected_desc = selected_descriptors(cfg_all)

    # Resolve the split grid from the actual dataset, just as the bootstrap and
    # single-split workflows do. This keeps one compact config portable across
    # datasets of different sizes.
    first_key, first_desc_cfg = next(iter(selected_desc.items()))
    first_ds = load_dataset(
        descriptor=first_key,
        features_dir=Path(first_desc_cfg["features_dir"]),
        excitations_dir=common["excitations_dir"],
        file_glob=common["file_glob"],
        wavelength_max_nm=common["wavelength_max_nm"],
        filter_wavelengths=common["filter_wavelengths"],
        pad_value=common["pad_value"],
    )
    n_samples = first_ds.X.shape[0]
    test_set_size, training_sizes = resolve_split_sizes(n_samples, cfg_all)
    print(
        f"📏 n_samples={n_samples} -> test_set_size={test_set_size}, "
        f"training_sizes={training_sizes}"
    )

    # Reuse single: block values as defaults so this points at the same split
    # (splits_path, iteration, train_size) you already validated for the manuscript,
    # unless a baselines: block overrides them explicitly.
    single_cfg = cfg_all.get("single", {}) or {}
    baselines_cfg = cfg_all.get("baselines", {}) or {}

    iteration = int(baselines_cfg.get("iteration", single_cfg.get("iteration", 8)))
    train_size = int(
        baselines_cfg.get("train_size", single_cfg.get("train_size", training_sizes[-1]))
    )
    if train_size not in training_sizes:
        raise ValueError(
            f"baselines.train_size={train_size} is not one of the resolved "
            f"training_sizes={training_sizes}. Pick one of those, or omit "
            "baselines.train_size to use the largest size."
        )
    splits_path_raw = baselines_cfg.get("splits_path", single_cfg.get("splits_path"))
    if not splits_path_raw:
        raise ValueError(
            "No splits_path found. Set baselines.splits_path (or single.splits_path) "
            "to your existing splits.npz file."
        )
    splits_path = Path(splits_path_raw)

    models_to_run = [str(m).lower() for m in baselines_cfg.get("models", BASELINE_MODELS)]
    model_params = baselines_cfg.get("model_params", {}) or {}

    all_results = []

    for key_l, desc_cfg in selected_desc.items():
        tag = DESCRIPTOR_TAGS[key_l]

        if key_l == first_key:
            ds = first_ds
        else:
            ds = load_dataset(
                descriptor=key_l,
                features_dir=Path(desc_cfg["features_dir"]),
                excitations_dir=common["excitations_dir"],
                file_glob=common["file_glob"],
                wavelength_max_nm=common["wavelength_max_nm"],
                filter_wavelengths=common["filter_wavelengths"],
                pad_value=common["pad_value"],
            )
            if ds.X.shape[0] != n_samples:
                raise ValueError(
                    f"Descriptor '{key_l}' loaded {ds.X.shape[0]} samples but "
                    f"'{first_key}' loaded {n_samples}. All descriptors must share "
                    "the same conformation set for a fair comparison."
                )

        method = str(desc_cfg.get("method", "none")).lower()
        max_components = int(desc_cfg.get("max_components", 30))
        umap_params = desc_cfg.get("umap", None)

        bcfg = BaselineCfg(
            seed_value=seed_value,
            n_bootstrap_iterations=n_boot,
            training_sizes=training_sizes,
            test_set_size=test_set_size,
            iteration=iteration,
            train_size=train_size,
            splits_path=splits_path,
            method=method,
            max_components=max_components,
            umap_params=umap_params,
            models=models_to_run,
            model_params=model_params,
        )

        print(f"\n📊 Baseline comparison | {tag} | train_size={train_size} iteration={iteration}")
        df = run_baseline_comparison(ds.X, ds.y, ds.filenames, bcfg)
        df.insert(0, "descriptor", tag)
        all_results.append(df)

        out_dir = out_root / key_l
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / f"{key_l}_baseline_comparison.csv"
        df.to_csv(csv_path, index=False)

        plt.figure(figsize=(6, 4), dpi=200)
        labels = [DISPLAY_NAMES.get(m, m) for m in df["model"]]
        plt.bar(labels, df["test_rmse"])
        plt.ylabel("Test RMSE")
        plt.title(f"{tag}: Linear vs. Nonlinear Baselines\n(train_size={train_size}, iteration={iteration})")
        plt.xticks(rotation=30, ha="right")
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        png_path = out_dir / f"{key_l}_baseline_comparison.png"
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(df.to_string(index=False))
        print(f"💾 Saved: {csv_path}")
        print(f"💾 Saved: {png_path}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = out_root / "all_descriptors_baseline_comparison.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n📁 Combined results: {combined_path.resolve()}")


if __name__ == "__main__":
    main()
