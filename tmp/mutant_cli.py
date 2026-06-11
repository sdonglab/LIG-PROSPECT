from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
import time

import yaml  # type: ignore

from .io import load_dataset
from .mutant_core import MutantCfg, run_mutant_prediction
from .mutant_outputs import save_predictions_tables, save_wavelength_histogram, analyze_and_save_peaks

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
    p = argparse.ArgumentParser(prog="spectra-mutant", description="Mutant held-out prediction runner.")
    p.add_argument("--config", required=True, type=str, help="YAML config file")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg_all = _load_yaml(Path(args.config))
    run_cfg = cfg_all.get("run", {}) or {}
    out_root = Path(run_cfg.get("out_root", "outputs")) / "mutant"
    out_root.mkdir(parents=True, exist_ok=True)

    descriptor_sel = str(run_cfg.get("descriptor", "all")).lower()
    print(f"📁 Saving mutant outputs to: {out_root.resolve()}")
    common = dict(
        file_glob=str(cfg_all.get("file_glob", "*cluster*")),
        wavelength_max_nm=float(cfg_all.get("wavelength_max_nm", 900)),
        pad_value=float(cfg_all.get("pad_value", -1)),
        excitations_dir=Path(cfg_all["excitations_dir"]),
    )

    mp_cfg = cfg_all.get("mutant_prediction", {}) or {}
    test_mutant = str(mp_cfg.get("test_mutant", "K283G"))
    y_mode_for_k = str(mp_cfg.get("y_mode_for_k", "wavelength_only"))
    sigma = float(mp_cfg.get("sigma", 0.2))
    peak_height = float(mp_cfg.get("peak_height", 0.0))
    bins = int(mp_cfg.get("bins", 25))
    max_wl = float(mp_cfg.get("max_wavelength_nm", common["wavelength_max_nm"]))
    do_peaks = bool(mp_cfg.get("do_peak_analysis", True))

    descriptors = cfg_all.get("descriptors", {}) or {}
    for key, desc_cfg in descriptors.items():
        key_l = str(key).lower()
        if key_l not in DESCRIPTOR_TAGS:
            raise ValueError(f"Unknown descriptor key: {key_l}. Use one of {sorted(DESCRIPTOR_TAGS)}")
        if descriptor_sel != "all" and key_l != descriptor_sel:
            continue

        tag = DESCRIPTOR_TAGS[key_l]

        ds = load_dataset(
            descriptor=key_l,
            features_dir=Path(desc_cfg["features_dir"]),
            excitations_dir=common["excitations_dir"],
            file_glob=common["file_glob"],
            wavelength_max_nm=common["wavelength_max_nm"],
            pad_value=common["pad_value"],
        )
        print("descriptor:", key_l)
        print("X shape:", ds.X.shape, "y shape:", ds.y.shape, "n files:", len(ds.filenames))
        stems = [Path(f).stem for f in ds.filenames]
        #print("PKG included n:", len(stems))
        #print("PKG included first 10:", stems[:10])
        #print("PKG included last 10:", stems[-10:])

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

        t1 = time.perf_counter()
        #print(f"✅ Done {tag} in {(t1 - t0):.2f}s | test_rmse={out.test_rmse:.4f}")
