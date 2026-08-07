from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
import time
import yaml  # type: ignore

from .io import load_dataset
from .bootstrap_core import BootstrapCfg, run_bootstrap_notebook_exact, write_bootstrap_csvs
from .sizing import resolve_split_sizes

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
    p = argparse.ArgumentParser(prog="spectra-bootstrap", description="Notebook-matching bootstrapping runner.")
    p.add_argument("--config", required=True, type=str, help="YAML config file")
    return p


def _normalize_optional_path(val: Any) -> Path | None:
    """Treat None/'none'/'' as None; otherwise Path(val)."""
    if val is None:
        return None
    if isinstance(val, str) and val.strip().lower() in {"", "none", "null"}:
        return None
    return Path(str(val))


def main() -> None:
    args = build_parser().parse_args()
    cfg_all = _load_yaml(Path(args.config))

    run_cfg = cfg_all.get("run", {}) or {}
    out_root = Path(run_cfg.get("out_root", "outputs")) / "bootstrap"
    out_root.mkdir(parents=True, exist_ok=True)
    descriptor_sel = str(run_cfg.get("descriptor", "all")).lower()

    file_glob = str(cfg_all.get("file_glob", "*cluster*"))
    wavelength_max_nm = float(cfg_all.get("wavelength_max_nm", 900))
    filter_wavelengths = bool(cfg_all.get("filter_wavelengths", False))
    pad_value = float(cfg_all.get("pad_value", -1))
    excitations_dir = Path(cfg_all["excitations_dir"])

    descriptors_cfg = cfg_all.get("descriptors", {}) or {}
    if not descriptors_cfg:
        raise ValueError("No 'descriptors' configured in YAML.")
    selected_desc = {
        k: v for k, v in descriptors_cfg.items()
        if descriptor_sel == "all" or str(k).lower() == descriptor_sel
    }
    if not selected_desc:
        raise ValueError(f"No descriptors matched selection '{descriptor_sel}'.")

    # Load ONE descriptor's dataset first, purely to learn n_samples, so
    # test_set_size / training_sizes can scale with whatever conformation
    # set is actually loaded instead of being fixed numbers.
    first_key, first_desc_cfg = next(iter(selected_desc.items()))
    first_ds = load_dataset(
        descriptor=str(first_key).lower(),
        features_dir=Path(first_desc_cfg["features_dir"]),
        excitations_dir=excitations_dir,
        file_glob=file_glob,
        wavelength_max_nm=wavelength_max_nm,
        filter_wavelengths=filter_wavelengths,
        pad_value=pad_value,
    )
    n_samples = first_ds.X.shape[0]
    test_set_size, training_sizes = resolve_split_sizes(n_samples, cfg_all)
    print(
        f"📏 n_samples={n_samples} -> test_set_size={test_set_size}, "
        f"training_sizes={training_sizes}"
    )

    common = dict(
        seed_value=int(cfg_all.get("seed", 123)),
        n_bootstrap_iterations=int(cfg_all.get("n_bootstrap_iterations", 100)),
        training_sizes=training_sizes,
        test_set_size=test_set_size,
        file_glob=file_glob,
        wavelength_max_nm=wavelength_max_nm,
        filter_wavelengths=filter_wavelengths,
        pad_value=pad_value,
        excitations_dir=excitations_dir,
    )

    # --- shared split file (stores BOTH train + test indices) ---
    bootstrap_cfg = cfg_all.get("bootstrap", {}) or {}
    use_splits = bool(bootstrap_cfg.get("use_splits", True))
    reuse_splits_if_exists = bool(bootstrap_cfg.get("reuse_splits_if_exists", True))

    splits_path: Path | None
    if not use_splits:
        splits_path = None
    else:
        splits_path_cfg = _normalize_optional_path(bootstrap_cfg.get("splits_path", None))
        if splits_path_cfg is not None:
            splits_path = splits_path_cfg
        else:
            # default shared file under out_root
            splits_path = out_root / "shared_splits.npz"

    for key, desc_cfg in selected_desc.items():
        key_l = str(key).lower()
        if key_l not in DESCRIPTOR_TAGS:
            raise ValueError(f"Unknown descriptor key: {key_l}. Use one of {sorted(DESCRIPTOR_TAGS)}")

        tag = DESCRIPTOR_TAGS[key_l]
        method = str(desc_cfg.get("method", "none")).lower()
        max_components = int(desc_cfg.get("max_components", 30))
        umap_params = desc_cfg.get("umap", None)

        if key == first_key:
            ds = first_ds  # already loaded above to determine n_samples
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
                    f"the same conformation set for shared splits to be valid."
                )

        cfg = BootstrapCfg(
            seed_value=common["seed_value"],
            n_bootstrap_iterations=common["n_bootstrap_iterations"],
            training_sizes=common["training_sizes"],
            test_set_size=common["test_set_size"],
            method=method,
            max_components=max_components,
            umap_params=umap_params,
            use_cross_val=bool(desc_cfg.get("use_cross_val", False)),
            splits_path=splits_path,  # None disables shared splits
            reuse_splits_if_exists=reuse_splits_if_exists,
        )

        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n🚀 Running bootstrapping for {tag} ...")
        t0 = time.perf_counter()

        out = run_bootstrap_notebook_exact(ds.X, ds.y, cfg, filenames=ds.filenames)
        write_bootstrap_csvs(out_dir, tag=tag, out=out)

        t1 = time.perf_counter()
        runtime = t1 - t0
        print(f"✅ {tag} bootstrapping finished in {runtime:.2f} seconds ({runtime/60:.2f} min)")

        (out_dir / "runtime_bootstrap.txt").write_text(
            f"{runtime:.4f} seconds\n{runtime/60:.4f} minutes\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
