from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
import time
import yaml  # type: ignore

from .io import load_dataset
from .bootstrap_core import BootstrapCfg, run_bootstrap_notebook_exact, write_bootstrap_csvs

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

    common = dict(
        seed_value=int(cfg_all.get("seed", 123)),
        n_bootstrap_iterations=int(cfg_all.get("n_bootstrap_iterations", 100)),
        training_sizes=[int(x) for x in cfg_all.get("training_sizes", list(range(60, 200, 10)))],
        test_set_size=int(cfg_all.get("test_set_size", 54)),
        file_glob=str(cfg_all.get("file_glob", "*cluster*")),
        wavelength_max_nm=float(cfg_all.get("wavelength_max_nm", 900)),
        pad_value=float(cfg_all.get("pad_value", -1)),
        excitations_dir=Path(cfg_all["excitations_dir"]),
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

    descriptors = cfg_all.get("descriptors", {}) or {}
    for key, desc_cfg in descriptors.items():
        key_l = str(key).lower()
        if key_l not in DESCRIPTOR_TAGS:
            raise ValueError(f"Unknown descriptor key: {key_l}. Use one of {sorted(DESCRIPTOR_TAGS)}")
        if descriptor_sel != "all" and key_l != descriptor_sel:
            continue

        tag = DESCRIPTOR_TAGS[key_l]
        features_dir = Path(desc_cfg["features_dir"])
        method = str(desc_cfg.get("method", "none")).lower()
        max_components = int(desc_cfg.get("max_components", 30))
        umap_params = desc_cfg.get("umap", None)

        ds = load_dataset(
            descriptor=key_l,
            features_dir=features_dir,
            excitations_dir=common["excitations_dir"],
            file_glob=common["file_glob"],
            wavelength_max_nm=common["wavelength_max_nm"],
            pad_value=common["pad_value"],
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

