"""Shared configuration helpers for LIG-PROSPECT command-line workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml

DESCRIPTOR_TAGS = {
    "pca_cc": "PCA-CC",
    "umap_ic": "UMAP-IC",
    "dd": "DD",
    "add": "ADD",
}


def normalize_descriptor(value: str) -> str:
    """Return the canonical descriptor key used throughout the package."""
    key = str(value).lower().replace("-", "_").strip()
    if key not in DESCRIPTOR_TAGS:
        choices = ", ".join(sorted(DESCRIPTOR_TAGS))
        raise ValueError(f"Unknown descriptor {value!r}. Choose one of: {choices}.")
    return key


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML mapping and report malformed or missing files clearly."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return config


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML configuration file")


def optional_path(value: Any) -> Path | None:
    """Convert a configured optional path, accepting null/none/empty as absent."""
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"}):
        return None
    return Path(str(value))


def data_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return data settings, supporting both current and legacy config layouts."""
    nested = config.get("data", {}) or {}
    if not isinstance(nested, Mapping):
        raise ValueError("data must be a YAML mapping.")

    def get(name: str, default: Any = None) -> Any:
        return nested.get(name, config.get(name, default))

    excitations_dir = get("excitations_dir")
    if not excitations_dir:
        raise ValueError("Set data.excitations_dir to the directory containing training-label CSV files.")
    return {
        "excitations_dir": Path(str(excitations_dir)),
        "file_glob": str(get("file_glob", "*cluster*")),
        "wavelength_max_nm": float(get("wavelength_max_nm", 900.0)),
        "filter_wavelengths": bool(get("filter_wavelengths", True)),
        "pad_value": float(get("pad_value", -1.0)),
    }


def selected_descriptors(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Select configured descriptors using ``run.descriptor`` (or ``all``)."""
    descriptors = config.get("descriptors", {}) or {}
    if not isinstance(descriptors, Mapping) or not descriptors:
        raise ValueError("Add a non-empty descriptors: mapping to the configuration.")
    run = config.get("run", {}) or {}
    selection = str(run.get("descriptor", "all")).lower().replace("-", "_").strip()
    if selection != "all":
        selection = normalize_descriptor(selection)

    result: dict[str, Mapping[str, Any]] = {}
    for name, settings in descriptors.items():
        key = normalize_descriptor(str(name))
        if selection in {"all", key}:
            if not isinstance(settings, Mapping):
                raise ValueError(f"descriptors.{name} must be a YAML mapping.")
            if not settings.get("features_dir"):
                raise ValueError(f"Set descriptors.{name}.features_dir.")
            result[key] = settings
    if not result:
        raise ValueError(f"No configured descriptor matches run.descriptor={selection!r}.")
    return result
