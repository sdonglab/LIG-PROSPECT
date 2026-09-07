"""Input readers and dataset assembly for LIG-PROSPECT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ATOM_NUMBERS = {"H": 1.0, "C": 6.0, "N": 7.0, "O": 8.0, "P": 15.0, "Cl": 17.0}
VALID_DESCRIPTORS = {"pca_cc", "umap_ic", "dd", "add"}


@dataclass(frozen=True)
class Dataset:
    """Features, labels, and matching filename stems for a training dataset."""

    X: np.ndarray
    y: np.ndarray
    filenames: list[str]


def _atom_number(symbol: str, path: Path) -> float:
    normalized = symbol.capitalize()
    if normalized not in ATOM_NUMBERS:
        raise ValueError(f"Unsupported element {symbol!r} in XYZ file: {path}")
    return ATOM_NUMBERS[normalized]


def _read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a standard XYZ file and return element symbols and coordinates."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError(f"Empty XYZ file: {path}")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as error:
        raise ValueError(f"First line of an XYZ file must be an atom count: {path}") from error

    coordinate_lines = lines[2 : 2 + atom_count]
    if len(coordinate_lines) != atom_count:
        raise ValueError(f"XYZ file declares {atom_count} atoms but contains {len(coordinate_lines)} coordinate rows: {path}")

    elements: list[str] = []
    coordinates: list[list[float]] = []
    for line_number, line in enumerate(coordinate_lines, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Expected element and three coordinates on line {line_number}: {path}")
        try:
            coordinates.append([float(value) for value in fields[1:4]])
        except ValueError as error:
            raise ValueError(f"Invalid coordinate on line {line_number}: {path}") from error
        elements.append(fields[0])
    return elements, np.asarray(coordinates, dtype=float)


def _read_xyz_tokens_as_numeric(path: Path) -> np.ndarray:
    """Encode XYZ data as atomic number followed by x, y, z for each atom."""
    elements, coordinates = _read_xyz(path)
    values = [value for element, xyz in zip(elements, coordinates) for value in (_atom_number(element, path), *xyz)]
    return np.asarray(values, dtype=float)


def _bond_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first, second = a - b, c - b
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        raise ValueError("Cannot calculate a bond angle with coincident atoms.")
    cosine = np.clip(np.dot(first, second) / denominator, -1.0, 1.0)
    return float(np.arccos(cosine))


def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    first, second, third = b - a, c - b, d - c
    normal_first, normal_second = np.cross(first, second), np.cross(second, third)
    norm_first, norm_second, norm_middle = np.linalg.norm(normal_first), np.linalg.norm(normal_second), np.linalg.norm(second)
    if norm_first == 0 or norm_second == 0 or norm_middle == 0:
        raise ValueError("Cannot calculate a dihedral angle with collinear or coincident atoms.")
    normal_first /= norm_first
    normal_second /= norm_second
    middle = np.cross(normal_first, second / norm_middle)
    return float(np.arctan2(np.dot(middle, normal_second), np.dot(normal_first, normal_second)))


def _internal_coords_from_xyz(path: Path) -> np.ndarray:
    """Build the notebook-compatible sequential bond, angle, and dihedral vector."""
    _, positions = _read_xyz(path)
    bonds = [np.linalg.norm(positions[index] - positions[index - 1]) for index in range(1, len(positions))]
    angles = [_bond_angle(positions[index - 2], positions[index - 1], positions[index]) for index in range(2, len(positions))]
    dihedrals = [_dihedral(positions[index - 3], positions[index - 2], positions[index - 1], positions[index]) for index in range(3, len(positions))]
    return np.asarray([*bonds, *angles, *dihedrals], dtype=float)


def _read_descriptor_floats(path: Path) -> np.ndarray:
    """Read a whitespace-separated DD or ADD descriptor file."""
    try:
        values = [float(value) for value in Path(path).read_text(encoding="utf-8", errors="ignore").split()]
    except ValueError as error:
        raise ValueError(f"Descriptor file contains a non-numeric value: {path}") from error
    if not values:
        raise ValueError(f"Descriptor file is empty: {path}")
    return np.asarray(values, dtype=float)


def _read_feature(path: Path, descriptor: str) -> np.ndarray:
    if descriptor == "pca_cc":
        return _read_xyz_tokens_as_numeric(path)
    if descriptor == "umap_ic":
        return _internal_coords_from_xyz(path)
    if descriptor in {"dd", "add"}:
        return _read_descriptor_floats(path)
    raise ValueError(f"descriptor must be one of: {', '.join(sorted(VALID_DESCRIPTORS))}")


def read_excitations_csv(path: Path, wavelength_max_nm: float, pad_value: float, filter_wavelengths: bool = True) -> np.ndarray:
    """Read and flatten wavelength/oscillator labels, padding filtered rows."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Excitation CSV is empty: {path}")
    expected_values = frame.size
    if filter_wavelengths:
        wavelength_column = "wavelength" if "wavelength" in frame.columns else frame.columns[0]
        frame = frame.loc[frame[wavelength_column] <= wavelength_max_nm]
    values = frame.to_numpy(dtype=float).ravel()
    return np.pad(values, (0, expected_values - len(values)), constant_values=pad_value)


def _pad_rows(rows: list[np.ndarray], pad_value: float) -> np.ndarray:
    width = max(len(row) for row in rows)
    return np.vstack([np.pad(row, (0, width - len(row)), constant_values=pad_value) for row in rows])


def load_dataset(*, descriptor: str, features_dir: Path, excitations_dir: Path, file_glob: str = "*cluster*", wavelength_max_nm: float = 900.0, pad_value: float = -1.0, filter_wavelengths: bool = True) -> Dataset:
    """Load matching feature and excitation-label files for a training workflow."""
    descriptor = descriptor.lower().replace("-", "_")
    if descriptor not in VALID_DESCRIPTORS:
        raise ValueError(f"descriptor must be one of: {', '.join(sorted(VALID_DESCRIPTORS))}")
    features_dir, excitations_dir = Path(features_dir), Path(excitations_dir)
    if not features_dir.is_dir():
        raise FileNotFoundError(f"Feature directory not found: {features_dir}")
    if not excitations_dir.is_dir():
        raise FileNotFoundError(f"Excitation-label directory not found: {excitations_dir}")

    feature_paths = sorted(path for path in features_dir.glob(file_glob) if path.is_file())
    if not feature_paths:
        raise FileNotFoundError(f"No feature files matched {file_glob!r} under {features_dir}")

    features, labels, names, skipped = [], [], [], []
    for feature_path in feature_paths:
        label_path = excitations_dir / f"{feature_path.stem}.csv"
        if not label_path.is_file():
            skipped.append(f"{feature_path.stem} (no matching CSV)")
            continue
        if filter_wavelengths:
            label_frame = pd.read_csv(label_path)
            wavelength_column = "wavelength" if "wavelength" in label_frame.columns else label_frame.columns[0]
            if (label_frame[wavelength_column] > wavelength_max_nm).any():
                skipped.append(f"{feature_path.stem} (wavelength exceeds cutoff)")
                continue
        label = read_excitations_csv(label_path, wavelength_max_nm, pad_value, filter_wavelengths)
        features.append(_read_feature(feature_path, descriptor))
        labels.append(label)
        names.append(feature_path.stem)

    if not features:
        detail = "; ".join(skipped[:3]) or "no matching inputs"
        raise RuntimeError(f"No usable samples were loaded from {features_dir}. Examples: {detail}")
    dataset = Dataset(X=_pad_rows(features, 0.0), y=_pad_rows(labels, pad_value), filenames=names)
    print(f"Loaded {len(names)} samples; skipped {len(skipped)}; X shape={dataset.X.shape}; y shape={dataset.y.shape}.")
    return dataset
