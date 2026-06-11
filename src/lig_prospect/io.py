from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

ATOM_MAP: Dict[str, float] = {"H": 1.0, "C": 6.0, "N": 7.0, "O": 8.0, "P": 15.0, "Cl": 17.0}

@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    filenames: List[str]

def _read_xyz_tokens_as_numeric(path: Path) -> np.ndarray:
    """
    Match PCA-CC notebook `get_xyz()` behavior:
    - read first line as atom count
    - keep ONLY the last `num_atoms` lines from the remainder (this drops the comment line)
    - split each line into tokens (Element x y z)
    - replace element symbols with atomic numbers, keep coordinates as floats
    Result length = 4 * num_atoms
    """
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
        if not first:
            raise ValueError(f"Empty xyz file: {path}")
        n = int(first)
        rest = f.read().splitlines()
    # Keep last n lines (drops comment line if present)
    coord_lines = rest[-n:] if len(rest) >= n else rest
    vals: List[float] = []
    for ln in coord_lines:
        parts = ln.split()
        if not parts:
            continue
        # expect: Element x y z
        el = parts[0]
        vals.append(float(ATOM_MAP.get(el, ATOM_MAP.get(el.capitalize(), np.nan))))
        # coords
        for tok in parts[1:4]:
            vals.append(float(tok))
    if len(vals) != 4 * n:
        # still return what we have, but be explicit
        raise ValueError(f"XYZ parsing mismatch for {path}: expected {4*n} values, got {len(vals)}")
    return np.asarray(vals, dtype=float)

def _bond_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a))

def _bond_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    cos_t = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    cos_t = max(-1.0, min(1.0, cos_t))
    return float(np.arccos(cos_t))

def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1 /= (np.linalg.norm(n1) + 1e-12)
    n2 /= (np.linalg.norm(n2) + 1e-12)

    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.arctan2(y, x))
#_DEBUG_FILE_INDEX = 34   # 35th file
#_FILE_COUNTER = 0

def _internal_coords_from_xyz(path: Path) -> np.ndarray:
    global _FILE_COUNTER

    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        n = int(f.readline().strip())
        _ = f.readline()
        atoms = []

        for _i in range(n):
            parts = f.readline().split()
            if len(parts) < 4:
                continue
            el = parts[0]
            x, y, z = parts[1:4]
            atoms.append([
                ATOM_MAP.get(el, ATOM_MAP.get(el.capitalize(), np.nan)),
                float(x), float(y), float(z)
            ])

    arr = np.asarray(atoms, dtype=float)
    pos = arr[:, 1:4]

    bl = []
    ba = []
    dh = []

    for i in range(1, len(pos)):
        bl.append(_bond_length(pos[i-1], pos[i]))

    for i in range(2, len(pos)):
        ba.append(_bond_angle(pos[i-2], pos[i-1], pos[i]))

    for i in range(3, len(pos)):
        dh.append(_dihedral(pos[i-3], pos[i-2], pos[i-1], pos[i]))

    # print only for the 35th file
 #   if _FILE_COUNTER == _DEBUG_FILE_INDEX:
 #       print("\n==============================")
 #       print("DEBUG: FIRST FILE INTERNAL COORDS")
 #       print("File:", path.name)
 #       print("Total atoms:", len(pos))
 #       print("Total bond angles:", len(ba))
 #       print("\nALL BOND ANGLES:")
 #       for i, angle in enumerate(ba):
 #           print(f"angle[{i}] = {angle}")
 #       print("==============================\n")
 #   _FILE_COUNTER +=1

    return np.asarray(bl + ba + dh, dtype=float)

def _read_descriptor_floats(path: Path) -> np.ndarray:
    """
    Match DD/ADD notebook behavior: read the whole file and split on whitespace.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    toks = text.split()
    vals = [float(t) for t in toks]
    return np.asarray(vals, dtype=float)

def read_excitations_csv(path: Path, wavelength_max_nm: float, pad_value: float) -> np.ndarray:
    """
    Match notebook behavior:
    - read csv to df
    - if ANY wavelength > threshold exists: caller will DROP the conformation entirely
    - else: drop rows with wavelength > threshold (no-op given the check), flatten, pad back to original size
    """
    df = pd.read_csv(path)
    num_elements = df.shape[0] * df.shape[1]
    df2 = df.drop(df[df["wavelength"] > wavelength_max_nm].index) if "wavelength" in df.columns else df
    arr = df2.to_numpy().astype(float).flatten()
    if len(arr) < num_elements:
        arr = np.pad(arr, (0, num_elements - len(arr)), mode="constant", constant_values=pad_value)
    return arr

def load_dataset(
    *,
    descriptor: str,
    features_dir: Path,
    excitations_dir: Path,
    file_glob: str = "*cluster*",
    wavelength_max_nm: float = 900.0,
    pad_value: float = -1.0,
) -> Dataset:
    """
    descriptor: one of {"pca_cc","umap_ic","dd","add"} controlling how features are built.
    - pca_cc: xyz -> tokens (Z, x, y, z) per atom (like PCA-CC notebook)
    - umap_ic: xyz -> internal coords (like PCA-IC notebook)
    - dd/add: read descriptor floats from text files
    """
    descriptor = descriptor.lower()
    features_dir = Path(features_dir)
    excitations_dir = Path(excitations_dir)

    feature_paths = sorted(features_dir.glob(file_glob))
    if not feature_paths:
        raise FileNotFoundError(f"No feature files matched {file_glob!r} under {features_dir}")

    X_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    stems: List[str] = []
    dropped: List[str] = []

    for fp in feature_paths:
        stem = fp.stem
        csv_path = excitations_dir / f"{stem}.csv"
        if not csv_path.exists():
            dropped.append(stem)
            continue

        df_check = pd.read_csv(csv_path)
        if "wavelength" in df_check.columns:
            if (df_check["wavelength"] > wavelength_max_nm).any():
                dropped.append(stem)
                continue
        else:
            # fallback: first column is wavelength-like
            if (df_check.iloc[:, 0] > wavelength_max_nm).any():
                dropped.append(stem)
                continue

        y = read_excitations_csv(csv_path, wavelength_max_nm=wavelength_max_nm, pad_value=pad_value)

        if descriptor in ("pca_cc",):
            x = _read_xyz_tokens_as_numeric(fp)
        elif descriptor in ("umap_ic",):
            x = _internal_coords_from_xyz(fp)
        elif descriptor in ("dd","add"):
            x = _read_descriptor_floats(fp)
        else:
            raise ValueError("descriptor must be one of: pca_cc, umap_ic, dd, add")

        X_rows.append(x)
        y_rows.append(y)
        stems.append(stem)

    if not X_rows:
        raise RuntimeError("No usable samples were loaded (all files dropped or missing excitations CSVs).")

    max_len = max(len(r) for r in X_rows)
    X = np.vstack([np.pad(r, (0, max_len - len(r)), mode="constant", constant_values=0.0) for r in X_rows])
    y = np.vstack(y_rows)

    return Dataset(X=X, y=y, filenames=stems)
