from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class SplitBundle:
    """
    Container for deterministic splits shared across descriptors.

    test_set_indices: shape (n_iter, test_size)
      - test_set_indices[it] gives the test indices for iteration it

    train_set_indices: dict mapping training_size -> array of shape (n_iter, training_size)
      - train_set_indices[size][it] gives the training indices for (size, iteration)
    """
    test_set_indices: np.ndarray
    train_set_indices: Dict[int, np.ndarray]
    training_sizes: List[int]
    filenames: List[str]
    seed: int
    n_iter: int
    test_size: int


def compute_test_set_indices(
    n_samples: int,
    seed: int,
    n_iter: int,
    test_size: int,
) -> np.ndarray:
    """
    Notebook-exact test split generation:

    for iteration in range(n_iter):
        np.random.seed(seed + iteration)
        test_idx = np.random.choice(np.arange(n_samples), size=test_size, replace=False)

    Returns:
        np.ndarray of shape (n_iter, test_size), dtype=int
    """
    if test_size <= 0:
        raise ValueError(f"test_size must be positive, got {test_size}")
    if n_iter <= 0:
        raise ValueError(f"n_iter must be positive, got {n_iter}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if test_size >= n_samples:
        raise ValueError(f"test_size ({test_size}) must be < n_samples ({n_samples})")

    all_indices = np.arange(n_samples, dtype=int)
    out = np.empty((n_iter, test_size), dtype=int)

    for it in range(n_iter):
        np.random.seed(seed + it)  # IMPORTANT: notebook behavior
        out[it, :] = np.random.choice(all_indices, size=test_size, replace=False).astype(int)

    return out


def compute_test_and_train_indices(
    n_samples: int,
    seed: int,
    n_iter: int,
    test_size: int,
    training_sizes: List[int],
) -> tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Notebook-exact combined generation for BOTH test and train indices.

    This matches the notebooks' RNG consumption order:
      1) build all test_set_indices with per-iteration reseeding
      2) without reseeding, for each training_size then each iteration,
         sample training indices from the non-test pool

    Returns:
      test_set_indices: (n_iter, test_size)
      train_set_indices: dict[size] -> (n_iter, size)
    """
    if not training_sizes:
        raise ValueError("training_sizes must be non-empty")

    # validate sizes
    for s in training_sizes:
        if s <= 0:
            raise ValueError(f"training size must be positive, got {s}")
        if s >= (n_samples - test_size):
            raise ValueError(
                f"training size {s} must be < n_samples - test_size ({n_samples - test_size})"
            )

    all_indices = np.arange(n_samples, dtype=int)

    # 1) Test indices (with reseeding each iteration)
    test_set_indices = np.empty((n_iter, test_size), dtype=int)
    for it in range(n_iter):
        np.random.seed(seed + it)
        test_set_indices[it, :] = np.random.choice(
            all_indices, size=test_size, replace=False
        ).astype(int)

    # 2) Training indices (NO reseeding; this is exactly what notebooks do)
    train_set_indices: Dict[int, np.ndarray] = {}
    for size in training_sizes:
        arr = np.empty((n_iter, size), dtype=int)
        for it in range(n_iter):
            pool = np.setdiff1d(all_indices, test_set_indices[it], assume_unique=False)
            arr[it, :] = np.random.choice(pool, size=size, replace=False).astype(int)
        train_set_indices[int(size)] = arr

    return test_set_indices, train_set_indices


def _save_npz(
    path: Path,
    test_set_indices: np.ndarray,
    train_set_indices: Dict[int, np.ndarray],
    training_sizes: List[int],
    filenames: List[str],
    seed: int,
    n_iter: int,
    test_size: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fn_arr = np.array(list(filenames), dtype=object)
    training_sizes_arr = np.array(list(training_sizes), dtype=int)

    # Save train arrays under deterministic keys
    payload = {
        "test_set_indices": np.asarray(test_set_indices, dtype=int),
        "filenames": fn_arr,
        "seed": int(seed),
        "n_iter": int(n_iter),
        "test_size": int(test_size),
        "n_samples": int(len(filenames)),
        "training_sizes": training_sizes_arr,
    }
    for size in training_sizes:
        key = f"train_set_indices__{int(size)}"
        payload[key] = np.asarray(train_set_indices[int(size)], dtype=int)

    np.savez(path, **payload)


def _load_npz(path: Path) -> SplitBundle:
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    test_set_indices = np.asarray(data["test_set_indices"], dtype=int)

    filenames = data["filenames"].tolist()
    if not isinstance(filenames, list):
        filenames = list(filenames)

    seed = int(data.get("seed", -1))
    n_iter = int(data.get("n_iter", test_set_indices.shape[0]))
    test_size = int(data.get("test_size", test_set_indices.shape[1]))

    training_sizes = data.get("training_sizes", None)
    if training_sizes is None:
        raise ValueError(
            f"Split file {path} does not contain training_sizes / train indices. "
            f"Delete and regenerate with the new splits.py."
        )
    training_sizes = [int(x) for x in np.asarray(training_sizes, dtype=int).tolist()]

    train_set_indices: Dict[int, np.ndarray] = {}
    for size in training_sizes:
        key = f"train_set_indices__{int(size)}"
        if key not in data:
            raise ValueError(f"Split file {path} missing key {key}")
        train_set_indices[int(size)] = np.asarray(data[key], dtype=int)

    return SplitBundle(
        test_set_indices=test_set_indices,
        train_set_indices=train_set_indices,
        training_sizes=training_sizes,
        filenames=[str(x) for x in filenames],
        seed=seed,
        n_iter=n_iter,
        test_size=test_size,
    )


def get_or_create_splits(
    split_path: Path,
    filenames: List[str],
    seed: int,
    n_iter: int,
    test_size: int,
    training_sizes: List[int],
    reuse_if_exists: bool = True,
    strict_filename_match: bool = True,
) -> SplitBundle:
    """
    Load precomputed splits from split_path, or create and save them.

    IMPORTANT: This version stores BOTH test and train indices, so all descriptors
    get identical (train, test) subsets for every (training_size, iteration).
    """
    split_path = Path(split_path)

    if reuse_if_exists and split_path.exists():
        bundle = _load_npz(split_path)

        # Shape checks
        if bundle.test_set_indices.shape != (n_iter, test_size):
            raise ValueError(
                f"Split file {split_path} has test shape {bundle.test_set_indices.shape}, "
                f"expected {(n_iter, test_size)}. Delete it or set reuse_if_exists=False."
            )

        # Training sizes checks
        if list(bundle.training_sizes) != list(training_sizes):
            raise ValueError(
                f"Split file {split_path} has training_sizes={bundle.training_sizes}, "
                f"expected {list(training_sizes)}. Delete and regenerate."
            )
        for s in training_sizes:
            arr = bundle.train_set_indices[int(s)]
            if arr.shape != (n_iter, int(s)):
                raise ValueError(
                    f"Split file {split_path} train shape for size {s} is {arr.shape}, "
                    f"expected {(n_iter, int(s))}. Delete and regenerate."
                )

        # Filename/order check
        if strict_filename_match:
            if bundle.filenames != list(filenames):
                min_len = min(len(bundle.filenames), len(filenames))
                first_diff = None
                for i in range(min_len):
                    if bundle.filenames[i] != filenames[i]:
                        first_diff = i
                        break
                msg = (
                    f"Filenames/order mismatch vs split file {split_path}.\n"
                    f"This would break cross-descriptor consistency.\n"
                )
                if first_diff is not None:
                    msg += (
                        f"First difference at index {first_diff}:\n"
                        f"  split file: {bundle.filenames[first_diff]}\n"
                        f"  current   : {filenames[first_diff]}\n"
                    )
                else:
                    msg += (
                        f"Lengths differ: split file has {len(bundle.filenames)}, "
                        f"current has {len(filenames)}.\n"
                    )
                msg += (
                    "Fix by ensuring all descriptors load the same conformations in the same order, "
                    "or regenerate splits by deleting the split file, or set strict_filename_match=False "
                    "(not recommended)."
                )
                raise ValueError(msg)

        if bundle.seed not in (-1, seed):
            raise ValueError(f"Split file seed={bundle.seed} but config seed={seed}: {split_path}")

        return SplitBundle(
            test_set_indices=bundle.test_set_indices,
            train_set_indices=bundle.train_set_indices,
            training_sizes=list(training_sizes),
            filenames=list(filenames),
            seed=seed,
            n_iter=n_iter,
            test_size=test_size,
        )

    # Create new
    n_samples = len(filenames)
    test_set_indices, train_set_indices = compute_test_and_train_indices(
        n_samples=n_samples,
        seed=seed,
        n_iter=n_iter,
        test_size=test_size,
        training_sizes=list(training_sizes),
    )

    _save_npz(
        split_path,
        test_set_indices=test_set_indices,
        train_set_indices=train_set_indices,
        training_sizes=list(training_sizes),
        filenames=list(filenames),
        seed=seed,
        n_iter=n_iter,
        test_size=test_size,
    )

    return SplitBundle(
        test_set_indices=test_set_indices,
        train_set_indices=train_set_indices,
        training_sizes=list(training_sizes),
        filenames=list(filenames),
        seed=seed,
        n_iter=n_iter,
        test_size=test_size,
    )

