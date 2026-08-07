from __future__ import annotations

from typing import Any, Dict, List, Tuple


def resolve_split_sizes(n_samples: int, cfg_all: Dict[str, Any]) -> Tuple[int, List[int]]:
    """
    Resolve (test_set_size, training_sizes) for a dataset with n_samples conformations.

    Backward compatible: if you still set explicit `test_set_size` and `training_sizes`
    in the YAML, those are used as-is (old behavior, unchanged).

    Otherwise, sizes are derived as fractions of n_samples so the SAME config file
    works across datasets with different numbers of conformations:

      test_fraction       (default 0.2)  -> test_set_size = round(test_fraction * n_samples)
      min_train_fraction  (default 0.3)  -> smallest training size, as a fraction of the
                                             usable pool (n_samples - test_set_size)
      n_training_points   (default 8)    -> number of points in the training-size sweep,
                                             evenly spaced from min_train up to max_train
    """
    explicit_test = cfg_all.get("test_set_size", None)
    explicit_sizes = cfg_all.get("training_sizes", None)

    if explicit_test is not None and explicit_sizes is not None:
        return int(explicit_test), [int(x) for x in explicit_sizes]

    test_fraction = float(cfg_all.get("test_fraction", 0.2))
    test_set_size = int(explicit_test) if explicit_test is not None else max(
        1, round(test_fraction * n_samples)
    )

    # splits.py requires training size STRICTLY LESS THAN (n_samples - test_set_size)
    max_train = n_samples - test_set_size - 1
    if max_train < 5:
        raise ValueError(
            f"Not enough samples ({n_samples}) for test_set_size={test_set_size}; "
            f"only {max_train} usable for training. Lower test_fraction/test_set_size."
        )

    if explicit_sizes is not None:
        training_sizes = [int(x) for x in explicit_sizes]
    else:
        min_train_fraction = float(cfg_all.get("min_train_fraction", 0.3))
        n_points = int(cfg_all.get("n_training_points", 8))
        min_train = max(5, round(min_train_fraction * max_train))
        min_train = min(min_train, max_train)

        if n_points <= 1 or min_train == max_train:
            training_sizes = [max_train]
        else:
            step = (max_train - min_train) / (n_points - 1)
            sizes = sorted({round(min_train + i * step) for i in range(n_points)})
            sizes[-1] = max_train
            training_sizes = sizes

    for s in training_sizes:
        if s <= 0 or s >= n_samples - test_set_size:
            raise ValueError(
                f"training size {s} invalid for n_samples={n_samples}, "
                f"test_set_size={test_set_size} (must be in (0, {n_samples - test_set_size}))."
            )

    return test_set_size, training_sizes
