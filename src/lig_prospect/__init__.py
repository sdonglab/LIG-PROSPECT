"""LIG-PROSPECT package initialization."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def _configure_numba_cache() -> None:
    """Give UMAP/Numba a writable cache outside the installed package directory."""
    if os.environ.get("NUMBA_CACHE_DIR"):
        return

    cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_dir = Path(cache_home) if cache_home else Path.home() / ".cache"
    cache_dir = cache_dir / "lig-prospect" / "numba"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(cache_dir, os.W_OK | os.X_OK):
            raise OSError(f"Cache directory is not writable: {cache_dir}")
    except OSError:
        cache_dir = Path(tempfile.gettempdir()) / "lig-prospect-numba"
        cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)


_configure_numba_cache()

__all__ = []
