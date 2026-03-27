"""NPZ-based caching of intermediate pipeline results.

Caches expensive computation outputs (flattened point clouds, registration
transforms, signed distances) to disk so that visualization and metric
extraction can run without re-computing upstream stages.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def cache_key(source_path: Path, stage: str) -> str:
    """Generate a cache filename from the source file and pipeline stage.

    Parameters
    ----------
    source_path
        Path to the input file (e.g., scan CSV).
    stage
        Pipeline stage name (e.g., 'flatten', 'register', 'distances').

    Returns
    -------
    str
        Cache filename without extension (e.g., 'm_VBN_05_cycle_001_flatten').
    """
    return f"{source_path.stem}_{stage}"


def save_cache(
    cache_dir: Path,
    key: str,
    **arrays: np.ndarray,
) -> Path:
    """Save arrays to a compressed NPZ file.

    Parameters
    ----------
    cache_dir
        Directory to store cache files. Created if it doesn't exist.
    key
        Cache key (filename stem).
    **arrays
        Named arrays to save.

    Returns
    -------
    Path
        Path to the saved NPZ file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{key}.npz"
    np.savez_compressed(out_path, **arrays)
    return out_path


def load_cache(
    cache_dir: Path,
    key: str,
    source_path: Path | None = None,
) -> dict[str, np.ndarray] | None:
    """Load cached arrays if they exist and are still valid.

    Parameters
    ----------
    cache_dir
        Directory containing cache files.
    key
        Cache key (filename stem).
    source_path
        If provided, cache is invalidated if source file is newer than cache.

    Returns
    -------
    dict or None
        Dictionary of named arrays, or None if cache miss or invalid.
    """
    cache_path = cache_dir / f"{key}.npz"
    if not cache_path.exists():
        return None
    if source_path is not None and not is_cache_valid(cache_path, source_path):
        return None
    data = np.load(cache_path, allow_pickle=False)
    return dict(data)


def is_cache_valid(cache_path: Path, source_path: Path) -> bool:
    """Check if a cache file is newer than its source file.

    Parameters
    ----------
    cache_path
        Path to the NPZ cache file.
    source_path
        Path to the original source file.

    Returns
    -------
    bool
        True if cache exists and is newer than source.
    """
    if not cache_path.exists():
        return False
    if not source_path.exists():
        return False
    return cache_path.stat().st_mtime >= source_path.stat().st_mtime
