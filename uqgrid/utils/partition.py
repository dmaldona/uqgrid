"""Utilities for working with ARKIMEX partitioning artifacts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence

TIME_KEY = "Time (sec):"


def sample_slow_indices(total_diff: int, min_frac: float, max_frac: float) -> List[int]:
    """Sample a sorted list of unique slow indices within the provided fraction range."""
    if total_diff == 0:
        return []

    fraction = random.uniform(min_frac, max_frac)
    desired_count = int(round(fraction * total_diff))
    desired_count = max(0, min(desired_count, total_diff))
    if desired_count == 0:
        return []

    all_indices = list(range(total_diff))
    return sorted(random.sample(all_indices, desired_count))


def format_index_list(indices: Sequence[int]) -> str:
    """Format a sequence of indices as a comma-separated string."""
    if not indices:
        return ""
    return ",".join(str(idx) for idx in indices)


def extract_runtime_from_log(log_path: Path) -> Optional[float]:
    """Parse the PETSc runtime entry from a log file, if present."""
    if not log_path.is_file():
        return None

    try:
        with log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if TIME_KEY in line:
                    parts = line.strip().split()
                    for token in parts:
                        try:
                            return float(token)
                        except ValueError:
                            continue
                    raise ValueError(f"Failed to parse runtime from line: {line.strip()}")
    except OSError:
        return None

    return None
