from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from .indexing import PFIndexCache


def build_param_selector(cache: PFIndexCache) -> csr_matrix:
    """Return the sparse load-parameter selector f_lambda for PQ buses."""

    n_rows = cache.n_unknowns
    n_cols = 2 * cache.n_pq

    if cache.n_pq == 0:
        return csr_matrix((n_rows, 0), dtype=float)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for local_idx, bus_idx in enumerate(cache.pq_buses):
        p_row = cache.pqv_indices[bus_idx]
        if p_row < 0:
            raise ValueError(f"PQ bus {bus_idx} missing active-power row index.")
        rows.append(cache.n_pq + int(p_row))
        cols.append(local_idx)
        data.append(1.0)

        q_row = cache.pq_indices[bus_idx]
        if q_row < 0:
            raise ValueError(f"PQ bus {bus_idx} missing reactive-power row index.")
        rows.append(int(q_row))
        cols.append(cache.n_pq + local_idx)
        data.append(1.0)

    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=float)
