"""Finite-difference Jacobian checker with human-readable index mapping."""

from __future__ import annotations

import numpy as np

from uqgrid.simulation.residual import residual_function


def _ext_bus_map(psys):
    ext2int = getattr(psys, "ext2int", None)
    if not ext2int:
        return {i: i for i in range(psys.nbuses)}
    return {v: k for k, v in ext2int.items()}


def build_index_map(psys) -> dict[int, str]:
    """Build a map from global index to a human-readable descriptor."""
    dif_size = psys.num_dof_dif
    alg_size = psys.num_dof_alg
    idx_map: dict[int, str] = {}

    int2ext = _ext_bus_map(psys)

    for dev in psys.devices:
        # Differential states
        for i in range(dev.dif_dim):
            name = _state_name(dev, i)
            idx_map[dev.dif_ptr + i] = f"{dev.model_type}:{name}@bus{int2ext.get(dev.bus, dev.bus)}"
        # Algebraic states
        for i in range(dev.alg_dim):
            name = _alg_name(dev, i)
            idx_map[dif_size + dev.alg_ptr + i] = f"{dev.model_type}:{name}@bus{int2ext.get(dev.bus, dev.bus)}"

    # Network equations (power/current balance)
    for i in range(psys.nbuses):
        ext_bus = int2ext.get(i, i)
        base = dif_size + alg_size + 2 * i
        if psys.power_injection:
            idx_map[base] = f"P_mismatch@bus{ext_bus}"
            idx_map[base + 1] = f"Q_mismatch@bus{ext_bus}"
        else:
            idx_map[base] = f"Ir_mismatch@bus{ext_bus}"
            idx_map[base + 1] = f"Ii_mismatch@bus{ext_bus}"

    return idx_map


def _state_name(dev, local_idx):
    if hasattr(dev, "state_list") and local_idx < len(dev.state_list):
        return dev.state_list[local_idx]
    return f"diff{local_idx}"


def _alg_name(dev, local_idx):
    if hasattr(dev, "state_list"):
        idx = dev.dif_dim + local_idx
        if idx < len(dev.state_list):
            return dev.state_list[idx]
    return f"alg{local_idx}"


def finite_difference_jacobian(psys, z, theta, eps=1e-6):
    """Compute a dense finite-difference Jacobian of residual_function."""
    n = z.shape[0]
    J = np.zeros((n, n), dtype=np.float64)
    f0 = np.zeros(n, dtype=np.float64)
    residual_function(f0, z, theta, psys)

    for j in range(n):
        z_p = z.copy()
        z_m = z.copy()
        z_p[j] += eps
        z_m[j] -= eps
        f_p = np.zeros(n, dtype=np.float64)
        f_m = np.zeros(n, dtype=np.float64)
        residual_function(f_p, z_p, theta, psys)
        residual_function(f_m, z_m, theta, psys)
        J[:, j] = (f_p - f_m) / (2 * eps)

    return J


def compare_jacobian_columns(
    psys, z, theta, J_analytical, columns, *, rows=None, eps=1e-6,
):
    """Compare selected Jacobian columns without allocating a dense matrix."""
    z = np.asarray(z, dtype=float)
    columns = tuple(dict.fromkeys(int(column) for column in columns))
    rows = np.arange(z.size, dtype=int) if rows is None else np.asarray(rows, dtype=int)
    index_map = build_index_map(psys)
    results = []
    for column in columns:
        step = float(eps) * max(1.0, abs(float(z[column])))
        increased = z.copy()
        decreased = z.copy()
        increased[column] += step
        decreased[column] -= step
        f_increased = np.zeros_like(z)
        f_decreased = np.zeros_like(z)
        residual_function(f_increased, increased, theta, psys)
        residual_function(f_decreased, decreased, theta, psys)
        numerical = (f_increased[rows] - f_decreased[rows]) / (2.0 * step)
        analytical = np.asarray(J_analytical[rows, column].toarray()).ravel()
        differences = np.abs(analytical - numerical)
        worst = int(np.argmax(differences))
        results.append(
            {
                "column": column,
                "column_desc": index_map.get(column, f"col{column}"),
                "maximum_absolute_error": float(differences[worst]),
                "worst_row": int(rows[worst]),
                "worst_row_desc": index_map.get(int(rows[worst]), f"row{rows[worst]}"),
                "analytical": float(analytical[worst]),
                "finite_difference": float(numerical[worst]),
            }
        )
    return results


def compare_jacobians(psys, z, theta, J_analytical, eps=1e-6, top_k=10, tol=0.0):
    """Compare analytical Jacobian against finite-difference Jacobian.

    Returns a list of top_k mismatches with descriptions.
    """
    J_fd = finite_difference_jacobian(psys, z, theta, eps=eps)
    J_an = J_analytical.toarray() if hasattr(J_analytical, "toarray") else np.array(J_analytical)

    diff = np.abs(J_an - J_fd)
    flat_order = np.argsort(diff.ravel())[::-1]
    idxs = np.dstack(np.unravel_index(flat_order, diff.shape))[0]

    idx_map = build_index_map(psys)
    mismatches = []
    count = 0
    for k in range(idxs.shape[0]):
        i, j = int(idxs[k][0]), int(idxs[k][1])
        if diff[i, j] < tol:
            continue
        mismatches.append(
            {
                "row": i,
                "col": j,
                "row_desc": idx_map.get(i, f"row{i}"),
                "col_desc": idx_map.get(j, f"col{j}"),
                "analytical": J_an[i, j],
                "finite_diff": J_fd[i, j],
                "abs_diff": diff[i, j],
            }
        )
        count += 1
        if count >= top_k:
            break
    return mismatches
