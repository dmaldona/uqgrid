"""Gradient helper functions for dynamic simulations."""

from __future__ import annotations

import numpy as np


def gradient_p(psys, z, theta, load_idx: int = 0) -> np.ndarray:
    """Gradient of the residual with respect to a single load parameter."""

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    sys_size = alg_size + dif_size + 2 * psys.nbuses
    dev = alg_size + dif_size

    v = z[dif_size + alg_size :]

    gradient = np.zeros(sys_size)
    psys.loads[load_idx].gradient_alpha(
        gradient[alg_size + dif_size :], z, v, theta, dev, psys.power_injection
    )

    return gradient


def gradient_xp(psys, z, theta, load_idx: int = 0) -> np.ndarray:
    """Mixed partial derivatives of the residual with respect to x and p."""

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    sys_size = alg_size + dif_size + 2 * psys.nbuses

    dev = alg_size + dif_size
    v = z[dif_size + alg_size :]

    mixed = np.zeros((sys_size, sys_size))

    psys.loads[load_idx].gradient_pp_alpha(mixed, z, v, theta, dev)

    return mixed


def gradient_pp(psys, z, theta, idx_a: int = 0, idx_b: int = 0) -> np.ndarray:
    """Second derivative of the residual with respect to load parameters."""

    del idx_a, idx_b

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    sys_size = alg_size + dif_size + 2 * psys.nbuses

    return np.zeros(sys_size)
