"""Residual assembly utilities for dynamic simulations."""

from __future__ import annotations

import numpy as np
from scipy.sparse._sparsetools import csr_matvec

from uqgrid.simulation.pflow import compute_pinj_alt


def residual_function(F: np.ndarray, z: np.ndarray, theta: np.ndarray, psys) -> None:
    """Populate the residual vector for the coupled DAE system."""

    F.fill(0.0)
    if z.flags.writeable:
        z = z.view()
        z.flags.writeable = False

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif

    v = z[dif_size + alg_size :]

    if psys.power_injection:
        compute_pinj_alt(
            v,
            F[alg_size + dif_size :],
            psys.ybus_mat,
            psys.graph_mat,
            psys.nbuses,
        )
    else:
        csr_matvec(
            psys.rybus.shape[0],
            psys.rybus.shape[1],
            psys.rybus.indptr,
            psys.rybus.indices,
            psys.rybus.data,
            v,
            F[alg_size + dif_size :],
        )
    F[alg_size + dif_size :] = -1.0 * F[alg_size + dif_size :]

    idxs = np.zeros(4, dtype=np.int64)

    for device in psys.devices:
        idxs[0] = device.dif_ptr
        idxs[1] = dif_size + device.alg_ptr
        idxs[2] = device.par_ptr
        idxs[3] = device.bus

        device.residual_diff(
            F,
            z,
            v,
            theta,
            idxs,
            psys.power_injection,
        )
        if psys.power_injection:
            device.residual_pinj(F[alg_size + dif_size :], z, v, theta, idxs)
        else:
            device.residual_cinj(F[alg_size + dif_size :], z, v, theta, idxs)

    # Gather controller outputs into generator-aligned buffers
    if getattr(psys, "gendyn", None):
        psys.p_m_ctrl_aligned.fill(0.0)
        for gi, col in enumerate(psys.gen_pm_ctrl_col):
            if col >= 0:
                psys.p_m_ctrl_aligned[gi] = z[col]

        psys.e_fd_ctrl_aligned.fill(0.0)
        for gi, col in enumerate(psys.gen_efd_ctrl_col):
            if col >= 0:
                psys.e_fd_ctrl_aligned[gi] = z[col]

        # Apply blend equations for each generator
        for gen in psys.gendyn:
            idxs[0] = gen.dif_ptr
            idxs[1] = dif_size + gen.alg_ptr
            gen.residual_blend(F, z, v, theta, idxs, psys)

    for fault in psys.fault_events:
        if fault.active:
            if psys.power_injection:
                fault.residual_pinj(F[alg_size + dif_size :], v)
            else:
                fault.residual_cinj(F[alg_size + dif_size :], v)