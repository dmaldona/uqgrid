"""Jacobian assembly helpers for the dynamic simulation pipeline."""

from __future__ import annotations

import numpy as np
from numba import jit

from uqgrid.utils.tools import csr_add_row, csr_set_row


@jit(nopython=True, cache=True)
def power_flow_jacobian(
    ybus_data,
    ybus_ptr,
    ybus_idx,
    J_data,
    J_ptr,
    J_idx,
    dev,
    v,
    nbus,
):
    """Assemble Jacobian contributions for power-injection formulation."""

    val = np.zeros(20)
    col = np.zeros(20)

    for fr in range(nbus):
        row = dev + 2 * fr

        col[0] = dev + 2 * fr
        col[1] = dev + 2 * fr + 1
        val[0] = 0.0
        val[1] = 0.0
        csr_set_row(J_data, J_ptr, J_idx, 2, row, col, val)

        row = dev + 2 * fr + 1
        col[0] = dev + 2 * fr
        col[1] = dev + 2 * fr + 1
        val[0] = 0.0
        val[1] = 0.0
        csr_set_row(J_data, J_ptr, J_idx, 2, row, col, val)

        conn = ybus_ptr[fr + 1] - ybus_ptr[fr]

        for i in range(conn):
            to = ybus_idx[ybus_ptr[fr] + i]
            if to == fr:
                gij = ybus_data[ybus_ptr[fr] + i].real
                bij = ybus_data[ybus_ptr[fr] + i].imag

                row = dev + 2 * fr
                col[0] = dev + 2 * fr
                col[1] = dev + 2 * fr + 1
                val[0] = -2 * v[2 * fr] * gij
                val[1] = 0.0
                csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)

                row = dev + 2 * fr + 1
                col[0] = dev + 2 * fr
                col[1] = dev + 2 * fr + 1
                val[0] = 2 * v[2 * fr] * bij
                val[1] = 0.0
                csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)

            else:
                angleij = v[2 * fr + 1] - v[2 * to + 1]

                gij = ybus_data[ybus_ptr[fr] + i].real
                bij = ybus_data[ybus_ptr[fr] + i].imag

                row = dev + 2 * fr
                col[0] = dev + 2 * to
                col[1] = dev + 2 * to + 1
                val[0] = -v[2 * fr] * (gij * np.cos(angleij) + bij * np.sin(angleij))
                val[1] = -v[2 * fr] * v[2 * to] * (gij * np.sin(angleij) - bij * np.cos(angleij))
                csr_set_row(J_data, J_ptr, J_idx, 2, row, col, val)

                col[0] = dev + 2 * fr
                col[1] = dev + 2 * fr + 1
                val[0] = -v[2 * to] * (gij * np.cos(angleij) + bij * np.sin(angleij))
                val[1] = -v[2 * fr] * v[2 * to] * (-gij * np.sin(angleij) + bij * np.cos(angleij))
                csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)

                row = dev + 2 * fr + 1
                col[0] = dev + 2 * to
                col[1] = dev + 2 * to + 1
                val[0] = -v[2 * fr] * (gij * np.sin(angleij) - bij * np.cos(angleij))
                val[1] = -v[2 * fr] * v[2 * to] * (-gij * np.cos(angleij) - bij * np.sin(angleij))
                csr_set_row(J_data, J_ptr, J_idx, 2, row, col, val)

                col[0] = dev + 2 * fr
                col[1] = dev + 2 * fr + 1
                val[0] = -v[2 * to] * (gij * np.sin(angleij) - bij * np.cos(angleij))
                val[1] = -v[2 * fr] * v[2 * to] * (gij * np.cos(angleij) + bij * np.sin(angleij))
                csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)


@jit(nopython=True, cache=True)
def current_injection_jacobian(ybus_data, ybus_ptr, ybus_idx, jac_data, jac_ptr, jac_idx, dev):
    """Jacobian assembly for current-injection formulation."""

    col = np.zeros(100, dtype=np.int32)
    val = np.zeros(100, dtype=np.double)

    for row_idx in range(len(ybus_ptr) - 1):
        row_ptr = ybus_ptr[row_idx]
        row_ptr_end = ybus_ptr[row_idx + 1]

        nvals = row_ptr_end - row_ptr
        row = row_idx + dev

        for j in range(nvals):
            col[j] = ybus_idx[row_ptr + j] + dev
            val[j] = -ybus_data[row_ptr + j]

        csr_set_row(jac_data, jac_ptr, jac_idx, nvals, row, col, val)


@jit(nopython=True, cache=True)
def _jacobian_diagonal_zeros(J_data, J_ptr, J_idx, ndim, ndiffeq):
    col = np.array([0])
    data = np.array([0.0])
    for i in range(ndiffeq):
        col[0] = i
        csr_set_row(J_data, J_ptr, J_idx, 1, i, col, data)


def residual_jacobian(J, z, theta, psys):
    """Top-level residual Jacobian assembly entry point."""

    if z.flags.writeable:
        z = z.view()
        z.flags.writeable = False

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif

    _jacobian_diagonal_zeros(J.data, J.indptr, J.indices, J.shape[0], dif_size)

    v = z[dif_size + alg_size :]

    dev = alg_size + dif_size

    if psys.power_injection:
        power_flow_jacobian(
            psys.ybus_spa.data,
            psys.ybus_spa.indptr,
            psys.ybus_spa.indices,
            J.data,
            J.indptr,
            J.indices,
            dev,
            v,
            psys.nbuses,
        )
    else:
        current_injection_jacobian(
            psys.rybus.data,
            psys.rybus.indptr,
            psys.rybus.indices,
            J.data,
            J.indptr,
            J.indices,
            dev,
        )

    idxs = np.zeros(5, dtype=np.int64)

    for device in psys.devices:
        idxs[0] = device.dif_ptr
        idxs[1] = dif_size + device.alg_ptr
        idxs[2] = alg_size + dif_size
        idxs[3] = device.par_ptr
        idxs[4] = device.bus

        ctrl_idx = device.ctrl_idx
        ctrl_var = device.ctrl_var

        device.residual_jac(J, z, v, theta, idxs, ctrl_idx, ctrl_var, psys.power_injection)

    for fault in psys.fault_events:
        if fault.active:
            fault.residual_jac(J, z, v, theta, dev, psys.power_injection)
