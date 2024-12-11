# IMPLEMENTATION OF LOAD MODELS
import numpy as np
import numba
from numba import jit
from .tools import csr_add_row, csr_set_row

@jit(nopython=True, cache=True)
def cinj_load(F, z, v, theta, idxs):

    pp = idxs[2]
    bus = idxs[3]

    vr = v[2*bus]
    vi = v[2*bus + 1]

    pl = theta[pp]
    ql = theta[pp + 1]
    alpha = theta[pp + 2]
    weight = theta[pp + 3]
    v0 = theta[pp + 4]

    yload_real = alpha*theta[pp + 5]
    yload_imag = alpha*theta[pp + 6]

    vm2 = vr*vr + vi*vi
    vm2_tld = 0.2

    F[2*bus] -= vr*yload_real - vi*yload_imag
    F[2*bus + 1] -= vr*yload_imag + vi*yload_real

    if vm2 > vm2_tld:
        F[2*bus] -= (1-alpha)*(pl*vr - ql*vi)/vm2
        F[2*bus + 1] -= (1-alpha)*(ql*vr + pl*vi)/vm2
    else:
        F[2*bus] -= (1-alpha)*(pl*vr - ql*vi)/vm2_tld
        F[2*bus + 1] -= (1-alpha)*(ql*vr + pl*vi)/vm2_tld

@jit(nopython=True, cache=True)
def jac_load(z, v, theta, idxs,
        ctrl_idx, ctrl_var, J_data, J_ptr, J_idx, power_injection):

    dp = idxs[0]
    ap = idxs[1]
    dev = idxs[2]
    pp = idxs[3]
    bus = idxs[4]

    vr = v[2*bus]
    vi = v[2*bus + 1]

    pl = theta[pp]
    ql = theta[pp + 1]
    alpha = theta[pp + 2]
    weight = theta[pp + 3]
    v0 = theta[pp + 4]

    yload_real = theta[pp + 5]
    yload_imag = theta[pp + 6]
    
    if power_injection:
        vm = v[2*bus]
        va = v[2*bus + 1]
    else:
        vr = v[2*bus]
        vi = v[2*bus + 1]
        vm = np.sqrt(vr**2.0 + vi**2.0)
        va = np.arctan2(vi, vr)

    col = np.zeros(2)
    val = np.zeros(2)

    if power_injection:
        # first row
        row = dev + 2*bus
        col[0] = dev + 2*bus
        val[0] = -alpha*2.0*pl*(vm/v0)**2.0/vm
        csr_add_row(J_data, J_ptr, J_idx, 1, row, col, val)

        # second row
        row = dev + 2*bus + 1
        col[0] = dev + 2*bus
        val[0] = alpha*(2.0*ql*(vm/v0)**2.0)/vm
        csr_add_row(J_data, J_ptr, J_idx, 1, row, col, val)

    else:
        # constant admittance contribution
        row = dev + 2*bus
        col[0] = dev + 2*bus
        col[1] = dev + 2*bus + 1
        val[0] = -alpha*yload_real
        val[1] = alpha*yload_imag
        csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)

        row = dev + 2*bus + 1
        col[0] = dev + 2*bus
        col[1] = dev + 2*bus + 1
        val[0] = -alpha*yload_imag
        val[1] = -alpha*yload_real
        csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)

        # constant power contribution
        vm2 = vr*vr + vi*vi
        vm2_tld = 0.2

        row = dev + 2*bus
        col[0] = dev + 2*bus
        col[1] = dev + 2*bus + 1
        if vm2 > vm2_tld:
            val[0] = (1-alpha)*((pl*vr - ql*vi)*2*vr - pl*vm2)/vm2**2.0
            val[1] = (1-alpha)*((pl*vr - ql*vi)*2*vi + ql*vm2)/vm2**2.0
        else:
            val[0] = (1-alpha)*(-pl)/vm2_tld
            val[1] = (1-alpha)*(ql)/vm2_tld
        csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)

        row = dev + 2*bus + 1
        col[0] = dev + 2*bus
        col[1] = dev + 2*bus + 1

        if vm2 > vm2_tld:
            val[0] = (1-alpha)*((ql*vr + pl*vi)*2*vr - ql*vm2)/vm2**2.0
            val[1] = (1-alpha)*((ql*vr + pl*vi)*2*vi - pl*vm2)/vm2**2.0
        else:
            val[0] = (1-alpha)*(-ql)/vm2_tld
            val[1] = (1-alpha)*(-pl)/vm2_tld
        csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)
