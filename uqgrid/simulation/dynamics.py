from __future__ import print_function

import copy
import math
import sys
from typing import Optional

import numdifftools as nd
import numpy as np
from numba import jit
from numpy import linalg as LA
from scipy import optimize
from scipy.sparse import csr_matrix
from scipy.sparse._sparsetools import csr_matvec
from scipy.sparse.linalg import factorized, spsolve

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Optional: PETSC4py
try:
    import petsc4py
    petsc4py.init(sys.argv)
    from petsc4py import PETSc
except ImportError:
    petsc4py = None
    logger.warning("PETSc4py not available. Some functionality will not be available.")

from uqgrid.simulation.config import IntegrationConfig, IntegrationCtx
from uqgrid.core import Psystem
from uqgrid.simulation.pflow import runpf, compute_pinj_alt, PowerFlowSolution
from uqgrid.simulation.gradients import gradient_p, gradient_xp, gradient_pp
from uqgrid.simulation.residual import residual_function
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.utils.tools import (
    matprint,
    csr_mult_row,
    csr_add_row,
    csr_set_row,
    csr_to_zeros,
)
# supress annoying LAPACK warning on MACOS
import warnings
warnings.filterwarnings(action="ignore",
                        module="scipy",
                        message="^internal gelsd")

# Test flags
TEST_JACOBIAN = False
VERIFY_HESSIAN = False
SECONDORDER = True


@jit(nopython=True, cache=True)
def power_flow_hessian(
    fr,
    ybus_data,
    ybus_ptr,
    ybus_idx,
    HP_data,
    HP_ptr,
    HP_idx,
    HQ_data,
    HQ_ptr,
    HQ_idx,
    dev,
    v,
    nbus,
):

    val = np.zeros(20)
    col = np.zeros(20)

    pinj_vf_vf = 0.0
    qinj_vf_vf = 0.0

    pinj_af_af = 0.0
    qinj_af_af = 0.0

    pinj_vf_af = 0.0
    qinj_vf_af = 0.0

    conn = ybus_ptr[fr + 1] - ybus_ptr[fr]

    for i in range(conn):

        to = ybus_idx[ybus_ptr[fr] + i]

        if to == fr:
            gij = ybus_data[ybus_ptr[fr] + i].real
            bij = ybus_data[ybus_ptr[fr] + i].imag
            pinj_vf_vf += -2 * gij
            qinj_vf_vf += 2 * bij

        else:
            angleij = v[2 * fr + 1] - v[2 * to + 1]
            gij = ybus_data[ybus_ptr[fr] + i].real
            bij = ybus_data[ybus_ptr[fr] + i].imag

            gsin = gij * np.sin(angleij)
            bsin = bij * np.sin(angleij)
            gcos = gij * np.cos(angleij)
            bcos = bij * np.cos(angleij)

            pinj_vf_af -= v[2 * to] * (-gsin + bcos)
            pinj_af_af -= v[2 * fr] * v[2 * to] * (-gcos - bsin)

            qinj_vf_af -= v[2 * to] * (gcos + bsin)
            qinj_af_af -= v[2 * fr] * v[2 * to] * (-gsin + bcos)

            pinj_vt_vt = 0.0
            pinj_vt_at = -v[2 * fr] * (gsin - bcos)
            pinj_at_at = -v[2 * fr] * v[2 * to] * (-gcos - bsin)

            qinj_vt_vt = 0.0
            qinj_vt_at = -v[2 * fr] * (-gcos - bsin)
            qinj_at_at = -v[2 * fr] * v[2 * to] * (-gsin + bcos)

            pinj_vt_vf = -(gcos + bsin)
            pinj_vt_af = -v[2 * fr] * (-gsin + bcos)
            pinj_vf_at = -v[2 * to] * (gsin - bcos)
            pinj_at_af = -v[2 * fr] * v[2 * to] * (gcos + bsin)

            qinj_vt_vf = -(gsin - bcos)
            qinj_vt_af = -v[2 * fr] * (gcos + bsin)
            qinj_vf_at = -v[2 * to] * (-gcos - bsin)
            qinj_at_af = -v[2 * fr] * v[2 * to] * (gsin - bcos)

            row = dev + 2 * fr

            col[0] = dev + 2 * to
            col[1] = dev + 2 * to + 1

            val[0] = pinj_vt_vf
            val[1] = pinj_vf_at
            csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
            val[0] = qinj_vt_vf
            val[1] = qinj_vf_at
            csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)

            row = dev + 2 * fr + 1
            val[0] = pinj_vt_af
            val[1] = pinj_at_af
            csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
            val[0] = qinj_vt_af
            val[1] = qinj_at_af
            csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)

            row = dev + 2 * to

            col[0] = dev + 2 * fr
            col[1] = dev + 2 * fr + 1

            val[0] = pinj_vt_vf
            val[1] = pinj_vt_af
            csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
            val[0] = qinj_vt_vf
            val[1] = qinj_vt_af
            csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)

            row = dev + 2 * to + 1
            val[0] = pinj_vf_at
            val[1] = pinj_at_af
            csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
            val[0] = qinj_vf_at
            val[1] = qinj_at_af
            csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)

            row = dev + 2 * to

            col[0] = dev + 2 * to
            col[1] = dev + 2 * to + 1

            val[0] = pinj_vt_vt
            val[1] = pinj_vt_at
            csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
            val[0] = qinj_vt_vt
            val[1] = qinj_vt_at
            csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)

            row = dev + 2 * to + 1
            val[0] = pinj_vt_at
            val[1] = pinj_at_at
            csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
            val[0] = qinj_vt_at
            val[1] = qinj_at_at
            csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)

    row = dev + 2 * fr

    col[0] = dev + 2 * fr
    col[1] = dev + 2 * fr + 1

    val[0] = pinj_vf_vf
    val[1] = pinj_vf_af
    csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
    val[0] = qinj_vf_vf
    val[1] = qinj_vf_af
    csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)

    row = dev + 2 * fr + 1
    val[0] = pinj_vf_af
    val[1] = pinj_af_af
    csr_set_row(HP_data, HP_ptr, HP_idx, 2, row, col, val)
    val[0] = qinj_vf_af
    val[1] = qinj_af_af
    csr_set_row(HQ_data, HQ_ptr, HQ_idx, 2, row, col, val)


def residual_hessian(H, z, theta, psys):

    # Lock system vector locally when necessary
    if z.flags.writeable:
        z = z.view()
        z.flags.writeable = False

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    # Assign vectors
    x = z[:dif_size]
    y = z[dif_size:dif_size + alg_size]
    v = z[dif_size + alg_size:]

    dev = alg_size + dif_size

    for fr in range(len(psys.graph_list)):

        # retrieve matrices
        Hp = H[dev + 2*fr]
        Hq = H[dev + 2*fr + 1]

        power_flow_hessian(fr, psys.ybus_spa.data, psys.ybus_spa.indptr,
                           psys.ybus_spa.indices, Hp.data, Hp.indptr,
                           Hp.indices, Hq.data, Hq.indptr, Hq.indices, dev, v,
                           psys.nbuses)

    # Load contribution
    for load in psys.loads:
        if load.dynamic == 0:
            load.residual_hes(H, z, v, theta, dev)

    idxs = np.zeros(5, dtype=np.int64)

    for i in range(psys.num_devices):
        idxs[0] = psys.devices[i].dif_ptr
        idxs[1] = dif_size + psys.devices[i].alg_ptr
        idxs[2] = alg_size + dif_size
        idxs[3] = psys.devices[i].par_ptr
        idxs[4] = psys.devices[i].bus

        psys.devices[i].residual_hess(H, z, v, theta, idxs)

    for fault in psys.fault_events:
        if fault.active:
            fault.residual_hes(H, z, v, theta, dev)

    # verify hessian with finite differences

    if VERIFY_HESSIAN == True:

        hes_nd = nd.Hessian(function_hessian_wrapper)
        for eq in range(sys_size):
            H_ND = hes_nd(z, psys, theta, eq)
            #matprint(H_ND)
            if H[eq] is None:
                H_US = np.zeros((sys_size, sys_size))
            else:
                H_US = H[eq].todense()
            #matprint(np.array(H_US))
            is_close = np.allclose(H_US, H_ND)

            if is_close == False:
                matprint(H_ND)
                matprint(np.array(H_US))
                print(H[eq])
                assert False
            else:
                print("True")

    # No need to restore write access because a local read-only view was used

    return None
###################################
#### Parametric jacobian (loads) ##
###################################

def preallocate_jacobian_parameters(psys):
    """Preallocates the Jacobian matrix with respect to load parameters (pl, ql).
    
    Args:
        psys: Power system object
        
    Returns:
        csr_matrix: Sparse Jacobian matrix structure
    """
    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    sys_size = alg_size + dif_size + 2*psys.nbuses
    
    # Number of parameters (pl, ql for each load)
    nparam = 2 * psys.nloads
    
    # Lists to build the sparse matrix
    row = []
    col = []
    
    # For each load, add the derivatives at the corresponding bus
    dev = alg_size + dif_size
    for i, load in enumerate(psys.loads):
        bus = load.bus
        # Each load affects only its bus equations in the power balance
        row.extend([dev + 2*bus, dev + 2*bus + 1])  # Real and reactive power equations
        col.extend([2*i, 2*i])  # pl parameter index
        
        row.extend([dev + 2*bus, dev + 2*bus + 1])  # Real and reactive power equations
        col.extend([2*i + 1, 2*i + 1])  # ql parameter index
    
    data = np.zeros(len(row))
    Jsparse = csr_matrix((data, (row, col)), shape=(sys_size, nparam))
    
    return Jsparse


@jit(nopython=True, cache=True)
def jac_load_params(z, v, theta, idxs, J_data, J_ptr, J_idx, load_idx):
    """
    Compute the Jacobian of the load model with respect to parameters pl and ql.
    
    Args:
        z: State vector
        v: Voltage vector
        theta: Parameter vector
        idxs: Array of indices [dp, ap, pp, bus]
        J_data, J_ptr, J_idx: CSR format arrays for the Jacobian matrix
        load_idx: Index of the load in the system
    """
    dev = idxs[2]
    pp = idxs[3]
    bus = idxs[4]

    vr = v[2*bus]
    vi = v[2*bus + 1]
    
    alpha = theta[pp + 2]
    
    vm2 = vr*vr + vi*vi
    vm2_tld = 0.2  # Voltage threshold for model switching
    
    col = np.zeros(2, dtype=np.int64)
    val = np.zeros(2, dtype=np.double)
    
    # Derivatives for real power balance equation
    row = dev + 2*bus
    
    # Derivative with respect to pl
    col[0] = 2*load_idx
    if vm2 > vm2_tld:
        val[0] = -(1-alpha)*vr/vm2
    else:
        val[0] = -(1-alpha)*vr/vm2_tld
    
    # Derivative with respect to ql
    col[1] = 2*load_idx + 1
    if vm2 > vm2_tld:
        val[1] = (1-alpha)*vi/vm2
    else:
        val[1] = (1-alpha)*vi/vm2_tld
    
    csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)

    # Derivatives for reactive power balance equation
    row = dev + 2*bus + 1
    
    # Derivative with respect to pl
    col[0] = 2*load_idx
    if vm2 > vm2_tld:
        val[0] = -(1-alpha)*vi/vm2
    else:
        val[0] = -(1-alpha)*vi/vm2_tld
    
    # Derivative with respect to ql
    col[1] = 2*load_idx + 1
    if vm2 > vm2_tld:
        val[1] = -(1-alpha)*vr/vm2
    else:
        val[1] = -(1-alpha)*vr/vm2_tld
    
    csr_add_row(J_data, J_ptr, J_idx, 2, row, col, val)


def residual_jacobian_parameters(Jp, z, theta, psys):
    """
    Compute the Jacobian of the residual function with respect to parameters pl and ql.
    
    Args:
        Jp: Sparse Jacobian matrix to be filled
        z: State vector
        theta: Parameter vector
        psys: Power system object
    """
    # Lock system vector locally when necessary
    if z.flags.writeable:
        z = z.view()
        z.flags.writeable = False
    
    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    dev = alg_size + dif_size
    
    # Clear the Jacobian
    Jp.data.fill(0.0)
    
    # Get voltage vector
    v = z[dif_size + alg_size:]
    
    # Compute Jacobian with respect to load parameters
    idxs = np.zeros(5, dtype=np.int64)
    
    for i, load in enumerate(psys.loads):
        idxs[0] = load.dif_ptr
        idxs[1] = dif_size + load.alg_ptr
        idxs[2] = alg_size + dif_size
        idxs[3] = load.par_ptr
        idxs[4] = load.bus

        jac_load_params(z, v, theta, idxs, Jp.data, Jp.indptr, Jp.indices, i)
    
    # No need to restore write access because a local read-only view was used
    
    return None

def test_jacobian_parameters(psys, z, theta):
    """
    Test the implementation of the Jacobian with respect to parameters using
    finite difference approximation.
    
    Args:
        psys: Power system object
        z: State vector
        theta: Parameter vector
        
    Returns:
        tuple: (Jp_analytical, Jp_finite_diff) - the analytical and finite difference Jacobians
    """
    # Preallocate the Jacobian with respect to parameters
    Jp = preallocate_jacobian_parameters(psys)
    
    # Compute the Jacobian
    residual_jacobian_parameters(Jp, z, theta, psys)
    
    # Compute the Jacobian using finite differences
    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    sys_size = alg_size + dif_size + 2*psys.nbuses
    nparam = 2 * psys.nloads
    
    Jp_fd = np.zeros((sys_size, nparam))
    F1 = np.zeros(sys_size)
    F2 = np.zeros(sys_size)
    
    eps = 1e-6
    
    for i, load in enumerate(psys.loads):
        # Derivative with respect to pl
        theta_p = theta.copy()
        theta_m = theta.copy()
        
        theta_p[load.par_ptr] += eps
        theta_m[load.par_ptr] -= eps
        
        residual_function(F1, z, theta_p, psys)
        residual_function(F2, z, theta_m, psys)
        
        Jp_fd[:, 2*i] = (F1 - F2) / (2*eps)
        
        # Derivative with respect to ql
        theta_p = theta.copy()
        theta_m = theta.copy()
        
        theta_p[load.par_ptr + 1] += eps
        theta_m[load.par_ptr + 1] -= eps
        
        residual_function(F1, z, theta_p, psys)
        residual_function(F2, z, theta_m, psys)
        
        Jp_fd[:, 2*i + 1] = (F1 - F2) / (2*eps)
    
    # Compare the results
    Jp_dense = Jp.todense()
    
    if np.allclose(Jp_dense, Jp_fd, rtol=1e-3, atol=1e-3):
        print("Jacobian with respect to parameters test passed!")
    else:
        print("Jacobian with respect to parameters test failed!")
        print("Maximum difference:", np.max(np.abs(Jp_dense - Jp_fd)))
        
        # Optionally show where the differences are
        diff = np.abs(Jp_dense - Jp_fd)
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"Max difference at {idx}: {Jp_dense[idx]} vs {Jp_fd[idx]}")
    
    return Jp_dense, Jp_fd

###################################
#### Integration              #####
###################################


def first_sensitivity(psys, z, sfact, uold, theta, h):
    """Computes first-order sensitivity using backward Euler

    Args:
        psys (psystem): power system object
        z (np.array): power system state
        J (csr_array): power system Jacobian matrix
        uold (np.array): sensitivity vector at step k-1
        theta (np.array): parameter array
        h (float): step size (seconds)
    """

    NDIFFEQ = psys.num_dof_dif

    rhs = np.zeros(z.size)

    
    for i in range(psys.nloads):
        rhs[:] = gradient_p(psys, z, theta, load_idx=i)
        rhs[:NDIFFEQ] = h*rhs[:NDIFFEQ]
        rhs[:NDIFFEQ] += uold[:NDIFFEQ, i]
        rhs = -rhs
        uold[:,i] = sfact(rhs)

@jit("f8(f8[:], f8[:], i8)", nopython=True, cache=True)
def numba_dot(x, y, n):
    res = 0.0
    for i in range(n):
        res += x[i]*y[i]
    return res


def second_sensitivity(psys, x, u, sfact, HES, vold, theta, h):
    """
    Name: second_sensitivity
    Description: computes second order sensitivities (self sensitivities)
    """

    # Hessian will be just a list of matrices. Since seems to be very sparse, we will
    # just write None for all the 0 matrices.

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    NDIFFEQ = psys.num_dof_dif
    NEQ = sys_size
    
    aux_vec = np.zeros(NEQ)
    mu = np.zeros(NEQ)
    ui = np.zeros(NEQ)

    for i in range(psys.nloads):
        GX = gradient_xp(psys, x, theta, load_idx=i)
        g = gradient_pp(psys, x, theta, idx_a=i, idx_b=i)
        ui[:] = u[:,i]
        mu.fill(0.0)

        for j in range(NEQ):
            if HES[j] is not None:
                
                aux_vec.fill(0.0)
                csr_matvec(NEQ, NEQ, HES[j].indptr, HES[j].indices, 
                    HES[j].data, ui, aux_vec)

                mu[j] = numba_dot(ui, aux_vec, NEQ)

        mu += 2.0*np.dot(GX, ui)
        mu += g
        mu[:NDIFFEQ] = h*mu[:NDIFFEQ]
        mu[:NDIFFEQ] += vold[:NDIFFEQ, i]

        mu = -mu

        vold[:,i] = sfact(mu)

def mixed_sensitivity(psys, x, u, sfact, HES, mold, theta, h):
    """
    Name: mixed_sensitivity
    Description: second order mixed sensitivities
    """

    # Hessian will be just a list of matrices. Since seems to be very sparse, we will
    # just write None for all the 0 matrices.

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    NDIFFEQ = psys.num_dof_dif
    NEQ = sys_size
    
    aux_vec = np.zeros(NEQ)
    ui = np.zeros(NEQ)
    uj = np.zeros(NEQ)
    mu = np.zeros(NEQ)

    k = 0
    for i in range(psys.nloads):
        for j in range(i + 1, psys.nloads):
            GXi = gradient_xp(psys, x, theta, load_idx=i)
            GXj = gradient_xp(psys, x, theta, load_idx=j)
            g = gradient_pp(psys, x, theta, idx_a=i, idx_b=j)
            ui[:] = u[:,i]
            uj[:] = u[:,j]
            mu.fill(0.0)

            for eq_idx in range(NEQ):
                if HES[eq_idx] is not None:
                    #mu[eq_idx] = ui.dot(HES[eq_idx].dot(uj))
                    aux_vec.fill(0.0)
                    csr_matvec(NEQ, NEQ, HES[eq_idx].indptr, HES[eq_idx].indices, 
                        HES[eq_idx].data, uj, aux_vec)
                    mu[eq_idx] = numba_dot(ui, aux_vec, NEQ)

            mu += np.dot(GXi, uj)
            mu += np.dot(GXj, ui)
            mu += g
            mu[:NDIFFEQ] = h*mu[:NDIFFEQ]
            mu[:NDIFFEQ] += mold[:NDIFFEQ, k]

            mu = -mu

            mold[:,k] = sfact(mu)
            k += 1

@jit(nopython=True, cache=True)
def _jacobian_beuler(J_data, J_ptr, J_idx, NDIFFEQ, h):
    for i in range(NDIFFEQ):
        csr_mult_row(J_data, J_ptr, J_idx, i, -h)
    col = np.array([0])
    data = np.array([1.0])
    for i in range(NDIFFEQ):
        col[0] = i
        csr_add_row(J_data, J_ptr, J_idx, 1, i, col, data)


def jacobian_beuler(J, NDIFFEQ, h):
    #J[:NDIFFEQ,:] = -h*J[:NDIFFEQ,:]
    #J[:NDIFFEQ, :NDIFFEQ] += sp.sparse.eye(NDIFFEQ)
    _jacobian_beuler(J.data, J.indptr, J.indices, NDIFFEQ, h)


@jit(nopython=True, cache=True)
def _jacobian_implicit(J_data, J_ptr, J_idx, ndim, NDIFFEQ, a):
    for i in range(ndim):
        csr_mult_row(J_data, J_ptr, J_idx, i, -1)
    
    col = np.array([0])
    data = np.array([a])
    for i in range(NDIFFEQ):
        col[0] = i
        csr_add_row(J_data, J_ptr, J_idx, 1, i, col, data)


def jacobian_implicit(J, NDIFFEQ, a):
    # Converts the Jacobian of the r.h.s into:
    # J = [a*I-df_dx, -df_dy
    #     -dg_dx, -dg_dy]
    _jacobian_implicit(J.data, J.indptr, J.indices, J.shape[0], NDIFFEQ, a)


def function_beuler_wrapper(z, zold, h, psys, theta):
    NDIFFEQ = psys.num_dof_dif
    F = np.zeros(len(z))

    residual_function(F, z, theta, psys)
    F[:NDIFFEQ] = z[:NDIFFEQ] - zold[:NDIFFEQ] - h*F[:NDIFFEQ]
    return F


def function_beuler_latin_wrapper(z, zold, h, psys, theta):
    NDIFFEQ = psys.num_dof_dif
    F = np.zeros(len(z))

    residual_function(F, z, theta, psys)
    F[:NDIFFEQ] = z[:NDIFFEQ] - zold[:NDIFFEQ] - h*F[:NDIFFEQ]

    bus_idx = psys.busmag_idx_set()

    for bidx in bus_idx:
        if z[bidx] < 0.1:
            F = F/z[bidx]

    return F


def function_hessian_wrapper(z, psys, theta, idx):
    NDIFFEQ = psys.num_dof_dif
    F = np.zeros(len(z))
    residual_function(F, z, theta, psys)
    return F[idx]


def integrate(zold,
              theta,
              h,
              psys,
              F,
              J,
              Hess,
              verbose=False,
              uold=None,
              vold=None,
              mold=None,
              fsolve=False):
    """
    Name: integrate
    Description: implements backward euler for the OMIB,
        returns x_{t + 1} given x_{t}, h and parameters.

    Notes:

        A = [In_x - h f_x, -hf]
    Args:
        xold (numpy array): state vector at (t).
        h (scalar): integration step in seconds
        e_fd (scalar): parameter
        p_m (scalar): parameter

    Output:
        x (numpy array): state vector at (t+1)
    """

    eps = 1e-10  # N-R tolerance
    max_iter = 500
    iteration = 0
    z = zold

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses
    NDIFFEQ = dif_size

    residual_function(F, z, theta, psys)
    F[:NDIFFEQ] = z[:NDIFFEQ] - zold[:NDIFFEQ] - h*F[:NDIFFEQ]
    norm_res = np.linalg.norm(F)

    if TEST_JACOBIAN:
        jac = nd.Jacobian(function_beuler_wrapper)

    if fsolve:
        sol, info, ier, msg = optimize.fsolve(function_beuler_wrapper,
                                              zold,
                                              args=(zold, h, psys, theta),
                                              full_output=True,
                                              epsfcn=1e-9)

        if ier == 1:
            if verbose: print("Fsolve converged.")
            z = sol
        else:
            raise NameError('Fsolve did not converge')

    else:

        if verbose:
            print("Iteration %d. Residual norm: %g" % (iteration, norm_res))

        # Iterate until residual norm is below tolerance
        while (norm_res > eps) and (iteration < max_iter):
            iteration = iteration + 1

            # Form sparse jacobian matrix
            residual_jacobian(J, z, theta, psys)
            jacobian_beuler(J, NDIFFEQ, h)

            if TEST_JACOBIAN:
                Jnd = jac(z, zold, h, psys, theta)
                jacobian_nd = np.allclose(J.todense(), Jnd)
                Jdiff = J.todense() - Jnd
                np.savetxt('jac_test.csv', Jdiff, delimiter=',')
                assert jacobian_nd == True

            # step
            zdelta = spsolve(J, F)
            z = z - zdelta

            # calculate new residual
            residual_function(F, z, theta, psys)
            F[:NDIFFEQ] = z[:NDIFFEQ] - zold[:NDIFFEQ] - h*F[:NDIFFEQ]

            # print residual norm
            norm_res = np.linalg.norm(F)

            if verbose:
                print("Iteration %d. Residual norm: %g" %
                      (iteration, norm_res))

        if iteration >= max_iter:
            raise NameError('N-R solver did not converge.')

    if uold is not None:
        # We need the Jacobian factorized and in CSC form
        
        csr_to_zeros(J.data, J.indptr, J.indices)
        residual_jacobian(J, z, theta, psys)
        
        for i in range(NDIFFEQ):
            csr_mult_row(J.data, J.indptr, J.indices, i, h)
        col = np.array([0])
        data = np.array([-1.0])
        for i in range(NDIFFEQ):
            col[0] = i
            csr_add_row(J.data, J.indptr, J.indices, 1, i, col, data)
        
        JJ = J.tocsc(copy=True)
        sfact = factorized(JJ)

    if uold is not None:
        # Integrate 1st order sensitivity equations
        first_sensitivity(psys, z, sfact, uold, theta, h)

    if vold is not None and SECONDORDER:
        # Integrate 2nd order sensitivity equations
        residual_hessian(Hess, z, theta, psys)
        second_sensitivity(psys, z, uold, sfact, Hess, vold, theta, h)
        mixed_sensitivity(psys, z, uold, sfact, Hess, mold, theta, h)
    else:
        v = None
        m = None

    return z, uold, vold, mold


def initialize_system(psys: Psystem, pf_solution: PowerFlowSolution):
    """ Based on system parameters and power flow solution, creates the
        initialized system vector and theta.
    """

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    sysvec = np.zeros(sys_size, dtype=np.float64)
    x = np.zeros(dif_size, dtype=np.float64)  #differential part
    y = np.zeros(alg_size, dtype=np.float64)  #algebraic part

    psys.initialize()

    assert psys.init_flag == True

    v = np.zeros(2*psys.nbuses, dtype=np.float64)
    for i in range(psys.nbuses):
        v[2*i] = pf_solution.v_magnitudes[i]
        v[2*i + 1] = pf_solution.v_angles[i]

    for device in psys.devices:
        vm = pf_solution.v_magnitudes[device.bus]
        va = pf_solution.v_angles[device.bus]

        if device.model_type  == "generator":
            # retrieve static gen id
            gen_static_id = device.static_gen_idx
            pi = pf_solution.gen_psch[gen_static_id]
            qi = pf_solution.gen_qsch[gen_static_id]

        elif device.model_type == "ZIPLoad":
            pi = -device.pload
            qi = device.qload
        elif device.model_type in ["governor", "exciter"]:
            # here we dont need pi and qi we just need to pass something
            # because we have the same signature for all the initialization
            pi = 0.0
            qi = 0.0
        else:
            #unknown device
            raise NameError("Unknown device type: %s" % device.model_type)

        device.initialize(vm, va, pi, qi, x, y, psys)

    sysvec[:dif_size] = x
    sysvec[dif_size:dif_size + alg_size] = y
    
    if psys.power_injection:
        sysvec[dif_size + alg_size:] = v
    else:
        for i in range(psys.nbuses):
            vm = v[2*i]
            va = v[2*i + 1]
            sysvec[dif_size + alg_size + 2*i] = vm*np.cos(va)
            sysvec[dif_size + alg_size + 2*i + 1] = vm*np.sin(va)

        # In addition, we will need the realified admittance matrix
        # Perhaps not the best place to put this as it might result
        # in extra overhead when doing MC sampling.
        psys.ybus_complex2real()

    # initialize theta
    theta = np.zeros(psys.num_pars)
    for i in range(psys.num_devices):
        psys.devices[i].initialize_theta(theta)

    return sysvec, theta


def initialize_sensitivities(volt, p_inj, psys, z, u, v):

    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations

    p_load = psys.get_loadvec()

    for device in psys.devices:
        vm = volt[2*device.bus]
        va = volt[2*device.bus + 1]

        if device.model_type == "generator":
            pi = p_inj[2*device.bus] - p_load[2*device.bus]
            qi = p_inj[2*device.bus + 1] - p_load[2*device.bus + 1]
        else:
            pi = p_load[2*device.bus]
            qi = p_load[2*device.bus + 1]

        if device.model_type == "motor":
            device.initialize_sens(vm, va, pi, qi, z, u, v, psys,
                                            dif_size)

    return None


def preallocate_jacobian(psys):

    # checks
    assert psys.init_flag == True
    assert psys.assembled == 1

    # system sizes
    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    # list of lists
    list_coordinates = [[] for i in range(sys_size)]

    # DIAGONAL ENTRIES
    # (NOTE): I am preallocating diagonal entries in the differential equation part.
    # This is because BEULER will need those entries. However, I am dubius I should
    # mix the structure of the Jacobian matrix of the r.h.s and the Jacobian of the
    # BEULER problem. Performance-wise, I will go with mixing for now.

    for i in range(sys_size):
        list_coordinates[i].extend([i])

    # network equations
    ptr = alg_size + dif_size
    for fr_bus in range(len(psys.graph_list)):
        list_coordinates[ptr + 2*fr_bus].extend(
            [ptr + 2*fr_bus, ptr + 2*fr_bus + 1])
        list_coordinates[ptr + 2*fr_bus + 1].extend(
            [ptr + 2*fr_bus, ptr + 2*fr_bus + 1])

        for to_bus in psys.graph_list[fr_bus]:
            list_coordinates[ptr + 2*fr_bus].extend(
                [ptr + 2*to_bus, ptr + 2*to_bus + 1])
            list_coordinates[ptr + 2*fr_bus + 1].extend(
                [ptr + 2*to_bus, ptr + 2*to_bus + 1])

    # each device returns a list of cordinates. we add this isto a list of lists.
    for i in range(psys.num_devices):
        idxs = np.array([
            psys.devices[i].dif_ptr, dif_size + psys.devices[i].alg_ptr,
            alg_size + dif_size
        ],
                        dtype=np.int32)
        coord = psys.devices[i].preallocate_jacobian(idxs, psys, psys.power_injection)

        for j in range(len(coord)):
            if not list_coordinates[coord[j][0]]:
                list_coordinates[coord[j][0]] = coord[j][1]
            else:
                list_coordinates[coord[j][0]].extend(coord[j][1])
                list_coordinates[coord[j][0]] = sorted(
                    set(list_coordinates[coord[j][0]]))

    # because the ZIP load depends only on the bus voltage, no need to
    # re-compute this.

    # form coordinate lists (row, col, data)
    row = []
    col = []

    for i in range(len(list_coordinates)):
        if list_coordinates[i]:
            row.extend([i for j in range(len(list_coordinates[i]))])
            col.extend(list_coordinates[i])
    data = np.zeros(len(row))

    Jsparse = csr_matrix((data, (row, col)), shape=(sys_size, sys_size))

    return Jsparse


def coord_to_sparse(rows, cols, sys_size):
    """
    This function returns a sparse matrix given arrays "rows" and "cols"
    that have the following structure:

    rows = [a, b, c]
    cols = [[a, b], [a, b], [c]]

    Thus, rows is a list that contains the indexes of those rows which
    have non-zero entries and cols a list of lists.
    """

    # we convert to coordinate format
    row_coor = []
    col_coor = []

    for i in range(len(rows)):
        row_coor.extend([rows[i] for j in range(len(cols[i]))])
        col_coor.extend(cols[i])
    data = np.zeros(len(row_coor))

    Jsparse = csr_matrix((data, (row_coor, col_coor)),
                         shape=(sys_size, sys_size))

    return Jsparse


def preallocate_hessian(psys):

    # checks
    assert psys.init_flag == True
    assert psys.assembled == 1

    # system sizes
    alg_size = psys.num_dof_alg
    dif_size = psys.num_dof_dif
    pow_size = 2*psys.nbuses  # power balance equations
    sys_size = alg_size + dif_size + 2*psys.nbuses

    # Hessian base structure
    Hsparse = sys_size*[None]

    # Indexing structure
    h_nnz = [{'rows': [], 'cols': []} for i in range(sys_size)]

    # NETWORK
    ptr = alg_size + dif_size

    for fr_bus, connect in enumerate(psys.graph_list):
        ncon = len(connect)
        # connected buses first
        rows = [0]*2*ncon
        cols = [[]]*2*ncon
        for i in range(ncon):
            rows[2*i] = ptr + 2*connect[i]
            rows[2*i + 1] = ptr + 2*connect[i] + 1
            cols[2*i] = [
                ptr + 2*connect[i], ptr + 2*connect[i] + 1, ptr + 2*fr_bus,
                ptr + 2*fr_bus + 1
            ]
            cols[2*i + 1] = [
                ptr + 2*connect[i], ptr + 2*connect[i] + 1, ptr + 2*fr_bus,
                ptr + 2*fr_bus + 1
            ]
        # add from_bus
        idx_tobus = rows.copy()
        idx_tobus.extend([ptr + 2*fr_bus, ptr + 2*fr_bus + 1])

        rows.extend([ptr + 2*fr_bus, ptr + 2*fr_bus + 1])
        cols.extend([idx_tobus.copy(), idx_tobus.copy()])

        # Both equations have same non-zero derivatives
        h_nnz[ptr + 2*fr_bus]['rows'] = rows
        h_nnz[ptr + 2*fr_bus]['cols'] = cols
        h_nnz[ptr + 2*fr_bus + 1]['rows'] = rows
        h_nnz[ptr + 2*fr_bus + 1]['cols'] = cols

    # DEVICES
    for i in range(psys.num_devices):
        idxs = np.array([
            psys.devices[i].dif_ptr, dif_size + psys.devices[i].alg_ptr,
            alg_size + dif_size
        ],
                        dtype=np.int32)
        psys.devices[i].preallocate_hessian(h_nnz, idxs, psys)

    # assemble sparse structures
    for i in range(sys_size):
        if len(h_nnz[i]['rows']) > 0:
            Hsparse[i] = coord_to_sparse(h_nnz[i]['rows'], h_nnz[i]['cols'],
                                         sys_size)

    return Hsparse

if petsc4py:
    class DAE_petsc(object):
        n = 1
        comm = PETSc.COMM_SELF
        def __init__(self, psys, theta, J, tfon, tfoff):
            self.psys = psys
            self.theta = theta
            self.J = J
            self.tfon = tfon
            self.tfoff = tfoff

            # ARKIMEX INFORMATION
            self.slow_indices = None
            self.fast_indices = None
            self.fast_indices_alg = None
            self.fast_indices_dif = None
            self.current_step = -1
            self.jfast_frozen = None # frozen Jacobian for fast system
            self.ff_arkimex = None

            # Pre-computed indices and structures for fast operations
            self._jfrozen_data_indices = None
            self._jfrozen_csr = None
            self._jfast_data_indices = None
            self._jfast_csr = None
            self._jfast_diag_indices_in_data = None
            self._indices_precomputed = False

        def set_ndiffeq_fast(self, ndiffeq_fast):
            self.ndiffeq_fast = ndiffeq_fast

        def set_fast_indices_split(self, fast_indices_alg, fast_indices_dif):
            self.fast_indices_alg = fast_indices_alg
            self.fast_indices_dif = fast_indices_dif

            if not self._indices_precomputed and self.J.nnz > 0:
                self._precompute_submatrix_indices()        

        def set_ts_ref(self, ts):
            self.ts_ref = ts
        
        def evalFunction(self, ts, t, x, xdot, f):
            # This operation is redundant but necessary for the correct
            # computation of the adjoint in the backward run. Might
            # not generalize when we consider multiple faults.
            # (TODO): consider using TSEvent.
            if t < self.tfon:
                self.psys.fault_events[0].remove()
            elif t > self.tfoff:
                self.psys.fault_events[0].remove()
            else:
                self.psys.fault_events[0].apply()

            start, end = x.getOwnershipRange()
            NDIFFEQ = self.psys.num_dof_dif
            xx = np.array(x[start:end])
            ff = np.zeros(xx.shape, dtype=np.float64)
            residual_function(ff, xx, self.theta, self.psys)
            f.setArray(-ff)
            f[:NDIFFEQ] += xdot[:NDIFFEQ]
            f.assemble()
        
        def evalJacobian(self, ts, t, x, xdot, a, J, P):
            if t < self.tfon:
                self.psys.fault_events[0].remove()
            elif t > self.tfoff:
                self.psys.fault_events[0].remove()
            else:
                self.psys.fault_events[0].apply()
            start, end = x.getOwnershipRange()
            NDIFFEQ = self.psys.num_dof_dif
            xx = np.array(x[start:end])
            residual_jacobian(self.J, xx, self.theta, self.psys)
            jacobian_implicit(self.J, NDIFFEQ, a)
            P.setValuesCSR(self.J.indptr, self.J.indices, self.J.data)
            P.assemble()
            if J != P: J.assemble()
            return True # same nonzero pattern
        
        def evalJacobianP(self, ts, t, x, xdot, a, P):
            start, end = x.getOwnershipRange()
            NDIFFEQ = self.psys.num_dof_dif
            xx = np.array(x[start:end])
            
            # Compute proper parameter Jacobian
            Jp_temp = preallocate_jacobian_parameters(self.psys)
            residual_jacobian_parameters(Jp_temp, xx, self.theta, self.psys)
            
            P.setValuesCSR(Jp_temp.indptr, Jp_temp.indices, -Jp_temp.data)
            P.assemble()
            return True

        #####################
        # ARKIMEX callbacks #
        #####################

        def _precompute_submatrix_indices(self):
            """
            Performs one-time computation of indices and CSR structures for submatrices.
            This avoids expensive sparse matrix slicing inside the time-stepper.
            """            
            # 1. Create a map where data contains its own index
            J_idx_map = self.J.copy()
            J_idx_map.data = np.arange(self.J.nnz, dtype=np.int32)

            # 2. Pre-compute for jfast_frozen = J[fast_indices_dif, fast_indices_dif]
            jfrozen_map = J_idx_map[self.fast_indices_dif][:, self.fast_indices_dif]
            self._jfrozen_data_indices = jfrozen_map.data.copy()
            self._jfrozen_csr = (jfrozen_map.indices.copy(), jfrozen_map.indptr.copy(), jfrozen_map.shape)

            # 3. Pre-compute for J_temp = J[fast_indices, fast_indices]
            jfast_map = J_idx_map[self.fast_indices][:, self.fast_indices]
            self._jfast_data_indices = jfast_map.data.copy()
            self._jfast_csr = (jfast_map.indices.copy(), jfast_map.indptr.copy(), jfast_map.shape)
            
            # Pre-allocate the temporary Jacobian for fast subsystem
            self._jfast_temp_data = np.zeros(len(self._jfast_data_indices), dtype=np.float64)

            # 4. Find the locations of the diagonal entries in J_temp's data array
            diag_indices = []
            for i in range(self.ndiffeq_fast):
                start, end = jfast_map.indptr[i], jfast_map.indptr[i+1]
                cols_in_row = jfast_map.indices[start:end]
                
                if i in cols_in_row:
                    loc_in_row = np.where(cols_in_row == i)[0][0]
                    diag_indices.append(start + loc_in_row)
            
            self._jfast_diag_indices_in_data = np.array(diag_indices, dtype=np.int32)
            
            # 5. NEW: Precompute indices for the differential-differential block
            # This is the [:ndiffeq_fast, :ndiffeq_fast] submatrix of jfast
            diff_diff_indices = []
            off_diag_indices = []  # For the [:ndiffeq_fast, ndiffeq_fast:] block
            
            for i in range(len(self.fast_indices)):
                start, end = jfast_map.indptr[i], jfast_map.indptr[i+1]
                cols_in_row = jfast_map.indices[start:end]
                
                if i < self.ndiffeq_fast:
                    # This is a differential equation row
                    for idx, col in enumerate(cols_in_row):
                        data_idx = start + idx
                        if col < self.ndiffeq_fast:
                            # This entry is in the diff-diff block
                            diff_diff_indices.append(data_idx)
                        else:
                            # This entry is in the off-diagonal block (diff-alg)
                            off_diag_indices.append(data_idx)
            
            self._jfast_diff_diff_indices = np.array(diff_diff_indices, dtype=np.int32)
            self._jfast_off_diag_indices = np.array(off_diag_indices, dtype=np.int32)
            
            # 6. Create mapping from jfrozen data indices to jfast data indices
            # Since both come from the same original matrix, we need to map between them
            jfrozen_indices, jfrozen_indptr, jfrozen_shape = self._jfrozen_csr
            jfast_indices, jfast_indptr, jfast_shape = self._jfast_csr
            
            jfrozen_to_jfast_map = []
            for i in range(self.ndiffeq_fast):
                # Find entries in jfrozen row i
                jfrozen_start, jfrozen_end = jfrozen_indptr[i], jfrozen_indptr[i+1]
                jfrozen_cols = jfrozen_indices[jfrozen_start:jfrozen_end]
                
                # Find corresponding entries in jfast row i  
                jfast_start, jfast_end = jfast_indptr[i], jfast_indptr[i+1]
                jfast_cols = jfast_indices[jfast_start:jfast_end]
                
                for idx, col in enumerate(jfrozen_cols):
                    if col < self.ndiffeq_fast:  # Only map entries in the diff-diff block
                        jfast_idx = np.where(jfast_cols == col)[0]
                        if len(jfast_idx) > 0:
                            jfrozen_to_jfast_map.append((jfrozen_start + idx, jfast_start + jfast_idx[0]))
            
            self._jfrozen_to_jfast_data_map = np.array(jfrozen_to_jfast_map, dtype=np.int32)
            
            self._indices_precomputed = True

        def evalRHSFunctionSlow(self, ts, t, x, f):
            start, end = x.getOwnershipRange()
            xx = np.array(x[start:end])
            ff = self.ff_arkimex
            f.setArray(ff[self.slow_indices])
            f.assemble()
        
        def evalIFunctionFast_split(self, ts, t, x, xdot, f):
            tstep = self.ts_ref.getStepNumber()
            ndiffeq_fast = self.ndiffeq_fast
            start, end = x.getOwnershipRange()
            xx = np.array(x[start:end])
            ff = np.zeros_like(xx)

            # We compute Jacobian once at the beginning of the time step
            if tstep != self.current_step:
                residual_jacobian(self.J, xx, self.theta, self.psys)
                # Extract data using pre-computed indices
                jfrozen_data = self.J.data[self._jfrozen_data_indices]
                # Reconstruct the sparse matrix from pre-computed CSR structure
                indices, indptr, shape = self._jfrozen_csr
                self.jfast_frozen = csr_matrix((jfrozen_data, indices, indptr), shape=shape)
                self.current_step = tstep

            residual_function(ff, xx, self.theta, self.psys)

            # Store residual for evalRHS
            self.ff_arkimex = ff

            f.setArray(-ff[self.fast_indices])
            f[:ndiffeq_fast] = np.zeros(len(self.fast_indices_dif))
            f[:ndiffeq_fast] += xdot[:ndiffeq_fast]
            f[:ndiffeq_fast] -= self.jfast_frozen.dot(xx[self.fast_indices_dif])
            f.assemble()

        def evalRHSFunctionFast_split(self, ts, t, x, f):
            start, end = x.getOwnershipRange()
            ndiffeq_fast = self.ndiffeq_fast
            xx = np.array(x[start:end])
            
            ff = self.ff_arkimex
            f.setArray(ff[self.fast_indices])
            f[ndiffeq_fast:] = np.zeros(len(self.fast_indices_alg))
            f[:ndiffeq_fast] -= self.jfast_frozen.dot(xx[self.fast_indices_dif])
            f.assemble()

        def evalIJacobianFast_split(self, ts, t, x, xdot, a, Jfast, Pfast):
            start, end = x.getOwnershipRange()
            xx = np.array(x[start:end])
            ndiffeq_fast = self.ndiffeq_fast
            
            # Compute full Jacobian
            residual_jacobian(self.J, xx, self.theta, self.psys)
            
            # Extract data for fast subsystem directly using precomputed indices
            # This replaces: J_temp = self.J[self.fast_indices][:, self.fast_indices]
            self._jfast_temp_data[:] = self.J.data[self._jfast_data_indices]
            
            # Zero out the off-diagonal block (differential-algebraic coupling)
            self._jfast_temp_data[self._jfast_off_diag_indices] = 0.0
            
            # Copy the frozen Jacobian values to the differential-differential block
            # This replaces: J_temp[:ndiffeq_fast, :ndiffeq_fast] = self.jfast_frozen[:, :]
            if self._jfrozen_to_jfast_data_map.size > 0:
                jfrozen_data_vals = self.jfast_frozen.data[self._jfrozen_to_jfast_data_map[:, 0]]
                self._jfast_temp_data[self._jfrozen_to_jfast_data_map[:, 1]] = jfrozen_data_vals
            
            # Add -a*I to the diagonal of the differential part
            self._jfast_temp_data[self._jfast_diag_indices_in_data] -= a
            
            # Set the values in the PETSc matrix using precomputed CSR structure
            indices, indptr, shape = self._jfast_csr
            Pfast.setValuesCSR(indptr, indices, -self._jfast_temp_data)
            Pfast.assemble()
            if Jfast != Pfast: 
                Jfast.assemble()
            return True

    class ALG_petsc(object):
        n = 1
        comm = PETSc.COMM_SELF
        def __init__(self, psys, theta, J):
            self.psys = psys
            self.theta = theta
            self.J = J
        
        def evalFunction(self, snes, x, f):
            start, end = x.getOwnershipRange()
            NDIFFEQ = self.psys.num_dof_dif
            xx = np.array(x[start:end])
            ff = np.zeros_like(xx)
            residual_function(ff, xx, self.theta, self.psys)
            ff[:NDIFFEQ] = 0.0
            f.setArray(-ff)
            f.assemble()

        def evalJacobian(self, snes, x, J, P):
            start, end = x.getOwnershipRange()
            NDIFFEQ = self.psys.num_dof_dif
            xx = np.array([x[i] for i in range(start, end)])
            residual_jacobian(self.J, xx, self.theta, self.psys)
            # The following has the effect of setting the differential part of the
            # jacobian to 0 and adding 1 to the diagonal hence keeping the differential
            # part constant (projection to manifold)
            jacobian_beuler(self.J, NDIFFEQ, 0.0)
            P.setValuesCSR(self.J.indptr, self.J.indices, self.J.data)
            P.assemble()

            if J != P: J.assemble()
            return True

    class ADJ_petsc(object):
        n = 1
        comm = PETSc.COMM_SELF
        def __init__(self, psys, theta):
            self.psys = psys
            self.theta = theta
        
        def evalCostIntegrand(self, ts, t, x, r):
            """Cost: integral of generator speed deviations squared"""
            speed_indices = self.psys.genspeed_idx_set()
            cost = 0.0
            for idx in speed_indices:
                cost += x[idx] * x[idx]
            r[0] = cost
            r.assemble()
        
        def evalJacobian(self, ts, t, x, A, B):
            """Gradient of cost w.r.t state: ∂g/∂x = 2*ω"""
            speed_indices = self.psys.genspeed_idx_set()
            A.zeroEntries()
            for idx in speed_indices:
                A[idx, 0] = 2.0 * x[idx]
            A.assemble()
            return True
        
        def evalJacobianP(self, ts, t, x, A):
            """Gradient of cost w.r.t parameters: ∂g/∂p = 0"""
            A.zeroEntries()
            A.assemble()
            return True

## Small-signal analysis
def compute_equilibrium(psys, power_injection=True):
    psys.power_injection=power_injection
    volt, Pinj = runpf(psys, verbose=False)
    z0, theta = initialize_system(volt, Pinj, psys)
    return z0, theta

def compute_rhs_jacobian(psys, z, theta, power_injection=True):
    J = preallocate_jacobian(psys)
    residual_jacobian(J, z, theta, psys)
    return J

def _eval_adjoint_z0_product(params_flat, psys_template, lambda_adjoint):
    """Evaluate λ^T * z0(p) for given load parameters."""
    psys = copy.deepcopy(psys_template)
    
    # Convert flat params to P, Q arrays
    p_loads = params_flat[::2]  # Even indices: P values
    q_loads = params_flat[1::2]  # Odd indices: Q values
    
    # Use existing psys method
    psys.set_load_pq(p_loads, q_loads)
    
    # Solve and initialize (using functions from same module)
    pf_sol = runpf(psys, verbose=False)
    z0, _ = initialize_system(psys, pf_sol)
    
    return np.dot(lambda_adjoint, z0)


def compute_initial_state_sensitivity(psys, lambda_adjoint, nominal_params, eps=1e-7):
    """
    Compute sensitivity A^T * λ where A = ∂z0/∂p_loads using centered differences.
    
    Args:
        psys: Power system object (template, will be copied)
        lambda_adjoint: Adjoint vector λ from DAE solution
        nominal_params: Flat array [P0, Q0, P1, Q1, ...] of nominal load parameters
        eps: Finite difference step size
        
    Returns:
        numpy.ndarray: Sensitivity vector A^T * λ
    """
    n_params = len(nominal_params)
    sensitivity = np.zeros(n_params)

    # Centered differences: (f(x+h) - f(x-h)) / (2h)
    for j in range(n_params):
        # Forward perturbation
        params_plus = nominal_params.copy()
        params_plus[j] += eps
        f_plus = _eval_adjoint_z0_product(params_plus, psys, lambda_adjoint)
        
        # Backward perturbation  
        params_minus = nominal_params.copy()
        params_minus[j] -= eps
        f_minus = _eval_adjoint_z0_product(params_minus, psys, lambda_adjoint)
        
        # Centered difference
        sensitivity[j] = (f_plus - f_minus) / (2 * eps)
    
    return sensitivity

def generate_default_partition_indices(psys, slow_diff_indices=None, fast_diff_indices=None):
    """Generate fast/slow index sets for ARKIMEX.

    Args:
        psys: Power system object with sizing information.
        slow_diff_indices: Optional iterable with the global indexes of
            differential equations that must belong to the slow subsystem.
        fast_diff_indices: Optional iterable with the global indexes of
            differential equations that must belong to the fast subsystem.

    Returns:
        Tuple containing the index lists expected by the ARKIMEX callbacks.
    """

    dif_size = psys.num_dof_dif
    alg_size = psys.num_dof_alg
    pow_size = 2 * psys.nbuses

    all_diff = list(range(dif_size))

    if slow_diff_indices is not None and fast_diff_indices is not None:
        raise ValueError("Specify only one of slow or fast differential index sets.")

    def _validate_diff_indices(indices, label):
        if indices is None:
            return None
        processed = [int(idx) for idx in indices]
        if len(set(processed)) != len(processed):
            raise ValueError(f"Duplicate entries detected in {label} indices.")
        for idx in processed:
            if idx < 0 or idx >= dif_size:
                raise ValueError(
                    f"{label.capitalize()} index {idx} is outside the valid range [0, {dif_size - 1}]."
                )
        return sorted(processed)

    slow_validated = _validate_diff_indices(slow_diff_indices, "slow differential")
    fast_validated = _validate_diff_indices(fast_diff_indices, "fast differential")

    if fast_validated is not None:
        fast_diff = fast_validated
        slow_indices = [idx for idx in all_diff if idx not in fast_diff]
        fast_diff_indices = fast_diff
    elif slow_validated is not None:
        slow_indices = slow_validated
        fast_diff_indices = [idx for idx in all_diff if idx not in slow_indices]
    else:
        midpoint = dif_size // 2
        slow_indices = list(range(midpoint))
        fast_diff_indices = list(range(midpoint, dif_size))

    fast_indices_alg = list(range(dif_size, dif_size + alg_size + pow_size))
    fast_indices = fast_diff_indices + fast_indices_alg
    fast_indices_dif = fast_diff_indices
    ndiff_fast = len(fast_diff_indices)

    return slow_indices, fast_indices, fast_indices_alg, fast_indices_dif, ndiff_fast

def integrate_system(
    psys: Psystem, config: IntegrationConfig, ctx: Optional[IntegrationCtx] = None
) -> dict:
    """Integrate power system dynamics

    Args:
        psys (Psystem): Power system object.
        config (IntegrationConfig): Configuration parameters.

    Returns:
        dict: Integration results.
    """
    tend = config.tend
    dt = config.dt
    steps = config.steps
    verbose = config.verbose
    comp_sens = config.comp_sens
    fsolve = config.fsolve
    ton = config.ton
    toff = config.toff
    petsc = config.petsc
    check_jacobian = config.check_jacobian
    jacobian_check_tol = config.jacobian_check_tol
    jacobian_check_top_k = config.jacobian_check_top_k
    jacobian_check_csv = config.jacobian_check_csv
    power_injection = config.power_injection
    solve_power_flow = config.solve_powerflow_dynamics
    arkimex = config.arkimex

    # check for arkimex enabled
    if arkimex and petsc:
        print("ARKIMEX activated.")

    results = {}
    psys.power_injection=power_injection

    # retrieve parameters
    pf_solution = runpf(psys, verbose=False)
    z0, theta = initialize_system(psys, pf_solution)

    # Use context if provided, otherwise fallback to config attributes
    z0_user = ctx.z0_user if ctx is not None else getattr(config, 'z0_user', None)
    theta_user = ctx.theta_user if ctx is not None else getattr(config, 'theta_user', None)

    if z0_user is not None:
        if z0_user.shape[0] != z0.shape[0]:
            raise ValueError("Provided initial state does not match system size.")
        z0 = z0_user

    if theta_user is not None:
        if theta_user.shape[0] != theta.shape[0]:
            raise ValueError("Provided theta does not match system parameters.")
        theta = theta_user

    system_size = z0.shape[0]
    jacobian = preallocate_jacobian(psys)
    residual = np.zeros(system_size)

    if check_jacobian:
        if petsc:
            raise ValueError("Jacobian check is only supported without PETSc.")
        from uqgrid.simulation.jacobian import residual_jacobian
        from uqgrid.simulation.jacobian_check import compare_jacobians
        residual_jacobian(jacobian, z0, theta, psys)
        mismatches = compare_jacobians(
            psys,
            z0,
            theta,
            jacobian,
            eps=1e-6,
            top_k=jacobian_check_top_k,
            tol=jacobian_check_tol,
        )
        print("== Jacobian FD check (top mismatches) ==")
        for m in mismatches:
            print(
                f"{m['row_desc']} <- {m['col_desc']}: "
                f"analytical={m['analytical']:.3e}, fd={m['finite_diff']:.3e}, "
                f"|diff|={m['abs_diff']:.3e}"
            )
        if jacobian_check_csv:
            import csv
            with open(jacobian_check_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "row",
                        "col",
                        "row_desc",
                        "col_desc",
                        "analytical",
                        "finite_diff",
                        "abs_diff",
                    ],
                )
                writer.writeheader()
                for m in mismatches:
                    writer.writerow(m)

    # calculate nsteps
    h = dt
    if steps > 0:
        nsteps = steps
    else:
        nsteps = int(math.floor(tend/dt)) + 1

    # hacky fault event time step calculation
    step_on = int(ton/h)
    step_off = int(toff/h)

    # Integration of D.A.E
    z = z0

    # Sensitivity parameters
    nparam = psys.nloads # For now, we only suport sensitivities of loads

    if comp_sens and not petsc:
        raise ValueError("Sensitivities can only be computed with PETSc.")

    if petsc4py and petsc:
        if verbose: print("Convert objects to PETSc format")
        nsize = jacobian.shape[0]
        jac_rhs = PETSc.Mat()
        jac_rhs.create(PETSc.COMM_WORLD)
        jac_rhs.setSizes([nsize, nsize])
        jac_rhs.setType('seqaij') # sparse
        csr = [jacobian.indptr, jacobian.indices, jacobian.data]
        jac_rhs.setPreallocationCSR(csr)
        jac_rhs.assemblyBegin()
        jac_rhs.assemblyEnd()

        nparam = 2 * psys.nloads
        jac_par_struct = preallocate_jacobian_parameters(psys)
        jac_par = PETSc.Mat()
        jac_par.create(PETSc.COMM_WORLD)
        jac_par.setSizes([nsize, nparam])
        jac_par.setType('seqaij')
        csr = [jac_par_struct.indptr, jac_par_struct.indices, jac_par_struct.data]
        jac_par.setPreallocationCSR(csr)
        jac_par.assemblyBegin()
        jac_par.assemblyEnd()

        z0p = PETSc.Vec()
        z0p.createSeq(nsize)
        z0p.setArray(z0)
        z0p.assemblyBegin()
        z0p.assemblyEnd()

        rhs_vec = z0p.duplicate()

        # Create integration object
        dae = DAE_petsc(psys, theta, jacobian, ton, toff)

        ts = PETSc.TS().create(comm=PETSc.COMM_WORLD)
        ts.setProblemType(ts.ProblemType.NONLINEAR)

        if arkimex:
            slow_indices, fast_indices, fast_indices_alg, fast_indices_dif, ndiff_fast = generate_default_partition_indices(
                psys,
                slow_diff_indices=config.arkimex_slow_differential,
                fast_diff_indices=config.arkimex_fast_differential,
            )
            
            # Set the optional fields in the DAE object
            dae.slow_indices = slow_indices
            dae.fast_indices = fast_indices
            dae.set_ndiffeq_fast(ndiff_fast)
            dae.set_fast_indices_split(fast_indices_alg, fast_indices_dif)
            dae.set_ts_ref(ts)

            # Provide stable default PETSc options for ARKIMEX unless the user overrides them.
            opts = PETSc.Options()

            if not opts.hasName("ts_adapt_type"):
                opts.setValue("ts_adapt_type", "none")

            if not opts.hasName("ts_arkimex_type"):
                opts.setValue("ts_arkimex_type", "a2")

            # Preallocate the Jacobian for the fast variables
            nfast = len(fast_indices)
            jac_fast = PETSc.Mat()
            jac_fast.create(PETSc.COMM_WORLD)
            jac_fast.setSizes([nfast, nfast])
            jac_fast.setType('seqaij')

            # Extract the fast part of the rhs Jacobian
            j_fast_pattern = jacobian[fast_indices][:, fast_indices]
            jac_fast.setPreallocationCSR([j_fast_pattern.indptr, j_fast_pattern.indices, j_fast_pattern.data])
            jac_fast.assemblyBegin()
            jac_fast.assemblyEnd()

            iss = PETSc.IS().createGeneral(slow_indices, comm=PETSc.COMM_WORLD)
            isf = PETSc.IS().createGeneral(fast_indices, comm=PETSc.COMM_WORLD)
            ts.setType(ts.Type.ARKIMEX)
            ts.setARKIMEXFastSlowSplit(True)
            ts.setRHSSplitIS("slow", iss)
            ts.setRHSSplitIS("fast", isf)
            ts.setRHSSplitRHSFunction("slow", dae.evalRHSFunctionSlow, None)
            ts.setRHSSplitIFunction("fast", dae.evalIFunctionFast_split, None)
            ts.setRHSSplitRHSFunction("fast", dae.evalRHSFunctionFast_split, None)
            ts.setRHSSplitIJacobian("fast", dae.evalIJacobianFast_split, jac_fast, jac_fast)
        else:
            ts.setType(ts.Type.THETA)
            ts.setIFunction(dae.evalFunction, rhs_vec)
            ts.setIJacobian(dae.evalJacobian, jac_rhs)
            ts.setIJacobianP(dae.evalJacobianP, jac_par)

        # create adjoint integrator
        if comp_sens:
            DRDX = PETSc.Mat().createDense([nsize, 1], comm=PETSc.COMM_WORLD)
            DRDX.setUp()
            DRDP = PETSc.Mat().createDense([nparam,1], comm=PETSc.COMM_WORLD)
            DRDP.setUp()
            quad = ADJ_petsc(psys, theta)
            quadts = ts.createQuadratureTS(forward=False)
            quadts.setRHSFunction(quad.evalCostIntegrand)
            quadts.setRHSJacobian(quad.evalJacobian, DRDX)
            quadts.setRHSJacobianP(quad.evalJacobianP, DRDP)
            v_lambda = z0p.duplicate()
            v_mu = PETSc.Vec()
            v_mu.createSeq(nparam)
            v_mu.assemblyBegin()
            v_mu.assemblyEnd()
            ts.setCostGradients(v_lambda, v_mu)
            ts.setSaveTrajectory()

        historyp = []
        tvecp = []
        def monitor(ts, i, t, x):
            xx = x[:].tolist()
            historyp.append(xx)
            tvecp.append(t)
        ts.setMonitor(monitor)
        ts.setTime(0.0)
        ts.setTimeStep(dt)
        ts.setMaxTime(ton)
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)
        ts.setFromOptions()
        ts.solve(z0p)
        
        if ton < tend:
            # fault application
            psys.fault_events[0].apply()
            alg = ALG_petsc(psys, theta, jacobian)
            fsp = z0p.duplicate()
            snes = PETSc.SNES()
            snes.create(PETSc.COMM_WORLD)
            snes.setFunction(alg.evalFunction, fsp)
            snes.setJacobian(alg.evalJacobian, jac_rhs)
            snes.setOptionsPrefix("alg_")
            snes.setFromOptions()
            snes.solve(None, z0p)

            # disturbance time
            ts.setTime(ton)
            ts.setMaxTime(toff)
            ts.solve(z0p)

            # fault removal
            psys.fault_events[0].remove()
            snes.solve(None, z0p)

            # post disturbance time
            ts.setTime(toff)
            ts.setMaxTime(tend)
            ts.solve(z0p)

        # adjoint computation
        if comp_sens:
            if verbose: print("Solving adjoint problem...")
            ts.adjointSolve()

            # extract results
            cst = ts.getCostIntegral()
            
            # Get the trajectory contribution
            cost_value = ts.getCostIntegral()
            mu_trajectory = np.array(v_mu[:])      # μᵢ term
            lambda_final = np.array(v_lambda[:])   # λᵢ term
            
            # Compute initial condition sensitivity ∂y₀/∂p
            p_loads, q_loads = psys.get_load_pq()
            nominal_params = np.zeros(2 * psys.nloads)
            nominal_params[::2] = p_loads   # P values at even indices
            nominal_params[1::2] = q_loads  # Q values at odd indices
            dy0_dp = compute_initial_state_sensitivity(
            psys, lambda_final, nominal_params
            )
            
            # Complete gradient = μᵢ + λᵢ(∂y₀/∂p)
            complete_gradient = mu_trajectory + dy0_dp
            
            results["adjoint_cost"] = float(cost_value[0])
            results["adjoint_gradient_trajectory"] = mu_trajectory  # Just μᵢ
            results["adjoint_gradient_initial"] = dy0_dp           # λᵢ(∂y₀/∂p)
            results["lambda_final"] = lambda_final
            results["adjoint_gradient_complete"] = complete_gradient # Total

        # Cast history to numpy arrays
        history = np.transpose(np.array(historyp))
        tvec = np.array(tvecp)

    else:
        tvec = np.linspace(0, nsteps*h, nsteps)
        history = np.zeros((system_size, nsteps))
        for i in range(nsteps):
            if verbose: print("Step: %i. Time: %g (sec)" % (i, i*h))
            z, u, v, m = integrate(z,
                                theta,
                                h,
                                psys,
                                residual,
                                jacobian,
                                None,
                                verbose=verbose,
                                fsolve=fsolve,
                                uold=None,
                                vold=None,
                                mold=None)
            history[:, i] = np.copy(z)

            if i == step_on:
                if verbose: print("Apply fault")
                if len(psys.fault_events) > 0:
                    psys.fault_events[0].apply()
                z, _, _, _ = integrate(z,
                                    theta,
                                    0.0,
                                    psys,
                                    residual,
                                    jacobian,
                                    None,
                                    verbose=verbose,
                                    fsolve=True,
                                    uold=None,
                                    vold=None,
                                    mold=None)
            if i == step_off:
                if verbose: print("Remove fault")
                if len(psys.fault_events) > 0:
                    psys.fault_events[0].remove()
                z, _, _, _ = integrate(z,
                                    theta,
                                    0.0,
                                    psys,
                                    residual,
                                    jacobian,
                                    None,
                                    verbose=verbose,
                                    fsolve=True,
                                    uold=None,
                                    vold=None,
                                    mold=None)

        # if tend < toff we remove fault before exiting
        if i < step_off:
            psys.fault_events[0].remove()

    # pack results into dict
    results["tvec"] = tvec
    results["history"] = history

    return results
