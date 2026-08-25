from __future__ import print_function

import copy
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

# Optional PETSc bindings are initialized lazily from IntegrationConfig.petsc_args.
petsc4py = None
PETSc = None


def _initialize_petsc(petsc_args=None):
    global petsc4py, PETSc
    if PETSc is not None:
        return PETSc

    try:
        import petsc4py as _petsc4py
    except ImportError as exc:
        raise ImportError(
            "PETSc integration requires petsc4py. Install UQGrid with the "
            "'petsc' extra or set petsc=False."
        ) from exc

    _petsc4py.init(list(petsc_args or []))
    from petsc4py import PETSc as _PETSc

    petsc4py = _petsc4py
    PETSc = _PETSc
    return PETSc


def _get_petsc_for_config(config):
    if not config.petsc:
        return None
    return _initialize_petsc(config.petsc_args)


def _petsc_ts_type(PETSc, method, arkimex):
    if arkimex:
        return PETSc.TS.Type.ARKIMEX
    if method == "beuler":
        return PETSc.TS.Type.BEULER
    if method == "cn":
        return PETSc.TS.Type.CN
    raise ValueError(f"Unsupported PETSc integration method: {method}")


def _algebraic_projection_adjoint(
        psys, state, theta, jacobian, lambda_values, mu_values):
    """Apply the adjoint jump for a fixed-differential algebraic projection."""
    ndiff = psys.num_dof_dif
    residual_jacobian(jacobian, state, theta, psys)
    algebraic_jacobian = jacobian[ndiff:, ndiff:].tocsc()
    differential_jacobian = jacobian[ndiff:, :ndiff]
    multiplier = spsolve(
        algebraic_jacobian.transpose().tocsc(),
        lambda_values[ndiff:],
    )

    parameter_jacobian = preallocate_jacobian_parameters(psys)
    residual_jacobian_parameters(parameter_jacobian, state, theta, psys)

    projected_lambda = np.array(lambda_values, copy=True)
    projected_lambda[:ndiff] -= np.asarray(
        differential_jacobian.transpose().dot(multiplier)
    ).ravel()
    projected_lambda[ndiff:] = 0.0
    projected_mu = np.array(mu_values, copy=True)
    projected_mu -= np.asarray(
        parameter_jacobian[ndiff:, :].transpose().dot(multiplier)
    ).ravel()
    return projected_lambda, projected_mu

from uqgrid.simulation.config import IntegrationConfig, IntegrationCtx
from uqgrid.core import Psystem
from uqgrid.simulation.pflow import (
    PowerFlowSolution,
    PowerFlowValidationError,
    compute_pinj_alt,
    runpf,
    validate_power_flow_solution,
)
from uqgrid.simulation.gradients import gradient_p, gradient_xp, gradient_pp
from uqgrid.simulation.residual import residual_function
from uqgrid.simulation.herk import integrate_system_herk
from uqgrid.simulation.dynamic_limits import (
    DynamicLimitError,
    DynamicLimitMode,
    _contextualize_dynamic_limit_runtime_error,
    collect_limited_state_descriptors,
    initialize_dynamic_limit_modes,
    project_limited_derivatives,
    update_dynamic_limit_active_set,
    validate_initial_dynamic_limits,
)
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.timing import build_integration_schedule
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
                logger.debug("H[%d]=%s", eq, H[eq])
                assert False
            else:
                logger.debug("True")

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
        logger.info("Jacobian with respect to parameters test passed!")
    else:
        logger.warning("Jacobian with respect to parameters test failed!")
        logger.warning("Maximum difference: %g", np.max(np.abs(Jp_dense - Jp_fd)))
        
        # Optionally show where the differences are
        diff = np.abs(Jp_dense - Jp_fd)
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        logger.warning("Max difference at %s: %s vs %s", idx, Jp_dense[idx], Jp_fd[idx])
    
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


def _assemble_beuler_residual(F, z, zold, h, psys, theta):
    """Assemble the ordinary backward-Euler endpoint residual in place."""
    residual_function(F, z, theta, psys)
    ndiff = psys.num_dof_dif
    F[:ndiff] = z[:ndiff] - zold[:ndiff] - h * F[:ndiff]


def _assemble_cn_residual(
    F, z, zold, h, start_derivative, psys, theta
):
    """Assemble the ordinary Crank-Nicolson endpoint residual in place."""
    residual_function(F, z, theta, psys)
    ndiff = psys.num_dof_dif
    F[:ndiff] = (
        z[:ndiff]
        - zold[:ndiff]
        - 0.5 * h * (start_derivative[:ndiff] + F[:ndiff])
    )


def _effective_cn_start_derivative(
    zold,
    theta,
    psys,
    descriptors,
    modes,
    *,
    tolerance,
    time=None,
):
    """Return the raw start derivative with inherited outward modes blocked."""
    raw_derivative = np.empty_like(zold)
    residual_function(raw_derivative, zold, theta, psys)
    active_descriptors = [
        descriptor
        for descriptor in descriptors
        if DynamicLimitMode(modes[descriptor.state_index])
        != DynamicLimitMode.FREE
    ]
    effective, _ = project_limited_derivatives(
        zold,
        raw_derivative,
        active_descriptors,
        tolerance=tolerance,
        time=time,
        stage_or_endpoint="start",
    )
    return effective


def _active_limit_bound(descriptor, mode):
    mode = DynamicLimitMode(mode)
    if mode == DynamicLimitMode.UPPER_ACTIVE:
        return descriptor.upper_bound
    if mode == DynamicLimitMode.LOWER_ACTIVE:
        return descriptor.lower_bound
    return None


def _apply_beuler_active_rows(F, J, z, descriptors, modes):
    """Replace active BE equations with bound equations and identity rows."""
    diagonal_column = np.zeros(1, dtype=np.int64)
    diagonal_value = np.ones(1, dtype=np.float64)
    for descriptor in descriptors:
        bound = _active_limit_bound(
            descriptor, modes[descriptor.state_index]
        )
        if bound is None:
            continue
        state_index = descriptor.state_index
        F[state_index] = z[state_index] - bound
        if J is not None:
            csr_mult_row(J.data, J.indptr, J.indices, state_index, 0.0)
            diagonal_column[0] = state_index
            csr_set_row(
                J.data,
                J.indptr,
                J.indices,
                1,
                state_index,
                diagonal_column,
                diagonal_value,
            )


def _limit_mode_signature(descriptors, modes):
    return tuple(
        DynamicLimitMode(modes[item.state_index]).value
        for item in descriptors
    )


def _json_finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _dynamic_limit_runtime_error(
    reason,
    *,
    method,
    operation,
    message,
    backend,
    nonlinear_solver,
    endpoint_time,
    step_size,
    active_set_iterations,
    newton_iterations,
    residual_norm,
    descriptors,
    modes,
    visited_signatures,
    complementarity,
    events,
    solver_diagnostics=None,
):
    solver_diagnostics = dict(solver_diagnostics or {})
    solver_diagnostics.pop("failure_reason", None)
    return DynamicLimitError(
        {
            "enabled": True,
            "phase": "runtime",
            "method": method,
            "backend": backend,
            "nonlinear_solver": nonlinear_solver,
            "operation": operation,
            "message": message,
            "failure_reasons": [reason],
            "time": _json_finite(endpoint_time),
            "step_size": _json_finite(step_size),
            "active_set_iterations": int(active_set_iterations),
            "newton_iterations": int(newton_iterations),
            "residual_norm": _json_finite(residual_norm),
            "modes": [
                {
                    "state_index": item.state_index,
                    "mode": DynamicLimitMode(modes[item.state_index]).value,
                }
                for item in descriptors
            ],
            "visited_mode_signatures": [
                list(signature) for signature in visited_signatures
            ],
            "complementarity": complementarity,
            "events": list(events),
            **solver_diagnostics,
        }
    )


def _solve_beuler_fixed_active_set(
    zold,
    theta,
    h,
    psys,
    F,
    J,
    *,
    descriptors,
    modes,
    newton_tol,
    newton_max_iter,
    verbose=False,
):
    """Solve one smooth BE system for a fixed dynamic-limit active set."""
    z = np.array(zold, dtype=float, copy=True)
    for descriptor in descriptors:
        bound = _active_limit_bound(
            descriptor, modes[descriptor.state_index]
        )
        if bound is not None:
            z[descriptor.state_index] = bound

    _assemble_beuler_residual(F, z, zold, h, psys, theta)
    _apply_beuler_active_rows(F, None, z, descriptors, modes)
    residual_norm = np.linalg.norm(F)
    iteration = 0
    if verbose:
        logger.info("Iteration %d. Residual norm: %g", iteration, residual_norm)

    while residual_norm > newton_tol and iteration < newton_max_iter:
        iteration += 1
        residual_jacobian(J, z, theta, psys)
        jacobian_beuler(J, psys.num_dof_dif, h)
        _apply_beuler_active_rows(F, J, z, descriptors, modes)
        delta = spsolve(J, F)
        if not np.all(np.isfinite(delta)):
            return z, iteration, np.inf, False
        z = z - delta
        _assemble_beuler_residual(F, z, zold, h, psys, theta)
        _apply_beuler_active_rows(F, None, z, descriptors, modes)
        residual_norm = np.linalg.norm(F)
        if not np.isfinite(residual_norm):
            return z, iteration, residual_norm, False
        if verbose:
            logger.info(
                "Iteration %d. Residual norm: %g", iteration, residual_norm
            )

    return z, iteration, residual_norm, residual_norm <= newton_tol


def _integrate_implicit_active_set(
    zold,
    theta,
    h,
    psys,
    F,
    *,
    descriptors,
    modes,
    state_tolerance,
    release_tolerance,
    max_active_set_iterations,
    endpoint_time,
    fixed_set_solver,
    assemble_free_residual,
    method,
    operation,
    failure_message,
    backend,
    nonlinear_solver,
    prior_events=(),
):
    """Advance one implicit interval with a backend-supplied fixed-set solve."""
    descriptors = list(descriptors)
    modes = dict(modes)
    interval_events = []
    signature = _limit_mode_signature(descriptors, modes)
    visited_signatures = [signature]
    seen_signatures = {signature}
    complementarity = None
    residual_norm = np.inf
    newton_iterations = 0
    solver_diagnostics = {}

    for active_set_iteration in range(1, max_active_set_iterations + 1):
        solve_result = fixed_set_solver(modes)
        if len(solve_result) == 4:
            z, newton_iterations, residual_norm, converged = solve_result
            solver_diagnostics = {}
        else:
            (
                z,
                newton_iterations,
                residual_norm,
                converged,
                solver_diagnostics,
            ) = solve_result
        if not converged:
            raise _dynamic_limit_runtime_error(
                solver_diagnostics.get(
                    "failure_reason", "newton_nonconvergence"
                ),
                method=method,
                operation=operation,
                message=failure_message,
                backend=backend,
                nonlinear_solver=nonlinear_solver,
                endpoint_time=endpoint_time,
                step_size=h,
                active_set_iterations=active_set_iteration,
                newton_iterations=newton_iterations,
                residual_norm=residual_norm,
                descriptors=descriptors,
                modes=modes,
                visited_signatures=visited_signatures,
                complementarity=complementarity,
                events=[*prior_events, *interval_events],
                solver_diagnostics=solver_diagnostics,
            )

        free_residual = np.empty_like(F)
        assemble_free_residual(free_residual, z)
        try:
            updated_modes, changed, complementarity, transition_events = (
                update_dynamic_limit_active_set(
                    z,
                    free_residual,
                    descriptors,
                    modes,
                    state_tolerance=state_tolerance,
                    release_tolerance=release_tolerance,
                    time=endpoint_time,
                    stage_or_endpoint="endpoint",
                    active_set_iterations=active_set_iteration,
                )
            )
        except DynamicLimitError as exc:
            reasons = exc.diagnostics.get("failure_reasons", [])
            reason = reasons[0] if reasons else "active_set_evaluation_failed"
            raise _dynamic_limit_runtime_error(
                reason,
                method=method,
                operation=operation,
                message=failure_message,
                backend=backend,
                nonlinear_solver=nonlinear_solver,
                endpoint_time=endpoint_time,
                step_size=h,
                active_set_iterations=active_set_iteration,
                newton_iterations=newton_iterations,
                residual_norm=residual_norm,
                descriptors=descriptors,
                modes=modes,
                visited_signatures=visited_signatures,
                complementarity=exc.diagnostics,
                events=[*prior_events, *interval_events],
                solver_diagnostics=solver_diagnostics,
            ) from exc
        for event in transition_events:
            if event["action"] == "activate":
                event["state_after"] = event["bound"]
        interval_events.extend(transition_events)

        if not changed:
            if complementarity["consistent"]:
                return z, updated_modes, interval_events
            raise _dynamic_limit_runtime_error(
                "inconsistent_active_set",
                method=method,
                operation=operation,
                message=failure_message,
                backend=backend,
                nonlinear_solver=nonlinear_solver,
                endpoint_time=endpoint_time,
                step_size=h,
                active_set_iterations=active_set_iteration,
                newton_iterations=newton_iterations,
                residual_norm=residual_norm,
                descriptors=descriptors,
                modes=updated_modes,
                visited_signatures=visited_signatures,
                complementarity=complementarity,
                events=[*prior_events, *interval_events],
                solver_diagnostics=solver_diagnostics,
            )

        signature = _limit_mode_signature(descriptors, updated_modes)
        if signature in seen_signatures:
            raise _dynamic_limit_runtime_error(
                "active_set_cycle",
                method=method,
                operation=operation,
                message=failure_message,
                backend=backend,
                nonlinear_solver=nonlinear_solver,
                endpoint_time=endpoint_time,
                step_size=h,
                active_set_iterations=active_set_iteration,
                newton_iterations=newton_iterations,
                residual_norm=residual_norm,
                descriptors=descriptors,
                modes=updated_modes,
                visited_signatures=[*visited_signatures, signature],
                complementarity=complementarity,
                events=[*prior_events, *interval_events],
                solver_diagnostics=solver_diagnostics,
            )
        seen_signatures.add(signature)
        visited_signatures.append(signature)
        modes = updated_modes

    raise _dynamic_limit_runtime_error(
        "active_set_iteration_limit",
        method=method,
        operation=operation,
        message=failure_message,
        backend=backend,
        nonlinear_solver=nonlinear_solver,
        endpoint_time=endpoint_time,
        step_size=h,
        active_set_iterations=max_active_set_iterations,
        newton_iterations=newton_iterations,
        residual_norm=residual_norm,
        descriptors=descriptors,
        modes=modes,
        visited_signatures=visited_signatures,
        complementarity=complementarity,
        events=[*prior_events, *interval_events],
        solver_diagnostics=solver_diagnostics,
    )


def _integrate_beuler_with_dynamic_limits(
    zold,
    theta,
    h,
    psys,
    F,
    J,
    *,
    descriptors,
    modes,
    state_tolerance,
    release_tolerance,
    max_active_set_iterations,
    endpoint_time,
    newton_tol,
    newton_max_iter,
    prior_events=(),
    verbose=False,
):
    """Advance one native BE interval with a directional active set."""

    def solve_fixed_set(current_modes):
        return _solve_beuler_fixed_active_set(
            zold,
            theta,
            h,
            psys,
            F,
            J,
            descriptors=descriptors,
            modes=current_modes,
            newton_tol=newton_tol,
            newton_max_iter=newton_max_iter,
            verbose=verbose,
        )

    def assemble_free_residual(free_residual, endpoint):
        _assemble_beuler_residual(
            free_residual, endpoint, zold, h, psys, theta
        )

    return _integrate_implicit_active_set(
        zold,
        theta,
        h,
        psys,
        F,
        descriptors=descriptors,
        modes=modes,
        state_tolerance=state_tolerance,
        release_tolerance=release_tolerance,
        max_active_set_iterations=max_active_set_iterations,
        endpoint_time=endpoint_time,
        fixed_set_solver=solve_fixed_set,
        assemble_free_residual=assemble_free_residual,
        method="beuler",
        operation="beuler_active_set",
        failure_message="Backward-Euler dynamic-limit solve failed",
        backend="native",
        nonlinear_solver="scipy_sparse_newton",
        prior_events=prior_events,
    )


def _petsc_snes_reason_name(PETSc, reason):
    reason_value = int(reason)
    for name in dir(PETSc.SNES.ConvergedReason):
        if not name.isupper():
            continue
        if int(getattr(PETSc.SNES.ConvergedReason, name)) == reason_value:
            return name.lower()
    return f"unknown_{reason_value}"


class _PETScBEActiveSetProblem:
    """Explicit BE residual and Jacobian callbacks for one SNES workspace."""

    def __init__(self, psys, theta, residual, jacobian):
        self.psys = psys
        self.theta = theta
        self.residual = residual
        self.jacobian = jacobian
        self.zold = None
        self.h = None
        self.descriptors = ()
        self.modes = {}

    def set_interval(self, zold, h, descriptors, modes):
        self.zold = zold
        self.h = h
        self.descriptors = descriptors
        self.modes = modes

    def function(self, snes, x, f):
        z = np.array(x[:], dtype=float, copy=True)
        _assemble_beuler_residual(
            self.residual,
            z,
            self.zold,
            self.h,
            self.psys,
            self.theta,
        )
        _apply_beuler_active_rows(
            self.residual,
            None,
            z,
            self.descriptors,
            self.modes,
        )
        f.setArray(np.array(self.residual, copy=True))
        f.assemble()

    def jacobian_function(self, snes, x, J, P):
        z = np.array(x[:], dtype=float, copy=True)
        residual_jacobian(self.jacobian, z, self.theta, self.psys)
        jacobian_beuler(self.jacobian, self.psys.num_dof_dif, self.h)
        _apply_beuler_active_rows(
            self.residual,
            self.jacobian,
            z,
            self.descriptors,
            self.modes,
        )
        P.setValuesCSR(
            self.jacobian.indptr,
            self.jacobian.indices,
            self.jacobian.data,
        )
        P.assemble()
        if J != P:
            J.setValuesCSR(
                self.jacobian.indptr,
                self.jacobian.indices,
                self.jacobian.data,
            )
            J.assemble()
        return True


class _PETScCNActiveSetProblem:
    """Explicit CN residual and Jacobian callbacks for one SNES workspace."""

    def __init__(self, psys, theta, residual, jacobian):
        self.psys = psys
        self.theta = theta
        self.residual = residual
        self.jacobian = jacobian
        self.zold = None
        self.h = None
        self.start_derivative = None
        self.descriptors = ()
        self.modes = {}

    def set_interval(
        self, zold, h, descriptors, modes, *, start_derivative
    ):
        self.zold = zold
        self.h = h
        self.start_derivative = np.array(
            start_derivative, dtype=float, copy=True
        )
        self.start_derivative.setflags(write=False)
        self.descriptors = descriptors
        self.modes = modes

    def function(self, snes, x, f):
        z = np.array(x[:], dtype=float, copy=True)
        _assemble_cn_residual(
            self.residual,
            z,
            self.zold,
            self.h,
            self.start_derivative,
            self.psys,
            self.theta,
        )
        _apply_beuler_active_rows(
            self.residual,
            None,
            z,
            self.descriptors,
            self.modes,
        )
        f.setArray(np.array(self.residual, copy=True))
        f.assemble()

    def jacobian_function(self, snes, x, J, P):
        z = np.array(x[:], dtype=float, copy=True)
        residual_jacobian(self.jacobian, z, self.theta, self.psys)
        jacobian_beuler(
            self.jacobian, self.psys.num_dof_dif, 0.5 * self.h
        )
        _apply_beuler_active_rows(
            self.residual,
            self.jacobian,
            z,
            self.descriptors,
            self.modes,
        )
        P.setValuesCSR(
            self.jacobian.indptr,
            self.jacobian.indices,
            self.jacobian.data,
        )
        P.assemble()
        if J != P:
            J.setValuesCSR(
                self.jacobian.indptr,
                self.jacobian.indices,
                self.jacobian.data,
            )
            J.assemble()
        return True


_PETSC_VI_SNES_TYPES = {"vinewtonrsls", "vinewtonssls"}


class _PETScActiveSetWorkspace:
    """Reusable ordinary-SNES objects for one limited implicit method."""

    _problem_class = None

    def __init__(
        self,
        PETSc,
        psys,
        theta,
        residual,
        jacobian,
        *,
        newton_tol,
        newton_max_iter,
    ):
        self.PETSc = PETSc
        self.problem = self._problem_class(
            psys, theta, residual, jacobian
        )
        system_size = jacobian.shape[0]
        communicator = PETSc.COMM_SELF

        self.solution = PETSc.Vec().createSeq(
            system_size, comm=communicator
        )
        self.function_vector = self.solution.duplicate()

        self.matrix = PETSc.Mat()
        self.matrix.create(comm=communicator)
        self.matrix.setSizes([system_size, system_size])
        self.matrix.setType("seqaij")
        self.matrix.setPreallocationCSR(
            [jacobian.indptr, jacobian.indices, jacobian.data]
        )
        self.matrix.assemblyBegin()
        self.matrix.assemblyEnd()

        self.snes = PETSc.SNES().create(comm=communicator)
        self.snes.setType(PETSc.SNES.Type.NEWTONLS)
        self.snes.setFunction(self.problem.function, self.function_vector)
        self.snes.setJacobian(
            self.problem.jacobian_function, self.matrix, self.matrix
        )
        self.snes.setTolerances(
            rtol=0.0,
            atol=newton_tol,
            stol=0.0,
            max_it=newton_max_iter,
        )
        self.snes.setFromOptions()
        self.snes_type = str(self.snes.getType()).lower()
        try:
            _validate_petsc_snes_type(self.snes_type)
        except ValueError:
            self.destroy()
            raise

    def solve_fixed_active_set(
        self, zold, h, descriptors, modes, **interval_data
    ):
        guess = np.array(zold, dtype=float, copy=True)
        for descriptor in descriptors:
            bound = _active_limit_bound(
                descriptor, modes[descriptor.state_index]
            )
            if bound is not None:
                guess[descriptor.state_index] = bound

        self.problem.set_interval(
            zold, h, descriptors, modes, **interval_data
        )
        self.solution.setArray(guess)
        self.solution.assemble()
        try:
            self.snes.solve(None, self.solution)
            reason = int(self.snes.getConvergedReason())
            iterations = int(self.snes.getIterationNumber())
            residual_norm = float(self.snes.getFunctionNorm())
            endpoint = np.array(self.solution[:], dtype=float, copy=True)
            diagnostics = {
                "snes_type": self.snes_type,
                "snes_converged_reason": reason,
                "snes_converged_reason_name": _petsc_snes_reason_name(
                    self.PETSc, reason
                ),
                "snes_iterations": iterations,
                "snes_function_norm": _json_finite(residual_norm),
            }
            if hasattr(self.snes, "getLinearSolveIterations"):
                diagnostics["snes_linear_iterations"] = int(
                    self.snes.getLinearSolveIterations()
                )
            if reason <= 0:
                diagnostics["failure_reason"] = "snes_nonconvergence"
            return (
                endpoint,
                iterations,
                residual_norm,
                reason > 0,
                diagnostics,
            )
        except self.PETSc.Error as exc:
            return (
                guess,
                0,
                np.inf,
                False,
                {
                    "failure_reason": "snes_nonconvergence",
                    "snes_type": self.snes_type,
                    "snes_converged_reason": None,
                    "snes_converged_reason_name": "petsc_error",
                    "snes_iterations": 0,
                    "snes_function_norm": None,
                    "snes_error": str(exc),
                },
            )

    def destroy(self):
        for obj in (
            getattr(self, "snes", None),
            getattr(self, "matrix", None),
            getattr(self, "function_vector", None),
            getattr(self, "solution", None),
        ):
            if obj is not None:
                obj.destroy()


class _PETScBEActiveSetWorkspace(_PETScActiveSetWorkspace):
    """Reusable ordinary-SNES objects for limited PETSc backward Euler."""

    _problem_class = _PETScBEActiveSetProblem


class _PETScCNActiveSetWorkspace(_PETScActiveSetWorkspace):
    """Reusable ordinary-SNES objects for limited PETSc Crank-Nicolson."""

    _problem_class = _PETScCNActiveSetProblem


def _validate_petsc_snes_type(snes_type):
    snes_type = str(snes_type).lower()
    if snes_type in _PETSC_VI_SNES_TYPES:
        raise ValueError(
            "PETSc variational-inequality SNES types are unsupported for "
            "hard dynamic limits. Use ordinary SNES/Newton options; UQGrid "
            "manages the active set explicitly."
        )
    return snes_type


def _integrate_petsc_beuler_with_dynamic_limits(
    zold,
    theta,
    h,
    psys,
    F,
    J,
    *,
    workspace,
    descriptors,
    modes,
    state_tolerance,
    release_tolerance,
    max_active_set_iterations,
    endpoint_time,
    newton_tol,
    newton_max_iter,
    prior_events=(),
    verbose=False,
):
    """Advance one PETSc SNES BE interval with a directional active set."""

    def solve_fixed_set(current_modes):
        return workspace.solve_fixed_active_set(
            zold, h, descriptors, current_modes
        )

    def assemble_free_residual(free_residual, endpoint):
        _assemble_beuler_residual(
            free_residual, endpoint, zold, h, psys, theta
        )

    return _integrate_implicit_active_set(
        zold,
        theta,
        h,
        psys,
        F,
        descriptors=descriptors,
        modes=modes,
        state_tolerance=state_tolerance,
        release_tolerance=release_tolerance,
        max_active_set_iterations=max_active_set_iterations,
        endpoint_time=endpoint_time,
        fixed_set_solver=solve_fixed_set,
        assemble_free_residual=assemble_free_residual,
        method="beuler",
        operation="beuler_active_set",
        failure_message="Backward-Euler dynamic-limit solve failed",
        backend="petsc",
        nonlinear_solver="petsc_snes",
        prior_events=prior_events,
    )


def _integrate_petsc_cn_with_dynamic_limits(
    zold,
    theta,
    h,
    psys,
    F,
    J,
    *,
    workspace,
    descriptors,
    modes,
    state_tolerance,
    release_tolerance,
    max_active_set_iterations,
    endpoint_time,
    newton_tol,
    newton_max_iter,
    prior_events=(),
    verbose=False,
):
    """Advance one PETSc SNES CN interval with a directional active set."""
    start_time = endpoint_time - h
    try:
        start_derivative = _effective_cn_start_derivative(
            zold,
            theta,
            psys,
            descriptors,
            modes,
            tolerance=state_tolerance,
            time=start_time,
        )
    except DynamicLimitError as exc:
        raise _contextualize_dynamic_limit_runtime_error(
            exc,
            method="cn",
            backend="petsc",
            time=start_time,
            stage_or_endpoint="start",
            prior_events=prior_events,
        ) from exc
    start_derivative.setflags(write=False)

    def solve_fixed_set(current_modes):
        return workspace.solve_fixed_active_set(
            zold,
            h,
            descriptors,
            current_modes,
            start_derivative=start_derivative,
        )

    def assemble_free_residual(free_residual, endpoint):
        _assemble_cn_residual(
            free_residual,
            endpoint,
            zold,
            h,
            start_derivative,
            psys,
            theta,
        )

    return _integrate_implicit_active_set(
        zold,
        theta,
        h,
        psys,
        F,
        descriptors=descriptors,
        modes=modes,
        state_tolerance=state_tolerance,
        release_tolerance=release_tolerance,
        max_active_set_iterations=max_active_set_iterations,
        endpoint_time=endpoint_time,
        fixed_set_solver=solve_fixed_set,
        assemble_free_residual=assemble_free_residual,
        method="cn",
        operation="cn_active_set",
        failure_message="Crank-Nicolson dynamic-limit solve failed",
        backend="petsc",
        nonlinear_solver="petsc_snes",
        prior_events=prior_events,
    )


def _integrate_system_petsc_implicit_limits(
    PETSc,
    psys,
    config,
    theta,
    z0,
    tvec,
    schedule,
    fault,
    residual,
    jacobian,
    descriptors,
    dynamic_limit_diagnostics,
    *,
    workspace_class,
    interval_stepper,
    method,
):
    """Run one limited PETSc implicit method over the shared schedule."""
    workspace = workspace_class(
        PETSc,
        psys,
        theta,
        residual,
        jacobian,
        newton_tol=config.newton_tol,
        newton_max_iter=config.newton_max_iter,
    )
    modes = initialize_dynamic_limit_modes(descriptors)
    history = np.zeros((z0.size, len(tvec)))
    z = np.array(z0, dtype=float, copy=True)
    history[:, 0] = z

    try:
        for i in range(1, len(tvec)):
            h_step = tvec[i] - tvec[i - 1]
            if config.verbose:
                logger.info(
                    "PETSc SNES %s step: %i. Time: %g (sec)",
                    method,
                    i,
                    tvec[i],
                )
            z, modes, events = interval_stepper(
                z,
                theta,
                h_step,
                psys,
                residual,
                jacobian,
                workspace=workspace,
                descriptors=descriptors,
                modes=modes,
                state_tolerance=config.dynamic_limit_tolerance,
                release_tolerance=config.dynamic_limit_release_tolerance,
                max_active_set_iterations=config.max_dynamic_limit_iterations,
                endpoint_time=tvec[i],
                newton_tol=config.newton_tol,
                newton_max_iter=config.newton_max_iter,
                prior_events=dynamic_limit_diagnostics["events"],
                verbose=config.verbose,
            )
            dynamic_limit_diagnostics["events"].extend(events)

            if i == schedule.fault_on_index:
                if config.verbose:
                    logger.info("Apply fault")
                fault.apply()
                z, _, _, _ = integrate(
                    z,
                    theta,
                    0.0,
                    psys,
                    residual,
                    jacobian,
                    None,
                    verbose=config.verbose,
                    fsolve=False,
                    newton_tol=config.newton_tol,
                    newton_max_iter=config.newton_max_iter,
                    uold=None,
                    vold=None,
                    mold=None,
                )
            if i == schedule.fault_off_index:
                if config.verbose:
                    logger.info("Remove fault")
                fault.remove()
                z, _, _, _ = integrate(
                    z,
                    theta,
                    0.0,
                    psys,
                    residual,
                    jacobian,
                    None,
                    verbose=config.verbose,
                    fsolve=False,
                    newton_tol=config.newton_tol,
                    newton_max_iter=config.newton_max_iter,
                    uold=None,
                    vold=None,
                    mold=None,
                )
            history[:, i] = z
    finally:
        if fault is not None:
            fault.remove()
        workspace.destroy()

    return history


def _integrate_system_petsc_beuler_limits(
    PETSc,
    psys,
    config,
    theta,
    z0,
    tvec,
    schedule,
    fault,
    residual,
    jacobian,
    descriptors,
    dynamic_limit_diagnostics,
):
    """Run limited PETSc BE directly over the normalized shared schedule."""
    return _integrate_system_petsc_implicit_limits(
        PETSc,
        psys,
        config,
        theta,
        z0,
        tvec,
        schedule,
        fault,
        residual,
        jacobian,
        descriptors,
        dynamic_limit_diagnostics,
        workspace_class=_PETScBEActiveSetWorkspace,
        interval_stepper=_integrate_petsc_beuler_with_dynamic_limits,
        method="beuler",
    )


def _integrate_system_petsc_cn_limits(
    PETSc,
    psys,
    config,
    theta,
    z0,
    tvec,
    schedule,
    fault,
    residual,
    jacobian,
    descriptors,
    dynamic_limit_diagnostics,
):
    """Run limited PETSc CN directly over the normalized shared schedule."""
    return _integrate_system_petsc_implicit_limits(
        PETSc,
        psys,
        config,
        theta,
        z0,
        tvec,
        schedule,
        fault,
        residual,
        jacobian,
        descriptors,
        dynamic_limit_diagnostics,
        workspace_class=_PETScCNActiveSetWorkspace,
        interval_stepper=_integrate_petsc_cn_with_dynamic_limits,
        method="cn",
    )


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
              fsolve=False,
              newton_tol=1e-10,
              newton_max_iter=500):
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

    eps = newton_tol
    max_iter = newton_max_iter
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
            if verbose:
                logger.info("Fsolve converged.")
            z = sol
        else:
            raise NameError('Fsolve did not converge')

    else:

        if verbose:
            logger.info("Iteration %d. Residual norm: %g", iteration, norm_res)

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
                logger.info("Iteration %d. Residual norm: %g", iteration, norm_res)

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

    initialization_order = [
        device for device in psys.devices if device.model_type == "generator"
    ]
    initialization_order.extend(
        device for device in psys.devices if device.model_type != "generator"
    )
    for device in initialization_order:
        vm = pf_solution.v_magnitudes[device.bus]
        va = pf_solution.v_angles[device.bus]

        if device.model_type  == "generator":
            # retrieve static gen id
            gen_static_id = device.static_gen_idx
            pi = pf_solution.gen_psch[gen_static_id]
            qi = pf_solution.gen_qsch[gen_static_id]

        elif device.model_type == "static_generator":
            pi = sum(pf_solution.gen_psch[idx] for idx in device.gen_idxs)
            qi = sum(pf_solution.gen_qsch[idx] for idx in device.gen_idxs)
        elif device.model_type == "ZIPLoad":
            pi = -device.pload
            qi = device.qload
        elif device.model_type in ["governor", "exciter", "stabilizer"]:
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


def _initialize_system_from_config(psys: Psystem, config: IntegrationConfig):
    """Solve and initialize the operating point specified by an integration config."""
    pf_solution = runpf(
        psys,
        verbose=False,
        enforce_q_limits=config.enforce_q_limits,
        q_limit_tolerance=config.q_limit_tolerance,
        max_q_limit_iterations=config.max_q_limit_iterations,
    )
    validation = config.power_flow_validation
    if validation.enabled:
        pf_solution.validation = validate_power_flow_solution(
            psys,
            pf_solution,
            residual_tolerance=validation.residual_tolerance,
            generator_limit_tolerance=validation.generator_limit_tolerance,
            voltage_min=validation.voltage_min,
            voltage_max=validation.voltage_max,
            branch_loading_max=validation.branch_loading_max,
            branch_limit_tolerance=validation.branch_limit_tolerance,
            active_set_voltage_tolerance=(
                validation.active_set_voltage_tolerance
            ),
        )
        if not pf_solution.validation["valid"]:
            raise PowerFlowValidationError(pf_solution.validation)
    z0, theta = initialize_system(psys, pf_solution)
    return pf_solution, z0, theta


def _initialize_integration_state(
    psys: Psystem,
    config: IntegrationConfig,
    ctx: Optional[IntegrationCtx] = None,
):
    """Initialize, apply caller overrides, and validate hard state limits."""
    pf_solution, z0, theta = _initialize_system_from_config(psys, config)

    z0_user = ctx.z0_user if ctx is not None else getattr(config, "z0_user", None)
    theta_user = (
        ctx.theta_user if ctx is not None else getattr(config, "theta_user", None)
    )
    if z0_user is not None:
        if z0_user.shape[0] != z0.shape[0]:
            raise ValueError("Provided initial state does not match system size.")
        z0 = z0_user
    if theta_user is not None:
        if theta_user.shape[0] != theta.shape[0]:
            raise ValueError("Provided theta does not match system parameters.")
        theta = theta_user

    descriptors = collect_limited_state_descriptors(psys, theta)
    dynamic_limit_diagnostics = validate_initial_dynamic_limits(
        z0,
        descriptors,
        enforce_dynamic_limits=config.enforce_dynamic_limits,
        dynamic_limit_tolerance=config.dynamic_limit_tolerance,
        dynamic_limit_release_tolerance=(
            config.dynamic_limit_release_tolerance
        ),
        max_dynamic_limit_iterations=config.max_dynamic_limit_iterations,
    )
    dynamic_limit_diagnostics["parameter_adjustments"] = [
        {
            "device_type": type(device).__name__,
            "device_id": str(device.id_tag).strip(),
            "bus": int(psys.buses[device.bus].id),
            **device.limit_initialization_diagnostics,
        }
        for device in psys.devices
        if getattr(device, "limit_initialization_diagnostics", None)
        and device.limit_initialization_diagnostics.get("bounds_adjusted", False)
    ]
    return pf_solution, z0, theta, dynamic_limit_diagnostics


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

class DAE_petsc(object):
        n = 1
        def __init__(self, psys, theta, J):
            self.psys = psys
            self.theta = theta
            self.J = J

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
            start, end = x.getOwnershipRange()
            NDIFFEQ = self.psys.num_dof_dif
            xx = np.array(x[start:end])
            ff = np.zeros(xx.shape, dtype=np.float64)
            residual_function(ff, xx, self.theta, self.psys)
            f.setArray(-ff)
            f[:NDIFFEQ] += xdot[:NDIFFEQ]
            f.assemble()
        
        def evalJacobian(self, ts, t, x, xdot, a, J, P):
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
    newton_tol = config.newton_tol
    newton_max_iter = config.newton_max_iter
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
    method = config.method

    if method in {"beuler", "cn"}:
        pass
    elif method in {"herk2", "herk4"}:
        return integrate_system_herk(psys, config, ctx)
    else:
        raise ValueError(f"Unknown integration method: {method}")

    # check for arkimex enabled
    if arkimex and petsc:
        logger.info("ARKIMEX activated.")

    results = {}
    psys.power_injection=power_injection

    # retrieve parameters
    pf_solution, z0, theta, dynamic_limit_diagnostics = (
        _initialize_integration_state(psys, config, ctx)
    )
    results["dynamic_limit_diagnostics"] = dynamic_limit_diagnostics
    if pf_solution.validation is not None:
        results["power_flow_diagnostics"] = pf_solution.validation

    limit_descriptors = []
    if config.enforce_dynamic_limits:
        limit_descriptors = [
            descriptor
            for descriptor in collect_limited_state_descriptors(psys, theta)
            if descriptor.enabled
        ]
    limit_modes = initialize_dynamic_limit_modes(limit_descriptors)

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

    has_fault = bool(psys.fault_events)
    schedule = build_integration_schedule(
        dt=dt,
        tend=tend,
        steps=steps,
        ton=ton,
        toff=toff,
        has_fault=has_fault,
    )
    tvec = schedule.times
    fault = psys.fault_events[0] if has_fault else None
    if fault is not None:
        fault.remove()

    # Integration of D.A.E
    z = z0

    # Sensitivity parameters
    nparam = psys.nloads # For now, we only suport sensitivities of loads

    if comp_sens and not petsc:
        raise ValueError("Sensitivities can only be computed with PETSc.")

    PETSc = _get_petsc_for_config(config)

    if PETSc is not None and method == "beuler" and limit_descriptors:
        history = _integrate_system_petsc_beuler_limits(
            PETSc,
            psys,
            config,
            theta,
            z0,
            tvec,
            schedule,
            fault,
            residual,
            jacobian,
            limit_descriptors,
            dynamic_limit_diagnostics,
        )
    elif PETSc is not None and method == "cn" and limit_descriptors:
        history = _integrate_system_petsc_cn_limits(
            PETSc,
            psys,
            config,
            theta,
            z0,
            tvec,
            schedule,
            fault,
            residual,
            jacobian,
            limit_descriptors,
            dynamic_limit_diagnostics,
        )
    elif PETSc is not None:
        if verbose:
            logger.info("Convert objects to PETSc format")
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
        dae = DAE_petsc(psys, theta, jacobian)

        ts = PETSc.TS().create(comm=PETSc.COMM_WORLD)
        ts.setProblemType(ts.ProblemType.NONLINEAR)
        opts = PETSc.Options()
        opts.setValue("ts_adapt_type", "none")

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
            ts.setType(_petsc_ts_type(PETSc, method, arkimex=False))
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

        expected_ts_type = _petsc_ts_type(PETSc, method, arkimex)
        ts.setFromOptions()
        if ts.getType() != expected_ts_type:
            raise ValueError(
                "PETSc options changed the configured integration method: "
                f"expected {expected_ts_type}, got {ts.getType()}"
            )
        ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

        transition_actions = {}
        if schedule.fault_on_index is not None:
            transition_actions.setdefault(schedule.fault_on_index, []).append("apply")
        if schedule.fault_off_index is not None:
            transition_actions.setdefault(schedule.fault_off_index, []).append("remove")

        historyp = [np.array(z0, copy=True)]
        tvecp = [float(tvec[0])]
        event_transitions = {}

        def monitor(ts, step, time, solution):
            value = np.array(solution[:], copy=True)
            if abs(time - tvecp[-1]) <= 1e-11:
                tvecp[-1] = float(time)
                historyp[-1] = value
            else:
                tvecp.append(float(time))
                historyp.append(value)

        ts.setMonitor(monitor)

        runs = []
        start_index = 0
        while start_index < len(tvec) - 1:
            step_width = tvec[start_index + 1] - tvec[start_index]
            boundary_index = start_index + 1
            while boundary_index < len(tvec) - 1:
                if boundary_index in transition_actions:
                    break
                next_width = tvec[boundary_index + 1] - tvec[boundary_index]
                if not np.isclose(next_width, step_width, rtol=0.0, atol=1e-12):
                    break
                boundary_index += 1
            runs.append((start_index, boundary_index, step_width))
            start_index = boundary_index

        for start_index, boundary_index, step_width in runs:
            ts.setTime(float(tvec[start_index]))
            ts.setTimeStep(float(step_width))
            ts.setMaxTime(float(tvec[boundary_index]))
            ts.solve(z0p)

            for action in transition_actions.get(boundary_index, ()):
                if action == "apply":
                    fault.apply()
                else:
                    fault.remove()
                projected, _, _, _ = integrate(
                    np.array(z0p[:], copy=True),
                    theta,
                    0.0,
                    psys,
                    residual,
                    jacobian,
                    None,
                    verbose=verbose,
                    fsolve=False,
                    newton_tol=newton_tol,
                    newton_max_iter=newton_max_iter,
                    uold=None,
                    vold=None,
                    mold=None,
                )
                z0p.setArray(projected)
                z0p.assemble()
                event_transitions.setdefault(boundary_index, []).append(
                    (action, np.array(z0p[:], copy=True))
                )
            if boundary_index in transition_actions:
                ts.setSolution(z0p)
                ts.setTime(float(tvec[boundary_index]))
                ts.restartStep()

        if (
            len(tvecp) != len(tvec)
            or not np.allclose(tvecp, tvec, rtol=0.0, atol=1e-11)
        ):
            raise RuntimeError("PETSc did not return the configured integration time grid")
        for event_index, transitions in event_transitions.items():
            historyp[event_index] = transitions[-1][1]

        # adjoint computation
        if comp_sens:
            if verbose:
                logger.info("Solving adjoint problem...")
            if fault is not None and ton <= tvec[-1] < toff:
                fault.apply()
            elif fault is not None:
                fault.remove()
            for start_index, boundary_index, _ in reversed(runs):
                ts.adjointSetSteps(boundary_index - start_index)
                ts.adjointSolve()
                for action, event_state in reversed(
                        event_transitions.get(start_index, ())):
                    lambda_values, mu_values = _algebraic_projection_adjoint(
                        psys,
                        event_state,
                        theta,
                        jacobian,
                        np.array(v_lambda[:], copy=True),
                        np.array(v_mu[:], copy=True),
                    )
                    v_lambda.setArray(lambda_values)
                    v_lambda.assemble()
                    v_mu.setArray(mu_values)
                    v_mu.assemble()
                    if action == "apply":
                        fault.remove()
                    else:
                        fault.apply()
            if fault is not None:
                fault.remove()

            # extract results
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

        if fault is not None:
            fault.remove()

        history = np.transpose(np.asarray(historyp))

    else:
        history = np.zeros((system_size, len(tvec)))
        history[:, 0] = np.copy(z)
        try:
            for i in range(1, len(tvec)):
                t_start = tvec[i - 1]
                h_step = tvec[i] - t_start
                if verbose:
                    logger.info("Step: %i. Time: %g (sec)", i, tvec[i])
                if psys.signal_injectors:
                    for inj in psys.signal_injectors:
                        inj.update(t_start, theta, psys)
                if limit_descriptors:
                    z, limit_modes, limit_events = (
                        _integrate_beuler_with_dynamic_limits(
                            z,
                            theta,
                            h_step,
                            psys,
                            residual,
                            jacobian,
                            descriptors=limit_descriptors,
                            modes=limit_modes,
                            state_tolerance=config.dynamic_limit_tolerance,
                            release_tolerance=(
                                config.dynamic_limit_release_tolerance
                            ),
                            max_active_set_iterations=(
                                config.max_dynamic_limit_iterations
                            ),
                            endpoint_time=tvec[i],
                            newton_tol=newton_tol,
                            newton_max_iter=newton_max_iter,
                            prior_events=dynamic_limit_diagnostics["events"],
                            verbose=verbose,
                        )
                    )
                    dynamic_limit_diagnostics["events"].extend(limit_events)
                else:
                    z, u, v, m = integrate(
                        z,
                        theta,
                        h_step,
                        psys,
                        residual,
                        jacobian,
                        None,
                        verbose=verbose,
                        fsolve=fsolve,
                        newton_tol=newton_tol,
                        newton_max_iter=newton_max_iter,
                        uold=None,
                        vold=None,
                        mold=None,
                    )

                if i == schedule.fault_on_index:
                    if verbose:
                        logger.info("Apply fault")
                    fault.apply()
                    z, _, _, _ = integrate(
                        z, theta, 0.0, psys, residual, jacobian, None,
                        verbose=verbose, fsolve=False,
                        newton_tol=newton_tol, newton_max_iter=newton_max_iter,
                        uold=None, vold=None, mold=None,
                    )
                if i == schedule.fault_off_index:
                    if verbose:
                        logger.info("Remove fault")
                    fault.remove()
                    z, _, _, _ = integrate(
                        z, theta, 0.0, psys, residual, jacobian, None,
                        verbose=verbose, fsolve=False,
                        newton_tol=newton_tol, newton_max_iter=newton_max_iter,
                        uold=None, vold=None, mold=None,
                    )
                history[:, i] = np.copy(z)
        finally:
            if fault is not None:
                fault.remove()

    # pack results into dict
    results["tvec"] = tvec
    results["history"] = history

    return results
