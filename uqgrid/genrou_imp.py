# IMPLEMENTATION OF GENROU
import numpy as np
import numba
from numba import jit
from .tools import csr_add_row, csr_set_row

@jit(nopython=True, cache=True)
def resdiff_genrou(F, z, v, theta, idxs, ctrl_idx, ctrl_var, power_injection):

    dp = idxs[0]
    ap = idxs[1]
    pp = idxs[2]
    bus = idxs[3]

    # parameters
    x_d = theta[pp]
    x_q = theta[pp + 1]
    x_dp = theta[pp + 2]
    x_qp = theta[pp + 3]
    x_ddp  = theta[pp + 4]
    x_qdp  = theta[pp + 5]
    xl = theta[pp + 6]
    H = theta[pp + 7]
    D = theta[pp + 8]
    T_d0p = theta[pp + 9]
    T_q0p = theta[pp + 10]
    T_d0dp = theta[pp + 11]
    T_q0dp = theta[pp + 12]

    # states
    e_qp     = z[dp]
    e_dp     = z[dp + 1]
    phi_1d   = z[dp + 2]
    phi_2q   = z[dp + 3]
    w        = z[dp + 4]
    delta    = z[dp + 5]

    v_q      = z[ap] 
    v_d      = z[ap + 1]
    i_q      = z[ap + 2]
    i_d      = z[ap + 3]

    if power_injection:
        vm = v[2*bus]
        va = v[2*bus + 1]
    else:
        vr = v[2*bus]
        vi = v[2*bus + 1]
        vm = np.sqrt(vr**2.0 + vi**2.0)
        va = np.arctan2(vi, vr)

    # control
    pm_idx = ctrl_idx[0]
    efd_idx = ctrl_idx[1]
    
    p_m = ctrl_var[0]
    e_fd = ctrl_var[1]

    if efd_idx >= 0:
        e_fd = z[efd_idx]

    if pm_idx >= 0:
        p_m = z[pm_idx]
    
    tmech = (p_m - D*w)/(1.0 + w)

    # auxiliary variables
    psi_de = (x_ddp - xl)/(x_dp - xl)*e_qp + \
        (x_dp - x_ddp)/(x_dp - xl)*phi_1d

    psi_qe = -(x_ddp - xl)/(x_qp - xl)*e_dp + \
        (x_qp - x_ddp)/(x_qp - xl)*phi_2q

    # equations
    F[dp] = (-e_qp + e_fd - (i_d - (-x_ddp + x_dp)*(-e_qp + i_d*(x_dp - xl) \
        + phi_1d)/((x_dp - xl)**2.0))*(x_d - x_dp))/T_d0p
    F[dp + 1] = (-e_dp + (i_q - (-x_qdp + x_qp)*( e_dp + i_q*(x_qp - xl) \
        + phi_2q)/((x_qp - xl)**2.0))*(x_q - x_qp))/T_q0p
    F[dp + 2] = ( e_qp - i_d*(x_dp - xl) - phi_1d)/T_d0dp
    F[dp + 3] = (-e_dp - i_q*(x_qp - xl) - phi_2q)/T_q0dp
    F[dp + 4] = (tmech - psi_de*i_q + psi_qe*i_d)/(2.0*H)
    F[dp + 5] = 2.0*np.pi*60.0*w

    # Stator currents
    F[ap] = i_d - ((x_ddp - xl)/(x_dp - xl)*e_qp + \
            (x_dp - x_ddp)/(x_dp - xl)*phi_1d - v_q)/x_ddp
    F[ap + 1] = i_q - (-(x_qdp - xl)/(x_qp - xl)*e_dp + \
            (x_qp - x_qdp)/(x_qp - xl)*phi_2q + v_d)/x_qdp

    # Stator voltage
    if power_injection:
        F[ap + 2] = v_d - vm*np.sin(delta - va)
        F[ap + 3] = v_q - vm*np.cos(delta - va)
    else:
        F[ap + 2] = v_d - (vr*np.sin(delta) - vi*np.cos(delta))
        F[ap + 3] = v_q - (vr*np.cos(delta) + vi*np.sin(delta))
    
#@jit(nopython=True, cache=True)
def jac_genrou(z, v, theta, idxs,
        ctrl_idx, ctrl_var, J_data, J_ptr, J_idx, power_injection):

    dp = idxs[0]
    ap = idxs[1]
    dev = idxs[2]
    pp = idxs[3]
    bus = idxs[4]

    # parameters
    x_d = theta[pp]
    x_q = theta[pp + 1]
    x_dp = theta[pp + 2]
    x_qp = theta[pp + 3]
    x_ddp  = theta[pp + 4]
    x_qdp  = theta[pp + 5]
    xl = theta[pp + 6]
    H = theta[pp + 7]
    D = theta[pp + 8]
    T_d0p = theta[pp + 9]
    T_q0p = theta[pp + 10]
    T_d0dp = theta[pp + 11]
    T_q0dp = theta[pp + 12]
    
    # states
    e_qp     = z[dp]
    e_dp     = z[dp + 1]
    phi_1d   = z[dp + 2]
    phi_2q   = z[dp + 3]
    w        = z[dp + 4]
    delta    = z[dp + 5]

    v_q      = z[ap] 
    v_d      = z[ap + 1]
    i_q      = z[ap + 2]
    i_d      = z[ap + 3]

    if power_injection:
        vm = v[2*bus]
        va = v[2*bus + 1]
    else:
        vr = v[2*bus]
        vi = v[2*bus + 1]

    # control
    pm_idx = ctrl_idx[0]
    efd_idx = ctrl_idx[1]
    
    p_m = ctrl_var[0]
    e_fd = ctrl_var[1]

    if efd_idx >= 0:
        e_fd = z[efd_idx]

    if pm_idx >= 0:
        p_m = z[pm_idx]

    # indexes
    e_qp_idx = dp
    e_dp_idx = dp + 1
    phi_1d_idx = dp + 2
    phi_2q_idx = dp + 3
    w_idx = dp + 4
    delta_idx = dp + 5
    v_q_idx = ap
    v_d_idx = ap + 1
    i_q_idx = ap + 2
    i_d_idx = ap + 3

    if power_injection:
        vm_idx = dev + 2*bus
        va_idx = dev + 2*bus + 1
    else:
        vr_idx = dev + 2*bus
        vi_idx = dev + 2*bus + 1

    # auxiliary variables
    psi_de = (x_ddp - xl)/(x_dp - xl)*e_qp + \
        (x_dp - x_ddp)/(x_dp - xl)*phi_1d

    psi_qe = -(x_ddp - xl)/(x_qp - xl)*e_dp + \
        (x_qp - x_ddp)/(x_qp - xl)*phi_2q

    # column and value vectors
    col = np.zeros(10)
    val = np.zeros(10)


    # first row
    row = dp
    col[0] = e_qp_idx
    val[0] = (-(x_d - x_dp)*(-x_ddp + x_dp)*(x_dp - xl)**(-2.0) - 1)/T_d0p
    col[1] = phi_1d_idx
    val[1] = (x_d - x_dp)*(-x_ddp + x_dp)*(x_dp - xl)**(-2.0)/T_d0p
    if efd_idx >= 0:
        col[2] = efd_idx
        val[2] = 1/T_d0p
        col[3] = i_d_idx
        val[3] = -(x_d - x_dp)*(-(-x_ddp + x_dp)*(x_dp - xl)**(-1.0) + 1)/T_d0p
        csr_set_row(J_data, J_ptr, J_idx, 4, row, col, val)
    else:
        col[2] = i_d_idx
        val[2] = -(x_d - x_dp)*(-(-x_ddp + x_dp)*(x_dp - xl)**(-1.0) + 1)/T_d0p
        csr_set_row(J_data, J_ptr, J_idx, 3, row, col, val)

    # second row
    row = dp + 1
    col[0] = e_dp_idx
    val[0] = (-(x_q - x_qp)*(-x_qdp + x_qp)*(x_qp - xl)**(-2.0) - 1)/T_q0p
    col[1] = phi_2q_idx
    val[1] = -(x_q - x_qp)*(-x_qdp + x_qp)*(x_qp - xl)**(-2.0)/T_q0p
    col[2] = i_q_idx
    val[2] = (x_q - x_qp)*(-(-x_qdp + x_qp)*(x_qp - xl)**(-1.0) + 1)/T_q0p
    csr_set_row(J_data, J_ptr, J_idx, 3, row, col, val)

    # third row
    row = dp + 2
    col[0] = e_qp_idx
    val[0] = 1.0/T_d0dp
    col[1] = phi_1d_idx
    val[1] = -1.0/T_d0dp
    col[2] = i_d_idx
    val[2] = (-x_dp + xl)/T_d0dp
    csr_set_row(J_data, J_ptr, J_idx, 3, row, col, val)

    # fourth
    row = dp + 3
    col[0] = e_dp_idx
    val[0] = -1.0/T_q0dp
    col[1] = phi_2q_idx
    val[1] = -1.0/T_q0dp
    col[2] = i_q_idx
    val[2] = (-x_qp + xl)/T_q0dp
    csr_set_row(J_data, J_ptr, J_idx, 3, row, col, val)
    
    # fifth
    row = dp + 4
    col[0] = e_qp_idx
    val[0] = -0.5*i_q*(x_ddp - xl)/(H*(x_dp - xl))
    col[1] = e_dp_idx
    val[1] = 0.5*i_d*(-x_ddp + xl)/(H*(x_qp - xl))
    col[2] = phi_1d_idx
    val[2] = -0.5*i_q*(-x_ddp + x_dp)/(H*(x_dp - xl))
    col[3] = phi_2q_idx
    val[3] = 0.5*i_d*(-x_ddp + x_qp)/(H*(x_qp - xl))
    col[4] = w_idx
    val[4] = 0.5*(-D/(w + 1.0) - (-D*w + p_m)/(w + 1.0)**2.0)/H
    
    if pm_idx >= 0:
        col[5] = i_q_idx
        val[5] = 0.5*(-e_qp*(x_ddp - xl)/(x_dp - xl) - phi_1d*(-x_ddp + x_dp)/(x_dp - xl))/H
        col[6] = i_d_idx
        val[6] = 0.5*(e_dp*(-x_ddp + xl)/(x_qp - xl) + phi_2q*(-x_ddp + x_qp)/(x_qp - xl))/H
        col[7] = pm_idx
        val[7] = 0.5/(H*(w + 1))
        csr_set_row(J_data, J_ptr, J_idx, 8, row, col, val)
    else:
        col[5] = i_q_idx
        val[5] = 0.5*(-e_qp*(x_ddp - xl)/(x_dp - xl) - phi_1d*(-x_ddp + x_dp)/(x_dp - xl))/H
        col[6] = i_d_idx
        val[6] = 0.5*(e_dp*(-x_ddp + xl)/(x_qp - xl) + phi_2q*(-x_ddp + x_qp)/(x_qp - xl))/H
        csr_set_row(J_data, J_ptr, J_idx, 7, row, col, val)

    # sixth
    row = dp + 5
    col[0] = w_idx
    val[0] = 120.0*np.pi
    csr_set_row(J_data, J_ptr, J_idx, 1, row, col, val)

    # algebraic first
    row = ap
    col[0] = e_qp_idx
    val[0] = -(x_ddp - xl)/(x_ddp*(x_dp - xl))
    col[1] = phi_1d_idx
    val[1] = -(-x_ddp + x_dp)/(x_ddp*(x_dp - xl))
    col[2] = v_q_idx
    val[2] = 1/x_ddp
    col[3] = i_d_idx
    val[3] = 1.0
    csr_set_row(J_data, J_ptr, J_idx, 4, row, col, val)

    # alg. second
    row = ap + 1
    col[0] = e_dp_idx
    val[0] = -(-x_qdp + xl)/(x_qdp*(x_qp - xl))
    col[1] = phi_2q_idx
    val[1] = -(-x_qdp + x_qp)/(x_qdp*(x_qp - xl))
    col[2] = v_d_idx
    val[2] = -1/x_qdp
    col[3] = i_q_idx
    val[3] = 1.0
    csr_set_row(J_data, J_ptr, J_idx, 4, row, col, val)

    if power_injection:
        
        # alg. third
        row = ap + 2
        col[0] = delta_idx
        val[0] = -vm*np.cos(delta - va)
        col[1] = v_d_idx
        val[1] = 1.0
        col[2] = vm_idx
        val[2] = -np.sin(delta - va)
        col[3] = va_idx
        val[3] = vm*np.cos(delta - va)
        csr_set_row(J_data, J_ptr, J_idx, 4, row, col, val)
    
        # alg. fourth
        row = ap + 3
        col[0] = delta_idx
        val[0] = vm*np.sin(delta - va)
        col[1] = v_q_idx
        val[1] = 1.0
        col[2] = vm_idx
        val[2] = -np.cos(delta - va)
        col[3] = va_idx
        val[3] = -vm*np.sin(delta - va)
        csr_set_row(J_data, J_ptr, J_idx, 4, row, col, val)
        
        # power injection
        row = dev + 2*bus
        col[0] = v_q_idx
        val[0] = i_q
        col[1] = v_d_idx
        val[1] = i_d
        col[2] = i_q_idx
        val[2] = v_q
        col[3] = i_d_idx
        val[3] = v_d
        csr_set_row(J_data, J_ptr, J_idx, 4, row, col, val)
        
        row = dev + 2*bus + 1
        col[0] = v_q_idx
        val[0] = i_d
        col[1] = v_d_idx
        val[1] = -i_q
        col[2] = i_q_idx
        val[2] = -v_d
        col[3] = i_d_idx
        val[3] = v_q
        csr_set_row(J_data, J_ptr, J_idx, 4, row, col, val)

    else:
        # alg. third
        row = ap + 2
        col[0] = delta_idx
        val[0] = -vr*np.cos(delta) - vi*np.sin(delta)
        col[1] = v_d_idx
        val[1] = 1.0
        col[2] = vr_idx
        val[2] = -np.sin(delta)
        col[3] = vi_idx
        val[3] = np.cos(delta)
        csr_set_row(J_data, J_ptr, J_idx, 4, row, col, val)
        
        # alg. fourth
        row = ap + 3
        col[0] = delta_idx
        val[0] = vr*np.sin(delta) - vi*np.cos(delta)
        col[1] = v_q_idx
        val[1] = 1.0
        col[2] = vr_idx
        val[2] = -np.cos(delta )
        col[3] = vi_idx
        val[3] = -np.sin(delta)
        csr_set_row(J_data, J_ptr, J_idx, 4, row, col, val)
        
        # power injection
        row = dev + 2*bus
        col[0] = delta_idx
        val[0] = i_d*np.cos(delta) - i_q*np.sin(delta)
        col[1] = i_q_idx
        val[1] = np.cos(delta)
        col[2] = i_d_idx
        val[2] = np.sin(delta)
        csr_set_row(J_data, J_ptr, J_idx, 3, row, col, val)
        
        # power injection
        row = dev + 2*bus + 1
        col[0] = delta_idx
        val[0] = i_d*np.sin(delta) + i_q*np.cos(delta)
        col[1] = i_q_idx
        val[1] = np.sin(delta)
        col[2] = i_d_idx
        val[2] = -np.cos(delta)
        csr_set_row(J_data, J_ptr, J_idx, 3, row, col, val)

@jit(nopython=True, cache=True)
def hes_genrou(z, v, theta, idxs,
            ctrl_idx, ctrl_var,
            H1_data, H1_indptr, H1_indices,
            H2_data, H2_indptr, H2_indices,
            H3_data, H3_indptr, H3_indices,
            H4_data, H4_indptr, H4_indices,
            H5_data, H5_indptr, H5_indices):

    dp = idxs[0]
    ap = idxs[1]
    dev = idxs[2]
    pp = idxs[3]
    bus = idxs[4]

    # parameters
    x_d = theta[pp]
    x_q = theta[pp + 1]
    x_dp = theta[pp + 2]
    x_qp = theta[pp + 3]
    x_ddp  = theta[pp + 4]
    x_qdp  = theta[pp + 5]
    xl = theta[pp + 6]
    H = theta[pp + 7]
    D = theta[pp + 8]
    T_d0p = theta[pp + 9]
    T_q0p = theta[pp + 10]
    T_d0dp = theta[pp + 11]
    T_q0dp = theta[pp + 12]

    # states
    e_qp     = z[dp]
    e_dp     = z[dp + 1]
    phi_1d   = z[dp + 2]
    phi_2q   = z[dp + 3]
    w        = z[dp + 4]
    delta    = z[dp + 5]

    v_q      = z[ap] 
    v_d      = z[ap + 1]
    i_q      = z[ap + 2]
    i_d      = z[ap + 3]

    vm = v[2*bus]
    va = v[2*bus + 1]

    # control
    pm_idx = ctrl_idx[0]
    efd_idx = ctrl_idx[1]
    
    p_m = ctrl_var[0]
    e_fd = ctrl_var[1]

    if efd_idx >= 0:
        e_fd = z[efd_idx]

    if pm_idx >= 0:
        p_m = z[pm_idx]
    
    # indexes
    e_qp_idx = dp
    e_dp_idx = dp + 1
    phi_1d_idx = dp + 2
    phi_2q_idx = dp + 3
    w_idx = dp + 4
    delta_idx = dp + 5
    v_q_idx = ap
    v_d_idx = ap + 1
    i_q_idx = ap + 2
    i_d_idx = ap + 3
    vm_idx = dev + 2*bus
    va_idx = dev + 2*bus + 1
    
    col = np.zeros(4)
    val = np.zeros(4)


    # SWING EQUATION
    row = e_qp_idx
    col[0] = i_q_idx
    val[0] = -0.5*(x_ddp - xl)/(H*(x_dp - xl))
    csr_set_row(H1_data, H1_indptr, H1_indices, 1, row, col, val)
    
    row = e_dp_idx
    col[0] = i_d_idx
    val[0] = -0.5*(x_ddp - xl)/(H*(x_qp - xl))
    csr_set_row(H1_data, H1_indptr, H1_indices, 1, row, col, val)
    
    row = phi_1d_idx
    col[0] = i_q_idx
    val[0] = 0.5*(x_ddp - x_dp)/(H*(x_dp - xl))
    csr_set_row(H1_data, H1_indptr, H1_indices, 1, row, col, val)
    
    row = phi_2q_idx
    col[0] = i_d_idx
    val[0] = -0.5*(x_ddp - x_qp)/(H*(x_qp - xl))
    csr_set_row(H1_data, H1_indptr, H1_indices, 1, row, col, val)
    
    row = w_idx
    col[0] = w_idx
    val[0] = 1.0*(D - (D*w - p_m)/(w + 1.0))/(H*(w + 1.0)**2)
    csr_set_row(H1_data, H1_indptr, H1_indices, 1, row, col, val)

    if pm_idx >= 0:
        row = w_idx
        col[0] = pm_idx
        val[0] = -0.5/(H*(w + 1.0)**2)
        csr_set_row(H1_data, H1_indptr, H1_indices, 1, row, col, val)
        
        row = pm_idx
        col[0] = w_idx
        val[0] = -0.5/(H*(w + 1.0)**2)
        csr_set_row(H1_data, H1_indptr, H1_indices, 1, row, col, val)
    
    row = i_q_idx
    col[0] = e_qp_idx
    val[0] = -0.5*(x_ddp - xl)/(H*(x_dp - xl))
    col[1] = phi_1d_idx
    val[1] = 0.5*(x_ddp - x_dp)/(H*(x_dp - xl))
    csr_set_row(H1_data, H1_indptr, H1_indices, 2, row, col, val)
    
    row = i_d_idx
    col[0] = e_dp_idx
    val[0] = -0.5*(x_ddp - xl)/(H*(x_qp - xl))
    col[1] = phi_2q_idx
    val[1] = -0.5*(x_ddp - x_qp)/(H*(x_qp - xl))
    csr_set_row(H1_data, H1_indptr, H1_indices, 2, row, col, val)

    # STATOR VOLTAGE 1
    row = delta_idx
    col[0] = delta_idx
    val[0] = vm*np.sin(delta - va)
    col[1] = vm_idx
    val[1] = -np.cos(delta - va)
    col[2] = va_idx
    val[2] = -vm*np.sin(delta - va)
    csr_set_row(H2_data, H2_indptr, H2_indices, 3, row, col, val)
    
    row = vm_idx
    col[0] = delta_idx
    val[0] = -np.cos(delta - va)
    col[1] = va_idx
    val[1] = np.cos(delta - va)
    csr_set_row(H2_data, H2_indptr, H2_indices, 2, row, col, val)
    
    row = va_idx
    col[0] = delta_idx
    val[0] = -vm*np.sin(delta - va)
    col[1] = vm_idx
    val[1] = np.cos(delta - va)
    col[2] = va_idx
    val[2] = vm*np.sin(delta - va)
    csr_set_row(H2_data, H2_indptr, H2_indices, 3, row, col, val)
    

    # STATOR VOLTAGE 2
    row = delta_idx
    col[0] = delta_idx
    val[0] = vm*np.cos(delta - va)
    col[1] = vm_idx
    val[1] = np.sin(delta - va)
    col[2] = va_idx
    val[2] = -vm*np.cos(delta - va)
    csr_set_row(H3_data, H3_indptr, H3_indices, 3, row, col, val)
    
    row = vm_idx
    col[0] = delta_idx
    val[0] = np.sin(delta - va)
    col[1] = va_idx
    val[1] = -np.sin(delta - va)
    csr_set_row(H3_data, H3_indptr, H3_indices, 2, row, col, val)
    
    row = va_idx
    col[0] = delta_idx
    val[0] = -vm*np.cos(delta - va)
    col[1] = vm_idx
    val[1] = -np.sin(delta - va)
    col[2] = va_idx
    val[2] = vm*np.cos(delta - va)
    csr_set_row(H3_data, H3_indptr, H3_indices, 3, row, col, val)
    

    # STATOR POWER INJECTION
    row = v_q_idx
    col[0] = i_q_idx
    val[0] = 1.0
    csr_set_row(H4_data, H4_indptr, H4_indices, 1, row, col, val)
    
    row = v_d_idx
    col[0] = i_d_idx
    val[0] = 1.0
    csr_set_row(H4_data, H4_indptr, H4_indices, 1, row, col, val)
    
    row = i_q_idx
    col[0] = v_q_idx
    val[0] = 1.0
    csr_set_row(H4_data, H4_indptr, H4_indices, 1, row, col, val)
    
    row = i_d_idx
    col[0] = v_d_idx
    val[0] = 1.0
    csr_set_row(H4_data, H4_indptr, H4_indices, 1, row, col, val)
    
    row = v_q_idx
    col[0] = i_d_idx
    val[0] = 1.0
    csr_set_row(H5_data, H5_indptr, H5_indices, 1, row, col, val)
    
    row = v_d_idx
    col[0] = i_q_idx
    val[0] = -1.0
    csr_set_row(H5_data, H5_indptr, H5_indices, 1, row, col, val)
    
    row = i_q_idx
    col[0] = v_d_idx
    val[0] = -1.0
    csr_set_row(H5_data, H5_indptr, H5_indices, 1, row, col, val)
    
    row = i_d_idx
    col[0] = v_q_idx
    val[0] = 1.0
    csr_set_row(H5_data, H5_indptr, H5_indices, 1, row, col, val)
