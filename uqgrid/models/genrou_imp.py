# IMPLEMENTATION OF GENROU
import numpy as np
from numba import jit
from uqgrid.utils.tools import csr_add_row, csr_set_row
from uqgrid.core.base_models import DynamicGenerator
from scipy import optimize

class GenGENROU(DynamicGenerator):
    def __init__(self, id_tag, x_d, x_q, x_dp, x_qp, x_ddp, xl, H, D, T_d0p,
                 T_q0p, T_d0dp, T_q0dp):

        self.x_d = x_d
        self.x_q = x_q
        self.x_dp = x_dp
        self.x_qp = x_qp
        self.x_ddp = x_ddp
        self.x_qdp = x_ddp
        self.xl = xl
        self.H = H
        self.D = D
        self.T_d0p = T_d0p
        self.T_q0p = T_q0p
        self.T_d0dp = T_d0dp
        self.T_q0dp = T_q0dp
        # (MBASE/SBASE) ratio. To be modified depending on MBASE.
        self.ratio = 1.0

        state_list = [
            'e_qp', 'e_dp', 'phi_1d', 'phi_2q', 'w', 'delta', 'v_q', 'v_d',
            'i_q', 'i_d'
        ]

        par_list = [
            'x_d', 'x_q', 'x_dp', 'x_qp', 'x_ddp', 'x_qdp', 'xl', 'H', 'D',
            'T_d0p', 'T_q0p', 'T_d0dp', 'T_q0dp'
        ]

        DynamicGenerator.__init__(self, id_tag, 12, 6, 4, len(par_list),
                                  state_list)
        
        # expose pure jit functions
        self.residual_diff_jit = resdiff_genrou
        self.residual_cinj_jit = cinj_genrou
        self.residual_jac_jit = jac_genrou

    def set_ratio(self, ratio):
        """ Modify machine parameters for a given MBASE/SBASE ratio"""

        self.ratio = ratio
        self.x_d = self.x_d*(1.0/ratio)
        self.x_q = self.x_q*(1.0/ratio)
        self.x_dp = self.x_dp*(1.0/ratio)
        self.x_qp = self.x_qp*(1.0/ratio)
        self.x_ddp = self.x_ddp*(1.0/ratio)
        self.x_qdp = self.x_qdp*(1.0/ratio)
        self.xl = self.xl*(1.0/ratio)
        self.H = self.H*ratio
        self.D = self.D*ratio

    def initialize_theta(self, theta):

        idx = self.par_ptr

        theta[idx] = self.x_d
        theta[idx + 1] = self.x_q
        theta[idx + 2] = self.x_dp
        theta[idx + 3] = self.x_qp
        theta[idx + 4] = self.x_ddp
        theta[idx + 5] = self.x_qdp
        theta[idx + 6] = self.xl
        theta[idx + 7] = self.H
        theta[idx + 8] = self.D
        theta[idx + 9] = self.T_d0p
        theta[idx + 10] = self.T_q0p
        theta[idx + 11] = self.T_d0dp
        theta[idx + 12] = self.T_q0dp

    def residualFinit(self, x, v, theta, p0, q0):

        F = np.zeros(self.initdim)

        # state variables
        e_qp = x[0]
        e_dp = x[1]
        phi_1d = x[2]
        phi_2q = x[3]
        w = x[4]
        delta = x[5]
        v_q = x[6]
        v_d = x[7]
        i_q = x[8]
        i_d = x[9]
        e_fd = x[10]
        p_m = x[11]

        # parameters
        x_d = self.x_d
        x_q = self.x_q
        x_dp = self.x_dp
        x_qp = self.x_qp
        x_ddp = self.x_ddp
        x_qdp = self.x_ddp
        xl = self.xl
        H = self.H
        D = self.D
        T_d0p = self.T_d0p
        T_q0p = self.T_q0p
        T_d0dp = self.T_d0dp
        T_q0dp = self.T_q0dp
        ratio = self.ratio

        # auxiliary variables
        psi_de = (x_ddp - xl)/(x_dp - xl)*e_qp + \
            (x_dp - x_ddp)/(x_dp - xl)*phi_1d

        psi_qe = -(x_ddp - xl)/(x_qp - xl)*e_dp + \
            (x_qp - x_ddp)/(x_qp - xl)*phi_2q

        # Machine states
        F[0] = (-e_qp + e_fd - (i_d - (-x_ddp + x_dp)*(-e_qp + i_d*
                                                       (x_dp - xl) + phi_1d)/
                                ((x_dp - xl)**2.0))*(x_d - x_dp))/T_d0p
        F[1] = (-e_dp + (i_q - (-x_qdp + x_qp)*
                         (e_dp + i_q*(x_qp - xl) + phi_2q)/((x_qp - xl)**2.0))*
                (x_q - x_qp))/T_q0p
        F[2] = (e_qp - i_d*(x_dp - xl) - phi_1d)/T_d0dp
        F[3] = (-e_dp - i_q*(x_qp - xl) - phi_2q)/T_q0dp

        F[4] = (p_m - psi_de*i_q + psi_qe*i_d)/(2.0*H)
        F[5] = 2.0*np.pi*60.0*w

        # Stator currents
        F[6] = i_d - ((x_ddp - xl)/(x_dp - xl)*e_qp + \
            (x_dp - x_ddp)/(x_dp - xl)*phi_1d - v_q)/x_ddp
        F[7] = i_q - (-(x_qdp - xl)/(x_qp - xl)*e_dp + \
            (x_qp - x_qdp)/(x_qp - xl)*phi_2q + v_d)/x_qdp

        # Stator voltage
        F[8] = v_d - v*np.sin(delta - theta)
        F[9] = v_q - v*np.cos(delta - theta)

        #Stator additional equations
        F[10] = v_d*i_d + v_q*i_q - p0
        F[11] = v_q*i_d - v_d*i_q - q0

        return F

    def initialize(self, vm, va, p, q, x, y, psys):

        # parameters
        x_d = self.x_d
        x_q = self.x_q
        x_dp = self.x_dp
        x_qp = self.x_qp
        x_ddp = self.x_ddp
        x_qdp = self.x_ddp
        xl = self.xl
        H = self.H
        D = self.D
        T_d0p = self.T_d0p
        T_q0p = self.T_q0p
        T_d0dp = self.T_d0dp
        T_q0dp = self.T_q0dp
        ratio = self.ratio

        x0 = np.ones(self.initdim)
        vt  = vm*np.cos(va) + 1j*vm*np.sin(va)
        ig = (p - 1j*q)/np.conjugate(vt)
        delta = np.angle(vt + (1j*x_q)*ig)

        v_d = vm*np.sin(delta - va)
        v_q = vm*np.cos(delta - va)
        i_d = (p*v_d + q*v_q)/(v_d**2 + v_q**2)
        i_q = (p*v_q - q*v_d)/(v_d**2 + v_q**2)

        phi_d = v_q
        phi_q = -v_d
    
        e_dp = (-x_qp)*i_q - phi_q
        e_qp = x_dp*i_d + phi_d

        phi_1d =  e_qp - (x_dp - xl)*i_d
        phi_2q =  -e_dp - (x_qp - xl)*i_q
    
        e_fd = e_qp + (x_d - x_dp)*i_d
        p_m = p

        x0[0] = e_qp
        x0[1] = e_dp
        x0[2] = phi_1d
        x0[3] = phi_2q
        x0[4] = 0.0
        x0[5] = delta
        x0[6] = v_q
        x0[7] = v_d
        x0[8] = i_q
        x0[9] = i_d
        x0[10] = e_fd
        x0[11] = p_m

        # REFINEMENT
        sol = optimize.root(
            self.residualFinit,
            x0,
            args=(vm, va, p, q),
            method='krylov',
            options={
                'xtol': 1e-8,
                'disp': False
            })

        self.e_fd = sol.x[10]
        self.p_m = sol.x[11]

        self.set_efd_val(sol.x[10])
        self.set_pm_val(sol.x[11])

        if self.exciter: self.exciter.e_fd0 = sol.x[10]
        if self.governor: self.governor.p_m0 = sol.x[11]

        self.initialized = True
        x[self.dif_ptr:self.dif_ptr + 6] = sol.x[0:6]
        y[self.alg_ptr:self.alg_ptr + 4] = sol.x[6:10]

        return None

    def residual_diff(self, F, z, v, theta, idxs, ctrl_idx, ctrl_var, power_injection):
        resdiff_genrou(F, z, v, theta, idxs, ctrl_idx, ctrl_var, power_injection)

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):

        dp = idxs[0]
        ap = idxs[1]
        v_q = z[ap]
        v_d = z[ap + 1]
        i_q = z[ap + 2]
        i_d = z[ap + 3]

        F[2*self.bus] += v_d*i_d + v_q*i_q
        F[2*self.bus + 1] += v_q*i_d - v_d*i_q
        return None
    
    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        cinj_genrou(F, z, v, theta, idxs)

    def preallocate_jacobian(self, idxs, psys, power_injection):

        coord = []

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]

        # these are INDEXES
        e_qp = dp
        e_dp = dp + 1
        phi_1d = dp + 2
        phi_2q = dp + 3
        w = dp + 4
        delta = dp + 5

        v_q = ap
        v_d = ap + 1
        i_q = ap + 2
        i_d = ap + 3

        if power_injection:
            vm = dev + 2*self.bus
            va = dev + 2*self.bus + 1
        else:
            vr = dev + 2*self.bus
            vi = dev + 2*self.bus + 1

        # first row
        row = dp
        if self.exciter:
            cols = [e_qp, phi_1d, self.efd_idx, i_d]
        else:
            cols = [e_qp, phi_1d, i_d]
        coord.append([row, cols])

        # second row
        row = dp + 1
        cols = [e_dp, phi_2q, i_q]
        coord.append([row, cols])

        # third row
        row = dp + 2
        cols = [e_qp, phi_1d, i_d]
        coord.append([row, cols])

        # fourth row
        row = dp + 3
        cols = [e_dp, phi_2q, i_q]
        coord.append([row, cols])

        # fifth row:
        row = dp + 4
        if self.governor:
            cols = [e_qp, e_dp, phi_1d, phi_2q, self.pm_idx, i_q, i_d, w]
        else:
            cols = [e_qp, e_dp, phi_1d, phi_2q, i_q, i_d, w]

        coord.append([row, cols])

        row = dp + 5
        cols = [w]
        coord.append([row, cols])

        # algebraic part:
        row = ap
        cols = [e_qp, phi_1d, v_q, i_d]
        coord.append([row, cols])

        row = ap + 1
        cols = [e_dp, phi_2q, v_d, i_q]
        coord.append([row, cols])
        
        if power_injection:
            row = ap + 2
            cols = [delta, v_d, vm, va]
            coord.append([row, cols])

            row = ap + 3
            cols = [delta, v_q, vm, va]
            coord.append([row, cols])

            row = dev + 2*self.bus
            cols = [v_q, v_d, i_q, i_d]
            coord.append([row, cols])

            row = dev + 2*self.bus + 1
            cols = [v_q, v_d, i_q, i_d]
            coord.append([row, cols])
        else:
            row = ap + 2
            cols = [delta, v_d, vr, vi]
            coord.append([row, cols])
            
            row = ap + 3
            cols = [delta, v_q, vr, vi]
            coord.append([row, cols])
            
            row = dev + 2*self.bus
            cols = [delta, i_q, i_d]
            coord.append([row, cols])

            row = dev + 2*self.bus + 1
            cols = [delta, i_q, i_d]
            coord.append([row, cols])

        return coord

    def preallocate_hessian(self, h_nnz, idxs, psys):

        coord = []

        dp = idxs[0]  # Differential pointer
        ap = idxs[1]  # Algebraic pointer (raw, not offset)
        pp = idxs[2]  # Parameter pointer
        bus = idxs[3] # Bus number
        dev = idxs[4]  # System offset

        # these are INDEXES
        e_qp = dp
        e_dp = dp + 1
        phi_1d = dp + 2
        phi_2q = dp + 3
        w = dp + 4
        delta = dp + 5
        p_m = self.pm_idx

        v_q = ap
        v_d = ap + 1
        i_q = ap + 2
        i_d = ap + 3

        vm = dev + 2*self.bus
        va = dev + 2*self.bus + 1

        # Torque equation
        h_nnz[w]['rows'].append(e_qp)
        h_nnz[w]['cols'].append([i_q])

        h_nnz[w]['rows'].append(e_dp)
        h_nnz[w]['cols'].append([i_d])

        h_nnz[w]['rows'].append(phi_1d)
        h_nnz[w]['cols'].append([i_q])

        h_nnz[w]['rows'].append(phi_2q)
        h_nnz[w]['cols'].append([i_d])

        if self.governor:
            h_nnz[w]['rows'].append(w)
            h_nnz[w]['cols'].append([w, p_m])

            h_nnz[w]['rows'].append(p_m)
            h_nnz[w]['cols'].append([w])
        else:
            h_nnz[w]['rows'].append(w)
            h_nnz[w]['cols'].append([w])

        h_nnz[w]['rows'].append(i_q)
        h_nnz[w]['cols'].append([e_qp, phi_1d])

        h_nnz[w]['rows'].append(i_d)
        h_nnz[w]['cols'].append([e_dp, phi_2q])

        # algebraic equations
        h_nnz[ap + 2]['rows'].append(delta)
        h_nnz[ap + 2]['cols'].append([delta, vm, va])
        h_nnz[ap + 2]['rows'].append(vm)
        h_nnz[ap + 2]['cols'].append([delta, va])
        h_nnz[ap + 2]['rows'].append(va)
        h_nnz[ap + 2]['cols'].append([delta, vm, va])

        h_nnz[ap + 3]['rows'].append(delta)
        h_nnz[ap + 3]['cols'].append([delta, vm, va])
        h_nnz[ap + 3]['rows'].append(vm)
        h_nnz[ap + 3]['cols'].append([delta, va])
        h_nnz[ap + 3]['rows'].append(va)
        h_nnz[ap + 3]['cols'].append([delta, vm, va])

        # power injection
        h_nnz[vm]['rows'].append(v_q)
        h_nnz[vm]['cols'].append([i_q])
        h_nnz[vm]['rows'].append(v_d)
        h_nnz[vm]['cols'].append([i_d])
        h_nnz[vm]['rows'].append(i_q)
        h_nnz[vm]['cols'].append([v_q])
        h_nnz[vm]['rows'].append(i_d)
        h_nnz[vm]['cols'].append([v_d])

        # (NOTE) This is wrong but it seems not to cause
        # any problem...
        h_nnz[vm]['rows'].append(v_q)
        h_nnz[vm]['cols'].append([i_d])
        h_nnz[vm]['rows'].append(v_d)
        h_nnz[vm]['cols'].append([i_q])
        h_nnz[vm]['rows'].append(i_q)
        h_nnz[vm]['cols'].append([v_d])
        h_nnz[vm]['rows'].append(i_d)
        h_nnz[vm]['cols'].append([v_q])

    def residual_jac(self, J, z, v, theta, idxs, ctrl_idx, ctrl_var,
            power_injection):

        jac_genrou(z, v, theta, idxs, ctrl_idx, ctrl_var, J.data, J.indptr,
                   J.indices, power_injection)

        return None

    def residual_hess(self, HESS, z, v, theta, idxs, ctrl_idx, ctrl_var):

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]
        pp = idxs[3]
        bus = idxs[4]

        H1 = HESS[dp + 4]
        H2 = HESS[ap + 2]
        H3 = HESS[ap + 3]
        H4 = HESS[dev + 2*bus]
        H5 = HESS[dev + 2*bus + 1]

        hes_genrou(z, v, theta, idxs, ctrl_idx, ctrl_var, H1.data, H1.indptr,
                   H1.indices, H2.data, H2.indptr, H2.indices, H3.data,
                   H3.indptr, H3.indices, H4.data, H4.indptr, H4.indices,
                   H5.data, H5.indptr, H5.indices)

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

@jit(nopython=True, cache=True)
def cinj_genrou(F, z, v, theta, idxs):

    dp = idxs[0]
    ap = idxs[1]
    bus = idxs[3]

    v_q = z[ap]
    v_d = z[ap + 1]
    i_q = z[ap + 2]
    i_d = z[ap + 3]
    delta = z[dp + 5]

    F[2*bus] += np.sin(delta)*i_d + np.cos(delta)*i_q
    F[2*bus + 1] += -np.cos(delta)*i_d + np.sin(delta)*i_q

    
@jit(nopython=True, cache=True)
def jac_genrou(z, v, theta, idxs,
        ctrl_idx, ctrl_var, J_data, J_ptr, J_idx, power_injection):

    dp = idxs[0]  # Differential pointer
    ap = idxs[1]  # Algebraic pointer (raw, not offset)
    pp = idxs[2]  # Parameter pointer
    bus = idxs[3] # Bus number
    dev = idxs[4]  # System offset

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

    dp = idxs[0]  # Differential pointer
    ap = idxs[1]  # Algebraic pointer (raw, not offset)
    pp = idxs[2]  # Parameter pointer
    bus = idxs[3] # Bus number
    dev = idxs[4]  # System offset

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
