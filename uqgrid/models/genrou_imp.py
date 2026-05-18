# IMPLEMENTATION OF GENROU
import numpy as np
from numba import jit
from uqgrid.utils.tools import csr_add_row, csr_set_row
from uqgrid.core.base_models import DynamicGenerator
from scipy import optimize


def sat_coefficients(s1, s2):
    """Return (A, B) for quadratic saturation S(E)=B*(E-A)^2/E with E1=1.0, E2=1.2."""
    if s1 <= 0.0 or s2 <= 0.0:
        return 0.0, 0.0
    e1 = 1.0
    e2 = 1.2
    a = np.sqrt(s1 * e1 / (s2 * e2))
    if a == 1.0:
        return 0.0, 0.0
    sat_a = e2 - (e1 - e2) / (a - 1.0)
    sat_b = s2 * e2 * (a - 1.0) ** 2.0 / (e1 - e2) ** 2.0
    return sat_a, sat_b


def sat_se(psi, sat_a, sat_b):
    if sat_b == 0.0 or psi <= sat_a or psi == 0.0:
        return 0.0
    return sat_b * (psi - sat_a) ** 2.0 / psi


@jit(nopython=True, cache=True)
def sat_coefficients_nb(s1, s2):
    if s1 <= 0.0 or s2 <= 0.0:
        return 0.0, 0.0
    e1 = 1.0
    e2 = 1.2
    a = np.sqrt(s1 * e1 / (s2 * e2))
    if a == 1.0:
        return 0.0, 0.0
    sat_a = e2 - (e1 - e2) / (a - 1.0)
    sat_b = s2 * e2 * (a - 1.0) ** 2.0 / (e1 - e2) ** 2.0
    return sat_a, sat_b


@jit(nopython=True, cache=True)
def sat_se_nb(psi, sat_a, sat_b):
    if sat_b == 0.0 or psi <= sat_a or psi == 0.0:
        return 0.0
    return sat_b * (psi - sat_a) ** 2.0 / psi
class GenGENROU(DynamicGenerator):
    def __init__(self, id_tag, x_d, x_q, x_dp, x_qp, x_ddp, xl, H, D, T_d0p,
                 T_q0p, T_d0dp, T_q0dp, S1, S2):

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
        self.S1 = S1
        self.S2 = S2
        # (MBASE/SBASE) ratio. To be modified depending on MBASE.
        self.ratio = 1.0

        state_list = [
            'e_qp',
            'e_dp',
            'phi_1d',
            'phi_2q',
            'w',
            'delta',
            'p_m0',
            'e_fd0',
            'v_q',
            'v_d',
            'i_q',
            'i_d',
            'p_m_out',
            'e_fd_out'
        ]

        par_list = [
            'x_d', 'x_q', 'x_dp', 'x_qp', 'x_ddp', 'x_qdp', 'xl', 'H', 'D',
            'T_d0p', 'T_q0p', 'T_d0dp', 'T_q0dp', 'S1', 'S2'
        ]
        INIT_DIM = 12
        DYN_DIM = 8
        ALG_DIM = 6
        DynamicGenerator.__init__(self, id_tag, INIT_DIM, DYN_DIM, ALG_DIM, len(par_list),
                                  state_list)

    def set_ratio(self, ratio):
        """Modify machine parameters for a given MBASE/SBASE ratio."""

        if ratio <= 0.0:
            return
        self.ratio = ratio
        inv_ratio = 1.0 / ratio
        self.x_d *= inv_ratio
        self.x_q *= inv_ratio
        self.x_dp *= inv_ratio
        self.x_qp *= inv_ratio
        self.x_ddp *= inv_ratio
        self.x_qdp *= inv_ratio
        self.xl *= inv_ratio
        self.H *= ratio
        self.D *= ratio

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
        theta[idx + 13] = self.S1
        theta[idx + 14] = self.S2

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

        # saturation
        sat_a, sat_b = sat_coefficients(self.S1, self.S2)
        psi2 = np.sqrt(psi_de*psi_de + psi_qe*psi_qe)
        Se = sat_se(psi2, sat_a, sat_b)

        # Machine states
        F[0] = (-e_qp + e_fd - (i_d - (-x_ddp + x_dp)*(-e_qp + i_d*
                                                       (x_dp - xl) + phi_1d)/
                                ((x_dp - xl)**2.0))*(x_d - x_dp) - Se*psi_de)/T_d0p
        F[1] = (-e_dp + (i_q - (-x_qdp + x_qp)*
                         (e_dp + i_q*(x_qp - xl) + phi_2q)/((x_qp - xl)**2.0))*
                (x_q - x_qp))/T_q0p
        F[2] = (e_qp - i_d*(x_dp - xl) - phi_1d)/T_d0dp
        F[3] = (-e_dp - i_q*(x_qp - xl) - phi_2q)/T_q0dp

        F[4] = (p_m - psi_de*i_q + psi_qe*i_d - D*w)/(2.0*H)
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
    
        psi_de = (x_ddp - xl)/(x_dp - xl)*e_qp + \
            (x_dp - x_ddp)/(x_dp - xl)*phi_1d
        psi_qe = -(x_ddp - xl)/(x_qp - xl)*e_dp + \
            (x_qp - x_ddp)/(x_qp - xl)*phi_2q
        sat_a, sat_b = sat_coefficients(self.S1, self.S2)
        psi2 = np.sqrt(psi_de*psi_de + psi_qe*psi_qe)
        Se0 = sat_se(psi2, sat_a, sat_b)
        e_fd = e_qp + (x_d - x_dp)*i_d + Se0*psi_de
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

        dp = self.dif_ptr
        ap = self.alg_ptr

        x[dp:dp + 6] = sol.x[0:6]
        x[dp + 6] = sol.x[11]  # p_m0
        x[dp + 7] = sol.x[10]  # e_fd0

        y[ap:ap + 4] = sol.x[6:10]
        y[ap + 4] = sol.x[11]  # p_m_out
        y[ap + 5] = sol.x[10]  # e_fd_out
        self.initialized = True

        return None

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        resdiff_genrou(F, z, v, theta, idxs, power_injection)

    def residual_blend(self, F, z, v, theta, idxs, psys):
        dp = idxs[0]
        ap = idxs[1]
        g = self.device_index

        pm_ref = z[dp + 6]
        efd_ref = z[dp + 7]

        pm_out = z[ap + 4]
        efd_out = z[ap + 5]

        pm_ctrl_col = psys.gen_pm_ctrl_col[g]
        efd_ctrl_col = psys.gen_efd_ctrl_col[g]

        pm_ctrl = psys.p_m_ctrl_aligned[g] if pm_ctrl_col >= 0 else 0.0
        efd_ctrl = psys.e_fd_ctrl_aligned[g] if efd_ctrl_col >= 0 else 0.0

        pm_target = psys.gov_mask[g] * pm_ctrl + (1.0 - psys.gov_mask[g]) * pm_ref
        efd_target = psys.exc_mask[g] * efd_ctrl + (1.0 - psys.exc_mask[g]) * efd_ref

        F[ap + 4] = pm_out - pm_target
        F[ap + 5] = efd_out - efd_target

    def residual_blend_jac(self, J, z, v, theta, idxs, psys):
        dp = idxs[0]
        ap = idxs[1]
        g = self.device_index

        pm_ctrl_col = psys.gen_pm_ctrl_col[g]
        efd_ctrl_col = psys.gen_efd_ctrl_col[g]

        # p_m_out row
        cols = [ap + 4, dp + 6]
        vals = [1.0, -(1.0 - psys.gov_mask[g])]
        if pm_ctrl_col >= 0:
            cols.append(pm_ctrl_col)
            vals.append(-psys.gov_mask[g])
        if len(cols) == 2:
            if cols[0] > cols[1]:
                cols[0], cols[1] = cols[1], cols[0]
                vals[0], vals[1] = vals[1], vals[0]
        elif len(cols) == 3:
            if cols[0] > cols[1]:
                cols[0], cols[1] = cols[1], cols[0]
                vals[0], vals[1] = vals[1], vals[0]
            if cols[1] > cols[2]:
                cols[1], cols[2] = cols[2], cols[1]
                vals[1], vals[2] = vals[2], vals[1]
            if cols[0] > cols[1]:
                cols[0], cols[1] = cols[1], cols[0]
                vals[0], vals[1] = vals[1], vals[0]
        cols_arr = np.array(cols, dtype=np.int32)
        vals_arr = np.array(vals, dtype=np.float64)
        csr_set_row(J.data, J.indptr, J.indices, len(cols_arr), ap + 4,
                    cols_arr, vals_arr)

        # e_fd_out row
        cols = [ap + 5, dp + 7]
        vals = [1.0, -(1.0 - psys.exc_mask[g])]
        if efd_ctrl_col >= 0:
            cols.append(efd_ctrl_col)
            vals.append(-psys.exc_mask[g])
        if len(cols) == 2:
            if cols[0] > cols[1]:
                cols[0], cols[1] = cols[1], cols[0]
                vals[0], vals[1] = vals[1], vals[0]
        elif len(cols) == 3:
            if cols[0] > cols[1]:
                cols[0], cols[1] = cols[1], cols[0]
                vals[0], vals[1] = vals[1], vals[0]
            if cols[1] > cols[2]:
                cols[1], cols[2] = cols[2], cols[1]
                vals[1], vals[2] = vals[2], vals[1]
            if cols[0] > cols[1]:
                cols[0], cols[1] = cols[1], cols[0]
                vals[0], vals[1] = vals[1], vals[0]
        cols_arr = np.array(cols, dtype=np.int32)
        vals_arr = np.array(vals, dtype=np.float64)
        csr_set_row(J.data, J.indptr, J.indices, len(cols_arr), ap + 5,
                    cols_arr, vals_arr)

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

        # local indices
        e_qp = dp
        e_dp = dp + 1
        phi_1d = dp + 2
        phi_2q = dp + 3
        w = dp + 4
        delta = dp + 5
        pm_ref = dp + 6
        efd_ref = dp + 7

        v_q = ap
        v_d = ap + 1
        i_q = ap + 2
        i_d = ap + 3
        pm_out = ap + 4
        efd_out = ap + 5

        gen_idx = getattr(self, "device_index", -1)
        pm_ctrl_col = -1 if gen_idx < 0 else psys.gen_pm_ctrl_col[gen_idx]
        efd_ctrl_col = -1 if gen_idx < 0 else psys.gen_efd_ctrl_col[gen_idx]

        if power_injection:
            vm = dev + 2 * self.bus
            va = dev + 2 * self.bus + 1
        else:
            vr = dev + 2 * self.bus
            vi = dev + 2 * self.bus + 1

        # Differential rows
        coord.append([dp, [e_qp, e_dp, phi_1d, phi_2q, i_d, efd_out]])
        coord.append([dp + 1, [e_dp, phi_2q, i_q]])
        coord.append([dp + 2, [e_qp, phi_1d, i_d]])
        coord.append([dp + 3, [e_dp, phi_2q, i_q]])
        coord.append([dp + 4, [e_qp, e_dp, phi_1d, phi_2q, w, i_q, i_d, pm_out]])
        coord.append([dp + 5, [w]])

        # Generator algebraic currents
        coord.append([ap, [e_qp, phi_1d, v_q, i_d]])
        coord.append([ap + 1, [e_dp, phi_2q, v_d, i_q]])

        if power_injection:
            coord.append([ap + 2, [delta, v_d, vm, va]])
            coord.append([ap + 3, [delta, v_q, vm, va]])
            coord.append([dev + 2 * self.bus, [v_q, v_d, i_q, i_d]])
            coord.append([dev + 2 * self.bus + 1, [v_q, v_d, i_q, i_d]])
        else:
            coord.append([ap + 2, [delta, v_d, vr, vi]])
            coord.append([ap + 3, [delta, v_q, vr, vi]])
            coord.append([dev + 2 * self.bus, [delta, i_q, i_d]])
            coord.append([dev + 2 * self.bus + 1, [delta, i_q, i_d]])

        # Frozen reference rows
        coord.append([pm_ref, [pm_ref]])
        coord.append([efd_ref, [efd_ref]])

        # Blend rows
        cols_pm = [pm_ref, pm_out]
        if pm_ctrl_col >= 0:
            cols_pm.append(pm_ctrl_col)
        coord.append([pm_out, cols_pm])

        cols_efd = [efd_ref, efd_out]
        if efd_ctrl_col >= 0:
            cols_efd.append(efd_ctrl_col)
        coord.append([efd_out, cols_efd])

        for row_cols in coord:
            row_cols[1] = sorted(row_cols[1])

        return coord

    def preallocate_hessian(self, h_nnz, idxs, psys):

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
        pm_out = ap + 4

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
            h_nnz[w]['cols'].append([w, pm_out])

            h_nnz[w]['rows'].append(pm_out)
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

    def residual_jac(self, J, z, v, theta, idxs, power_injection):

        jac_genrou(z, v, theta, idxs, J.data, J.indptr,
                   J.indices, power_injection)

        return None

    def residual_hess(self, HESS, z, v, theta, idxs):

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

        hes_genrou(
            z,
            v,
            theta,
            idxs,
            H1.data,
            H1.indptr,
            H1.indices,
            H2.data,
            H2.indptr,
            H2.indices,
            H3.data,
            H3.indptr,
            H3.indices,
            H4.data,
            H4.indptr,
            H4.indices,
            H5.data,
            H5.indptr,
            H5.indices,
        )

        return None

@jit(nopython=True, cache=True)
def resdiff_genrou(F, z, v, theta, idxs, power_injection):

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
    S1 = theta[pp + 13]
    S2 = theta[pp + 14]

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

    # control outputs (blend results)
    p_m = z[ap + 4]
    e_fd = z[ap + 5]
    
    # auxiliary variables
    psi_de = (x_ddp - xl)/(x_dp - xl)*e_qp + \
        (x_dp - x_ddp)/(x_dp - xl)*phi_1d

    psi_qe = -(x_ddp - xl)/(x_qp - xl)*e_dp + \
        (x_qp - x_ddp)/(x_qp - xl)*phi_2q

    sat_a, sat_b = sat_coefficients_nb(S1, S2)
    psi2 = np.sqrt(psi_de*psi_de + psi_qe*psi_qe)
    Se = sat_se_nb(psi2, sat_a, sat_b)

    # equations
    F[dp] = (-e_qp + e_fd - (i_d - (-x_ddp + x_dp)*(-e_qp + i_d*(x_dp - xl) \
        + phi_1d)/((x_dp - xl)**2.0))*(x_d - x_dp) - Se*psi_de)/T_d0p
    F[dp + 1] = (-e_dp + (i_q - (-x_qdp + x_qp)*( e_dp + i_q*(x_qp - xl) \
        + phi_2q)/((x_qp - xl)**2.0))*(x_q - x_qp))/T_q0p
    F[dp + 2] = ( e_qp - i_d*(x_dp - xl) - phi_1d)/T_d0dp
    F[dp + 3] = (-e_dp - i_q*(x_qp - xl) - phi_2q)/T_q0dp
    F[dp + 4] = (p_m - psi_de*i_q + psi_qe*i_d - D*w)/(2.0*H)
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
def jac_genrou(z, v, theta, idxs, J_data, J_ptr, J_idx, power_injection):

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
    S1 = theta[pp + 13]
    S2 = theta[pp + 14]

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

    # blended outputs
    p_m = z[ap + 4]
    e_fd = z[ap + 5]

    # saturation terms
    psi_de = (x_ddp - xl)/(x_dp - xl)*e_qp + (x_dp - x_ddp)/(x_dp - xl)*phi_1d
    psi_qe = -(x_ddp - xl)/(x_qp - xl)*e_dp + (x_qp - x_ddp)/(x_qp - xl)*phi_2q

    sat_a, sat_b = sat_coefficients_nb(S1, S2)
    psi2 = np.sqrt(psi_de*psi_de + psi_qe*psi_qe)
    Se = sat_se_nb(psi2, sat_a, sat_b)

    if sat_b == 0.0 or psi2 <= sat_a or psi2 == 0.0:
        dSe_dpsi = 0.0
    else:
        g = psi2 - sat_a
        dSe_dpsi = sat_b * (2.0 * g * psi2 - g * g) / (psi2 * psi2)

    if psi2 == 0.0:
        dpsi_dpsi_de = 0.0
        dpsi_dpsi_qe = 0.0
    else:
        dpsi_dpsi_de = psi_de / psi2
        dpsi_dpsi_qe = psi_qe / psi2

    dpsi_de_deqp = (x_ddp - xl) / (x_dp - xl)
    dpsi_de_phi1d = (x_dp - x_ddp) / (x_dp - xl)
    dpsi_qe_dedp = -(x_ddp - xl) / (x_qp - xl)
    dpsi_qe_phi2q = (x_qp - x_ddp) / (x_qp - xl)

    dSe_dpsi_de = dSe_dpsi * dpsi_dpsi_de
    dSe_dpsi_qe = dSe_dpsi * dpsi_dpsi_qe

    dT_dpsi_de = -(dSe_dpsi_de * psi_de + Se)
    dT_dpsi_qe = -(dSe_dpsi_qe * psi_de)

    dT_deqp = dT_dpsi_de * dpsi_de_deqp
    dT_dphi1d = dT_dpsi_de * dpsi_de_phi1d
    dT_dedp = dT_dpsi_qe * dpsi_qe_dedp
    dT_dphi2q = dT_dpsi_qe * dpsi_qe_phi2q

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
    pm_idx = ap + 4
    efd_idx = ap + 5

    if power_injection:
        vm_idx = dev + 2*bus
        va_idx = dev + 2*bus + 1
    else:
        vr_idx = dev + 2*bus
        vi_idx = dev + 2*bus + 1

    # differential rows
    cols = np.empty(6, dtype=np.int32)
    vals = np.empty(6, dtype=np.float64)
    cols[0] = e_qp_idx
    vals[0] = (-(x_d - x_dp)*(-x_ddp + x_dp)*(x_dp - xl)**(-2.0) - 1)/T_d0p + dT_deqp / T_d0p
    cols[1] = e_dp_idx
    vals[1] = dT_dedp / T_d0p
    cols[2] = phi_1d_idx
    vals[2] = (x_d - x_dp)*(-x_ddp + x_dp)*(x_dp - xl)**(-2.0)/T_d0p + dT_dphi1d / T_d0p
    cols[3] = phi_2q_idx
    vals[3] = dT_dphi2q / T_d0p
    cols[4] = i_d_idx
    vals[4] = -(x_d - x_dp)*(-(-x_ddp + x_dp)*(x_dp - xl)**(-1.0) + 1)/T_d0p
    cols[5] = efd_idx
    vals[5] = 1.0/T_d0p
    csr_set_row(J_data, J_ptr, J_idx, 6, dp, cols, vals)

    cols = np.empty(3, dtype=np.int32)
    vals = np.empty(3, dtype=np.float64)
    cols[0] = e_dp_idx
    vals[0] = (-(x_q - x_qp)*(-x_qdp + x_qp)*(x_qp - xl)**(-2.0) - 1)/T_q0p
    cols[1] = phi_2q_idx
    vals[1] = -(x_q - x_qp)*(-x_qdp + x_qp)*(x_qp - xl)**(-2.0)/T_q0p
    cols[2] = i_q_idx
    vals[2] = (x_q - x_qp)*(-(-x_qdp + x_qp)*(x_qp - xl)**(-1.0) + 1)/T_q0p
    csr_set_row(J_data, J_ptr, J_idx, 3, dp + 1, cols, vals)

    cols = np.empty(3, dtype=np.int32)
    vals = np.empty(3, dtype=np.float64)
    cols[0] = e_qp_idx
    vals[0] = 1.0/T_d0dp
    cols[1] = phi_1d_idx
    vals[1] = -1.0/T_d0dp
    cols[2] = i_d_idx
    vals[2] = (-x_dp + xl)/T_d0dp
    csr_set_row(J_data, J_ptr, J_idx, 3, dp + 2, cols, vals)

    cols = np.empty(3, dtype=np.int32)
    vals = np.empty(3, dtype=np.float64)
    cols[0] = e_dp_idx
    vals[0] = -1.0/T_q0dp
    cols[1] = phi_2q_idx
    vals[1] = -1.0/T_q0dp
    cols[2] = i_q_idx
    vals[2] = (-x_qp + xl)/T_q0dp
    csr_set_row(J_data, J_ptr, J_idx, 3, dp + 3, cols, vals)

    cols = np.empty(8, dtype=np.int32)
    vals = np.empty(8, dtype=np.float64)
    cols[0] = e_qp_idx
    vals[0] = -0.5*i_q*(x_ddp - xl)/(H*(x_dp - xl))
    cols[1] = e_dp_idx
    vals[1] = 0.5*i_d*(-x_ddp + xl)/(H*(x_qp - xl))
    cols[2] = phi_1d_idx
    vals[2] = -0.5*i_q*(-x_ddp + x_dp)/(H*(x_dp - xl))
    cols[3] = phi_2q_idx
    vals[3] = 0.5*i_d*(-x_ddp + x_qp)/(H*(x_qp - xl))
    cols[4] = w_idx
    vals[4] = -0.5*D/H
    cols[5] = i_q_idx
    vals[5] = 0.5*(-e_qp*(x_ddp - xl)/(x_dp - xl) - phi_1d*(-x_ddp + x_dp)/(x_dp - xl))/H
    cols[6] = i_d_idx
    vals[6] = 0.5*(e_dp*(-x_ddp + xl)/(x_qp - xl) + phi_2q*(-x_ddp + x_qp)/(x_qp - xl))/H
    cols[7] = pm_idx
    vals[7] = 0.5/H
    csr_set_row(J_data, J_ptr, J_idx, 8, dp + 4, cols, vals)

    cols = np.empty(1, dtype=np.int32)
    vals = np.empty(1, dtype=np.float64)
    cols[0] = w_idx
    vals[0] = 120.0*np.pi
    csr_set_row(J_data, J_ptr, J_idx, 1, dp + 5, cols, vals)

    # algebraic rows
    cols = np.empty(4, dtype=np.int32)
    vals = np.empty(4, dtype=np.float64)
    cols[0] = e_qp_idx
    vals[0] = -(x_ddp - xl)/(x_ddp*(x_dp - xl))
    cols[1] = phi_1d_idx
    vals[1] = -(-x_ddp + x_dp)/(x_ddp*(x_dp - xl))
    cols[2] = v_q_idx
    vals[2] = 1.0/x_ddp
    cols[3] = i_d_idx
    vals[3] = 1.0
    csr_set_row(J_data, J_ptr, J_idx, 4, ap, cols, vals)

    cols = np.empty(4, dtype=np.int32)
    vals = np.empty(4, dtype=np.float64)
    cols[0] = e_dp_idx
    vals[0] = -(-x_qdp + xl)/(x_qdp*(x_qp - xl))
    cols[1] = phi_2q_idx
    vals[1] = -(-x_qdp + x_qp)/(x_qdp*(x_qp - xl))
    cols[2] = v_d_idx
    vals[2] = -1.0/x_qdp
    cols[3] = i_q_idx
    vals[3] = 1.0
    csr_set_row(J_data, J_ptr, J_idx, 4, ap + 1, cols, vals)

    if power_injection:
        cols = np.empty(4, dtype=np.int32)
        vals = np.empty(4, dtype=np.float64)
        cols[0] = delta_idx
        vals[0] = -vm*np.cos(delta - va)
        cols[1] = v_d_idx
        vals[1] = 1.0
        cols[2] = vm_idx
        vals[2] = -np.sin(delta - va)
        cols[3] = va_idx
        vals[3] = vm*np.cos(delta - va)
        csr_set_row(J_data, J_ptr, J_idx, 4, ap + 2, cols, vals)

        cols = np.empty(4, dtype=np.int32)
        vals = np.empty(4, dtype=np.float64)
        cols[0] = delta_idx
        vals[0] = vm*np.sin(delta - va)
        cols[1] = v_q_idx
        vals[1] = 1.0
        cols[2] = vm_idx
        vals[2] = -np.cos(delta - va)
        cols[3] = va_idx
        vals[3] = -vm*np.sin(delta - va)
        csr_set_row(J_data, J_ptr, J_idx, 4, ap + 3, cols, vals)

        cols = np.empty(4, dtype=np.int32)
        vals = np.empty(4, dtype=np.float64)
        cols[0] = v_q_idx
        vals[0] =  i_q
        cols[1] = v_d_idx
        vals[1] =  i_d
        cols[2] = i_q_idx
        vals[2] =  v_q
        cols[3] = i_d_idx
        vals[3] =  v_d
        csr_set_row(J_data, J_ptr, J_idx, 4, dev + 2*bus, cols, vals)

        cols = np.empty(4, dtype=np.int32)
        vals = np.empty(4, dtype=np.float64)
        cols[0] = v_q_idx
        vals[0] =  i_d
        cols[1] = v_d_idx
        vals[1] = -i_q
        cols[2] = i_q_idx
        vals[2] = -v_d
        cols[3] = i_d_idx
        vals[3] =  v_q
        csr_set_row(J_data, J_ptr, J_idx, 4, dev + 2*bus + 1, cols, vals)
    else:
        cols = np.empty(4, dtype=np.int32)
        vals = np.empty(4, dtype=np.float64)
        cols[0] = delta_idx
        vals[0] = -vr*np.cos(delta) - vi*np.sin(delta)
        cols[1] = v_d_idx
        vals[1] = 1.0
        cols[2] = vr_idx
        vals[2] = -np.sin(delta)
        cols[3] = vi_idx
        vals[3] = np.cos(delta)
        csr_set_row(J_data, J_ptr, J_idx, 4, ap + 2, cols, vals)

        cols = np.empty(4, dtype=np.int32)
        vals = np.empty(4, dtype=np.float64)
        cols[0] = delta_idx
        vals[0] = vr*np.sin(delta) - vi*np.cos(delta)
        cols[1] = v_q_idx
        vals[1] = 1.0
        cols[2] = vr_idx
        vals[2] = -np.cos(delta)
        cols[3] = vi_idx
        vals[3] = -np.sin(delta)
        csr_set_row(J_data, J_ptr, J_idx, 4, ap + 3, cols, vals)

        cols = np.empty(3, dtype=np.int32)
        vals = np.empty(3, dtype=np.float64)
        cols[0] = delta_idx
        vals[0] = i_d*np.cos(delta) - i_q*np.sin(delta)
        cols[1] = i_q_idx
        vals[1] = np.cos(delta)
        cols[2] = i_d_idx
        vals[2] = np.sin(delta)
        csr_set_row(J_data, J_ptr, J_idx, 3, dev + 2*bus, cols, vals)

        cols = np.empty(3, dtype=np.int32)
        vals = np.empty(3, dtype=np.float64)
        cols[0] = delta_idx
        vals[0] = i_d*np.sin(delta) + i_q*np.cos(delta)
        cols[1] = i_q_idx
        vals[1] = np.sin(delta)
        cols[2] = i_d_idx
        vals[2] = -np.cos(delta)
        csr_set_row(J_data, J_ptr, J_idx, 3, dev + 2*bus + 1, cols, vals)

@jit(nopython=True, cache=True)
def hes_genrou(z, v, theta, idxs,
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

    # control outputs live in the generator algebraic block
    pm_idx = ap + 4
    efd_idx = ap + 5
    p_m = z[pm_idx]
    e_fd = z[efd_idx]
    
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
