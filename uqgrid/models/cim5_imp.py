import numpy as np
from scipy import optimize
from uqgrid.core.base_models import Motor
from uqgrid.utils.tools import csr_add_row, csr_set_row
ws = 2*np.pi*60

class MotCIM5(Motor):
    def __init__(self, id_tag, ra, xa, xm, r1, x1, H, D):

        self.ra = ra
        self.xa = xa
        self.xm = xm
        self.r1 = r1
        self.x1 = x1
        self.H = H
        self.D = D

        self.tp = (x1 + xm)/(r1*ws)
        self.x0 = (xa + xm)
        self.x_p = xa + (x1*xm)/(x1 + xm)

        state_list = ['e_dqp', 'e_qp', 's', 'i_ds', 'i_qs', 't_m', 'ysh']
        param_list = [
            'ra', 'xa', 'xm', 'r1', 'x1', 'H', 'D', 'tp', 'x0', 'x_p'
        ]

        Motor.__init__(self, id_tag, 7, 5, 2, len(param_list), state_list)

    def initialize_theta(self, theta):

        idx = self.par_ptr

        theta[idx] = self.ra
        theta[idx + 1] = self.xa
        theta[idx + 2] = self.xm
        theta[idx + 3] = self.r1
        theta[idx + 4] = self.x1
        theta[idx + 5] = self.H
        theta[idx + 6] = self.D
        theta[idx + 7] = self.tp
        theta[idx + 8] = self.x0
        theta[idx + 9] = self.x_p

    def init_sens(self, x, v, va, p0, q0, weight):

        e_dp = x[0]
        e_qp = x[1]
        s = x[2]
        t_m = x[3]
        ysh = x[4]
        i_ds = x[5]
        i_qs = x[6]

        tp = self.tp
        x_0 = self.x0
        x_p = self.x_p
        Hm = self.H
        ra = self.ra

        v_ds = -v*np.sin(va)
        v_qs = v*np.cos(va)

        J = np.array(
            [[-1.0/tp, s*ws, e_qp*ws, 0, 0, 0, -1.0*(x_0 - x_p)/tp],
             [-s*ws, -1.0/tp, -e_dp*ws, 0, 0, -1.0*(-x_0 + x_p)/tp, 0], [
                 -0.5*i_ds/Hm, -0.5*i_qs/Hm, 0, 0.5/Hm, 0, -0.5*e_dp/Hm,
                 -0.5*e_qp/Hm
             ], [1, 0, 0, 0, 0, ra, -x_p], [0, 1, 0, 0, 0, x_p, ra],
             [0, 0, 0, 0, 0, -v*np.sin(va),
              v*np.cos(va)], [0, 0, 0, 0, v**2, v*np.cos(va), v*np.sin(va)]])

        JA = np.array([0, 0, 0, 0, 0, -p0, -q0])

        # Hessian
        nstate = 7
        H = nstate*[None]
        H[0] = np.zeros((nstate, nstate))
        H[1] = np.zeros((nstate, nstate))
        H[2] = np.zeros((nstate, nstate))

        H[0][1, 2] = ws
        H[0][2, 1] = ws
        H[1][0, 2] = -ws
        H[1][2, 0] = -ws
        H[2][0, 5] = -0.5/Hm
        H[2][1, 6] = -0.5/Hm
        H[2][5, 0] = -0.5/Hm
        H[2][6, 1] = -0.5/Hm

        return J, JA, H

    def residualFinit(self, x, v, va, p0, q0):

        F = np.zeros(self.initdim)

        e_dp = x[0]
        e_qp = x[1]
        s = x[2]
        t_m = x[3]
        ysh = x[4]
        i_ds = x[5]
        i_qs = x[6]

        tp = self.tp
        x0 = self.x0
        x_p = self.x_p
        Hm = self.H
        ra = self.ra

        v_ds = -v*np.sin(va)
        v_qs = v*np.cos(va)

        F[0] = (-1.0/tp)*(e_dp + (x0 - x_p)*i_qs) + s*ws*e_qp
        F[1] = (-1.0/tp)*(e_qp - (x0 - x_p)*i_ds) - s*ws*e_dp

        F[2] = (1.0/(2.0*Hm))*(t_m - e_dp*i_ds - e_qp*i_qs)

        F[3] = ra*i_ds - x_p*i_qs + e_dp - v_ds
        F[4] = ra*i_qs + x_p*i_ds + e_qp - v_qs

        F[5] = v_ds*i_ds + v_qs*i_qs + p0
        F[6] = (v_qs*i_ds - v_ds*i_qs + ysh*v*v) + q0

        return F

    def initialize(self, vm, va, p, q, x, y, psys):

        x0 = np.ones(self.initdim)
        w = self.weight
        x0[2] = 0.01
        sol = optimize.root(
            self.residualFinit,
            x0,
            args=(vm, va, w*p, w*q),
            method='krylov',
            options={
                'xtol': 1e-10,
                'disp': False
            })
        assert sol.success == True
        self.initialized = True
        x[self.dif_ptr:self.dif_ptr + 5] = sol.x[0:5]
        y[self.alg_ptr:self.alg_ptr + 2] = sol.x[5:7]

        return None

    def initialize_sens(self, vm, va, p, q, z, u, v, psys, diff_size):

        mot_x = np.zeros(self.initdim)
        mot_x[0:5] = z[self.dif_ptr:self.dif_ptr + 5]
        mot_x[5:7] = z[self.alg_ptr + diff_size:self.alg_ptr + diff_size + 2]
        w = self.weight

        # compute initial sensitivity vectors
        J, JA, H = self.init_sens(mot_x, vm, va, p, q, w)
        u_mot = -np.dot(np.linalg.inv(J), JA)

        # set sensitivities
        u[self.dif_ptr:self.dif_ptr + 5] = u_mot[0:5]
        u[self.alg_ptr + diff_size:self.alg_ptr + diff_size + 2] = u_mot[5:7]

        # second order
        b = np.zeros(7)
        for i in range(len(H)):
            if H[i] is not None:
                b[i] += u_mot.dot(H[i].dot(u_mot))

        v_mot = -np.dot(np.linalg.inv(J), b)
        v[self.dif_ptr:self.dif_ptr + 5] = v_mot[0:5]
        v[self.alg_ptr + diff_size:self.alg_ptr + diff_size + 2] = v_mot[5:7]

        return None

    def residual_diff(self, F, z, v, theta, idxs, ctrl_idx, ctrl_var):

        dp = idxs[0]
        ap = idxs[1]

        # paramters
        tp = self.tp
        x_0 = self.x0
        x_p = self.x_p
        Hm = self.H
        ra = self.ra

        # states
        e_dpm = z[dp]
        e_qpm = z[dp + 1]
        s = z[dp + 2]
        t_m = z[dp + 3]
        ysh = z[dp + 4]
        i_dm = z[ap]
        i_qm = z[ap + 1]

        vm = v[2*self.bus]
        va = v[2*self.bus + 1]

        v_dm = -vm*np.sin(va)
        v_qm = vm*np.cos(va)

        F[dp] = (-1.0/tp)*(e_dpm + (x_0 - x_p)*i_qm) + s*ws*e_qpm
        F[dp + 1] = (-1.0/tp)*(e_qpm - (x_0 - x_p)*i_dm) - s*ws*e_dpm
        F[dp + 2] = (1.0/(2.0*Hm))*(t_m - e_dpm*i_dm - e_qpm*i_qm)
        F[dp + 3] = 0.0
        F[dp + 4] = 0.0

        F[ap] = ra*i_dm - x_p*i_qm + e_dpm - v_dm
        F[ap + 1] = ra*i_qm + x_p*i_dm + e_qpm - v_qm

        return None

    def residual_pinj(self, F, z, v, theta, idxs, alpha=0.0):

        dp = idxs[0]
        ap = idxs[1]

        ysh = z[dp + 4]
        i_dm = z[ap]
        i_qm = z[ap + 1]

        vm = v[2*self.bus]
        va = v[2*self.bus + 1]

        v_dm = -vm*np.sin(va)
        v_qm = vm*np.cos(va)

        F[2*self.bus] -= (v_dm*i_dm + v_qm*i_qm)
        F[2*self.bus + 1] -= (v_qm*i_dm - v_dm*i_qm)
        F[2*self.bus + 1] -= ysh*(vm*vm)

        return None

    def preallocate_jacobian(self, idxs, psys):

        coord = []

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]

        # these are INDEXES
        e_dpm = dp
        e_qpm = dp + 1
        s = dp + 2
        t_m = dp + 3
        ysh = dp + 4
        i_dm = ap
        i_qm = ap + 1

        vm = dev + 2*self.bus
        va = dev + 2*self.bus + 1

        # first row
        row = dp
        cols = [e_dpm, e_qpm, s, i_qm]
        coord.append([row, cols])

        # second row
        row = dp + 1
        cols = [e_dpm, e_qpm, s, i_dm]
        coord.append([row, cols])

        # third row
        row = dp + 2
        cols = [e_dpm, e_qpm, t_m, i_dm, i_qm]
        coord.append([row, cols])

        row = ap
        cols = [e_dpm, i_dm, i_qm, vm, va]
        coord.append([row, cols])

        row = ap + 1
        cols = [e_qpm, i_dm, i_qm, vm, va]
        coord.append([row, cols])

        row = dev + 2*self.bus
        cols = [i_dm, i_qm]
        coord.append([row, cols])

        row = dev + 2*self.bus + 1
        cols = [ysh, i_dm, i_qm]
        coord.append([row, cols])

        return coord

    def residual_jac(self, J, z, v, theta, idxs, ctrl_idx, ctrl_var):

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]

        # parameters
        tp = self.tp
        x_0 = self.x0
        x_p = self.x_p
        Hm = self.H
        ra = self.ra

        # states
        e_dpm = z[dp]
        e_qpm = z[dp + 1]
        s = z[dp + 2]
        t_m = z[dp + 3]
        ysh = z[dp + 4]
        i_dm = z[ap]
        i_qm = z[ap + 1]

        vm = v[2*self.bus]
        va = v[2*self.bus + 1]

        # indeces
        e_dpm_idx = dp
        e_qpm_idx = dp + 1
        s_idx = dp + 2
        t_m_idx = dp + 3
        ysh_idx = dp + 4
        i_dm_idx = ap
        i_qm_idx = ap + 1
        vm_idx = dev + 2*self.bus
        va_idx = dev + 2*self.bus + 1

        # column and value vectors
        col = np.zeros(10)
        val = np.zeros(10)

        # first row
        row = dp
        col[0] = e_dpm_idx
        val[0] = -1.0/tp
        col[1] = e_qpm_idx
        val[1] = s*ws
        col[2] = s_idx
        val[2] = e_qpm*ws
        col[3] = i_qm_idx
        val[3] = -1.0*(x_0 - x_p)/tp
        csr_set_row(J.data, J.indptr, J.indices, 4, row, col, val)

        # second row
        row = dp + 1
        col[0] = e_dpm_idx
        val[0] = -s*ws
        col[1] = e_qpm_idx
        val[1] = -1.0/tp
        col[2] = s_idx
        val[2] = -e_dpm*ws
        col[3] = i_dm_idx
        val[3] = -1.0*(-x_0 + x_p)/tp
        csr_set_row(J.data, J.indptr, J.indices, 4, row, col, val)

        # third row
        row = dp + 2
        col[0] = e_dpm_idx
        val[0] = -0.5*i_dm/Hm
        col[1] = e_qpm_idx
        val[1] = -0.5*i_qm/Hm
        col[2] = t_m_idx
        val[2] = 0.5/Hm
        col[3] = i_dm_idx
        val[3] = -0.5*e_dpm/Hm
        col[4] = i_qm_idx
        val[4] = -0.5*e_qpm/Hm
        csr_set_row(J.data, J.indptr, J.indices, 5, row, col, val)

        # algebraic fist  row
        row = ap
        col[0] = e_dpm_idx
        val[0] = 1.0
        col[1] = i_dm_idx
        val[1] = ra
        col[2] = i_qm_idx
        val[2] = -x_p
        col[3] = vm_idx
        val[3] = np.sin(va)
        col[4] = va_idx
        val[4] = vm*np.cos(va)
        csr_set_row(J.data, J.indptr, J.indices, 5, row, col, val)

        # algebraic fist  row
        row = ap + 1
        col[0] = e_qpm_idx
        val[0] = 1.0
        col[1] = i_dm_idx
        val[1] = x_p
        col[2] = i_qm_idx
        val[2] = ra
        col[3] = vm_idx
        val[3] = -np.cos(va)
        col[4] = va_idx
        val[4] = vm*np.sin(va)
        csr_set_row(J.data, J.indptr, J.indices, 5, row, col, val)

        # POWER INJECTION (SET ENTRIES)
        alpha = 1.0

        row = dev + 2*self.bus
        col[0] = i_dm_idx
        col[0] = i_dm_idx
        val[0] = alpha*vm*np.sin(va)
        col[1] = i_qm_idx
        val[1] = -alpha*vm*np.cos(va)
        csr_set_row(J.data, J.indptr, J.indices, 2, row, col, val)

        row = dev + 2*self.bus + 1
        col[0] = ysh_idx
        val[0] = -vm*vm
        col[1] = i_dm_idx
        val[1] = -alpha*vm*np.cos(va)
        col[2] = i_qm_idx
        val[2] = -alpha*vm*np.sin(va)
        csr_set_row(J.data, J.indptr, J.indices, 3, row, col, val)

        # POWER INJECTION (ADD ENTRIES)
        row = dev + 2*self.bus
        col[0] = dev + 2*self.bus
        val[0] = -alpha*(-i_dm*np.sin(va) + i_qm*np.cos(va))
        col[1] = dev + 2*self.bus + 1
        val[1] = -alpha*(-i_dm*vm*np.cos(va) - i_qm*vm*np.sin(va))
        csr_add_row(J.data, J.indptr, J.indices, 2, row, col, val)

        row = dev + 2*self.bus + 1
        col[0] = dev + 2*self.bus
        val[0] = -alpha*(i_dm*np.cos(va) + i_qm*np.sin(va)) - 2*ysh*vm
        col[1] = dev + 2*self.bus + 1
        val[1] = -alpha*(-i_dm*vm*np.sin(va) + i_qm*vm*np.cos(va))
        csr_add_row(J.data, J.indptr, J.indices, 2, row, col, val)

    def preallocate_hessian(self, h_nnz, idxs, psys):

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]

        # these are INDEXES
        e_dpm = dp
        e_qpm = dp + 1
        s = dp + 2
        t_m = dp + 3
        ysh = dp + 4
        i_dm = ap
        i_qm = ap + 1

        vm = dev + 2*self.bus
        va = dev + 2*self.bus + 1

        # F0
        h_nnz[dp]['rows'].append(e_qpm)
        h_nnz[dp]['cols'].append([s])

        h_nnz[dp]['rows'].append(s)
        h_nnz[dp]['cols'].append([e_qpm])

        # F1
        h_nnz[dp + 1]['rows'].append(e_dpm)
        h_nnz[dp + 1]['cols'].append([s])

        h_nnz[dp + 1]['rows'].append(s)
        h_nnz[dp + 1]['cols'].append([e_dpm])

        # F2
        h_nnz[dp + 2]['rows'].append(e_dpm)
        h_nnz[dp + 2]['cols'].append([i_dm])

        h_nnz[dp + 2]['rows'].append(e_qpm)
        h_nnz[dp + 2]['cols'].append([i_qm])

        h_nnz[dp + 2]['rows'].append(i_dm)
        h_nnz[dp + 2]['cols'].append([e_dpm])

        h_nnz[dp + 2]['rows'].append(i_qm)
        h_nnz[dp + 2]['cols'].append([e_qpm])

        # F3
        h_nnz[ap]['rows'].append(vm)
        h_nnz[ap]['cols'].append([va])

        h_nnz[ap]['rows'].append(va)
        h_nnz[ap]['cols'].append([vm, va])

        # F4
        h_nnz[ap + 1]['rows'].append(vm)
        h_nnz[ap + 1]['cols'].append([va])

        h_nnz[ap + 1]['rows'].append(va)
        h_nnz[ap + 1]['cols'].append([vm, va])

        # F5
        h_nnz[dev + 2*self.bus]['rows'].append(i_dm)
        h_nnz[dev + 2*self.bus]['cols'].append([vm, va])

        h_nnz[dev + 2*self.bus]['rows'].append(i_qm)
        h_nnz[dev + 2*self.bus]['cols'].append([vm, va])

        h_nnz[dev + 2*self.bus]['rows'].append(vm)
        h_nnz[dev + 2*self.bus]['cols'].append([i_dm, i_qm, va])

        h_nnz[dev + 2*self.bus]['rows'].append(va)
        h_nnz[dev + 2*self.bus]['cols'].append([i_dm, i_qm, vm, va])

        # F6
        h_nnz[dev + 2*self.bus + 1]['rows'].append(ysh)
        h_nnz[dev + 2*self.bus + 1]['cols'].append([vm])

        h_nnz[dev + 2*self.bus + 1]['rows'].append(i_dm)
        h_nnz[dev + 2*self.bus + 1]['cols'].append([vm, va])

        h_nnz[dev + 2*self.bus + 1]['rows'].append(i_qm)
        h_nnz[dev + 2*self.bus + 1]['cols'].append([vm, va])

        h_nnz[dev + 2*self.bus + 1]['rows'].append(vm)
        h_nnz[dev + 2*self.bus + 1]['cols'].append([ysh, i_dm, i_qm, vm, va])

        h_nnz[dev + 2*self.bus + 1]['rows'].append(va)
        h_nnz[dev + 2*self.bus + 1]['cols'].append([i_dm, i_qm, vm, va])

    def residual_hess(self, HESS, z, v, theta, idxs, ctrl_idx, ctrl_var):

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]
        pp = idxs[3]
        bus = idxs[4]

        H0 = HESS[dp]
        H1 = HESS[dp + 1]
        H2 = HESS[dp + 2]
        H3 = HESS[dp + 3]
        H4 = HESS[dp + 4]
        H5 = HESS[ap]
        H6 = HESS[ap + 1]
        H7 = HESS[dev + 2*bus]
        H8 = HESS[dev + 2*bus + 1]

        # parameters
        tp = self.tp
        x_0 = self.x0
        x_p = self.x_p
        Hm = self.H
        ra = self.ra

        # states
        e_dp = z[dp]
        e_qp = z[dp + 1]
        s = z[dp + 2]
        t_m = z[dp + 3]
        ysh = z[dp + 4]
        i_ds = z[ap]
        i_qs = z[ap + 1]

        vm = v[2*bus]
        va = v[2*bus + 1]

        # indeces
        e_dp_idx = dp
        e_qp_idx = dp + 1
        s_idx = dp + 2
        t_m_idx = dp + 3
        ysh_idx = dp + 4
        i_ds_idx = ap
        i_qs_idx = ap + 1
        vm_idx = dev + 2*bus
        va_idx = dev + 2*bus + 1

        # column and value vectors
        col = np.zeros(10)
        val = np.zeros(10)

        ### HESSIAN OF F0 ###

        row = e_qp_idx
        col[0] = s_idx
        val[0] = ws
        csr_set_row(H0.data, H0.indptr, H0.indices, 1, row, col, val)

        row = s_idx
        col[0] = e_qp_idx
        val[0] = ws
        csr_set_row(H0.data, H0.indptr, H0.indices, 1, row, col, val)

        ### HESSIAN OF F1 ###

        row = e_dp_idx
        col[0] = s_idx
        val[0] = -ws
        csr_set_row(H1.data, H1.indptr, H1.indices, 1, row, col, val)

        row = s_idx
        col[0] = e_dp_idx
        val[0] = -ws
        csr_set_row(H1.data, H1.indptr, H1.indices, 1, row, col, val)

        ### HESSIAN OF F2 ###

        row = e_dp_idx
        col[0] = i_ds_idx
        val[0] = -0.5/Hm
        csr_set_row(H2.data, H2.indptr, H2.indices, 1, row, col, val)

        row = e_qp_idx
        col[0] = i_qs_idx
        val[0] = -0.5/Hm
        csr_set_row(H2.data, H2.indptr, H2.indices, 1, row, col, val)

        row = i_ds_idx
        col[0] = e_dp_idx
        val[0] = -0.5/Hm
        csr_set_row(H2.data, H2.indptr, H2.indices, 1, row, col, val)

        row = i_qs_idx
        col[0] = e_qp_idx
        val[0] = -0.5/Hm
        csr_set_row(H2.data, H2.indptr, H2.indices, 1, row, col, val)

        ### HESSIAN OF F3 ###

        ### HESSIAN OF F4 ###

        ### HESSIAN OF F5 ###

        row = vm_idx
        col[0] = va_idx
        val[0] = np.cos(va)
        csr_set_row(H5.data, H5.indptr, H5.indices, 1, row, col, val)

        row = va_idx
        col[0] = vm_idx
        val[0] = np.cos(va)
        col[1] = va_idx
        val[1] = -vm*np.sin(va)
        csr_set_row(H5.data, H5.indptr, H5.indices, 2, row, col, val)

        ### HESSIAN OF F6 ###

        row = vm_idx
        col[0] = va_idx
        val[0] = np.sin(va)
        csr_set_row(H6.data, H6.indptr, H6.indices, 1, row, col, val)

        row = va_idx
        col[0] = vm_idx
        val[0] = np.sin(va)
        col[1] = va_idx
        val[1] = vm*np.cos(va)
        csr_set_row(H6.data, H6.indptr, H6.indices, 2, row, col, val)

        ### HESSIAN OF F7 ###

        row = i_ds_idx
        col[0] = vm_idx
        val[0] = np.sin(va)
        col[1] = va_idx
        val[1] = vm*np.cos(va)
        csr_set_row(H7.data, H7.indptr, H7.indices, 2, row, col, val)

        row = i_qs_idx
        col[0] = vm_idx
        val[0] = -np.cos(va)
        col[1] = va_idx
        val[1] = vm*np.sin(va)
        csr_set_row(H7.data, H7.indptr, H7.indices, 2, row, col, val)

        row = vm_idx
        col[0] = i_ds_idx
        val[0] = np.sin(va)
        col[1] = i_qs_idx
        val[1] = -np.cos(va)
        csr_set_row(H7.data, H7.indptr, H7.indices, 2, row, col, val)

        row = vm_idx
        col[0] = va_idx
        val[0] = i_ds*np.cos(va) + i_qs*np.sin(va)
        csr_add_row(H7.data, H7.indptr, H7.indices, 1, row, col, val)

        row = va_idx
        col[0] = i_ds_idx
        val[0] = vm*np.cos(va)
        col[1] = i_qs_idx
        val[1] = vm*np.sin(va)
        csr_set_row(H7.data, H7.indptr, H7.indices, 2, row, col, val)

        row = va_idx
        col[0] = vm_idx
        val[0] = i_ds*np.cos(va) + i_qs*np.sin(va)
        col[1] = va_idx
        val[1] = vm*(-i_ds*np.sin(va) + i_qs*np.cos(va))
        csr_add_row(H7.data, H7.indptr, H7.indices, 2, row, col, val)

        ### HESSIAN OF F8 ###

        row = ysh_idx
        col[0] = vm_idx
        val[0] = -2*vm
        csr_set_row(H8.data, H8.indptr, H8.indices, 1, row, col, val)

        row = i_ds_idx
        col[0] = vm_idx
        val[0] = -np.cos(va)
        col[1] = va_idx
        val[1] = vm*np.sin(va)
        csr_set_row(H8.data, H8.indptr, H8.indices, 2, row, col, val)

        row = i_qs_idx
        col[0] = vm_idx
        val[0] = -np.sin(va)
        col[1] = va_idx
        val[1] = -vm*np.cos(va)
        csr_set_row(H8.data, H8.indptr, H8.indices, 2, row, col, val)

        row = vm_idx
        col[0] = ysh_idx
        val[0] = -2*vm
        col[1] = i_ds_idx
        val[1] = -np.cos(va)
        col[2] = i_qs_idx
        val[2] = -np.sin(va)
        csr_set_row(H8.data, H8.indptr, H8.indices, 3, row, col, val)

        row = va_idx
        col[0] = i_ds_idx
        val[0] = vm*np.sin(va)
        col[1] = i_qs_idx
        val[1] = -vm*np.cos(va)
        csr_set_row(H8.data, H8.indptr, H8.indices, 2, row, col, val)

        row = vm_idx
        col[0] = vm_idx
        val[0] = -2*ysh
        col[1] = va_idx
        val[1] = i_ds*np.sin(va) - i_qs*np.cos(va)
        csr_add_row(H8.data, H8.indptr, H8.indices, 2, row, col, val)

        row = va_idx
        col[0] = vm_idx
        val[0] = i_ds*np.sin(va) - i_qs*np.cos(va)
        col[1] = va_idx
        val[1] = vm*(i_ds*np.cos(va) + i_qs*np.sin(va))
        csr_add_row(H8.data, H8.indptr, H8.indices, 2, row, col, val)

def residualFinit_cim5(x, theta, v, va, p0, q0):

    e_dp = x[0]
    e_qp = x[1]
    s    = x[2]
    t_m  = x[3]
    ysh  = x[4]
    i_ds = x[5]
    i_qs = x[6]

    tp = theta[0]
    x0 = theta[1]
    x_p = theta[2]
    Hm = theta[3]
    ra = theta[4]

    v_ds = -v*np.sin(va)
    v_qs = v*np.cos(va)

    F0 = (-1.0/tp)*(e_dp + (x0 - x_p)*i_qs) + s*ws*e_qp
    F1 = (-1.0/tp)*(e_qp - (x0 - x_p)*i_ds) - s*ws*e_dp
    F2 = (t_m - e_dp*i_ds - e_qp*i_qs)
    F3 = ra*i_ds - x_p*i_qs + e_dp - v_ds
    F4 = ra*i_qs + x_p*i_ds + e_qp - v_qs
    F5 = v_ds*i_ds + v_qs*i_qs + p0
    F6 = (v_qs*i_ds - v_ds*i_qs + ysh*v*v) + q0

    return np.array([F0, F1, F2, F3, F4, F5, F6])



########## tools #####################

if __name__ == "__main__":
    import sympy as sp
    from sympy.printing.pycode import pycode

    # states
    e_dp = sp.symbols("e_dp")
    e_qp = sp.symbols("e_qp")
    s = sp.symbols("s")
    ysh = sp.symbols("ysh")
    i_ds = sp.symbols("i_ds")
    i_qs = sp.symbols("i_qs")
    t_m = sp.symbols("t_m")

    # parameters
    tp, x0, x_p, Hm, ra, t_m, ws = sp.symbols("tp, x_0, x_p, Hm, ra, t_m, ws")
    vm, va = sp.symbols("vm, va")
    p0 = sp.symbols("p0")
    q0 = sp.symbols("q0")
    weight = sp.symbols("weight")
    v_ds = -vm * sp.sin(va)
    v_qs = vm * sp.cos(va)

    F0 = (-1.0 / tp) * (e_dp + (x0 - x_p) * i_qs) + s * ws * e_qp
    F1 = (-1.0 / tp) * (e_qp - (x0 - x_p) * i_ds) - s * ws * e_dp
    F2 = (1.0 / (2.0 * Hm)) * (t_m - e_dp * i_ds - e_qp * i_qs)
    F3 = ra * i_ds - x_p * i_qs + e_dp - v_ds
    F4 = ra * i_qs + x_p * i_ds + e_qp - v_qs
    F5 = v_ds * i_ds + v_qs * i_qs + (1 - weight) * p0
    F6 = (v_qs * i_ds - v_ds * i_qs + ysh * vm * vm) + (1 - weight) * q0

    dF0d0 = sp.diff(F0, e_dp)
    dF0d1 = sp.diff(F0, e_qp)
    dF0d2 = sp.diff(F0, s)
    dF0d3 = sp.diff(F0, t_m)
    dF0d4 = sp.diff(F0, ysh)
    dF0d5 = sp.diff(F0, i_ds)
    dF0d6 = sp.diff(F0, i_qs)

    dF1d0 = sp.diff(F1, e_dp)
    dF1d1 = sp.diff(F1, e_qp)
    dF1d2 = sp.diff(F1, s)
    dF1d3 = sp.diff(F1, t_m)
    dF1d4 = sp.diff(F1, ysh)
    dF1d5 = sp.diff(F1, i_ds)
    dF1d6 = sp.diff(F1, i_qs)

    dF2d0 = sp.diff(F2, e_dp)
    dF2d1 = sp.diff(F2, e_qp)
    dF2d2 = sp.diff(F2, s)
    dF2d3 = sp.diff(F2, t_m)
    dF2d4 = sp.diff(F2, ysh)
    dF2d5 = sp.diff(F2, i_ds)
    dF2d6 = sp.diff(F2, i_qs)

    dF3d0 = sp.diff(F3, e_dp)
    dF3d1 = sp.diff(F3, e_qp)
    dF3d2 = sp.diff(F3, s)
    dF3d3 = sp.diff(F3, t_m)
    dF3d4 = sp.diff(F3, ysh)
    dF3d5 = sp.diff(F3, i_ds)
    dF3d6 = sp.diff(F3, i_qs)

    dF4d0 = sp.diff(F4, e_dp)
    dF4d1 = sp.diff(F4, e_qp)
    dF4d2 = sp.diff(F4, s)
    dF4d3 = sp.diff(F4, t_m)
    dF4d4 = sp.diff(F4, ysh)
    dF4d5 = sp.diff(F4, i_ds)
    dF4d6 = sp.diff(F4, i_qs)

    dF5d0 = sp.diff(F5, e_dp)
    dF5d1 = sp.diff(F5, e_qp)
    dF5d2 = sp.diff(F5, s)
    dF5d3 = sp.diff(F5, t_m)
    dF5d4 = sp.diff(F5, ysh)
    dF5d5 = sp.diff(F5, i_ds)
    dF5d6 = sp.diff(F5, i_qs)

    dF6d0 = sp.diff(F6, e_dp)
    dF6d1 = sp.diff(F6, e_qp)
    dF6d2 = sp.diff(F6, s)
    dF6d3 = sp.diff(F6, t_m)
    dF6d4 = sp.diff(F6, ysh)
    dF6d5 = sp.diff(F6, i_ds)
    dF6d6 = sp.diff(F6, i_qs)

    J = sp.Matrix(
        [
            [dF0d0, dF0d1, dF0d2, dF0d3, dF0d4, dF0d5, dF0d6],
            [dF1d0, dF1d1, dF1d2, dF1d3, dF1d4, dF1d5, dF1d6],
            [dF2d0, dF2d1, dF2d2, dF2d3, dF2d4, dF2d5, dF2d6],
            [dF3d0, dF3d1, dF3d2, dF3d3, dF3d4, dF3d5, dF3d6],
            [dF4d0, dF4d1, dF4d2, dF4d3, dF4d4, dF4d5, dF4d6],
            [dF5d0, dF5d1, dF5d2, dF5d3, dF5d4, dF5d5, dF5d6],
            [dF6d0, dF6d1, dF6d2, dF6d3, dF6d4, dF6d5, dF6d6],
        ]
    )

    dF0da = sp.diff(F0, weight)
    dF1da = sp.diff(F1, weight)
    dF2da = sp.diff(F2, weight)
    dF3da = sp.diff(F3, weight)
    dF4da = sp.diff(F4, weight)
    dF5da = sp.diff(F5, weight)
    dF6da = sp.diff(F6, weight)

    JA = sp.Matrix([[dF0da], [dF1da], [dF2da], [dF3da], [dF4da], [dF5da], [dF6da]])

    sp.pprint(J)
    sp.pprint(JA)

    print(pycode(J))
    print(pycode(JA))

    print("HESSIAN OF INTIALIZATION")

    FF = [F0, F1, F2, F3, F4, F5, F6]
    state_vars = [e_dp, e_qp, s, t_m, ysh, i_ds, i_qs]
    state_name = ["e_dp", "e_qp", "s", "t_m", "ysh", "i_ds", "i_qs"]
    nvars = len(state_vars)
    for m in range(len(FF)):
        for i in range(nvars):
            for j in range(nvars):
                differential = sp.diff(FF[m], state_vars[i], state_vars[j])
                if (differential.is_zero is None) or (differential.is_zero is False):
                    print("H[%d][%d, %d] = %s" % (m, i, j, str(differential)))

    # RESIDUAL
    F0 = (-1.0 / tp) * (e_dp + (x0 - x_p) * i_qs) + s * ws * e_qp
    F1 = (-1.0 / tp) * (e_qp - (x0 - x_p) * i_ds) - s * ws * e_dp
    F2 = (1.0 / (2.0 * Hm)) * (t_m - e_dp * i_ds - e_qp * i_qs)
    F3 = 0.0
    F4 = 0.0
    F5 = ra * i_ds - x_p * i_qs + e_dp - v_ds
    F6 = ra * i_qs + x_p * i_ds + e_qp - v_qs
    F7 = -(v_ds * i_ds + v_qs * i_qs)
    F8 = -((v_qs * i_ds - v_ds * i_qs + ysh * vm * vm))

    FF = [F0, F1, F2, F3, F4, F5, F6, F7, F8]
    state_vars = [e_dp, e_qp, s, t_m, ysh, i_ds, i_qs, vm, va]
    state_name = ["e_dp", "e_qp", "s", "t_m", "ysh", "i_ds", "i_qs", "vm", "va"]
    nvars = len(state_vars)

    print("HESSIAN CALCULATION")
    for m in range(len(FF)):
        print("### HESSIAN OF F%d ###\n" % (m))
        for i in range(nvars):
            differential_var = []
            differential_val = []
            for j in range(nvars):
                differential = sp.diff(FF[m], state_vars[i], state_vars[j])
                if (differential.is_zero is None) or (differential.is_zero is False):
                    differential_var.append(state_name[j])
                    differential_val.append(str(differential))

            if len(differential_var) > 0:
                print("row = %s_idx" % (state_name[i]))
                for k in range(len(differential_var)):
                    print("col[%d] = %s_idx" % (k, differential_var[k]))
                    print("val[%d] = %s" % (k, differential_val[k]))
                print(
                    "csr_set_row(H%d.data, H%d.indptr, H%d.indices, %d, row, col, val)\n"
                    % (m, m, m, len(differential_var))
                )
