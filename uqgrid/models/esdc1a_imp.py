import numpy as np
from numba import jit
from uqgrid.utils.tools import csr_add_row, csr_set_row
from uqgrid.core.base_models import Exciter
from scipy import optimize

class ExcESDC1A(Exciter):
    def __init__(self, id_tag, Ka, Ta, Kf, Tf, Ke, Te, Tr, Ae, Be):

        self.Ka = Ka
        self.Ta = Ta
        self.Kf = Kf
        self.Tf = Tf
        self.Ke = Ke
        self.Te = Te
        self.Tr = Tr
        self.Ae = Ae
        self.Be = Be

        # control variables
        self.vref = None
        self.efd_idx = 2

        parameter_list = ['Ka', 'Ta', 'Kf', 'Tf', 'Ke', 'Te', 'Tr', 'Ae', 'Be', 'vref']
        state_list = ['vr1', 'vr2', 'e_fd']

        Exciter.__init__(self, id_tag, 3, 3, 0, len(parameter_list), state_list)

    def residualFinit(self, x, v, theta, p0, q0):

        F = np.zeros(self.initdim)

        # parameters
        Ka = self.Ka
        Ta = self.Ta
        Kf = self.Kf
        Tf = self.Tf
        Ke = self.Ke
        Te = self.Te
        Tr = self.Tr
        Ae = self.Ae
        Be = self.Be
        e_fd = self.e_fd0

        vr1 = x[0]
        vr2 = x[1]
        vref = x[2]

        F[0] = (Ka*(vref - v - vr2 - (Kf/Tf)*e_fd) - vr1)/Ta
        F[1] = -((Kf/Tf)*e_fd + vr2)/Tf
        F[2] = -(e_fd*(Ke + Ae*np.exp(Be*v)) - vr1)/Te

        return F

    def initialize(self, vm, va, p, q, x, y, psys):

        x0 = np.ones(self.initdim)
        sol = optimize.root(
            self.residualFinit,
            x0,
            args=(vm, va, p, q),
            method='krylov',
            options={
                'xtol': 1e-8,
                'disp': False
            })

        self.initialized = True
        x[self.dif_ptr:self.dif_ptr + 2] = sol.x[0:2]
        x[self.dif_ptr + 2] = self.e_fd0
        self.vref = sol.x[2]
        return None

    def initialize_theta(self, theta):

        idx = self.par_ptr

        theta[idx] = self.Ka
        theta[idx + 1] = self.Ta
        theta[idx + 2] = self.Kf
        theta[idx + 3] = self.Tf
        theta[idx + 4] = self.Ke
        theta[idx + 5] = self.Te
        theta[idx + 6] = self.Tr
        theta[idx + 7] = self.Ae
        theta[idx + 8] = self.Be
        theta[idx + 9] = self.vref

    def residual_diff(self, F, z, v, theta, idxs, power_injection):

        dp = idxs[0]
        ap = idxs[1]
        pp = idxs[2]

        # parameters
        Ka = self.Ka
        Ta = self.Ta
        Kf = self.Kf
        Tf = self.Tf
        Ke = self.Ke
        Te = self.Te
        Tr = self.Tr
        Ae = self.Ae
        Be = self.Be
        vref = theta[pp + 9]

        # states
        vr1 = z[dp]
        vr2 = z[dp + 1]
        e_fd = z[dp + 2]

        if power_injection:
            vm = v[2*self.bus]
        else:
            vr = v[2*self.bus]
            vi = v[2*self.bus + 1]
            vm = np.sqrt(vr*vr + vi*vi)
            if vm == 0.0:
                vm = 1e-12

        F[dp] = (Ka*(vref - vm - vr2 - (Kf/Tf)*e_fd) - vr1)/Ta
        F[dp + 1] = -((Kf/Tf)*e_fd + vr2)/Tf
        F[dp + 2] = -(e_fd*(Ke + Ae*np.exp(Be*vm)) - vr1)/Te

        return None

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys, power_injection):

        coord = []

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]

        # these are INDEXES
        vr1 = dp
        vr2 = dp + 1
        e_fd = dp + 2

        if power_injection:
            vm = dev + 2*self.bus
        else:
            vr = dev + 2*self.bus
            vi = dev + 2*self.bus + 1

        # first row
        row = dp
        if power_injection:
            cols = [vr1, vr2, e_fd, vm]
        else:
            cols = [vr1, vr2, e_fd, vr, vi]
        coord.append([row, cols])

        # second row
        row = dp + 1
        cols = [vr2, e_fd]
        coord.append([row, cols])

        # third row
        row = dp + 2
        if power_injection:
            cols = [vr1, e_fd, vm]
        else:
            cols = [vr1, e_fd, vr, vi]
        coord.append([row, cols])

        return coord

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]

        # parameters
        Ka = self.Ka
        Ta = self.Ta
        Kf = self.Kf
        Tf = self.Tf
        Ke = self.Ke
        Te = self.Te
        Tr = self.Tr
        Ae = self.Ae
        Be = self.Be

        # states
        vr1 = z[dp]
        vr2 = z[dp + 1]
        e_fd = z[dp + 2]

        if power_injection:
            vm = v[2*self.bus]
        else:
            vr = v[2*self.bus]
            vi = v[2*self.bus + 1]
            vm = np.sqrt(vr*vr + vi*vi)
            if vm == 0.0:
                vm = 1e-12

        # indexes
        vr1_idx = dp
        vr2_idx = dp + 1
        e_fd_idx = dp + 2
        if power_injection:
            vm_idx = dev + 2*self.bus
        else:
            vr_idx = dev + 2*self.bus
            vi_idx = dev + 2*self.bus + 1

        col = np.zeros(10)
        val = np.zeros(10)

        # first row
        row = dp
        col[0] = vr1_idx
        val[0] = -1/Ta
        col[1] = vr2_idx
        val[1] = -Ka/Ta
        col[2] = e_fd_idx
        val[2] = -Ka*Kf/(Ta*Tf)
        if power_injection:
            col[3] = vm_idx
            val[3] = -Ka/Ta
            ncols = 4
        else:
            dvm_dvr = vr / vm
            dvm_dvi = vi / vm
            col[3] = vr_idx
            val[3] = (-Ka/Ta) * dvm_dvr
            col[4] = vi_idx
            val[4] = (-Ka/Ta) * dvm_dvi
            ncols = 5
        csr_set_row(J.data, J.indptr, J.indices, ncols, row, col, val)

        # second row
        row = dp + 1
        col[0] = vr2_idx
        val[0] = -1/Tf
        col[1] = e_fd_idx
        val[1] = -Kf/Tf**2
        csr_set_row(J.data, J.indptr, J.indices, 2, row, col, val)

        # third row
        row = dp + 2
        col[0] = vr1_idx
        val[0] = 1/Te
        col[1] = e_fd_idx
        val[1] = -(Ke + Ae*np.exp(Be*vm))/Te
        if power_injection:
            col[2] = vm_idx
            val[2] = -e_fd*(Ae*Be*np.exp(Be*vm))/Te
            ncols = 3
        else:
            dvm_dvr = vr / vm
            dvm_dvi = vi / vm
            col[2] = vr_idx
            val[2] = (-e_fd*(Ae*Be*np.exp(Be*vm))/Te) * dvm_dvr
            col[3] = vi_idx
            val[3] = (-e_fd*(Ae*Be*np.exp(Be*vm))/Te) * dvm_dvi
            ncols = 4
        csr_set_row(J.data, J.indptr, J.indices, ncols, row, col, val)

    def preallocate_hessian(self, h_nnz, idxs, psys):

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]

        # these are INDEXES
        vr1 = dp
        vr2 = dp + 1
        e_fd = dp + 2

        vm = dev + 2*self.bus
        va = dev + 2*self.bus + 1

        # F0
        h_nnz[dp + 2]['rows'].append(e_fd)
        h_nnz[dp + 2]['cols'].append([vm])

        h_nnz[dp + 2]['rows'].append(vm)
        h_nnz[dp + 2]['cols'].append([e_fd, vm])

    def residual_hess(self, HESS, z, v, theta, idxs):

        dp = idxs[0]
        ap = idxs[1]
        dev = idxs[2]
        pp = idxs[3]
        bus = idxs[4]

        H0 = HESS[dp]
        H1 = HESS[dp + 1]
        H2 = HESS[dp + 2]

        # parameters
        Ka = self.Ka
        Ta = self.Ta
        Kf = self.Kf
        Tf = self.Tf
        Ke = self.Ke
        Te = self.Te
        Tr = self.Tr
        Ae = self.Ae
        Be = self.Be

        # states
        vr1 = z[dp]
        vr2 = z[dp + 1]
        e_fd = z[dp + 2]

        vm = v[2*self.bus]
        va = v[2*self.bus + 1]

        # indexes
        vr1_idx = dp
        vr2_idx = dp + 1
        e_fd_idx = dp + 2
        vm_idx = dev + 2*self.bus
        va_idx = dev + 2*self.bus + 1

        # column and value vectors
        col = np.zeros(10)
        val = np.zeros(10)

        ### HESSIAN OF F0 ###

        row = e_fd_idx
        col[0] = vm_idx
        val[0] = -Ae*Be*np.exp(Be*vm)/Te
        csr_set_row(H2.data, H2.indptr, H2.indices, 1, row, col, val)

        row = vm_idx
        col[0] = e_fd_idx
        val[0] = -Ae*Be*np.exp(Be*vm)/Te
        col[1] = vm_idx
        val[1] = -Ae*Be**2*e_fd*np.exp(Be*vm)/Te
        csr_set_row(H2.data, H2.indptr, H2.indices, 2, row, col, val)

if __name__ == "__main__":
    import sympy as sp
    from sympy.printing.pycode import pycode

    Ka, Ta, Kf, Tf, Ke, Te, Tr, Ae, Be = sp.symbols("Ka, Ta, Kf, Tf, Ke, Te, Tr, Ae, Be")

    vr1, vr2, vref, vm, e_fd = sp.symbols("vr1, vr2, vref, vm, e_fd")

    # RESIDUAL
    F1 = (Ka * (vref - vm - vr2 - (Kf / Tf) * e_fd) - vr1) / Ta
    F2 = -((Kf / Tf) * e_fd + vr2) / Tf
    F3 = -(e_fd * (Ke + Ae * sp.exp(Be * vm)) - vr1) / Te

    FF = [F1, F2, F3]
    state_vars = [vr1, vr2, e_fd, vm]
    state_name = ["vr1", "vr2", "e_fd", "vm"]

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
