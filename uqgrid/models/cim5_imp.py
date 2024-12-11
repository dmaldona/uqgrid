import numpy as np
ws = 2*np.pi*60

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
    from sympy import *
    from sympy.printing.pycode import pycode

    # states
    e_dp = symbols('e_dp')
    e_qp = symbols('e_qp')
    s = symbols('s')
    ysh = symbols('ysh')
    i_ds = symbols('i_ds')
    i_qs = symbols('i_qs')
    t_m = symbols('t_m')

    # parameters
    tp, x0, x_p, Hm, ra, t_m, ws = symbols('tp, x_0, x_p, Hm, ra, t_m, ws')
    vm, va = symbols('vm, va')
    p0 = symbols('p0')
    q0 = symbols('q0')
    weight = symbols('weight')
    v_ds = -vm*sin(va)
    v_qs = vm*cos(va)
    
    F0 = (-1.0/tp)*(e_dp + (x0 - x_p)*i_qs) + s*ws*e_qp
    F1 = (-1.0/tp)*(e_qp - (x0 - x_p)*i_ds) - s*ws*e_dp
    F2 = (1.0/(2.0*Hm))*(t_m - e_dp*i_ds - e_qp*i_qs)
    F3 = ra*i_ds - x_p*i_qs + e_dp - v_ds
    F4 = ra*i_qs + x_p*i_ds + e_qp - v_qs
    F5 = v_ds*i_ds + v_qs*i_qs + (1 - weight)*p0
    F6 = (v_qs*i_ds - v_ds*i_qs + ysh*vm*vm) + (1 - weight)*q0

    dF0d0 = diff(F0, e_dp)
    dF0d1 = diff(F0, e_qp)
    dF0d2 = diff(F0, s)
    dF0d3 = diff(F0, t_m)
    dF0d4 = diff(F0, ysh)
    dF0d5 = diff(F0, i_ds)
    dF0d6 = diff(F0, i_qs)
    
    dF1d0 = diff(F1, e_dp)
    dF1d1 = diff(F1, e_qp)
    dF1d2 = diff(F1, s)
    dF1d3 = diff(F1, t_m)
    dF1d4 = diff(F1, ysh)
    dF1d5 = diff(F1, i_ds)
    dF1d6 = diff(F1, i_qs)
    
    dF2d0 = diff(F2, e_dp)
    dF2d1 = diff(F2, e_qp)
    dF2d2 = diff(F2, s)
    dF2d3 = diff(F2, t_m)
    dF2d4 = diff(F2, ysh)
    dF2d5 = diff(F2, i_ds)
    dF2d6 = diff(F2, i_qs)
    
    dF3d0 = diff(F3, e_dp)
    dF3d1 = diff(F3, e_qp)
    dF3d2 = diff(F3, s)
    dF3d3 = diff(F3, t_m)
    dF3d4 = diff(F3, ysh)
    dF3d5 = diff(F3, i_ds)
    dF3d6 = diff(F3, i_qs)
    
    dF4d0 = diff(F4, e_dp)
    dF4d1 = diff(F4, e_qp)
    dF4d2 = diff(F4, s)
    dF4d3 = diff(F4, t_m)
    dF4d4 = diff(F4, ysh)
    dF4d5 = diff(F4, i_ds)
    dF4d6 = diff(F4, i_qs)
    
    dF5d0 = diff(F5, e_dp)
    dF5d1 = diff(F5, e_qp)
    dF5d2 = diff(F5, s)
    dF5d3 = diff(F5, t_m)
    dF5d4 = diff(F5, ysh)
    dF5d5 = diff(F5, i_ds)
    dF5d6 = diff(F5, i_qs)
    
    dF6d0 = diff(F6, e_dp)
    dF6d1 = diff(F6, e_qp)
    dF6d2 = diff(F6, s)
    dF6d3 = diff(F6, t_m)
    dF6d4 = diff(F6, ysh)
    dF6d5 = diff(F6, i_ds)
    dF6d6 = diff(F6, i_qs)
    

    J = Matrix([
        [dF0d0, dF0d1, dF0d2, dF0d3, dF0d4, dF0d5, dF0d6],
        [dF1d0, dF1d1, dF1d2, dF1d3, dF1d4, dF1d5, dF1d6],
        [dF2d0, dF2d1, dF2d2, dF2d3, dF2d4, dF2d5, dF2d6],
        [dF3d0, dF3d1, dF3d2, dF3d3, dF3d4, dF3d5, dF3d6],
        [dF4d0, dF4d1, dF4d2, dF4d3, dF4d4, dF4d5, dF4d6],
        [dF5d0, dF5d1, dF5d2, dF5d3, dF5d4, dF5d5, dF5d6],
        [dF6d0, dF6d1, dF6d2, dF6d3, dF6d4, dF6d5, dF6d6]])

    
    dF0da = diff(F0, weight)
    dF1da = diff(F1, weight)
    dF2da = diff(F2, weight)
    dF3da = diff(F3, weight)
    dF4da = diff(F4, weight)
    dF5da = diff(F5, weight)
    dF6da = diff(F6, weight)

    JA = Matrix([
        [dF0da],
        [dF1da],
        [dF2da],
        [dF3da],
        [dF4da],
        [dF5da],
        [dF6da]])

    pprint(J)
    pprint(JA)

    print(pycode(J))
    print(pycode(JA))
    
    print("HESSIAN OF INTIALIZATION")

    FF = [F0, F1, F2, F3, F4, F5, F6]
    state_vars = [e_dp, e_qp, s, t_m, ysh, i_ds, i_qs]
    state_name = ['e_dp', 'e_qp', 's', 't_m', 'ysh', 'i_ds', 'i_qs']
    nvars = len(state_vars)
    for m in range(len(FF)):
        for i in range(nvars):
            for j in range(nvars):
                differential = diff(FF[m], state_vars[i],  state_vars[j])
                if (differential.is_zero is None) or (differential.is_zero is False):
                    print("H[%d][%d, %d] = %s" % (m, i, j, str(differential)))

    # RESIDUAL
    F0 = (-1.0/tp)*(e_dp + (x0 - x_p)*i_qs) + s*ws*e_qp
    F1 = (-1.0/tp)*(e_qp - (x0 - x_p)*i_ds) - s*ws*e_dp
    F2 = (1.0/(2.0*Hm))*(t_m - e_dp*i_ds - e_qp*i_qs)
    F3 = 0.0
    F4 = 0.0
    F5 = ra*i_ds - x_p*i_qs + e_dp - v_ds
    F6 = ra*i_qs + x_p*i_ds + e_qp - v_qs
    F7 = -(v_ds*i_ds + v_qs*i_qs)
    F8 = -((v_qs*i_ds - v_ds*i_qs + ysh*vm*vm))

    FF = [F0, F1, F2, F3, F4, F5, F6, F7, F8]
    state_vars = [e_dp, e_qp, s, t_m, ysh, i_ds, i_qs, vm, va]
    state_name = ['e_dp', 'e_qp', 's', 't_m', 'ysh', 'i_ds', 'i_qs', 'vm', 'va']
    nvars = len(state_vars)
    
    print("HESSIAN CALCULATION")
    for m in range(len(FF)):
        print ("### HESSIAN OF F%d ###\n" % (m))
        for i in range(nvars):
            differential_var = []
            differential_val = []
            for j in range(nvars):
                differential = diff(FF[m], state_vars[i],  state_vars[j])
                if (differential.is_zero is None) or (differential.is_zero is False):
                    differential_var.append(state_name[j])
                    differential_val.append(str(differential))

            if len(differential_var) > 0:
                print("row = %s_idx" % (state_name[i]))
                for k in range(len(differential_var)):
                    print("col[%d] = %s_idx" % (k, differential_var[k]))
                    print("val[%d] = %s" % (k, differential_val[k]))
                print("csr_set_row(H%d.data, H%d.indptr, H%d.indices, %d, row, col, val)\n" %
                        (m, m, m, len(differential_var)))
