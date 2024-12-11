import numpy as np

########## tools #####################

if __name__ == "__main__":
    from sympy import *
    from sympy.printing.pycode import pycode

    T1, T2, T3, T4, T5, T6, K1, K2, K3 = symbols('T1, T2, T3, T4, T5, T6, K1, K2, K3')
    PF0, PLL, TP1, TP2, TP3, p_m = symbols('PF0, PLL, TP1, TP2, TP3, p_m')
    w, pref = symbols('w, p_ref')

    # RESIDUAL
    F0 = (1.0/T1)*(K1*w - PF0)
    F1 = (1.0/T3)*((1.0 - (T2/T3))*PF0 - PLL)

    SatP = pref - (T2/T3)*PF0 - PLL

    F2 = (1.0/T4)*(SatP - TP1)
    F3 = (1.0/T5)*(K2*TP1 - TP2)
    F4 = (1.0/T6)*(K3*TP2 - TP3)

    F5 = TP1*(1 - K2) + TP2*(1 - K3) + TP3 - p_m

    FF = [F1, F1, F2, F3, F4, F5]
    state_vars = [PF0, PLL, TP1, TP2, TP3, p_m]
    state_name = ['PF0', 'PLL', 'TP1', 'TP2', 'TP3', 'p_m']
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
