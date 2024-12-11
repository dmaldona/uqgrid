import numpy as np

########## tools #####################

if __name__ == "__main__":
    from sympy import *
    from sympy.printing.pycode import pycode


    Ka, Ta, Kf, Tf, Ke, Te, Tr, Ae, Be = symbols('Ka, Ta, Kf, Tf, Ke, Te, Tr, Ae, Be')

    vr1, vr2, vref, vm, e_fd =  symbols('vr1, vr2, vref, vm, e_fd')

    # RESIDUAL
    F1 = (Ka*(vref - vm - vr2 - (Kf/Tf)*e_fd) - vr1)/Ta
    F2 = -((Kf/Tf)*e_fd + vr2)/Tf
    F3 = -(e_fd*(Ke + Ae*exp(Be*vm)) - vr1)/Te

    FF = [F1, F2, F3]
    state_vars = [vr1, vr2, e_fd, vm]
    state_name = ['vr1', 'vr2', 'e_fd', 'vm']
    
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
