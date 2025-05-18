from numba import njit

# must match: def resdiff_genrou(F, z, v, theta, idxs, ctrl_idx, ctrl_var, power_injection)
@njit(nopython=True, cache=True)
def resdiff_void(F, z, v, theta, idxs, ctrl_idx, ctrl_var, power_injection):
    return

# must match: def residual_pinj(F, z, v, theta, idxs)
@njit(nopython=True, cache=True)
def pinj_void(F, z, v, theta, idxs):
    return

# must match: def residual_cinj(F, z, v, theta, idxs)
@njit(nopython=True, cache=True)
def cinj_void(F, z, v, theta, idxs):
    return

# must match: def jac_genrou(z, v, theta, idxs, ctrl_idx, ctrl_var, J_data, J_ptr, J_idx, power_injection)
@njit(nopython=True, cache=True)
def jac_void(z, v, theta, idxs, ctrl_idx, ctrl_var, J_data, J_ptr, J_idx, power_injection):
    return