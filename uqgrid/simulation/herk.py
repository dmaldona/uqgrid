"""Half-explicit Runge-Kutta integrator for UQGrid.

Heun (HERK2) and classical RK4 (HERK4) are wired through
``integrate_system``.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

from uqgrid.simulation.residual import residual_function
from uqgrid.simulation.jacobian import residual_jacobian


# ---------------------------------------------------------------------------
# Tableaus
# ---------------------------------------------------------------------------

# Heun (explicit RK2)
A_HEUN = np.array([[0.0, 0.0],
                   [1.0, 0.0]])
B_HEUN = np.array([0.5, 0.5])
C_HEUN = np.array([0.0, 1.0])

# Classical RK4
A_RK4 = np.array([[0.0, 0.0, 0.0, 0.0],
                  [0.5, 0.0, 0.0, 0.0],
                  [0.0, 0.5, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]])
B_RK4 = np.array([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])
C_RK4 = np.array([0.0, 0.5, 0.5, 1.0])


TABLEAUS = {
    "herk2": (A_HEUN, B_HEUN, C_HEUN),
    "herk4": (A_RK4, B_RK4, C_RK4),
}


# ---------------------------------------------------------------------------
# Stage algebraic Newton
# ---------------------------------------------------------------------------


def solve_stage_algebraic(X_i, y0, v0, theta, psys, F_full, J_full,
                          tol=1e-10, max_iter=50):
    """Solve g(X_i, y, v) = 0 for (y, v) with X_i frozen.

    Returns (y, v, iterations). Raises RuntimeError on non-convergence.
    """
    NDIFFEQ = psys.num_dof_dif
    alg_size = psys.num_dof_alg

    z = np.concatenate([X_i, y0, v0])

    for it in range(max_iter):
        residual_function(F_full, z, theta, psys)
        g = F_full[NDIFFEQ:]
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            y = z[NDIFFEQ:NDIFFEQ + alg_size].copy()
            v = z[NDIFFEQ + alg_size:].copy()
            return y, v, it

        residual_jacobian(J_full, z, theta, psys)
        Jgg = J_full[NDIFFEQ:, NDIFFEQ:].tocsc()
        dy = spsolve(Jgg, g)
        z[NDIFFEQ:] -= dy

    raise RuntimeError(
        f"HERK stage algebraic Newton did not converge in {max_iter} iters "
        f"(||g||={g_norm:.3e}, tol={tol:.3e})"
    )


# ---------------------------------------------------------------------------
# Per-step driver
# ---------------------------------------------------------------------------


def herk_step(z_old, theta, h, psys, A, b, c, F, J, tol, max_iter):
    """Advance one HERK step from ``z_old`` by ``h``.

    Returns the new combined state vector ``z_new = [x_new; y_new; v_new]``.
    """
    NDIFFEQ = psys.num_dof_dif
    alg_size = psys.num_dof_alg
    s = len(b)

    x_n = z_old[:NDIFFEQ].copy()
    y_prev = z_old[NDIFFEQ:NDIFFEQ + alg_size].copy()
    v_prev = z_old[NDIFFEQ + alg_size:].copy()

    K = np.zeros((s, NDIFFEQ))
    Y_last, V_last = y_prev, v_prev

    for i in range(s):
        X_i = x_n.copy()
        for j in range(i):
            a_ij = A[i, j]
            if a_ij != 0.0:
                X_i += h * a_ij * K[j]

        Y_i, V_i, _ = solve_stage_algebraic(
            X_i, Y_last, V_last, theta, psys, F, J, tol, max_iter)
        Y_last, V_last = Y_i, V_i

        z_stage = np.concatenate([X_i, Y_i, V_i])
        residual_function(F, z_stage, theta, psys)
        K[i] = F[:NDIFFEQ].copy()

    x_new = x_n + h * (b @ K)

    Y_new, V_new, _ = solve_stage_algebraic(
        x_new, Y_last, V_last, theta, psys, F, J, tol, max_iter)

    return np.concatenate([x_new, Y_new, V_new])


# ---------------------------------------------------------------------------
# Integration driver
# ---------------------------------------------------------------------------


def integrate_system_herk(psys, config, ctx=None):
    """HERK driver mirroring the non-PETSc branch of ``integrate_system``.

    Only the no-PETSc, no-sensitivity, no-Jacobian-check path is supported.
    Fault on/off events are handled between steps via an algebraic resolve.
    """
    import math

    from uqgrid.simulation.dynamics import (
        initialize_system,
        preallocate_jacobian,
    )
    from uqgrid.simulation.pflow import runpf

    if config.petsc:
        raise ValueError("HERK driver does not support PETSc.")
    if config.comp_sens:
        raise ValueError("HERK driver does not support sensitivities.")

    method = config.method
    if method not in TABLEAUS:
        raise ValueError(f"HERK tableau not registered for method: {method}")
    A, b, c = TABLEAUS[method]

    psys.power_injection = config.power_injection

    pf_solution = runpf(psys, verbose=False)
    z0, theta = initialize_system(psys, pf_solution)

    z0_user = ctx.z0_user if ctx is not None else getattr(config, "z0_user", None)
    theta_user = ctx.theta_user if ctx is not None else getattr(config, "theta_user", None)
    if z0_user is not None:
        if z0_user.shape[0] != z0.shape[0]:
            raise ValueError("Provided initial state does not match system size.")
        z0 = z0_user
    if theta_user is not None:
        if theta_user.shape[0] != theta.shape[0]:
            raise ValueError("Provided theta does not match system parameters.")
        theta = theta_user

    system_size = z0.shape[0]
    J = preallocate_jacobian(psys)
    F = np.zeros(system_size)

    h = config.dt
    if config.steps > 0:
        nsteps = config.steps
    else:
        nsteps = int(math.floor(config.tend / config.dt)) + 1

    step_on = int(config.ton / h)
    step_off = int(config.toff / h)

    tol = config.herk_alg_tol
    max_iter = config.herk_alg_max_iter

    tvec = np.linspace(0, nsteps * h, nsteps)
    history = np.zeros((system_size, nsteps))
    z = z0

    NDIFFEQ = psys.num_dof_dif
    alg_size = psys.num_dof_alg

    for i in range(nsteps):
        z = herk_step(z, theta, h, psys, A, b, c, F, J, tol, max_iter)
        history[:, i] = z

        if i == step_on and len(psys.fault_events) > 0:
            psys.fault_events[0].apply()
            x = z[:NDIFFEQ]
            y = z[NDIFFEQ:NDIFFEQ + alg_size]
            v = z[NDIFFEQ + alg_size:]
            y_new, v_new, _ = solve_stage_algebraic(
                x, y, v, theta, psys, F, J, tol, max_iter)
            z = np.concatenate([x, y_new, v_new])

        if i == step_off and len(psys.fault_events) > 0:
            psys.fault_events[0].remove()
            x = z[:NDIFFEQ]
            y = z[NDIFFEQ:NDIFFEQ + alg_size]
            v = z[NDIFFEQ + alg_size:]
            y_new, v_new, _ = solve_stage_algebraic(
                x, y, v, theta, psys, F, J, tol, max_iter)
            z = np.concatenate([x, y_new, v_new])

    if nsteps - 1 < step_off and len(psys.fault_events) > 0:
        psys.fault_events[0].remove()

    return {"tvec": tvec, "history": history}
