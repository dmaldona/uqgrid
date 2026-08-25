"""Half-explicit Runge-Kutta integrator for UQGrid.

Heun (HERK2) and classical RK4 (HERK4) are wired through
``integrate_system``. Enabled hard state limits are projected at each stage
and accepted endpoint while the raw device equations remain unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

from uqgrid.simulation.dynamic_limits import (
    DynamicLimitError,
    _contextualize_dynamic_limit_runtime_error,
    collect_limited_state_descriptors,
    initialize_dynamic_limit_modes,
    project_limited_derivatives,
    project_limited_states,
    update_explicit_dynamic_limit_modes,
)
from uqgrid.simulation.residual import residual_function
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.timing import build_integration_schedule


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


def _solve_algebraics_with_dynamic_bounds(
    x, y, v, theta, psys, F, J, tol, max_iter, descriptors, *, time, stage,
):
    """Iterate algebraic solves and moving-bound projections to consistency."""
    events = []
    for _ in range(10):
        y, v, _ = solve_stage_algebraic(
            x, y, v, theta, psys, F, J, tol, max_iter
        )
        complete = np.concatenate([x, y, v])
        projected, projection_events = project_limited_states(
            complete, descriptors, time=time, stage_or_endpoint=stage
        )
        events.extend(projection_events)
        x = projected[:psys.num_dof_dif]
        if not projection_events:
            return x, y, v, complete, events
    raise RuntimeError("Moving dynamic-limit projection did not converge.")


def _herk_step_with_limits(
    z_old,
    theta,
    h,
    psys,
    A,
    b,
    c,
    F,
    J,
    tol,
    max_iter,
    *,
    limit_descriptors=(),
    limit_tolerance=0.0,
    limit_modes=None,
    t_start=0.0,
):
    """Advance one HERK step and return limiter state and events."""
    limit_descriptors = list(limit_descriptors)
    if limit_modes is None:
        limit_modes = initialize_dynamic_limit_modes(limit_descriptors)
    has_dynamic_bounds = any(
        descriptor.bound_scale is not None for descriptor in limit_descriptors
    )

    NDIFFEQ = psys.num_dof_dif
    alg_size = psys.num_dof_alg
    s = len(b)

    x_n = z_old[:NDIFFEQ].copy()
    y_prev = z_old[NDIFFEQ:NDIFFEQ + alg_size].copy()
    v_prev = z_old[NDIFFEQ + alg_size:].copy()

    K = np.zeros((s, NDIFFEQ))
    Y_last, V_last = y_prev, v_prev
    events = []

    for i in range(s):
        X_i = x_n.copy()
        for j in range(i):
            a_ij = A[i, j]
            if a_ij != 0.0:
                X_i += h * a_ij * K[j]

        stage_time = t_start + c[i] * h
        stage_name = f"stage_{i + 1}"
        if limit_descriptors and not has_dynamic_bounds:
            X_i, projection_events = project_limited_states(
                X_i,
                limit_descriptors,
                time=stage_time,
                stage_or_endpoint=stage_name,
            )
            events.extend(projection_events)
        if limit_descriptors and has_dynamic_bounds:
            X_i, Y_i, V_i, z_stage, projection_events = (
                _solve_algebraics_with_dynamic_bounds(
                    X_i, Y_last, V_last, theta, psys, F, J, tol, max_iter,
                    limit_descriptors, time=stage_time, stage=stage_name,
                )
            )
            events.extend(projection_events)
        else:
            Y_i, V_i, _ = solve_stage_algebraic(
                X_i, Y_last, V_last, theta, psys, F, J, tol, max_iter
            )
            z_stage = np.concatenate([X_i, Y_i, V_i])
        Y_last, V_last = Y_i, V_i

        z_stage = np.concatenate([X_i, Y_i, V_i])
        residual_function(F, z_stage, theta, psys)
        raw_derivative = F[:NDIFFEQ].copy()
        if limit_descriptors:
            K[i], _ = project_limited_derivatives(
                z_stage,
                raw_derivative,
                limit_descriptors,
                tolerance=limit_tolerance,
                time=stage_time,
                stage_or_endpoint=stage_name,
            )
            limit_modes, _, transition_events = (
                update_explicit_dynamic_limit_modes(
                    z_stage,
                    raw_derivative,
                    limit_descriptors,
                    limit_modes,
                    tolerance=limit_tolerance,
                    time=stage_time,
                    stage_or_endpoint=stage_name,
                )
            )
            events.extend(transition_events)
        else:
            K[i] = raw_derivative

    x_new = x_n + h * (b @ K)
    endpoint_time = t_start + h
    if limit_descriptors and not has_dynamic_bounds:
        x_new, projection_events = project_limited_states(
            x_new,
            limit_descriptors,
            time=endpoint_time,
            stage_or_endpoint="endpoint",
        )
        events.extend(projection_events)

    if limit_descriptors and has_dynamic_bounds:
        x_new, Y_new, V_new, z_new, projection_events = (
            _solve_algebraics_with_dynamic_bounds(
                x_new, Y_last, V_last, theta, psys, F, J, tol, max_iter,
                limit_descriptors, time=endpoint_time, stage="endpoint",
            )
        )
        events.extend(projection_events)
    else:
        Y_new, V_new, _ = solve_stage_algebraic(
            x_new, Y_last, V_last, theta, psys, F, J, tol, max_iter
        )
        z_new = np.concatenate([x_new, Y_new, V_new])

    if limit_descriptors:
        residual_function(F, z_new, theta, psys)
        raw_derivative = F[:NDIFFEQ].copy()
        limit_modes, _, transition_events = update_explicit_dynamic_limit_modes(
            z_new,
            raw_derivative,
            limit_descriptors,
            limit_modes,
            tolerance=limit_tolerance,
            time=endpoint_time,
            stage_or_endpoint="endpoint",
        )
        events.extend(transition_events)

    return z_new, limit_modes, events


def herk_step(z_old, theta, h, psys, A, b, c, F, J, tol, max_iter):
    """Advance one unconstrained HERK step from ``z_old`` by ``h``.

    Returns the new combined state vector ``z_new = [x_new; y_new; v_new]``.
    """
    z_new, _, _ = _herk_step_with_limits(
        z_old,
        theta,
        h,
        psys,
        A,
        b,
        c,
        F,
        J,
        tol,
        max_iter,
    )
    return z_new


# ---------------------------------------------------------------------------
# Integration driver
# ---------------------------------------------------------------------------


def integrate_system_herk(psys, config, ctx=None):
    """HERK driver mirroring the non-PETSc branch of ``integrate_system``.

    Only the no-PETSc, no-sensitivity, no-fsolve, no-ARKIMEX,
    no-Jacobian-check path is supported. Fault on/off events are handled
    between steps via an algebraic resolve. Enabled hard limits are enforced
    at RK stages and weighted endpoints without changing the time grid.
    """
    from uqgrid.simulation.dynamics import (
        _initialize_integration_state,
        preallocate_jacobian,
    )

    if config.petsc:
        raise ValueError("HERK driver does not support PETSc.")
    if config.comp_sens:
        raise ValueError("HERK driver does not support sensitivities.")
    if config.fsolve:
        raise ValueError(
            "HERK driver does not support config.fsolve; it uses sparse "
            "Newton solves for algebraic stages."
        )
    if config.arkimex:
        raise ValueError("HERK driver does not support ARKIMEX.")
    if config.check_jacobian:
        raise ValueError("HERK driver does not support Jacobian checking.")

    method = config.method
    if method not in TABLEAUS:
        raise ValueError(f"HERK tableau not registered for method: {method}")
    A, b, c = TABLEAUS[method]

    psys.power_injection = config.power_injection

    pf_solution, z0, theta, dynamic_limit_diagnostics = (
        _initialize_integration_state(psys, config, ctx)
    )
    limit_descriptors = []
    if config.enforce_dynamic_limits:
        limit_descriptors = [
            descriptor
            for descriptor in collect_limited_state_descriptors(psys, theta)
            if descriptor.enabled
        ]
    limit_modes = initialize_dynamic_limit_modes(limit_descriptors)

    system_size = z0.shape[0]
    J = preallocate_jacobian(psys)
    F = np.zeros(system_size)

    has_fault = bool(psys.fault_events)
    schedule = build_integration_schedule(
        dt=config.dt,
        tend=config.tend,
        steps=config.steps,
        ton=config.ton,
        toff=config.toff,
        has_fault=has_fault,
    )
    tvec = schedule.times
    fault = psys.fault_events[0] if has_fault else None
    if fault is not None:
        fault.remove()

    tol = config.herk_alg_tol
    max_iter = config.herk_alg_max_iter

    history = np.zeros((system_size, len(tvec)))
    z = z0
    history[:, 0] = z

    NDIFFEQ = psys.num_dof_dif
    alg_size = psys.num_dof_alg

    try:
        for i in range(1, len(tvec)):
            t_start = tvec[i - 1]
            h_step = tvec[i] - t_start
            if psys.signal_injectors:
                for inj in psys.signal_injectors:
                    inj.update(t_start, theta, psys)
            try:
                z, limit_modes, limit_events = _herk_step_with_limits(
                    z,
                    theta,
                    h_step,
                    psys,
                    A,
                    b,
                    c,
                    F,
                    J,
                    tol,
                    max_iter,
                    limit_descriptors=limit_descriptors,
                    limit_tolerance=config.dynamic_limit_tolerance,
                    limit_modes=limit_modes,
                    t_start=t_start,
                )
            except DynamicLimitError as exc:
                raise _contextualize_dynamic_limit_runtime_error(
                    exc,
                    method=method,
                    backend="native",
                    time=tvec[i],
                    stage_or_endpoint="endpoint",
                    prior_events=dynamic_limit_diagnostics["events"],
                ) from exc
            dynamic_limit_diagnostics["events"].extend(limit_events)

            if i == schedule.fault_on_index:
                fault.apply()
                x = z[:NDIFFEQ]
                y = z[NDIFFEQ:NDIFFEQ + alg_size]
                v = z[NDIFFEQ + alg_size:]
                y_new, v_new, _ = solve_stage_algebraic(
                    x, y, v, theta, psys, F, J, tol, max_iter)
                z = np.concatenate([x, y_new, v_new])
                if any(item.bound_scale is not None for item in limit_descriptors):
                    x, y_new, v_new, z, events = _solve_algebraics_with_dynamic_bounds(
                        x, y_new, v_new, theta, psys, F, J, tol, max_iter,
                        limit_descriptors, time=tvec[i], stage="fault_on",
                    )
                    dynamic_limit_diagnostics["events"].extend(events)

            if i == schedule.fault_off_index:
                fault.remove()
                x = z[:NDIFFEQ]
                y = z[NDIFFEQ:NDIFFEQ + alg_size]
                v = z[NDIFFEQ + alg_size:]
                y_new, v_new, _ = solve_stage_algebraic(
                    x, y, v, theta, psys, F, J, tol, max_iter)
                z = np.concatenate([x, y_new, v_new])
                if any(item.bound_scale is not None for item in limit_descriptors):
                    x, y_new, v_new, z, events = _solve_algebraics_with_dynamic_bounds(
                        x, y_new, v_new, theta, psys, F, J, tol, max_iter,
                        limit_descriptors, time=tvec[i], stage="fault_off",
                    )
                    dynamic_limit_diagnostics["events"].extend(events)
            history[:, i] = z
    finally:
        if fault is not None:
            fault.remove()

    results = {
        "tvec": tvec,
        "history": history,
        "dynamic_limit_diagnostics": dynamic_limit_diagnostics,
    }
    if pf_solution.validation is not None:
        results["power_flow_diagnostics"] = pf_solution.validation
    return results
