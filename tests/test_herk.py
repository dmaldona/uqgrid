import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.simulation.config import IntegrationConfig
from uqgrid.simulation.dynamics import (
    initialize_system,
    integrate_system,
    preallocate_jacobian,
)
from uqgrid.simulation.herk import solve_stage_algebraic
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


@pytest.fixture
def data_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )


def _build_two_bus_psys(data_dir):
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, "GENROU.dyr"))
    psys.createYbusComplex()
    psys.add_busfault(1, 1.0)
    return psys


def test_config_defaults_preserve_beuler():
    cfg = IntegrationConfig()

    assert cfg.method == "beuler"
    assert cfg.herk_alg_tol == pytest.approx(1e-10)
    assert cfg.herk_alg_max_iter == 50


def test_unknown_method_raises(data_dir):
    psys = _build_two_bus_psys(data_dir)
    cfg = IntegrationConfig(method="bogus", tend=0.0, steps=1, petsc=False)

    with pytest.raises(ValueError, match="Unknown integration method: bogus"):
        integrate_system(psys, cfg)


def test_beuler_regression_unchanged(data_dir):
    base_config = dict(
        tend=0.1,
        dt=1.0 / 120.0,
        steps=3,
        power_injection=True,
        verbose=False,
        comp_sens=False,
        fsolve=False,
        petsc=False,
        ton=10.0,
        toff=11.0,
    )

    psys_default = _build_two_bus_psys(data_dir)
    res_default = integrate_system(psys_default, IntegrationConfig(**base_config))

    psys_explicit = _build_two_bus_psys(data_dir)
    res_explicit = integrate_system(
        psys_explicit, IntegrationConfig(method="beuler", **base_config)
    )

    assert np.array_equal(res_default["tvec"], res_explicit["tvec"])
    assert np.allclose(res_default["history"], res_explicit["history"])


# ---------------------------------------------------------------------------
# Phase 2 — stage algebraic Newton
# ---------------------------------------------------------------------------


def _init_psys_at_steady_state(psys):
    """Return (sysvec, theta, F, J) initialized from a power-flow solution."""
    pf_solution = runpf(psys, verbose=False)
    sysvec, theta = initialize_system(psys, pf_solution)
    F = np.zeros_like(sysvec)
    J = preallocate_jacobian(psys)
    return sysvec, theta, F, J


def _build_two_bus_no_fault(data_dir):
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, "GENROU.dyr"))
    psys.createYbusComplex()
    return psys


def _build_ieee9_no_fault(data_dir):
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus.dyr"))
    psys.createYbusComplex()
    return psys


def _build_ieee9_gov_no_fault(data_dir):
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus_gov.dyr"))
    psys.createYbusComplex()
    return psys


def _build_ieee9_fault(data_dir):
    psys = _build_ieee9_no_fault(data_dir)
    psys.add_busfault(1, 1.0)
    return psys


def _build_ieee39_no_fault(data_dir):
    psys = load_psse(raw_filename=os.path.join(data_dir, "IEEE39_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "IEEE39.dyr"))
    psys.createYbusComplex()
    return psys


def _split_z(sysvec, psys):
    NDIFFEQ = psys.num_dof_dif
    alg_size = psys.num_dof_alg
    x = sysvec[:NDIFFEQ].copy()
    y = sysvec[NDIFFEQ:NDIFFEQ + alg_size].copy()
    v = sysvec[NDIFFEQ + alg_size:].copy()
    return x, y, v


def test_stage_solve_at_init_2bus(data_dir):
    psys = _build_two_bus_no_fault(data_dir)
    sysvec, theta, F, J = _init_psys_at_steady_state(psys)
    x0, y0, v0 = _split_z(sysvec, psys)

    y, v, iters = solve_stage_algebraic(x0, y0, v0, theta, psys, F, J,
                                        tol=1e-10, max_iter=10)

    # At a well-initialized state, expect at most 1 Newton iteration.
    assert iters <= 1
    z_check = np.concatenate([x0, y, v])
    residual_function(F, z_check, theta, psys)
    assert np.linalg.norm(F[psys.num_dof_dif:]) < 1e-10


def test_stage_solve_at_init_ieee9(data_dir):
    psys = _build_ieee9_no_fault(data_dir)
    sysvec, theta, F, J = _init_psys_at_steady_state(psys)
    x0, y0, v0 = _split_z(sysvec, psys)

    y, v, iters = solve_stage_algebraic(x0, y0, v0, theta, psys, F, J,
                                        tol=1e-10, max_iter=10)

    z_check = np.concatenate([x0, y, v])
    residual_function(F, z_check, theta, psys)
    assert np.linalg.norm(F[psys.num_dof_dif:]) < 1e-10


def test_stage_solve_recovers_from_perturbation(data_dir):
    psys = _build_ieee9_no_fault(data_dir)
    sysvec, theta, F, J = _init_psys_at_steady_state(psys)
    x0, y0, v0 = _split_z(sysvec, psys)

    rng = np.random.default_rng(0)
    y_pert = y0 + 1e-3 * rng.standard_normal(y0.shape)
    v_pert = v0 + 1e-3 * rng.standard_normal(v0.shape)

    y, v, iters = solve_stage_algebraic(x0, y_pert, v_pert, theta, psys, F, J,
                                        tol=1e-10, max_iter=50)

    assert iters >= 1
    z_check = np.concatenate([x0, y, v])
    residual_function(F, z_check, theta, psys)
    assert np.linalg.norm(F[psys.num_dof_dif:]) < 1e-10
    # Should recover close to the original consistent (y, v).
    assert np.linalg.norm(y - y0) < 1e-6
    assert np.linalg.norm(v - v0) < 1e-6


def test_stage_solve_freezes_x(data_dir):
    psys = _build_ieee9_no_fault(data_dir)
    sysvec, theta, F, J = _init_psys_at_steady_state(psys)
    x0, y0, v0 = _split_z(sysvec, psys)

    rng = np.random.default_rng(1)
    X_i = x0 + 1e-3 * rng.standard_normal(x0.shape)
    X_i_before = X_i.copy()

    y, v, _ = solve_stage_algebraic(X_i, y0, v0, theta, psys, F, J,
                                    tol=1e-10, max_iter=50)

    # X_i must not have been modified by the stage solve.
    assert np.array_equal(X_i, X_i_before)
    # Residual is g(X_i, y, v); verify by rebuilding z with the perturbed X_i.
    z_check = np.concatenate([X_i, y, v])
    residual_function(F, z_check, theta, psys)
    assert np.linalg.norm(F[psys.num_dof_dif:]) < 1e-10


# ---------------------------------------------------------------------------
# Phase 3 — HERK2 (Heun) per-step driver and flat-line check
# ---------------------------------------------------------------------------


def test_first_step_residual_herk2(data_dir):
    from uqgrid.simulation.herk import A_HEUN, B_HEUN, C_HEUN, herk_step

    psys = _build_ieee9_no_fault(data_dir)
    sysvec, theta, F, J = _init_psys_at_steady_state(psys)
    x0, y0, v0 = _split_z(sysvec, psys)

    h = 1e-5
    z1 = herk_step(sysvec, theta, h, psys, A_HEUN, B_HEUN, C_HEUN,
                   F, J, tol=1e-10, max_iter=50)

    residual_function(F, z1, theta, psys)
    assert np.linalg.norm(F[psys.num_dof_dif:]) < 1e-8

    x1 = z1[:psys.num_dof_dif]
    assert np.linalg.norm(x1 - x0, np.inf) < 1e-4


def _run_flatline_herk2(psys, tend=5.0, h=1e-3):
    cfg = IntegrationConfig(
        method="herk2",
        tend=tend,
        dt=h,
        steps=-1,
        power_injection=True,
        verbose=False,
        comp_sens=False,
        fsolve=False,
        petsc=False,
        ton=tend + 10.0,  # never trigger
        toff=tend + 11.0,
    )
    res = integrate_system(psys, cfg)
    return res


def test_flatline_herk2_2bus(data_dir):
    psys = _build_two_bus_no_fault(data_dir)
    res = _run_flatline_herk2(psys, tend=5.0, h=1e-3)
    NDIFFEQ = psys.num_dof_dif
    x_traj = res["history"][:NDIFFEQ, :]
    x0 = x_traj[:, 0]
    drift = np.max(np.abs(x_traj - x0[:, None]))
    assert drift < 1e-6, f"flat-line drift on 2-bus = {drift:.3e}"


def test_flatline_herk2_ieee9(data_dir):
    psys = _build_ieee9_no_fault(data_dir)
    res = _run_flatline_herk2(psys, tend=5.0, h=1e-3)
    NDIFFEQ = psys.num_dof_dif
    x_traj = res["history"][:NDIFFEQ, :]
    x0 = x_traj[:, 0]
    drift = np.max(np.abs(x_traj - x0[:, None]))
    assert drift < 1e-6, f"flat-line drift on IEEE9 = {drift:.3e}"


# ---------------------------------------------------------------------------
# Phase 4 — HERK4 (classical RK4) driver
# ---------------------------------------------------------------------------


def test_first_step_residual_herk4(data_dir):
    from uqgrid.simulation.herk import A_RK4, B_RK4, C_RK4, herk_step

    psys = _build_ieee9_no_fault(data_dir)
    sysvec, theta, F, J = _init_psys_at_steady_state(psys)
    x0, y0, v0 = _split_z(sysvec, psys)

    h = 1e-5
    z1 = herk_step(sysvec, theta, h, psys, A_RK4, B_RK4, C_RK4,
                   F, J, tol=1e-10, max_iter=50)

    residual_function(F, z1, theta, psys)
    assert np.linalg.norm(F[psys.num_dof_dif:]) < 1e-8

    x1 = z1[:psys.num_dof_dif]
    assert np.linalg.norm(x1 - x0, np.inf) < 1e-4


def _run_flatline_herk4(psys, tend=5.0, h=1e-3):
    cfg = IntegrationConfig(
        method="herk4",
        tend=tend,
        dt=h,
        steps=-1,
        power_injection=True,
        verbose=False,
        comp_sens=False,
        fsolve=False,
        petsc=False,
        ton=tend + 10.0,
        toff=tend + 11.0,
    )
    res = integrate_system(psys, cfg)
    return res


def test_flatline_herk4_ieee9(data_dir):
    psys = _build_ieee9_no_fault(data_dir)
    res = _run_flatline_herk4(psys, tend=5.0, h=1e-3)
    NDIFFEQ = psys.num_dof_dif
    x_traj = res["history"][:NDIFFEQ, :]
    x0 = x_traj[:, 0]
    drift = np.max(np.abs(x_traj - x0[:, None]))
    assert drift < 1e-6, f"flat-line drift on IEEE9 = {drift:.3e}"


def test_flatline_herk4_ieee39(data_dir):
    psys = _build_ieee39_no_fault(data_dir)
    res = _run_flatline_herk4(psys, tend=5.0, h=1e-3)
    NDIFFEQ = psys.num_dof_dif
    x_traj = res["history"][:NDIFFEQ, :]
    x0 = x_traj[:, 0]
    drift = np.max(np.abs(x_traj - x0[:, None]))
    assert drift < 1e-6, f"flat-line drift on IEEE39 = {drift:.3e}"


# ---------------------------------------------------------------------------
# Phase 5 — fault response cross-validated against backward Euler
# ---------------------------------------------------------------------------


def _run_ieee9_fault(data_dir, method):
    psys = _build_ieee9_fault(data_dir)
    cfg = IntegrationConfig(
        method=method,
        tend=5.0,
        dt=1e-4,
        steps=-1,
        power_injection=True,
        verbose=False,
        comp_sens=False,
        fsolve=False,
        petsc=False,
        ton=1.0,
        toff=1.1,
    )
    return psys, integrate_system(psys, cfg)


def _gen_angle_speed_indices(psys):
    angle_idx = [gen.dif_ptr + 5 for gen in psys.gendyn]
    speed_idx = psys.genspeed_idx_set()
    return angle_idx + speed_idx


def _save_ieee9_fault_plot(out_path, tvec, beuler, herk4, indices):
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ngen = len(indices) // 2
    angle_idx = indices[:ngen]
    speed_idx = indices[ngen:]

    for k, idx in enumerate(angle_idx):
        axes[0].plot(tvec, beuler[idx, :], color=f"C{k}", linewidth=1.0)
        axes[0].plot(tvec, herk4[idx, :], color=f"C{k}", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("delta")

    for k, idx in enumerate(speed_idx):
        axes[1].plot(tvec, beuler[idx, :], color=f"C{k}", linewidth=1.0)
        axes[1].plot(tvec, herk4[idx, :], color=f"C{k}", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("w")
    axes[1].set_xlabel("time [s]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def test_fault_response_herk4_vs_beuler_ieee9(data_dir):
    psys_be, res_be = _run_ieee9_fault(data_dir, "beuler")
    psys_rk, res_rk = _run_ieee9_fault(data_dir, "herk4")

    assert np.array_equal(res_be["tvec"], res_rk["tvec"])
    indices = _gen_angle_speed_indices(psys_be)
    rk_indices = _gen_angle_speed_indices(psys_rk)
    assert indices == rk_indices

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmarks",
        "herk_vs_beuler_ieee9.png",
    )
    _save_ieee9_fault_plot(
        out_path, res_be["tvec"], res_be["history"], res_rk["history"], indices
    )
    assert os.path.exists(out_path)

    ngen = len(indices) // 2
    diff = np.abs(res_be["history"][indices, :] - res_rk["history"][indices, :])
    angle_err = np.max(diff[:ngen, :])
    speed_err = np.max(diff[ngen:, :])
    assert angle_err < 5e-4, f"IEEE9 fault HERK4 vs BE rotor angle error = {angle_err:.3e}"
    assert speed_err < 1e-5, f"IEEE9 fault HERK4 vs BE speed error = {speed_err:.3e}"


def test_controller_blend_propagates_under_herk(data_dir):
    from uqgrid.simulation.herk import A_RK4, B_RK4, C_RK4, herk_step

    psys = _build_ieee9_gov_no_fault(data_dir)
    sysvec, theta, F, J = _init_psys_at_steady_state(psys)
    gov_indices = np.flatnonzero(psys.gov_mask)
    assert gov_indices.size > 0

    gen_idx = int(gov_indices[0])
    gen = psys.gendyn[gen_idx]
    gov = gen.governor

    h = 1e-4
    eps = 1e-3
    z_pert = sysvec.copy()
    z_pert[gov.dif_ptr + 4] += eps  # IEESGO TP3 directly shifts p_m output.

    base = herk_step(sysvec, theta, h, psys, A_RK4, B_RK4, C_RK4,
                     F, J, tol=1e-10, max_iter=50)
    pert = herk_step(z_pert, theta, h, psys, A_RK4, B_RK4, C_RK4,
                     F, J, tol=1e-10, max_iter=50)

    speed_idx = gen.dif_ptr + 4
    speed_delta = pert[speed_idx] - base[speed_idx]
    expected = h * eps / (2.0 * gen.H)

    assert speed_delta > 0.0
    assert speed_delta == pytest.approx(expected, rel=5e-2, abs=1e-11)


# ---------------------------------------------------------------------------
# Phase 6 — convergence-order study and Jacobian sanity
# ---------------------------------------------------------------------------


def test_stage_jacobian_matches_finite_difference(data_dir):
    from uqgrid.simulation.jacobian import residual_jacobian

    psys = _build_ieee9_no_fault(data_dir)
    sysvec, theta, F, J = _init_psys_at_steady_state(psys)
    residual_jacobian(J, sysvec, theta, psys)

    ndiff = psys.num_dof_dif
    Jgg = J[ndiff:, ndiff:].toarray()
    Jfd = np.zeros_like(Jgg)
    eps = 1e-6

    F_plus = np.zeros_like(sysvec)
    F_minus = np.zeros_like(sysvec)
    for col in range(Jgg.shape[1]):
        z_plus = sysvec.copy()
        z_minus = sysvec.copy()
        z_plus[ndiff + col] += eps
        z_minus[ndiff + col] -= eps
        residual_function(F_plus, z_plus, theta, psys)
        residual_function(F_minus, z_minus, theta, psys)
        Jfd[:, col] = (F_plus[ndiff:] - F_minus[ndiff:]) / (2.0 * eps)

    max_abs = np.max(np.abs(Jgg - Jfd))
    assert max_abs < 1e-6, f"stage algebraic Jacobian FD mismatch = {max_abs:.3e}"
