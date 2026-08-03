import os
import pytest
import numpy as np
from uqgrid.core.psydef import Psystem
from uqgrid.models import GenGENROU, GenGENSAL, GovTGOV1
from uqgrid.models.esdc1a_imp import esdc1a_sat_coefficients
from uqgrid.simulation.dynamics import initialize_system, integrate_system, preallocate_jacobian
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.jacobian_check import compare_jacobians
from uqgrid.io.parse import load_psse, add_dyr
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function
from uqgrid.simulation.config import IntegrationConfig, IntegrationCtx

EPS = 1e-10  # Tolerance for floating-point comparisons


@pytest.fixture
def data_dir():
    """
    Fixture to provide the absolute path to the data directory.
    Ensures that data files are correctly located regardless of the current working directory.
    """
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'
    )


def test_two_bus_system(data_dir):
    """
    TEST CASE 001: Two-bus system validation.
    
    This test loads a two-bus system from a PSSE raw file, applies a fault,
    adds generator dynamics, integrates the system using IntegrationConfig,
    and compares the simulation results against expected PSSE data.
    """
    zfault = 1.0  # Fault impedance (p.u.)
    fault_time = 1.0  # Fault initiation time in seconds

    # Define the integration configuration using IntegrationConfig
    config = IntegrationConfig(
        tend=10.0,                 # Integration end time in seconds
        dt=1.0 / 120.0,            # Time step in seconds
        steps=-1,                # Number of integration steps (-1 for automatic)
        power_injection=True,     # Adjust based on your requirements
        verbose=False,             # Disable verbose output for testing
        comp_sens=False,           # Disable sensitivity computation
        fsolve=False,              # Disable fsolve for nonlinear equations
        petsc=False,               # Disable PETSc integration
        zfault=1.0,                # Fault impedance (p.u.)
        ton=0.25,                   # Fault activation time
        toff=0.4                    # Fault deactivation time
        # Include other necessary fields if added to IntegrationConfig
    )

    # Load power system from PSSE raw file
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    psys.createYbusComplex()
    psys.add_busfault(1, zfault)

    # Add generator dynamics
    psys.add_gen_dynamics(
        psys.gens[0],
        GenGENROU(
            0, 1.575, 1.512, 0.291, 0.39, 0.1733,
            0.0787, 3.38, 0.0, 6.1, 1.0, 0.05, 0.15, 0.0, 0.0
        )
    )

    # Integrate the system using IntegrationConfig
    res = integrate_system(psys, config=config)

    # Load expected PSSE data
    psse = np.loadtxt(os.path.join(data_dir, '2bus_GENROU.csv'), delimiter=',')

    # Retrieve PSSE values. Delete erroneous or switching event indices
    indices_to_remove = [0, 1, 33, 52]
    time_p = np.delete(psse[0, :], indices_to_remove)
    volt1_p = np.delete(psse[1, :], indices_to_remove)
    volt2_p = np.delete(psse[3, :], indices_to_remove)
    eq_p = np.delete(psse[5, :], indices_to_remove)
    speed = np.delete(psse[9, :], indices_to_remove)

    # Extract simulation history
    history = res["history"]
    np.testing.assert_allclose(res["tvec"][[0, -1]], [0.0, config.tend])

    # The PSSE trace repeats the pre-switch state at each event time. UQGrid's
    # normalized contract stores the post-switch algebraic state there, so
    # compare every sample except the two topology boundaries.
    comparison_mask = np.ones(history.shape[1], dtype=bool)
    comparison_mask[np.isclose(res["tvec"], config.ton)] = False
    comparison_mask[np.isclose(res["tvec"], config.toff)] = False
    history = history[:, comparison_mask]
    volt1_p = volt1_p[comparison_mask]
    volt2_p = volt2_p[comparison_mask]
    eq_p = eq_p[comparison_mask]
    speed = speed[comparison_mask]

    gen = psys.gendyn[0]
    dif_ptr = gen.dif_ptr

    eqp_idx = dif_ptr  # e_qp
    speed_idx = dif_ptr + 4

    busmag_idx = psys.busmag_idx_set()
    volt1_idx = busmag_idx[0]
    volt2_idx = busmag_idx[1]

    # Calculate relative errors using programmatic indices
    error_volt1 = np.linalg.norm(np.abs((volt1_p - history[volt1_idx, :]) / history[volt1_idx, :]))
    error_volt2 = np.linalg.norm(np.abs((volt2_p - history[volt2_idx, :]) / history[volt2_idx, :]))
    error_eqp = np.linalg.norm(np.abs((eq_p - history[eqp_idx, :]) / history[eqp_idx, :]))
    error_speed = np.linalg.norm(np.abs((speed - history[speed_idx, :]) / (history[speed_idx, :] + 1)))

    # Assertions to verify simulation accuracy
    assert error_volt1 < 0.01, 'Voltage 1 trajectory differs'
    assert error_volt2 < 0.01, 'Voltage 2 trajectory differs'
    assert error_eqp < 0.01, 'E\'q trajectory differs'
    assert error_speed < 0.01, 'Speed trajectory differs'


def test_tgov1_dt_uses_machine_to_system_base_scaling(data_dir, tmp_path):
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    dyr_path = tmp_path / "ieee9_tgov1_scaling.dyr"
    dyr_path.write_text(
        """
1 'GENROU' 1 7.729 0.047 0.859 0.068 4.31 0.0 1.9266 1.8442 0.3812 0.5469 0.2889 0.2443 0.115 0.627 /
1 'TGOV1' 1 0.05 0.1 1.2 -0.1 0.2 10.0 0.3 /
""".lstrip()
    )

    add_dyr(psys, str(dyr_path))

    gov = psys.gov[0]
    mbase = psys.gens[0].mbase
    sbase = psys.basemva
    system_to_machine = sbase / mbase
    machine_to_system = mbase / sbase

    assert gov.R == pytest.approx(0.05 * system_to_machine)
    assert gov.VMAX == pytest.approx(1.2 * system_to_machine)
    assert gov.VMIN == pytest.approx(-0.1 * system_to_machine)
    assert gov.DT == pytest.approx(0.3 * machine_to_system)


def test_tgov1_limits_are_stored_but_disabled_by_default():
    gov = GovTGOV1("1", R=0.05, T1=0.1, VMAX=1.2, VMIN=-0.1, T2=0.2, T3=10.0, DT=0.3)
    gov.par_ptr = 0
    gov.pref = 0.7
    theta = np.zeros(gov.par_dim)

    gov.initialize_theta(theta)

    assert gov.VMAX == 1.2
    assert gov.VMIN == -0.1
    assert gov.enable_limits is False
    assert theta[7] == 0.0


def test_gensal_maps_to_salient_generator_parameters(data_dir, tmp_path):
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    dyr_path = tmp_path / "ieee9_gensal.dyr"
    dyr_path.write_text(
        """
1 'GENSAL' 1 6.0 0.05 0.04 4.31 0.1 1.8 1.1 0.3 0.25 0.12 0.2 0.6 /
""".lstrip()
    )

    add_dyr(psys, str(dyr_path))

    gen = psys.gendyn[0]
    assert isinstance(gen, GenGENSAL)
    assert gen.T_d0p == pytest.approx(6.0)
    assert gen.T_q0p == pytest.approx(6.0)
    assert gen.T_d0dp == pytest.approx(0.05)
    assert gen.T_q0dp == pytest.approx(0.04)
    assert gen.x_d == pytest.approx(1.8 * psys.basemva / psys.gens[0].mbase)
    assert gen.x_q == pytest.approx(1.1 * psys.basemva / psys.gens[0].mbase)
    assert gen.x_dp == pytest.approx(0.3 * psys.basemva / psys.gens[0].mbase)
    assert gen.x_qp == pytest.approx(gen.x_dp)
    assert gen.x_ddp == pytest.approx(0.25 * psys.basemva / psys.gens[0].mbase)
    assert gen.x_qdp == pytest.approx(gen.x_ddp)


def test_gensal_initialization_and_jacobian(data_dir, tmp_path):
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    dyr_path = tmp_path / "ieee9_gensal_init.dyr"
    dyr_path.write_text(
        """
1 'GENSAL' 1 6.0 0.05 0.04 4.31 0.1 1.8 1.1 0.3 0.25 0.12 0.2 0.6 /
""".lstrip()
    )

    add_dyr(psys, str(dyr_path))
    psys.createYbusComplex()
    pf_solution = runpf(psys, verbose=False)
    sysvec, theta = initialize_system(psys, pf_solution)
    residual = np.zeros_like(sysvec)
    residual_function(residual, sysvec, theta, psys)
    jacobian = preallocate_jacobian(psys)
    residual_jacobian(jacobian, sysvec, theta, psys)
    residual_jacobian(jacobian, sysvec, theta, psys)
    mismatches = compare_jacobians(
        psys, sysvec, theta, jacobian, eps=1e-6, top_k=10, tol=1e-5,
    )

    assert np.linalg.norm(residual, np.inf) < 1e-8
    assert mismatches == []


def test_unmatched_generators_use_aggregated_static_devices(data_dir, tmp_path):
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    psys.add_gen(
        bus=1,
        idx_name="extra",
        psch=0.0,
        qsch=0.0,
        pgub=10.0,
        pglb=-10.0,
        qgub=5.0,
        qglb=-5.0,
    )
    dyr_path = tmp_path / "empty.dyr"
    dyr_path.write_text("")

    add_dyr(psys, str(dyr_path))

    assert len(psys.gendyn) == 0
    assert len(psys.static_gens) == 3
    static_by_bus = {gen.bus: gen for gen in psys.static_gens}
    assert len(static_by_bus[1].gen_idxs) == 2
    assert static_by_bus[1].enable_limits is False
    assert static_by_bus[1].pmax == pytest.approx(
        psys.gens[1].pgub + psys.gens[3].pgub
    )
    assert static_by_bus[1].qmin == pytest.approx(
        psys.gens[1].qglb + psys.gens[3].qglb
    )


def test_static_generator_initialization_and_jacobian(data_dir, tmp_path):
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    dyr_path = tmp_path / "empty.dyr"
    dyr_path.write_text("")

    add_dyr(psys, str(dyr_path))
    psys.createYbusComplex()
    pf_solution = runpf(psys, verbose=False)
    sysvec, theta = initialize_system(psys, pf_solution)
    residual = np.zeros_like(sysvec)
    residual_function(residual, sysvec, theta, psys)
    jacobian = preallocate_jacobian(psys)
    residual_jacobian(jacobian, sysvec, theta, psys)
    residual_jacobian(jacobian, sysvec, theta, psys)
    mismatches = compare_jacobians(
        psys, sysvec, theta, jacobian, eps=1e-6, top_k=10, tol=1e-5,
    )

    assert np.linalg.norm(residual, np.inf) < 1e-8
    assert mismatches == []


@pytest.mark.parametrize("method", ["beuler", "cn"])
def test_petsc_uses_q_limited_initial_power_flow(data_dir, monkeypatch, method):
    pytest.importorskip("petsc4py")
    from uqgrid.simulation import dynamics

    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus.dyr"))
    psys.gens[1].qgub = 0.02
    psys.add_busfault(4, 1e-4)
    psys.createYbusComplex()
    captured = []
    original_runpf = dynamics.runpf

    def recording_runpf(*args, **kwargs):
        result = original_runpf(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(dynamics, "runpf", recording_runpf)
    config = IntegrationConfig(
        steps=2,
        dt=1.0 / 120.0,
        power_injection=False,
        petsc=True,
        method=method,
        ton=10.0,
        toff=11.0,
        enforce_q_limits=True,
        power_flow_validation={
            "enabled": True,
            "voltage_min": 0.9,
            "voltage_max": 1.1,
        },
    )

    result = integrate_system(psys, config)

    assert len(captured) == 1
    assert captured[0].q_limit_events[0]["side"] == "upper"
    assert captured[0].gen_qsch[1] == pytest.approx(psys.gens[1].qgub)
    assert result["power_flow_diagnostics"] == captured[0].validation
    assert result["dynamic_limit_diagnostics"]["initialization"]["valid"] is True
    assert result["history"].shape[1] == 3
    np.testing.assert_allclose(result["tvec"], [0.0, config.dt, 2 * config.dt])


def test_petsc_steps_truncate_fault_interval(data_dir, monkeypatch):
    pytest.importorskip("petsc4py")

    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus.dyr"))
    psys.add_busfault(4, 1e-4)
    psys.createYbusComplex()
    fault = psys.fault_events[0]
    events = []
    original_apply = fault.apply
    original_remove = fault.remove

    def recording_apply():
        events.append("apply")
        original_apply()

    def recording_remove():
        events.append("remove")
        original_remove()

    monkeypatch.setattr(fault, "apply", recording_apply)
    monkeypatch.setattr(fault, "remove", recording_remove)
    config = IntegrationConfig(
        steps=3,
        dt=1.0 / 120.0,
        power_injection=False,
        petsc=True,
        ton=1.0 / 120.0,
        toff=10.0,
    )

    result = integrate_system(psys, config)

    assert result["tvec"][-1] == pytest.approx(3.0 / 120.0)
    assert np.max(result["tvec"]) <= 3.0 / 120.0 + 1e-12
    assert "apply" in events
    assert events[-1] == "remove"
    assert fault.active is False


@pytest.mark.parametrize("method", ["beuler", "cn"])
def test_petsc_uses_exact_off_grid_fault_transitions(data_dir, method):
    pytest.importorskip("petsc4py")

    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus.dyr"))
    psys.add_busfault(4, 1e-4)
    psys.createYbusComplex()
    pf_solution = runpf(psys, verbose=False, enforce_q_limits=True)
    z0, theta = initialize_system(psys, pf_solution)
    ctx = IntegrationCtx()
    ctx.set_initial_conditions(z0.copy())
    ctx.set_theta(theta.copy())
    psys.fault_events[0].apply()
    dt = 1.0 / 120.0
    config = IntegrationConfig(
        method=method,
        steps=4,
        dt=dt,
        power_injection=False,
        petsc=True,
        ton=1.5 * dt,
        toff=2.7 * dt,
    )

    result = integrate_system(psys, config, ctx)

    expected_times = [0.0, dt, 1.5 * dt, 2 * dt, 2.7 * dt, 3 * dt, 4 * dt]
    np.testing.assert_allclose(result["tvec"], expected_times)
    np.testing.assert_allclose(result["history"][:, 0], z0)
    assert np.all(np.diff(result["tvec"]) > 0.0)

    residual = np.zeros_like(z0)
    fault = psys.fault_events[0]
    fault.remove()
    residual_function(residual, result["history"][:, 1], theta, psys)
    assert np.linalg.norm(residual[psys.num_dof_dif:], np.inf) < 1e-8
    fault.apply()
    residual_function(residual, result["history"][:, 2], theta, psys)
    assert np.linalg.norm(residual[psys.num_dof_dif:], np.inf) < 1e-8
    fault.remove()
    residual_function(residual, result["history"][:, 4], theta, psys)
    assert np.linalg.norm(residual[psys.num_dof_dif:], np.inf) < 1e-8
    assert fault.active is False


def test_petsc_arkimex_uses_common_grid_without_fault(data_dir):
    pytest.importorskip("petsc4py")

    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, "GENROU.dyr"))
    psys.createYbusComplex()
    config = IntegrationConfig(
        petsc=True,
        arkimex=True,
        enforce_dynamic_limits=False,
        steps=2,
        dt=0.01,
        ton=10.0,
        toff=11.0,
        power_injection=True,
    )

    result = integrate_system(psys, config)

    np.testing.assert_allclose(result["tvec"], [0.0, 0.01, 0.02])
    assert result["history"].shape[1] == 3


def test_petsc_arkimex_uses_exact_off_grid_fault_transitions(data_dir):
    pytest.importorskip("petsc4py")

    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, "GENROU.dyr"))
    psys.add_busfault(1, 1.0)
    psys.createYbusComplex()
    pf_solution = runpf(psys, verbose=False)
    z0, theta = initialize_system(psys, pf_solution)
    ctx = IntegrationCtx()
    ctx.set_initial_conditions(z0.copy())
    ctx.set_theta(theta.copy())
    config = IntegrationConfig(
        petsc=True,
        arkimex=True,
        enforce_dynamic_limits=False,
        steps=4,
        dt=0.01,
        ton=0.015,
        toff=0.027,
        power_injection=True,
    )

    result = integrate_system(psys, config, ctx)

    np.testing.assert_allclose(
        result["tvec"], [0.0, 0.01, 0.015, 0.02, 0.027, 0.03, 0.04]
    )
    residual = np.zeros_like(z0)
    fault = psys.fault_events[0]
    fault.apply()
    residual_function(residual, result["history"][:, 2], theta, psys)
    assert np.linalg.norm(residual[psys.num_dof_dif:], np.inf) < 1e-8
    fault.remove()
    residual_function(residual, result["history"][:, 4], theta, psys)
    assert np.linalg.norm(residual[psys.num_dof_dif:], np.inf) < 1e-8
    assert fault.active is False


def test_static_generators_reject_inconsistent_voltage_setpoints(data_dir, tmp_path):
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    psys.add_gen(bus=1, idx_name="extra", psch=0.0, qsch=0.0, vset=0.95)
    dyr_path = tmp_path / "empty.dyr"
    dyr_path.write_text("")

    with pytest.raises(ValueError, match="inconsistent voltage setpoints"):
        add_dyr(psys, str(dyr_path))


def test_unmatched_tgov1_logs_warning_and_is_skipped(data_dir, tmp_path, caplog):
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    dyr_path = tmp_path / "unmatched_tgov1.dyr"
    dyr_path.write_text(
        """
1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /
1 'TGOV1' 2 0.05 0.1 1.2 -0.1 0.2 10.0 0.3 /
""".lstrip()
    )

    with caplog.at_level("WARNING", logger="uqgrid.io.parse"):
        add_dyr(psys, str(dyr_path))

    assert len(psys.gov) == 0
    assert "Cannot pair TGOV1" in caplog.text


def test_unmatched_sexs_logs_warning_and_is_skipped(data_dir, tmp_path, caplog):
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    dyr_path = tmp_path / "unmatched_sexs.dyr"
    dyr_path.write_text(
        """
1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /
1 'SEXS' 2 0.1 0.2 100.0 0.05 -999.0 999.0 /
""".lstrip()
    )

    with caplog.at_level("WARNING", logger="uqgrid.io.parse"):
        add_dyr(psys, str(dyr_path))

    assert len(psys.exc) == 0
    assert "Cannot pair SEXS" in caplog.text


def test_valid_tgov1_and_sexs_records_attach_without_warning(data_dir, tmp_path, caplog, capsys):
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    dyr_path = tmp_path / "valid_controllers.dyr"
    dyr_path.write_text(
        """
1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /
1 'TGOV1' 1 0.05 0.1 1.2 -0.1 0.2 10.0 0.3 /
1 'SEXS' 1 0.1 0.2 100.0 0.05 -999.0 999.0 /
""".lstrip()
    )

    with caplog.at_level("WARNING", logger="uqgrid.io.parse"):
        add_dyr(psys, str(dyr_path))

    captured = capsys.readouterr()
    assert len(psys.gov) == 1
    assert len(psys.exc) == 1
    assert psys.exc[0].enable_limits is True
    assert "Cannot pair TGOV1" not in caplog.text
    assert "Cannot pair SEXS" not in caplog.text
    assert captured.out == ""


@pytest.mark.parametrize(
    "emin, emax, message",
    [
        ("nan", "1.0", "EMIN and EMAX must be finite"),
        ("0.5", "0.5", "EMIN (0.5) must be less than EMAX (0.5)"),
        ("1.0", "0.5", "EMIN (1.0) must be less than EMAX (0.5)"),
    ],
)
def test_invalid_parsed_sexs_limits_fail_with_device_identity(
    data_dir, tmp_path, emin, emax, message
):
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    dyr_path = tmp_path / "invalid_sexs_limits.dyr"
    dyr_path.write_text(
        f"""
1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /
1 'SEXS' 1 0.1 0.2 100.0 0.05 {emin} {emax} /
""".lstrip()
    )

    with pytest.raises(ValueError) as exc_info:
        add_dyr(psys, str(dyr_path))

    assert "bus 1, generator 1" in str(exc_info.value)
    assert message in str(exc_info.value)


def test_add_dyr_verbose_uses_info_logging_not_stdout(data_dir, tmp_path, caplog, capsys):
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    dyr_path = tmp_path / "verbose_controllers.dyr"
    dyr_path.write_text(
        """
1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /
1 'TGOV1' 1 0.05 0.1 1.2 -0.1 0.2 10.0 0.3 /
1 'SEXS' 1 0.1 0.2 100.0 0.05 -999.0 999.0 /
""".lstrip()
    )

    with caplog.at_level("INFO", logger="uqgrid.io.parse"):
        add_dyr(psys, str(dyr_path), verbose=True)

    captured = capsys.readouterr()
    assert "Adding GENROU at bus 1. GENID 1." in caplog.text
    assert "Adding TGOV1 at bus 1. GENID 1." in caplog.text
    assert "Adding SEXS at bus 1. GENID 1." in caplog.text
    assert captured.out == ""


def test_esdc1a_parser_uses_dyr_parameters(data_dir, tmp_path):
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    dyr_path = tmp_path / "esdc1a_distinctive.dyr"
    dyr_path.write_text(
        """
1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /
1 'ESDC1A' 1 0.03 31.0 1.4 0.8 0.9 12.0 -11.0 6.5 0.45 0.6 0.75 0.0 1.3 0.02 1.7 0.04 /
""".lstrip()
    )

    add_dyr(psys, str(dyr_path))

    assert len(psys.exc) == 1
    exc = psys.exc[0]
    assert exc.Ka == pytest.approx(31.0)
    assert exc.Ta == pytest.approx(1.4)
    assert exc.Kf == pytest.approx(0.6)
    assert exc.Tf == pytest.approx(0.75)
    assert exc.Ke == pytest.approx(6.5)
    assert exc.Te == pytest.approx(0.45)
    assert exc.Tr == pytest.approx(0.03)
    assert exc.Tb == pytest.approx(0.8)
    assert exc.Tc == pytest.approx(0.9)
    assert exc.Vrmax == pytest.approx(12.0)
    assert exc.Vrmin == pytest.approx(-11.0)
    assert exc.Sw == pytest.approx(0.0)
    assert exc.E1 == pytest.approx(1.3)
    assert exc.SE1 == pytest.approx(0.02)
    assert exc.E2 == pytest.approx(1.7)
    assert exc.SE2 == pytest.approx(0.04)

    sat_a, sat_b = esdc1a_sat_coefficients(exc.E1, exc.SE1, exc.E2, exc.SE2)
    assert exc.sat_a == pytest.approx(sat_a)
    assert exc.sat_b == pytest.approx(sat_b)
    assert sat_b * (exc.E1 - sat_a) ** 2 == pytest.approx(exc.E1 * exc.SE1)
    assert sat_b * (exc.E2 - sat_a) ** 2 == pytest.approx(exc.E2 * exc.SE2)


def test_esdc1a_initialization_residual_uses_saturation_points(data_dir, tmp_path):
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    dyr_path = tmp_path / "esdc1a_saturated.dyr"
    dyr_path.write_text(
        """
1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /
1 'ESDC1A' 1 0.03 31.0 1.4 0.0 0.0 12.0 -11.0 6.5 0.45 0.6 0.75 0.0 1.0 0.02 1.2 0.04 /
""".lstrip()
    )

    add_dyr(psys, str(dyr_path))
    psys.createYbusComplex()
    pf_solution = runpf(psys, verbose=False)
    sysvec, theta = initialize_system(psys, pf_solution)

    F = np.zeros_like(sysvec)
    residual_function(F, sysvec, theta, psys)

    exc = psys.exc[0]
    exc_slice = slice(exc.dif_ptr, exc.dif_ptr + exc.dif_dim)
    assert np.linalg.norm(F[exc_slice], np.inf) < 1e-10
