import json
from pathlib import Path

import numpy as np
import pytest

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.simulation.config import IntegrationConfig
from uqgrid.simulation.dynamics import integrate_system
from uqgrid.simulation.pflow import runpf


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "golden"


def _load_golden(name):
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


def test_phase0_power_flow_golden_output():
    expected = _load_golden("mcp_phase0_power_flow.json")
    psys = load_psse(str(ROOT / expected["case"]))
    psys.createYbusComplex()

    result = runpf(psys, verbose=False)

    assert psys.nbuses == expected["bus_count"]
    assert psys.ngens == expected["generator_count"]
    assert result.residual_norm == pytest.approx(expected["residual_norm"], abs=1e-8)
    assert np.min(result.v_magnitudes) == pytest.approx(expected["voltage_min_pu"])
    assert np.max(result.v_magnitudes) == pytest.approx(expected["voltage_max_pu"])
    np.testing.assert_allclose(result.v_magnitudes, expected["voltage_magnitudes_pu"])
    np.testing.assert_allclose(
        np.degrees(result.v_angles), expected["voltage_angles_deg"], atol=1e-9
    )
    assert result.q_limit_iterations == expected["q_limit_iterations"]
    assert len(result.q_limit_events) == expected["q_limit_event_count"]


def test_phase0_dynamics_golden_output():
    expected = _load_golden("mcp_phase0_dynamics.json")
    psys = load_psse(str(ROOT / expected["case"]))
    add_dyr(psys, str(ROOT / expected["dynamics"]))
    psys.add_busfault(psys.ext2int[expected["fault_bus_id"]], expected["fault_impedance_pu"])
    psys.createYbusComplex()
    integration = expected["integration"]

    result = integrate_system(
        psys,
        IntegrationConfig(
            method=integration["method"],
            dt=integration["dt_s"],
            tend=integration["end_s"],
            ton=integration["start_s"],
            toff=integration["clear_s"],
            petsc=integration["petsc"],
        ),
    )
    history = result["history"]
    speeds = history[psys.genspeed_idx_set(), :]
    voltages = history[psys.busmag_idx_set(), :]

    assert len(result["tvec"]) == expected["step_count"]
    assert history.shape[0] == expected["state_count"]
    np.testing.assert_allclose(result["tvec"], expected["time_s"])
    np.testing.assert_allclose(
        speeds[:, -1], expected["final_generator_speeds_pu"], atol=1e-10
    )
    assert np.min(voltages) == pytest.approx(expected["minimum_bus_voltage_pu"], abs=1e-9)
    assert np.max(np.abs(speeds)) == pytest.approx(
        expected["maximum_abs_generator_speed_pu"], abs=1e-9
    )
    assert np.linalg.norm(history) == pytest.approx(expected["history_l2_norm"], abs=1e-8)
