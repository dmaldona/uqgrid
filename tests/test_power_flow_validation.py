import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from uqgrid.core.psydef import Bus
from uqgrid.io.parse import load_matpower
from uqgrid.simulation import dynamics
from uqgrid.simulation.config import IntegrationConfig
from uqgrid.simulation.pflow import (
    PowerFlowValidationError,
    compute_branch_loading_diagnostics,
    runpf,
    validate_power_flow_solution,
)


@pytest.fixture
def case9():
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    psys = load_matpower(os.path.join(data_dir, "case9.m"))
    psys.createYbusComplex()
    solution = runpf(psys, enforce_q_limits=True)
    return psys, solution


def _validate(psys, solution, **overrides):
    options = {
        "residual_tolerance": 1e-8,
        "generator_limit_tolerance": 1e-6,
        "voltage_min": 0.9,
        "voltage_max": 1.1,
        "branch_loading_max": None,
        "branch_limit_tolerance": 1e-5,
        "active_set_voltage_tolerance": 1e-6,
    }
    options.update(overrides)
    return validate_power_flow_solution(psys, solution, **options)


def test_valid_solution_produces_json_safe_diagnostics(case9):
    psys, solution = case9

    diagnostics = _validate(psys, solution)

    assert diagnostics["valid"]
    assert diagnostics["failure_reasons"] == []
    assert diagnostics["island_slack"]["invalid_island_count"] == 0
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize(
    "options",
    [
        {"residual_tolerance": -1e-8},
        {"generator_limit_tolerance": -1e-6},
        {"branch_loading_max": 0.0},
        {"branch_limit_tolerance": -1e-5},
        {"active_set_voltage_tolerance": -1e-6},
        {"voltage_min": 1.1, "voltage_max": 0.9},
    ],
)
def test_direct_validation_rejects_invalid_options(case9, options):
    psys, solution = case9

    with pytest.raises(ValueError):
        validate_power_flow_solution(psys, solution, **options)


def test_validation_accumulates_residual_voltage_and_generator_failures(case9):
    psys, solution = case9
    solution.residual_norm = 1e-3
    solution.v_magnitudes[3] = 0.8
    solution.v_magnitudes[4] = 1.2
    solution.gen_psch[1] = psys.gens[1].pgub + 0.1
    solution.gen_qsch[2] = psys.gens[2].qglb - 0.1

    diagnostics = _validate(psys, solution)

    assert diagnostics["valid"] is False
    assert diagnostics["failure_reasons"] == [
        "pf_residual",
        "gen_p_limit",
        "gen_q_limit",
        "voltage_low",
        "voltage_high",
    ]
    assert diagnostics["gen_p"]["violation_count"] == 1
    assert diagnostics["gen_q"]["violation_count"] == 1


def test_validation_rejects_nonfinite_voltage_with_json_safe_output(case9):
    psys, solution = case9
    solution.v_magnitudes[3] = np.nan

    diagnostics = _validate(psys, solution)

    assert "nonfinite_voltage" in diagnostics["failure_reasons"]
    json.dumps(diagnostics, allow_nan=False)


def test_branch_loading_tolerance_accepts_and_rejects_same_solution(case9):
    psys, solution = case9
    branch = compute_branch_loading_diagnostics(psys, solution)["loading_top"][0]
    branch_index = branch["branch_index"]
    apparent_power = max(branch["s_from_mva"], branch["s_to_mva"])
    psys.branches[branch_index].rateA = apparent_power / 1.0000027227879946

    accepted = _validate(
        psys,
        solution,
        branch_loading_max=1.0,
        branch_limit_tolerance=1e-5,
    )
    rejected = _validate(
        psys,
        solution,
        branch_loading_max=1.0,
        branch_limit_tolerance=1e-7,
    )

    assert accepted["branch"]["loading_max"] == pytest.approx(1.0000027227879946)
    assert accepted["valid"]
    assert "branch_overload" in rejected["failure_reasons"]


@pytest.mark.parametrize(
    ("gen_index", "bound_name", "bound_value", "side", "voltage_offset"),
    [
        (1, "qgub", 0.02, "upper", 1e-3),
        (2, "qglb", -0.05, "lower", -1e-3),
    ],
)
def test_active_set_complementarity_is_validated(
        case9, gen_index, bound_name, bound_value, side, voltage_offset):
    psys, _ = case9
    setattr(psys.gens[gen_index], bound_name, bound_value)
    solution = runpf(psys, enforce_q_limits=True)
    bus_index = psys.gens[gen_index].bus
    assert solution.q_limit_events[0]["side"] == side
    solution.v_magnitudes[bus_index] = (
        psys.buses[bus_index].v0m + voltage_offset
    )

    diagnostics = _validate(psys, solution)

    assert "active_set_inconsistent" in diagnostics["failure_reasons"]
    assert diagnostics["active_set"]["violation_count"] == 1


def test_validation_requires_one_slack_per_island(case9):
    psys, solution = case9
    psys.buses[1].type = Bus.SLACK

    diagnostics = _validate(psys, solution)

    assert "invalid_slack_topology" in diagnostics["failure_reasons"]
    assert diagnostics["island_slack"]["invalid_island_count"] == 1


def test_validation_error_is_raised_before_dynamic_initialization(
        case9, monkeypatch):
    psys, solution = case9
    solution.residual_norm = 1e-3
    initialized = False

    def fake_runpf(*args, **kwargs):
        return solution

    def fake_initialize_system(*args, **kwargs):
        nonlocal initialized
        initialized = True
        raise AssertionError("initialize_system must not run")

    monkeypatch.setattr(dynamics, "runpf", fake_runpf)
    monkeypatch.setattr(dynamics, "initialize_system", fake_initialize_system)
    config = IntegrationConfig(
        power_flow_validation={
            "enabled": True,
            "residual_tolerance": 1e-8,
        },
    )

    with pytest.raises(PowerFlowValidationError) as exc_info:
        dynamics._initialize_system_from_config(psys, config)

    assert initialized is False
    assert exc_info.value.diagnostics["failure_reasons"] == ["pf_residual"]


def test_disabled_validation_preserves_initialization(case9, monkeypatch):
    psys, solution = case9
    solution.residual_norm = 1e-3
    z0 = np.array([1.0])
    theta = np.array([2.0])

    monkeypatch.setattr(dynamics, "runpf", lambda *args, **kwargs: solution)
    monkeypatch.setattr(
        dynamics,
        "initialize_system",
        lambda *args, **kwargs: (z0, theta),
    )

    actual_solution, actual_z0, actual_theta = (
        dynamics._initialize_system_from_config(psys, IntegrationConfig())
    )

    assert actual_solution.validation is None
    assert actual_z0 is z0
    assert actual_theta is theta


def test_validation_failure_precedes_petsc_setup(monkeypatch):
    diagnostics = {"valid": False, "failure_reasons": ["gen_q_limit"]}
    petsc_requested = False

    def fail_initialization(*args, **kwargs):
        raise PowerFlowValidationError(diagnostics)

    def request_petsc(*args, **kwargs):
        nonlocal petsc_requested
        petsc_requested = True
        return object()

    monkeypatch.setattr(
        dynamics, "_initialize_system_from_config", fail_initialization,
    )
    monkeypatch.setattr(dynamics, "_get_petsc_for_config", request_petsc)
    psys = SimpleNamespace(power_injection=False)
    config = IntegrationConfig(
        petsc=True,
        power_flow_validation={"enabled": True},
    )

    with pytest.raises(PowerFlowValidationError):
        dynamics.integrate_system(psys, config)

    assert petsc_requested is False
