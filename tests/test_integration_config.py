import numpy as np
import pytest

from uqgrid.simulation import dynamics
from uqgrid.simulation.config import IntegrationConfig


def test_integration_config_accepts_slow_partition():
    cfg = IntegrationConfig(arkimex=True, arkimex_slow_differential=[0, 2, 4])
    assert cfg.arkimex_slow_differential == [0, 2, 4]
    assert cfg.arkimex_fast_differential is None


def test_integration_config_accepts_fast_partition():
    cfg = IntegrationConfig(arkimex=True, arkimex_fast_differential=[1, 3])
    assert cfg.arkimex_fast_differential == [1, 3]
    assert cfg.arkimex_slow_differential is None


def test_integration_config_rejects_both_fast_and_slow_lists():
    with pytest.raises(ValueError):
        IntegrationConfig(
            arkimex=True,
            arkimex_fast_differential=[1, 3],
            arkimex_slow_differential=[0, 2],
        )


def test_integration_config_q_limit_defaults():
    cfg = IntegrationConfig()

    assert cfg.enforce_q_limits is False
    assert cfg.q_limit_tolerance == pytest.approx(1e-8)
    assert cfg.max_q_limit_iterations is None
    assert cfg.power_flow_validation.enabled is False
    assert cfg.power_flow_validation.voltage_min is None
    assert cfg.power_flow_validation.branch_loading_max is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"q_limit_tolerance": -1e-8},
        {"max_q_limit_iterations": 0},
    ],
)
def test_integration_config_rejects_invalid_q_limit_controls(kwargs):
    with pytest.raises(ValueError):
        IntegrationConfig(**kwargs)


@pytest.mark.parametrize(
    "validation",
    [
        {"residual_tolerance": -1e-8},
        {"generator_limit_tolerance": -1e-6},
        {"branch_loading_max": 0.0},
        {"branch_limit_tolerance": -1e-5},
        {"active_set_voltage_tolerance": -1e-6},
        {"voltage_min": 1.1, "voltage_max": 0.9},
    ],
)
def test_integration_config_rejects_invalid_power_flow_validation(validation):
    with pytest.raises(ValueError):
        IntegrationConfig(power_flow_validation=validation)


@pytest.mark.parametrize("petsc", [False, True])
def test_shared_initializer_forwards_q_limit_controls(monkeypatch, petsc):
    calls = []
    pf_solution = object()
    z0 = np.array([1.0])
    theta = np.array([2.0])
    psys = object()

    def fake_runpf(received_psys, **kwargs):
        calls.append((received_psys, kwargs))
        return pf_solution

    def fake_initialize_system(received_psys, received_pf):
        assert received_psys is psys
        assert received_pf is pf_solution
        return z0, theta

    monkeypatch.setattr(dynamics, "runpf", fake_runpf)
    monkeypatch.setattr(dynamics, "initialize_system", fake_initialize_system)
    cfg = IntegrationConfig(
        petsc=petsc,
        enforce_q_limits=True,
        q_limit_tolerance=2e-7,
        max_q_limit_iterations=9,
    )

    actual_pf, actual_z0, actual_theta = dynamics._initialize_system_from_config(
        psys, cfg,
    )

    assert actual_pf is pf_solution
    assert actual_z0 is z0
    assert actual_theta is theta
    assert calls == [(
        psys,
        {
            "verbose": False,
            "enforce_q_limits": True,
            "q_limit_tolerance": 2e-7,
            "max_q_limit_iterations": 9,
        },
    )]
