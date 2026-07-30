import numpy as np
import pytest

from uqgrid.simulation import dynamics
from uqgrid.simulation.config import IntegrationConfig


def test_integration_config_accepts_slow_partition():
    cfg = IntegrationConfig(
        petsc=True,
        arkimex=True,
        enforce_dynamic_limits=False,
        arkimex_slow_differential=[0, 2, 4],
    )
    assert cfg.arkimex_slow_differential == [0, 2, 4]
    assert cfg.arkimex_fast_differential is None


def test_integration_config_accepts_fast_partition():
    cfg = IntegrationConfig(
        petsc=True,
        arkimex=True,
        enforce_dynamic_limits=False,
        arkimex_fast_differential=[1, 3],
    )
    assert cfg.arkimex_fast_differential == [1, 3]
    assert cfg.arkimex_slow_differential is None


def test_integration_config_rejects_both_fast_and_slow_lists():
    with pytest.raises(ValueError):
        IntegrationConfig(
            petsc=True,
            arkimex=True,
            enforce_dynamic_limits=False,
            arkimex_fast_differential=[1, 3],
            arkimex_slow_differential=[0, 2],
        )


def test_integration_config_q_limit_defaults():
    cfg = IntegrationConfig()

    assert cfg.enforce_q_limits is True
    assert cfg.q_limit_tolerance == pytest.approx(1e-8)
    assert cfg.max_q_limit_iterations is None
    assert cfg.power_flow_validation.enabled is False
    assert cfg.power_flow_validation.voltage_min is None
    assert cfg.power_flow_validation.branch_loading_max is None
    assert IntegrationConfig(enforce_q_limits=False).enforce_q_limits is False


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dt": 0.0}, "greater than 0"),
        ({"steps": 0}, "positive integer"),
        ({"steps": -2}, "positive integer"),
        ({"ton": 0.2, "toff": 0.1}, "toff"),
        ({"method": "bogus"}, "method"),
        ({"method": "cn"}, "requires `petsc=True`"),
        ({"method": "herk2", "petsc": True}, "requires `petsc=False`"),
        (
            {"comp_sens": True, "petsc": True, "enforce_dynamic_limits": False},
            "method='cn'",
        ),
        (
            {"comp_sens": True, "petsc": False, "enforce_dynamic_limits": False},
            "petsc=True",
        ),
        (
            {"arkimex": True, "enforce_dynamic_limits": False},
            "requires `petsc=True`",
        ),
        (
            {
                "arkimex": True,
                "petsc": True,
                "method": "cn",
                "enforce_dynamic_limits": False,
            },
            "cannot be combined",
        ),
        (
            {"petsc": True, "petsc_args": ["-ts_type", "cn"]},
            "cannot override",
        ),
        (
            {"petsc": True, "petsc_args": ["-ts_dt=0.01"]},
            "cannot override",
        ),
    ],
)
def test_integration_config_rejects_invalid_time_contract(kwargs, match):
    with pytest.raises(ValueError, match=match):
        IntegrationConfig(**kwargs)


def test_integration_config_accepts_explicit_petsc_methods():
    assert IntegrationConfig(petsc=True, method="beuler").method == "beuler"
    assert IntegrationConfig(petsc=True, method="cn").method == "cn"
    assert IntegrationConfig(
        petsc=True,
        method="cn",
        comp_sens=True,
        enforce_dynamic_limits=False,
    ).comp_sens is True


def test_integration_config_dynamic_limit_defaults_and_opt_out():
    cfg = IntegrationConfig()

    assert cfg.enforce_dynamic_limits is True
    assert cfg.dynamic_limit_tolerance == pytest.approx(1e-8)
    assert cfg.dynamic_limit_release_tolerance == pytest.approx(1e-10)
    assert cfg.max_dynamic_limit_iterations == 20
    assert IntegrationConfig(
        enforce_dynamic_limits=False,
        fsolve=True,
    ).fsolve is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dynamic_limit_tolerance": -1e-8},
        {"dynamic_limit_release_tolerance": -1e-10},
        {"max_dynamic_limit_iterations": 0},
        {"max_dynamic_limit_iterations": -1},
    ],
)
def test_integration_config_rejects_invalid_dynamic_limit_controls(kwargs):
    with pytest.raises(ValueError):
        IntegrationConfig(**kwargs)


@pytest.mark.parametrize("option", ["arkimex", "comp_sens", "fsolve"])
def test_dynamic_limits_reject_unsupported_legacy_paths(option):
    with pytest.raises(
        ValueError,
        match=rf"`{option}=True`.*`enforce_dynamic_limits=False`",
    ):
        IntegrationConfig(**{option: True})


def test_petsc_method_mapping_is_explicit():
    class FakePETSc:
        class TS:
            class Type:
                BEULER = "beuler"
                CN = "cn"
                ARKIMEX = "arkimex"

    assert dynamics._petsc_ts_type(FakePETSc, "beuler", False) == "beuler"
    assert dynamics._petsc_ts_type(FakePETSc, "cn", False) == "cn"
    assert dynamics._petsc_ts_type(FakePETSc, "beuler", True) == "arkimex"


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
