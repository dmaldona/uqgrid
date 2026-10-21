import json
from collections import UserDict

import numpy as np
import pytest

from uqgrid.simulation import dynamics
from uqgrid.simulation.config import IntegrationConfig


def test_integration_config_defaults_and_schema_after_arkimex_removal():
    cfg = IntegrationConfig()
    removed = {
        "arkimex", "arkimex_slow_differential", "arkimex_fast_differential",
    }

    assert cfg.method == "cn"
    assert cfg.petsc is False
    assert removed.isdisjoint(IntegrationConfig.model_fields)
    assert removed.isdisjoint(IntegrationConfig.model_json_schema()["properties"])
    assert removed.isdisjoint(cfg.model_dump())
    assert all(not hasattr(cfg, name) for name in removed)


@pytest.mark.parametrize(
    "key,value",
    [
        ("arkimex", True),
        ("arkimex", False),
        ("arkimex", None),
        ("arkimex_slow_differential", None),
        ("arkimex_slow_differential", []),
        ("arkimex_slow_differential", [0, 2]),
        ("arkimex_fast_differential", None),
        ("arkimex_fast_differential", []),
        ("arkimex_fast_differential", [1, 3]),
    ],
)
@pytest.mark.parametrize("petsc", [False, True])
def test_integration_config_rejects_removed_arkimex_keys(key, value, petsc):
    with pytest.raises(ValueError, match=rf"ARKIMEX has been removed.*{key}"):
        IntegrationConfig(petsc=petsc, **{key: value})


def test_integration_config_rejects_removed_arkimex_keys_in_mapping():
    with pytest.raises(ValueError, match="ARKIMEX has been removed.*arkimex"):
        IntegrationConfig.model_validate(UserDict({"arkimex": False}))


def test_integration_config_rejects_serialized_removed_arkimex_keys():
    payload = {"arkimex": False, "arkimex_slow_differential": None}

    with pytest.raises(ValueError, match="ARKIMEX has been removed.*arkimex_slow_differential"):
        IntegrationConfig.model_validate(payload)
    with pytest.raises(ValueError, match="ARKIMEX has been removed.*arkimex_slow_differential"):
        IntegrationConfig.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "petsc_args",
    [
        ["-ts_arkimex_type", "a2"],
        ["-ts_arkimex_type=a2"],
        ["--ts_arkimex_type=a2"],
        ["-ts_arkimex_fastslowsplit"],
        ["-ts_arkimex_fully_implicit", "true"],
        ["-ts_arkimex"],
    ],
)
@pytest.mark.parametrize("petsc", [False, True])
def test_integration_config_rejects_removed_arkimex_petsc_options(petsc_args, petsc):
    with pytest.raises(ValueError, match="ARKIMEX has been removed.*PETSc options"):
        IntegrationConfig(petsc=petsc, petsc_args=petsc_args)


@pytest.mark.parametrize(
    "method,petsc",
    [("beuler", False), ("beuler", True), ("cn", False), ("cn", True), ("herk2", False), ("herk4", False)],
)
def test_integration_config_preserves_supported_methods_and_solver_options(method, petsc):
    options = ["-ksp_type", "gmres", "-snes_rtol=1e-9", "-ts_monitor"]
    cfg = IntegrationConfig(method=method, petsc=petsc, petsc_args=options)

    assert cfg.method == method
    assert cfg.petsc is petsc
    assert cfg.petsc_args == options
    assert IntegrationConfig(unrelated_option=True).method == "cn"


def test_integration_config_q_limit_defaults():
    cfg = IntegrationConfig()

    assert cfg.enforce_q_limits is True
    assert cfg.q_limit_tolerance == pytest.approx(1e-8)
    assert cfg.max_q_limit_iterations is None
    assert cfg.power_flow_validation.enabled is False
    assert cfg.power_flow_validation.voltage_min is None
    assert cfg.power_flow_validation.branch_loading_max is None
    assert IntegrationConfig(enforce_q_limits=False).enforce_q_limits is False
    assert IntegrationConfig().jacobian_mode == "analytical"


@pytest.mark.parametrize("method", ["beuler", "cn"])
def test_integration_config_accepts_native_finite_difference_jacobian(method):
    cfg = IntegrationConfig(method=method, jacobian_mode="finite_difference")
    assert cfg.finite_difference_epsilon == pytest.approx(1e-7)


def test_herk_resets_cached_jacobian_mode(monkeypatch):
    psys = type("System", (), {})()
    psys.jacobian_mode = "finite_difference"
    monkeypatch.setattr(
        dynamics,
        "integrate_system_herk",
        lambda system, config, ctx: system.jacobian_mode,
    )

    result = dynamics.integrate_system(
        psys, IntegrationConfig(method="herk2", jacobian_mode="analytical")
    )

    assert result == "analytical"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"dt": 0.0}, "greater than 0"),
        ({"steps": 0}, "positive integer"),
        ({"steps": -2}, "positive integer"),
        ({"ton": 0.2, "toff": 0.1}, "toff"),
        ({"method": "bogus"}, "method"),
        ({"method": "arkimex", "petsc": True}, "method"),
        ({"method": "herk2", "petsc": True}, "requires `petsc=False`"),
        ({"jacobian_mode": "invalid"}, "jacobian_mode"),
        ({"jacobian_mode": "finite_difference", "petsc": True}, "native backward Euler"),
        ({"jacobian_mode": "finite_difference", "petsc": True, "method": "cn"}, "native backward Euler"),
        ({"jacobian_mode": "finite_difference", "method": "herk2"}, "native backward Euler"),
        ({"jacobian_mode": "finite_difference", "method": "herk4"}, "native backward Euler"),
        (
            {"jacobian_mode": "finite_difference", "check_jacobian": True},
            "requires `jacobian_mode='analytical'`",
        ),
        ({"finite_difference_epsilon": float("inf")}, "finite number"),
        (
            {"method": "beuler", "comp_sens": True, "petsc": True, "enforce_dynamic_limits": False},
            "method='cn'",
        ),
        (
            {"comp_sens": True, "petsc": False, "enforce_dynamic_limits": False},
            "petsc=True",
        ),
        (
            {"method": "cn", "comp_sens": True, "enforce_dynamic_limits": False},
            "petsc=True",
        ),
        (
            {"petsc": True, "petsc_args": ["-ts_type", "cn"]},
            "cannot override",
        ),
        (
            {"petsc": True, "petsc_args": ["-ts_dt=0.01"]},
            "cannot override",
        ),
        (
            {"petsc": True, "petsc_args": ["-ts_type", "arkimex"]},
            "cannot override",
        ),
        (
            {"petsc": True, "petsc_args": ["-ts_type=arkimex"]},
            "cannot override",
        ),
    ],
)
def test_integration_config_rejects_invalid_time_contract(kwargs, match):
    with pytest.raises(ValueError, match=match):
        IntegrationConfig(**kwargs)


def test_integration_config_accepts_native_cn_with_legacy_fsolve_opt_out():
    cfg = IntegrationConfig(method="cn", fsolve=True, enforce_dynamic_limits=False)

    assert cfg.method == "cn"
    assert cfg.petsc is False
    assert cfg.fsolve is True


def test_integration_config_accepts_explicit_petsc_methods():
    assert IntegrationConfig(petsc=True, method="beuler").method == "beuler"
    assert IntegrationConfig(petsc=True, method="cn").method == "cn"
    assert IntegrationConfig(
        petsc=True,
        method="cn",
        comp_sens=True,
        enforce_dynamic_limits=False,
    ).comp_sens is True


def test_integration_config_default_cn_accepts_petsc_sensitivities():
    cfg = IntegrationConfig(
        petsc=True,
        comp_sens=True,
        enforce_dynamic_limits=False,
    )

    assert cfg.method == "cn"
    assert cfg.comp_sens is True


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


@pytest.mark.parametrize("method", ["beuler", "cn"])
@pytest.mark.parametrize("option", ["comp_sens", "fsolve"])
def test_dynamic_limits_reject_unsupported_legacy_paths(option, method):
    with pytest.raises(
        ValueError,
        match=rf"`{option}=True`.*`enforce_dynamic_limits=False`",
    ):
        IntegrationConfig(method=method, **{option: True})


def test_petsc_method_mapping_is_explicit():
    class FakePETSc:
        class TS:
            class Type:
                BEULER = "beuler"
                CN = "cn"

    assert dynamics._petsc_ts_type(FakePETSc, "beuler") == "beuler"
    assert dynamics._petsc_ts_type(FakePETSc, "cn") == "cn"
    with pytest.raises(ValueError, match="Unsupported PETSc integration method"):
        dynamics._petsc_ts_type(FakePETSc, "arkimex")


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
