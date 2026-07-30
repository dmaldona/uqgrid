import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.models.sexs_imp import ExcSEXS
from uqgrid.simulation.config import IntegrationConfig, IntegrationCtx
from uqgrid.simulation.dynamic_limits import (
    DynamicLimitError,
    DynamicLimitMode,
    LimitedStateDescriptor,
    collect_limited_state_descriptors,
    evaluate_dynamic_limit_complementarity,
    initialize_dynamic_limit_modes,
    project_limited_derivatives,
    project_limited_states,
    update_dynamic_limit_active_set,
    validate_initial_dynamic_limits,
)
from uqgrid.simulation.dynamics import initialize_system, integrate_system
from uqgrid.simulation.pflow import runpf


@pytest.fixture
def data_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )


def _descriptor(**overrides):
    values = {
        "state_index": 0,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "device_type": "SEXS",
        "bus": 101,
        "device_id": "1",
        "enabled": True,
    }
    values.update(overrides)
    return LimitedStateDescriptor(**values)


def _validate(z0, descriptors, **overrides):
    values = {
        "enforce_dynamic_limits": True,
        "dynamic_limit_tolerance": 1e-8,
        "dynamic_limit_release_tolerance": 1e-10,
        "max_dynamic_limit_iterations": 20,
    }
    values.update(overrides)
    return validate_initial_dynamic_limits(z0, descriptors, **values)


def test_collect_sexs_descriptors_uses_effective_theta_and_external_bus():
    first = ExcSEXS("A", 0.1, 1.0, 10.0, 0.1, -1.0, 1.0)
    first.set_pointers(2, 0, 0, 0)
    first.set_bus(0)
    second = ExcSEXS("B", 0.1, 1.0, 10.0, 0.1, -2.0, 2.0)
    second.set_pointers(8, 0, 8, 1)
    second.set_bus(1)
    theta = np.zeros(16)
    theta[4:7] = [-0.5, 0.75, 1.0]
    theta[12:15] = [-1.5, 1.25, 0.0]
    psys = SimpleNamespace(
        devices=[first, second],
        buses=[SimpleNamespace(id=101), SimpleNamespace(id=205)],
    )

    descriptors = collect_limited_state_descriptors(psys, theta)

    assert descriptors == [
        LimitedStateDescriptor(3, -0.5, 0.75, "SEXS", 101, "A", True),
        LimitedStateDescriptor(9, -1.5, 1.25, "SEXS", 205, "B", False),
    ]
    json.dumps([item.to_dict() for item in descriptors], allow_nan=False)


def test_sexs_exposes_efd_bounded_state_metadata():
    metadata = ExcSEXS.bounded_state_metadata

    assert len(metadata) == 1
    assert metadata[0].state_name == "Efd"
    assert metadata[0].state_offset == 1
    assert metadata[0].lower_parameter_offset == 4
    assert metadata[0].upper_parameter_offset == 5
    assert metadata[0].enabled_parameter_offset == 6
    assert metadata[0].device_type == "SEXS"


@pytest.mark.parametrize("value", [0.0, 1.0, -5e-9, 1.0 + 5e-9, 0.5])
def test_initial_dynamic_limit_validation_accepts_bounds_and_tolerance(value):
    z0 = np.asarray([value])
    original = z0.copy()

    diagnostics = _validate(z0, [_descriptor()])

    assert diagnostics["initialization"]["valid"] is True
    assert diagnostics["initialization"]["checked_state_count"] == 1
    assert diagnostics["initialization"]["violations"] == []
    np.testing.assert_array_equal(z0, original)


@pytest.mark.parametrize(
    "value, side, reason",
    [
        (-2e-8, "lower", "initial_state_below_lower_bound"),
        (1.0 + 2e-8, "upper", "initial_state_above_upper_bound"),
    ],
)
def test_initial_dynamic_limit_validation_rejects_bound_violations(
    value, side, reason
):
    z0 = np.asarray([value])
    original = z0.copy()

    with pytest.raises(DynamicLimitError) as exc_info:
        _validate(z0, [_descriptor()])

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["initialization"]["failure_reasons"] == [reason]
    assert diagnostics["initialization"]["violations"][0]["side"] == side
    assert diagnostics["initialization"]["violations"][0]["state_index"] == 0
    json.dumps(diagnostics, allow_nan=False)
    np.testing.assert_array_equal(z0, original)


@pytest.mark.parametrize(
    "descriptor, value, reason",
    [
        (_descriptor(lower_bound=-np.inf), 0.5, "non_finite_bounds"),
        (_descriptor(lower_bound=1.0, upper_bound=1.0), 1.0, "degenerate_bounds"),
        (_descriptor(lower_bound=2.0, upper_bound=1.0), 1.5, "inverted_bounds"),
        (_descriptor(), np.nan, "non_finite_initial_state"),
    ],
)
def test_initial_dynamic_limit_validation_rejects_invalid_data(
    descriptor, value, reason
):
    with pytest.raises(DynamicLimitError) as exc_info:
        _validate(np.asarray([value]), [descriptor])

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["initialization"]["failure_reasons"] == [reason]
    json.dumps(diagnostics, allow_nan=False)


def test_disabled_global_or_device_limit_skips_initial_check():
    globally_disabled = _validate(
        np.asarray([2.0]),
        [_descriptor()],
        enforce_dynamic_limits=False,
    )
    device_disabled = _validate(
        np.asarray([2.0]),
        [_descriptor(enabled=False)],
    )

    assert globally_disabled["enabled"] is False
    assert globally_disabled["initialization"]["checked_state_count"] == 0
    assert device_disabled["enabled_state_count"] == 0
    assert device_disabled["initialization"]["valid"] is True

    no_states = _validate(np.asarray([]), [])
    assert no_states["discovered_state_count"] == 0
    assert no_states["initialization"]["valid"] is True


def test_state_projection_clamps_both_sides_without_mutating_inputs():
    descriptors = [
        _descriptor(state_index=0, device_id="lower"),
        _descriptor(state_index=1, device_id="upper"),
        _descriptor(state_index=2, device_id="disabled", enabled=False),
    ]
    state = np.asarray([-0.25, 1.25, 4.0])
    original = state.copy()

    projected, events = project_limited_states(
        state,
        descriptors,
        time=0.2,
        stage_or_endpoint="endpoint",
        active_set_iterations=2,
    )

    np.testing.assert_array_equal(state, original)
    np.testing.assert_array_equal(projected, [0.0, 1.0, 4.0])
    assert [event["side"] for event in events] == ["lower", "upper"]
    assert all(event["action"] == "project" for event in events)
    assert events[0]["time"] == pytest.approx(0.2)
    assert events[0]["stage_or_endpoint"] == "endpoint"
    assert events[0]["active_set_iterations"] == 2
    json.dumps(events, allow_nan=False)


@pytest.mark.parametrize(
    "state, raw_derivative, expected, blocked_sides",
    [
        ([1.0, 0.0], [2.0, -3.0], [0.0, 0.0], ["upper", "lower"]),
        ([1.0, 0.0], [-2.0, 3.0], [-2.0, 3.0], []),
        ([1.0 - 5e-9, 5e-9], [2.0, -3.0], [0.0, 0.0], ["upper", "lower"]),
    ],
)
def test_directional_projection_blocks_outward_and_releases_inward(
    state, raw_derivative, expected, blocked_sides
):
    descriptors = [_descriptor(state_index=0), _descriptor(state_index=1)]
    raw_derivative = np.asarray(raw_derivative, dtype=float)
    original = raw_derivative.copy()

    projected, events = project_limited_derivatives(
        np.asarray(state, dtype=float),
        raw_derivative,
        descriptors,
        tolerance=1e-8,
    )

    np.testing.assert_array_equal(raw_derivative, original)
    np.testing.assert_array_equal(projected, expected)
    assert [event["side"] for event in events] == blocked_sides
    assert all(
        event["action"] == "block_outward_derivative" for event in events
    )


def test_disabled_limit_helpers_leave_state_and_derivative_unchanged():
    descriptor = _descriptor(enabled=False)
    state = np.asarray([2.0])
    derivative = np.asarray([3.0])

    projected_state, state_events = project_limited_states(state, [descriptor])
    projected_derivative, derivative_events = project_limited_derivatives(
        state,
        derivative,
        [descriptor],
        tolerance=1e-8,
    )

    np.testing.assert_array_equal(projected_state, state)
    np.testing.assert_array_equal(projected_derivative, derivative)
    assert state_events == []
    assert derivative_events == []


@pytest.mark.parametrize(
    "descriptor, reason",
    [
        (_descriptor(upper_bound=np.inf), "non_finite_bounds"),
        (_descriptor(lower_bound=1.0, upper_bound=1.0), "degenerate_bounds"),
        (_descriptor(lower_bound=2.0, upper_bound=1.0), "inverted_bounds"),
    ],
)
def test_projection_helpers_reject_invalid_effective_bounds(descriptor, reason):
    with pytest.raises(DynamicLimitError) as exc_info:
        project_limited_states(np.asarray([0.5]), [descriptor])

    assert exc_info.value.diagnostics["failure_reasons"] == [reason]


def test_active_set_activates_retains_and_releases_multiple_limits():
    descriptors = [
        _descriptor(state_index=0, device_id="upper"),
        _descriptor(state_index=1, device_id="lower"),
    ]
    modes = initialize_dynamic_limit_modes(descriptors)

    modes, changed, complementarity, events = update_dynamic_limit_active_set(
        np.asarray([1.1, -0.1]),
        np.asarray([0.0, 0.0]),
        descriptors,
        modes,
        state_tolerance=1e-8,
        release_tolerance=1e-10,
        time=0.1,
        stage_or_endpoint="endpoint",
        active_set_iterations=1,
    )

    assert changed is True
    assert complementarity["consistent"] is False
    assert modes == {
        0: DynamicLimitMode.UPPER_ACTIVE,
        1: DynamicLimitMode.LOWER_ACTIVE,
    }
    assert [(event["side"], event["action"]) for event in events] == [
        ("upper", "activate"),
        ("lower", "activate"),
    ]

    modes, changed, complementarity, events = update_dynamic_limit_active_set(
        np.asarray([1.0, 0.0]),
        np.asarray([-1e-3, 1e-3]),
        descriptors,
        modes,
        state_tolerance=1e-8,
        release_tolerance=1e-10,
    )

    assert changed is False
    assert complementarity["consistent"] is True
    assert events == []

    modes, changed, complementarity, events = update_dynamic_limit_active_set(
        np.asarray([1.0, 0.0]),
        np.asarray([1e-3, -1e-3]),
        descriptors,
        modes,
        state_tolerance=1e-8,
        release_tolerance=1e-10,
    )

    assert changed is True
    assert complementarity["consistent"] is False
    assert modes == {0: DynamicLimitMode.FREE, 1: DynamicLimitMode.FREE}
    assert [(event["side"], event["action"]) for event in events] == [
        ("upper", "release"),
        ("lower", "release"),
    ]
    json.dumps(events, allow_nan=False)


def test_active_set_tolerances_do_not_activate_or_release_at_threshold():
    descriptor = _descriptor()
    modes = initialize_dynamic_limit_modes([descriptor])

    modes, changed, complementarity, _ = update_dynamic_limit_active_set(
        np.asarray([1.0 + 1e-8]),
        np.asarray([0.0]),
        [descriptor],
        modes,
        state_tolerance=1e-8,
        release_tolerance=1e-10,
    )

    assert changed is False
    assert complementarity["consistent"] is True

    modes[0] = DynamicLimitMode.UPPER_ACTIVE
    modes, changed, complementarity, _ = update_dynamic_limit_active_set(
        np.asarray([1.0]),
        np.asarray([1e-10]),
        [descriptor],
        modes,
        state_tolerance=1e-8,
        release_tolerance=1e-10,
    )

    assert changed is False
    assert complementarity["consistent"] is True
    assert modes[0] == DynamicLimitMode.UPPER_ACTIVE


def test_complementarity_rejects_non_finite_free_residual():
    descriptor = _descriptor()
    modes = initialize_dynamic_limit_modes([descriptor])

    with pytest.raises(DynamicLimitError) as exc_info:
        evaluate_dynamic_limit_complementarity(
            np.asarray([0.5]),
            np.asarray([np.nan]),
            [descriptor],
            modes,
            state_tolerance=1e-8,
            release_tolerance=1e-10,
        )

    assert exc_info.value.diagnostics["failure_reasons"] == [
        "non_finite_free_residual"
    ]


def _load_enabled_sexs_system(data_dir):
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, "2bus_SEXS.dyr"))
    assert psys.exc[0].enable_limits is True
    psys.createYbusComplex()
    return psys


def test_native_be_and_herk_return_identical_initial_limit_diagnostics(data_dir):
    results = []
    for method in ("beuler", "herk2"):
        psys = _load_enabled_sexs_system(data_dir)
        results.append(
            integrate_system(
                psys,
                IntegrationConfig(
                    method=method,
                    petsc=False,
                    steps=1,
                    dt=0.01,
                    ton=10.0,
                    toff=11.0,
                ),
            )["dynamic_limit_diagnostics"]
        )

    assert results[0] == results[1]
    assert results[0]["enabled_state_count"] == 1
    assert results[0]["initialization"]["valid"] is True
    assert results[0]["events"] == []


def _invalid_initial_context(psys):
    pf_solution = runpf(psys, verbose=False)
    z0, theta = initialize_system(psys, pf_solution)
    exciter = psys.exc[0]
    state_index = exciter.dif_ptr + exciter.efd_idx
    par_ptr = exciter.par_ptr
    theta[par_ptr + 4] = z0[state_index] - 1.0
    theta[par_ptr + 5] = z0[state_index] - 0.1
    theta[par_ptr + 6] = 1.0
    ctx = IntegrationCtx()
    ctx.set_initial_conditions(z0.copy())
    ctx.set_theta(theta)
    return ctx, state_index, z0[state_index]


def test_invalid_context_fails_before_petsc_setup_or_schedule(data_dir, monkeypatch):
    from uqgrid.simulation import dynamics

    psys = _load_enabled_sexs_system(data_dir)
    ctx, state_index, initial_value = _invalid_initial_context(psys)
    monkeypatch.setattr(
        dynamics,
        "_get_petsc_for_config",
        lambda config: pytest.fail("PETSc setup must not run"),
    )
    monkeypatch.setattr(
        dynamics,
        "build_integration_schedule",
        lambda **kwargs: pytest.fail("schedule must not be built"),
    )

    with pytest.raises(DynamicLimitError) as exc_info:
        integrate_system(
            psys,
            IntegrationConfig(petsc=True, method="cn", steps=1),
            ctx,
        )

    violation = exc_info.value.diagnostics["initialization"]["violations"][0]
    assert violation["state_index"] == state_index
    assert violation["initial_value"] == pytest.approx(initial_value)
    assert ctx.z0_user[state_index] == pytest.approx(initial_value)


def test_invalid_context_fails_before_native_allocation_or_schedule(
    data_dir, monkeypatch
):
    from uqgrid.simulation import dynamics

    psys = _load_enabled_sexs_system(data_dir)
    ctx, _, _ = _invalid_initial_context(psys)
    monkeypatch.setattr(
        dynamics,
        "preallocate_jacobian",
        lambda psys: pytest.fail("Jacobian allocation must not run"),
    )
    monkeypatch.setattr(
        dynamics,
        "build_integration_schedule",
        lambda **kwargs: pytest.fail("schedule must not be built"),
    )

    with pytest.raises(DynamicLimitError):
        integrate_system(
            psys,
            IntegrationConfig(petsc=False, method="beuler", steps=1),
            ctx,
        )


def test_invalid_context_fails_before_herk_allocation_or_schedule(
    data_dir, monkeypatch
):
    from uqgrid.simulation import dynamics, herk

    psys = _load_enabled_sexs_system(data_dir)
    ctx, _, _ = _invalid_initial_context(psys)
    monkeypatch.setattr(
        dynamics,
        "preallocate_jacobian",
        lambda psys: pytest.fail("Jacobian allocation must not run"),
    )
    monkeypatch.setattr(
        herk,
        "build_integration_schedule",
        lambda **kwargs: pytest.fail("schedule must not be built"),
    )

    with pytest.raises(DynamicLimitError):
        integrate_system(
            psys,
            IntegrationConfig(petsc=False, method="herk2", steps=1),
            ctx,
        )
