import importlib
import json
import os

import numpy as np
import pytest

import uqgrid
import uqgrid.simulation
from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.simulation import dynamics
from uqgrid.simulation.config import IntegrationConfig, IntegrationCtx
from uqgrid.simulation.dynamic_limits import (
    DYNAMIC_LIMIT_EVENT_ACTIONS,
    DYNAMIC_LIMIT_EVENT_FIELDS,
    DynamicLimitMode,
    collect_limited_state_descriptors,
    initialize_dynamic_limit_modes,
)
from uqgrid.simulation.dynamics import initialize_system, integrate_system
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


METHODS = (
    pytest.param("beuler", False, id="native-be"),
    pytest.param("herk2", False, id="herk2"),
    pytest.param("herk4", False, id="herk4"),
    pytest.param("beuler", True, id="petsc-be"),
    pytest.param("cn", True, id="petsc-cn"),
)
IMPLICIT_METHODS = (
    pytest.param("beuler", False, id="native-be"),
    pytest.param("beuler", True, id="petsc-be"),
    pytest.param("cn", True, id="petsc-cn"),
)


@pytest.fixture
def data_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )


def _require_backend(petsc):
    if petsc:
        pytest.importorskip("petsc4py")


def _build_two_bus(data_dir, *, fault=False):
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, "2bus_SEXS.dyr"))
    if fault:
        psys.add_busfault(1, 1e-4)
    psys.createYbusComplex()
    psys.power_injection = False
    return psys


def _initialize_context(psys):
    solution = runpf(psys, verbose=False)
    z0, theta = initialize_system(psys, solution)
    ctx = IntegrationCtx()
    ctx.set_initial_conditions(z0.copy())
    ctx.set_theta(theta.copy())
    return z0, theta, ctx


def _set_biased_limits(psys, z0, theta, *, width=0.01, offset=0.05):
    exciter = psys.exc[0]
    state_index = exciter.dif_ptr + exciter.efd_idx
    initial = float(z0[state_index])
    ptr = exciter.par_ptr
    theta[ptr + 4] = initial - width
    theta[ptr + 5] = initial + width
    theta[ptr + 6] = 1.0
    theta[ptr + 7] += offset
    return state_index, initial - width, initial + width


def _run_two_bus(
    data_dir,
    method,
    petsc,
    *,
    fault=False,
    biased=False,
    dt=0.005,
    steps=4,
    tend=10.0,
    ton=10.0,
    toff=11.0,
):
    _require_backend(petsc)
    psys = _build_two_bus(data_dir, fault=fault)
    z0, theta, ctx = _initialize_context(psys)
    limits = None
    if biased:
        limits = _set_biased_limits(psys, z0, theta)
        ctx.set_theta(theta.copy())
    config = IntegrationConfig(
        method=method,
        petsc=petsc,
        dt=dt,
        steps=steps,
        tend=tend,
        ton=ton,
        toff=toff,
        newton_tol=1e-10,
        newton_max_iter=100,
        herk_alg_tol=1e-10,
        herk_alg_max_iter=100,
    )
    result = integrate_system(psys, config, ctx)
    return psys, ctx, config, result, limits


def _run_limited_ieee9(data_dir, method, petsc):
    _require_backend(petsc)
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus_SEXS.dyr"))
    psys.createYbusComplex()
    psys.power_injection = False
    z0, theta, ctx = _initialize_context(psys)
    limits = []
    for index, exciter in enumerate(psys.exc):
        state_index = exciter.dif_ptr + exciter.efd_idx
        initial = float(z0[state_index])
        ptr = exciter.par_ptr
        theta[ptr + 4] = initial - 0.01
        theta[ptr + 5] = initial + 0.01
        theta[ptr + 6] = 1.0
        expected_side = "upper" if index % 2 == 0 else "lower"
        theta[ptr + 7] += 0.05 if expected_side == "upper" else -0.05
        limits.append(
            {
                "state_index": state_index,
                "lower": initial - 0.01,
                "upper": initial + 0.01,
                "expected_side": expected_side,
                "bus": int(psys.buses[exciter.bus].id),
            }
        )
    ctx.set_theta(theta.copy())
    config = IntegrationConfig(
        method=method,
        petsc=petsc,
        dt=0.005,
        steps=3,
        ton=10.0,
        toff=11.0,
        newton_tol=1e-10,
        newton_max_iter=100,
        herk_alg_tol=1e-10,
        herk_alg_max_iter=100,
    )
    return psys, ctx, config, integrate_system(psys, config, ctx), limits


def _assert_event_contract(events, *, implicit):
    required = set(DYNAMIC_LIMIT_EVENT_FIELDS)
    assert events
    previous_time = -np.inf
    for event in events:
        assert required <= set(event)
        assert event["action"] in DYNAMIC_LIMIT_EVENT_ACTIONS
        assert event["side"] in {"lower", "upper"}
        assert isinstance(event["device_type"], str)
        assert isinstance(event["device_id"], str)
        assert isinstance(event["bus"], int)
        assert isinstance(event["state_index"], int)
        assert np.isfinite(event["time"])
        assert event["time"] >= previous_time - 1e-14
        previous_time = event["time"]
        if implicit:
            assert event["stage_or_endpoint"] == "endpoint"
            assert isinstance(event["active_set_iterations"], int)
        else:
            context = event["stage_or_endpoint"]
            assert context == "endpoint" or context.startswith("stage_")
            assert event["active_set_iterations"] is None
    json.dumps(events, allow_nan=False)


def test_event_schema_constants_are_exported():
    assert uqgrid.DYNAMIC_LIMIT_EVENT_FIELDS == DYNAMIC_LIMIT_EVENT_FIELDS
    assert uqgrid.DYNAMIC_LIMIT_EVENT_ACTIONS == DYNAMIC_LIMIT_EVENT_ACTIONS
    assert (
        uqgrid.simulation.DYNAMIC_LIMIT_EVENT_FIELDS
        == DYNAMIC_LIMIT_EVENT_FIELDS
    )


def test_standard_ieee9_is_an_explicit_zero_limit_negative_control(data_dir):
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus.dyr"))
    psys.createYbusComplex()
    _, theta, ctx = _initialize_context(psys)

    assert collect_limited_state_descriptors(psys, theta) == []

    result = integrate_system(
        psys,
        IntegrationConfig(
            method="beuler",
            petsc=False,
            steps=1,
            dt=0.005,
            ton=10.0,
            toff=11.0,
        ),
        ctx,
    )
    diagnostics = result["dynamic_limit_diagnostics"]
    assert diagnostics["discovered_state_count"] == 0
    assert diagnostics["enabled_state_count"] == 0
    assert diagnostics["events"] == []
    assert (
        uqgrid.simulation.DYNAMIC_LIMIT_EVENT_ACTIONS
        == DYNAMIC_LIMIT_EVENT_ACTIONS
    )


@pytest.mark.parametrize("method,petsc", METHODS)
def test_all_integrators_return_the_common_event_schema(
    data_dir, method, petsc
):
    psys, ctx, config, result, limits = _run_two_bus(
        data_dir, method, petsc, biased=True
    )
    events = result["dynamic_limit_diagnostics"]["events"]

    _assert_event_contract(
        events, implicit=method in {"beuler", "cn"}
    )
    state_index, lower, upper = limits
    values = result["history"][state_index]
    assert np.min(values) >= lower - config.dynamic_limit_tolerance
    assert np.max(values) <= upper + config.dynamic_limit_tolerance
    assert "stage_history" not in result
    assert "limit_stage_history" not in result


@pytest.mark.parametrize("method,petsc", METHODS)
def test_all_integrators_share_the_schema_for_multiple_ieee9_exciters(
    data_dir, method, petsc
):
    psys, ctx, config, result, limits = _run_limited_ieee9(
        data_dir, method, petsc
    )
    events = result["dynamic_limit_diagnostics"]["events"]

    descriptors = collect_limited_state_descriptors(psys, ctx.theta_user)
    assert len(psys.exc) == 3
    assert len(descriptors) == 3
    assert all(descriptor.device_type == "SEXS" for descriptor in descriptors)
    assert all(descriptor.enabled for descriptor in descriptors)
    _assert_event_contract(events, implicit=method in {"beuler", "cn"})
    event_buses = {event["bus"] for event in events}
    assert event_buses == {psys.buses[exciter.bus].id for exciter in psys.exc}
    for limit in limits:
        state_index = limit["state_index"]
        lower = limit["lower"]
        upper = limit["upper"]
        side = limit["expected_side"]
        bound = upper if side == "upper" else lower
        descriptor = next(
            item for item in descriptors if item.state_index == state_index
        )
        assert descriptor.lower_bound == pytest.approx(lower)
        assert descriptor.upper_bound == pytest.approx(upper)
        activations = [
            event
            for event in events
            if event["state_index"] == state_index
            and event["action"] == "activate"
        ]
        assert activations
        assert {event["side"] for event in activations} == {side}
        assert all(event["bus"] == limit["bus"] for event in activations)
        values = result["history"][state_index]
        assert np.min(values) >= lower - config.dynamic_limit_tolerance
        assert np.max(values) <= upper + config.dynamic_limit_tolerance
        np.testing.assert_allclose(values[1:], bound, atol=1e-8, rtol=0.0)
        if method in {"herk2", "herk4"}:
            projections = [
                event
                for event in events
                if event["state_index"] == state_index
                and event["action"] == "project"
                and event["side"] == side
            ]
            assert projections
            assert all(
                event["state_after"] == pytest.approx(bound)
                for event in projections
            )


@pytest.mark.parametrize("method,petsc", METHODS)
def test_no_fault_limited_trajectories_remain_flat(data_dir, method, petsc):
    psys, ctx, config, result, _ = _run_two_bus(
        data_dir,
        method,
        petsc,
        dt=1.0 / 120.0,
        steps=5,
    )
    history = result["history"]
    drift = np.abs(history - history[:, [0]])
    ndiff = psys.num_dof_dif
    nalg = psys.num_dof_alg
    slices = (
        drift[:ndiff],
        drift[ndiff : ndiff + nalg],
        drift[ndiff + nalg :],
        drift,
    )
    for values in slices:
        assert np.max(values, initial=0.0) < 1e-8

    residual = np.zeros(history.shape[0])
    for state in history.T:
        residual_function(residual, state, ctx.theta_user, psys)
        assert np.linalg.norm(residual, np.inf) < 1e-8
    assert result["dynamic_limit_diagnostics"]["events"] == []


@pytest.mark.parametrize("method", ["herk2", "herk4"])
def test_herk_stages_are_bound_feasible_and_algebraically_consistent(
    data_dir, method, monkeypatch
):
    herk = importlib.import_module("uqgrid.simulation.herk")
    projected_states = []
    algebraic_norms = []
    original_project = herk.project_limited_states
    original_solve = herk.solve_stage_algebraic

    def recording_project(state, descriptors, **kwargs):
        projected, events = original_project(state, descriptors, **kwargs)
        projected_states.append((projected.copy(), list(descriptors)))
        return projected, events

    def recording_solve(X_i, y0, v0, theta, psys, F, J, *args, **kwargs):
        y, v, iterations = original_solve(
            X_i, y0, v0, theta, psys, F, J, *args, **kwargs
        )
        state = np.concatenate([X_i, y, v])
        check = np.zeros_like(state)
        residual_function(check, state, theta, psys)
        algebraic_norms.append(
            np.linalg.norm(check[psys.num_dof_dif :], np.inf)
        )
        return y, v, iterations

    monkeypatch.setattr(herk, "project_limited_states", recording_project)
    monkeypatch.setattr(herk, "solve_stage_algebraic", recording_solve)
    psys, ctx, config, result, _ = _run_two_bus(
        data_dir, method, False, biased=True
    )

    assert projected_states
    for state, descriptors in projected_states:
        for descriptor in descriptors:
            value = state[descriptor.state_index]
            assert value >= descriptor.lower_bound - config.dynamic_limit_tolerance
            assert value <= descriptor.upper_bound + config.dynamic_limit_tolerance
    assert algebraic_norms
    assert max(algebraic_norms) < 1e-8
    assert "stage_history" not in result


def _apply_endpoint_events(modes, descriptors, events, endpoint_time):
    updated = dict(modes)
    for event in events:
        if not np.isclose(event["time"], endpoint_time, atol=1e-14, rtol=0.0):
            continue
        state_index = event["state_index"]
        if event["action"] == "activate":
            updated[state_index] = (
                DynamicLimitMode.UPPER_ACTIVE
                if event["side"] == "upper"
                else DynamicLimitMode.LOWER_ACTIVE
            )
        elif event["action"] == "release":
            updated[state_index] = DynamicLimitMode.FREE
    return updated


@pytest.mark.parametrize("method,petsc", IMPLICIT_METHODS)
def test_implicit_endpoint_equations_and_complementarity(
    data_dir, method, petsc
):
    psys, ctx, config, result, _ = _run_two_bus(
        data_dir, method, petsc, biased=True
    )
    history = result["history"]
    times = result["tvec"]
    theta = ctx.theta_user
    descriptors = collect_limited_state_descriptors(psys, theta)
    modes = initialize_dynamic_limit_modes(descriptors)
    events = result["dynamic_limit_diagnostics"]["events"]

    for index in range(1, len(times)):
        zold = history[:, index - 1]
        endpoint = history[:, index]
        h = times[index] - times[index - 1]
        start_modes = dict(modes)
        if method == "cn":
            start_derivative = dynamics._effective_cn_start_derivative(
                zold,
                theta,
                psys,
                descriptors,
                start_modes,
                tolerance=config.dynamic_limit_tolerance,
            )
            free_residual = np.zeros_like(endpoint)
            dynamics._assemble_cn_residual(
                free_residual,
                endpoint,
                zold,
                h,
                start_derivative,
                psys,
                theta,
            )
        else:
            free_residual = np.zeros_like(endpoint)
            dynamics._assemble_beuler_residual(
                free_residual, endpoint, zold, h, psys, theta
            )

        modes = _apply_endpoint_events(modes, descriptors, events, times[index])
        free_rows = np.ones(psys.num_dof_dif, dtype=bool)
        for descriptor in descriptors:
            state_index = descriptor.state_index
            mode = modes[state_index]
            if mode == DynamicLimitMode.FREE:
                continue
            free_rows[state_index] = False
            if mode == DynamicLimitMode.UPPER_ACTIVE:
                assert endpoint[state_index] == pytest.approx(
                    descriptor.upper_bound, abs=1e-8
                )
                assert free_residual[state_index] <= (
                    config.dynamic_limit_release_tolerance + 1e-8
                )
            else:
                assert endpoint[state_index] == pytest.approx(
                    descriptor.lower_bound, abs=1e-8
                )
                assert free_residual[state_index] >= -(
                    config.dynamic_limit_release_tolerance + 1e-8
                )
        assert np.linalg.norm(
            free_residual[: psys.num_dof_dif][free_rows], np.inf
        ) < 1e-8
        assert np.linalg.norm(
            free_residual[psys.num_dof_dif :], np.inf
        ) < 1e-8


def test_native_and_petsc_be_match_on_the_common_fault_case(data_dir):
    native = _run_two_bus(
        data_dir,
        "beuler",
        False,
        fault=True,
        biased=True,
        dt=0.005,
        steps=4,
        ton=0.0075,
        toff=0.0135,
    )
    petsc = _run_two_bus(
        data_dir,
        "beuler",
        True,
        fault=True,
        biased=True,
        dt=0.005,
        steps=4,
        ton=0.0075,
        toff=0.0135,
    )

    np.testing.assert_allclose(petsc[3]["tvec"], native[3]["tvec"])
    np.testing.assert_allclose(
        petsc[3]["history"],
        native[3]["history"],
        atol=1e-8,
        rtol=1e-8,
    )
    native_events = native[3]["dynamic_limit_diagnostics"]["events"]
    petsc_events = petsc[3]["dynamic_limit_diagnostics"]["events"]
    identities = lambda records: [
        (
            item["device_type"],
            item["device_id"],
            item["bus"],
            item["state_index"],
            item["side"],
            item["action"],
            item["time"],
            item["stage_or_endpoint"],
        )
        for item in records
    ]
    assert identities(petsc_events) == identities(native_events)


@pytest.mark.parametrize("method,petsc", METHODS)
def test_all_integrators_preserve_the_normalized_fault_schedule(
    data_dir, method, petsc
):
    psys, ctx, config, result, limits = _run_two_bus(
        data_dir,
        method,
        petsc,
        fault=True,
        biased=True,
        dt=0.005,
        steps=4,
        ton=0.0075,
        toff=0.0135,
    )
    expected = [0.0, 0.005, 0.0075, 0.01, 0.0135, 0.015, 0.02]
    np.testing.assert_allclose(result["tvec"], expected)
    assert result["history"].shape[1] == len(expected)
    state_index, lower, upper = limits
    values = result["history"][state_index]
    assert np.min(values) >= lower - config.dynamic_limit_tolerance
    assert np.max(values) <= upper + config.dynamic_limit_tolerance
    activations = [
        event
        for event in result["dynamic_limit_diagnostics"]["events"]
        if event["action"] == "activate" and event["state_index"] == state_index
    ]
    assert activations
    assert any(
        np.isclose(values, event["bound"], atol=1e-8, rtol=0.0).any()
        for event in activations
    )

    residual = np.zeros(result["history"].shape[0])
    fault = psys.fault_events[0]
    for time, state in zip(result["tvec"], result["history"].T):
        if config.ton <= time < config.toff:
            fault.apply()
        else:
            fault.remove()
        residual_function(residual, state, ctx.theta_user, psys)
        assert np.linalg.norm(residual[psys.num_dof_dif :], np.inf) < 1e-8
    fault.remove()
    assert fault.active is False


def _sample_reference(reference_times, reference_history, sample_times):
    indices = []
    for time in sample_times:
        matches = np.flatnonzero(
            np.isclose(reference_times, time, atol=1e-13, rtol=0.0)
        )
        assert len(matches) == 1
        indices.append(matches[0])
    return reference_history[:, indices]


def _normalized_error(candidate, reference):
    scale = max(1.0, float(np.linalg.norm(reference, np.inf)))
    return float(np.linalg.norm(candidate - reference, np.inf) / scale)


def test_all_integrators_converge_toward_one_faulted_limited_trajectory(data_dir):
    dts = (1.0 / 120.0, 1.0 / 240.0, 1.0 / 480.0)
    reference_run = _run_two_bus(
        data_dir,
        "herk4",
        False,
        fault=True,
        biased=True,
        dt=1.0 / 1920.0,
        steps=-1,
        tend=0.1,
        ton=0.025,
        toff=0.05,
    )
    reference = reference_run[3]
    ndiff = reference_run[0].num_dof_dif
    errors = {f"{method}:{petsc}": [] for method, petsc in [
        ("beuler", False),
        ("herk2", False),
        ("herk4", False),
        ("beuler", True),
        ("cn", True),
    ]}
    endpoint_states = {dt: [] for dt in dts}
    activation_times = []
    reference_events = reference["dynamic_limit_diagnostics"]["events"]
    reference_activation = next(
        event["time"]
        for event in reference_events
        if event["action"] == "activate"
    )

    for method, petsc in [
        ("beuler", False),
        ("herk2", False),
        ("herk4", False),
        ("beuler", True),
        ("cn", True),
    ]:
        _require_backend(petsc)
        key = f"{method}:{petsc}"
        for dt in dts:
            run = _run_two_bus(
                data_dir,
                method,
                petsc,
                fault=True,
                biased=True,
                dt=dt,
                steps=-1,
                tend=0.1,
                ton=0.025,
                toff=0.05,
            )
            result = run[3]
            reference_sample = _sample_reference(
                reference["tvec"],
                reference["history"][:ndiff],
                result["tvec"],
            )
            errors[key].append(
                _normalized_error(result["history"][:ndiff], reference_sample)
            )
            endpoint_states[dt].append(result["history"][:ndiff, -1])
            activation = next(
                event["time"]
                for event in result["dynamic_limit_diagnostics"]["events"]
                if event["action"] == "activate"
            )
            activation_times.append((activation, reference_activation, dt))

    for method_errors in errors.values():
        assert method_errors[-1] < 0.75 * method_errors[0]

    spreads = []
    for dt in dts:
        states = endpoint_states[dt]
        spreads.append(
            max(
                _normalized_error(left, right)
                for left in states
                for right in states
            )
        )
    assert spreads[-1] < 0.75 * spreads[0]
    for activation, fine_activation, dt in activation_times:
        assert abs(activation - fine_activation) <= dt + 1e-12


def test_arkimex_requires_legacy_unconstrained_dynamic_limits():
    with pytest.raises(ValueError, match="enforce_dynamic_limits=False"):
        IntegrationConfig(petsc=True, arkimex=True)

    config = IntegrationConfig(
        petsc=True,
        arkimex=True,
        enforce_dynamic_limits=False,
    )
    assert config.arkimex is True
    assert config.enforce_dynamic_limits is False
