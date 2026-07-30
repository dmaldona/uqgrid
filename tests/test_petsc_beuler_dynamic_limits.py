import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.simulation import dynamics
from uqgrid.simulation.config import IntegrationConfig, IntegrationCtx
from uqgrid.simulation.dynamic_limits import (
    DynamicLimitError,
    DynamicLimitMode,
    collect_limited_state_descriptors,
    initialize_dynamic_limit_modes,
)
from uqgrid.simulation.dynamics import (
    initialize_system,
    integrate_system,
    preallocate_jacobian,
)
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


@pytest.fixture
def data_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )


class _FakeVector:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float).copy()

    def __getitem__(self, key):
        return self.values[key]

    def setArray(self, values):
        self.values = np.asarray(values, dtype=float).copy()

    def assemble(self):
        return None


class _FakeMatrix:
    def __init__(self):
        self.value = None

    def setValuesCSR(self, indptr, indices, data):
        self.value = csr_matrix(
            (np.array(data, copy=True), np.array(indices, copy=True),
             np.array(indptr, copy=True))
        )

    def assemble(self):
        return None


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


def _set_limits(theta, exciter, lower, upper, *, vref_offset=0.0):
    ptr = exciter.par_ptr
    theta[ptr + 4] = lower
    theta[ptr + 5] = upper
    theta[ptr + 6] = 1.0
    theta[ptr + 7] += vref_offset


def _biased_two_bus(data_dir, side, *, petsc, fault=False, width=0.01):
    psys = _build_two_bus(data_dir, fault=fault)
    z0, theta, ctx = _initialize_context(psys)
    exciter = psys.exc[0]
    state_index = exciter.dif_ptr + exciter.efd_idx
    initial = float(z0[state_index])
    lower = initial - width
    upper = initial + width
    _set_limits(
        theta,
        exciter,
        lower,
        upper,
        vref_offset=0.05 if side == "upper" else -0.05,
    )
    ctx.set_theta(theta.copy())
    config = IntegrationConfig(
        method="beuler",
        petsc=petsc,
        steps=4,
        dt=0.005,
        ton=10.0,
        toff=11.0,
        newton_tol=1e-10,
        newton_max_iter=100,
    )
    return psys, ctx, config, state_index, lower, upper


def _event_identity(event):
    return (
        event["state_index"],
        event["bus"],
        event["device_id"],
        event["side"],
        event["action"],
        event["time"],
        event["stage_or_endpoint"],
        event["active_set_iterations"],
    )


def test_petsc_callbacks_match_native_be_rows_for_multiple_active_limits(
    data_dir,
):
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus_SEXS.dyr"))
    psys.createYbusComplex()
    psys.power_injection = False
    z0, theta, _ = _initialize_context(psys)
    descriptors = collect_limited_state_descriptors(psys, theta)
    assert len(descriptors) >= 2
    modes = initialize_dynamic_limit_modes(descriptors)
    modes[descriptors[0].state_index] = DynamicLimitMode.UPPER_ACTIVE
    modes[descriptors[1].state_index] = DynamicLimitMode.LOWER_ACTIVE
    endpoint = z0.copy()
    endpoint[descriptors[0].state_index] = descriptors[0].upper_bound
    endpoint[descriptors[1].state_index] = descriptors[1].lower_bound
    h = 0.005

    ordinary_residual = np.zeros_like(endpoint)
    dynamics._assemble_beuler_residual(
        ordinary_residual, endpoint, z0, h, psys, theta
    )
    ordinary_jacobian = preallocate_jacobian(psys)
    residual_jacobian(ordinary_jacobian, endpoint, theta, psys)
    dynamics.jacobian_beuler(ordinary_jacobian, psys.num_dof_dif, h)

    callback_residual = np.zeros_like(endpoint)
    callback_jacobian = preallocate_jacobian(psys)
    problem = dynamics._PETScBEActiveSetProblem(
        psys, theta, callback_residual, callback_jacobian
    )
    problem.set_interval(z0, h, descriptors, modes)
    x = _FakeVector(endpoint)
    f = _FakeVector(np.zeros_like(endpoint))
    matrix = _FakeMatrix()

    problem.function(None, x, f)
    problem.jacobian_function(None, x, matrix, matrix)

    active_indices = {
        descriptors[0].state_index,
        descriptors[1].state_index,
    }
    for state_index in active_indices:
        expected = np.zeros(endpoint.size)
        expected[state_index] = 1.0
        np.testing.assert_allclose(matrix.value.toarray()[state_index], expected)
    inactive = [
        index for index in range(endpoint.size) if index not in active_indices
    ]
    np.testing.assert_allclose(
        matrix.value.toarray()[inactive], ordinary_jacobian.toarray()[inactive]
    )
    np.testing.assert_allclose(
        f.values[inactive], ordinary_residual[inactive]
    )


@pytest.mark.parametrize("snes_type", ["vinewtonrsls", "vinewtonssls"])
def test_petsc_vi_snes_types_are_rejected(snes_type):
    with pytest.raises(ValueError, match="variational-inequality"):
        dynamics._validate_petsc_snes_type(snes_type)


@pytest.mark.parametrize("side", ["upper", "lower"])
def test_native_and_petsc_be_match_limit_crossings(data_dir, side):
    pytest.importorskip("petsc4py")
    native = _biased_two_bus(data_dir, side, petsc=False)
    petsc = _biased_two_bus(data_dir, side, petsc=True)

    native_result = integrate_system(native[0], native[2], native[1])
    petsc_result = integrate_system(petsc[0], petsc[2], petsc[1])

    np.testing.assert_allclose(petsc_result["tvec"], native_result["tvec"])
    np.testing.assert_allclose(
        petsc_result["history"], native_result["history"], atol=1e-8, rtol=1e-8
    )
    lower, upper = petsc[4], petsc[5]
    values = petsc_result["history"][petsc[3]]
    assert np.min(values) >= lower - 1e-12
    assert np.max(values) <= upper + 1e-12
    native_events = native_result["dynamic_limit_diagnostics"]["events"]
    petsc_events = petsc_result["dynamic_limit_diagnostics"]["events"]
    assert [_event_identity(event) for event in petsc_events] == [
        _event_identity(event) for event in native_events
    ]
    free_residual = np.zeros(petsc_result["history"].shape[0])
    dynamics._assemble_beuler_residual(
        free_residual,
        petsc_result["history"][:, -1],
        petsc_result["history"][:, -2],
        petsc_result["tvec"][-1] - petsc_result["tvec"][-2],
        petsc[0],
        petsc[1].theta_user,
    )
    if side == "upper":
        assert free_residual[petsc[3]] <= 1e-10
    else:
        assert free_residual[petsc[3]] >= -1e-10


def _run_release_pair(data_dir, *, petsc, side):
    psys = _build_two_bus(data_dir)
    z0, theta, _ = _initialize_context(psys)
    exciter = psys.exc[0]
    state_index = exciter.dif_ptr + exciter.efd_idx
    ptr = exciter.par_ptr
    initial = float(z0[state_index])
    base_vref = float(theta[ptr + 7])
    if side == "upper":
        _set_limits(
            theta, exciter, initial - 0.1, initial, vref_offset=0.05
        )
    else:
        _set_limits(
            theta, exciter, initial, initial + 0.1, vref_offset=-0.05
        )
    descriptors = collect_limited_state_descriptors(psys, theta)
    modes = initialize_dynamic_limit_modes(descriptors)
    residual = np.zeros_like(z0)
    jacobian = preallocate_jacobian(psys)
    workspace = None
    try:
        if petsc:
            config = IntegrationConfig(petsc=True, method="beuler")
            PETSc = dynamics._get_petsc_for_config(config)
            workspace = dynamics._PETScBEActiveSetWorkspace(
                PETSc,
                psys,
                theta,
                residual,
                jacobian,
                newton_tol=1e-10,
                newton_max_iter=100,
            )
            stepper = dynamics._integrate_petsc_beuler_with_dynamic_limits
            extra = {"workspace": workspace}
        else:
            stepper = dynamics._integrate_beuler_with_dynamic_limits
            extra = {}

        pinned, modes, first_events = stepper(
            z0,
            theta,
            0.005,
            psys,
            residual,
            jacobian,
            descriptors=descriptors,
            modes=modes,
            state_tolerance=1e-8,
            release_tolerance=1e-10,
            max_active_set_iterations=20,
            endpoint_time=0.005,
            newton_tol=1e-10,
            newton_max_iter=100,
            **extra,
        )
        theta[ptr + 7] = (
            base_vref - 0.05 if side == "upper" else base_vref + 0.05
        )
        released, modes, second_events = stepper(
            pinned,
            theta,
            0.005,
            psys,
            residual,
            jacobian,
            descriptors=descriptors,
            modes=modes,
            state_tolerance=1e-8,
            release_tolerance=1e-10,
            max_active_set_iterations=20,
            endpoint_time=0.01,
            newton_tol=1e-10,
            newton_max_iter=100,
            **extra,
        )
        return pinned, released, modes, first_events + second_events, state_index
    finally:
        if workspace is not None:
            workspace.destroy()


@pytest.mark.parametrize("side", ["upper", "lower"])
def test_native_and_petsc_be_match_activation_and_release(data_dir, side):
    pytest.importorskip("petsc4py")
    native = _run_release_pair(data_dir, petsc=False, side=side)
    petsc = _run_release_pair(data_dir, petsc=True, side=side)

    np.testing.assert_allclose(petsc[0], native[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(petsc[1], native[1], atol=1e-8, rtol=1e-8)
    assert petsc[2][petsc[4]] == native[2][native[4]] == DynamicLimitMode.FREE
    assert [_event_identity(event) for event in petsc[3]] == [
        _event_identity(event) for event in native[3]
    ]


def _build_limited_ieee9(data_dir, *, petsc):
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus_SEXS.dyr"))
    psys.createYbusComplex()
    psys.power_injection = False
    z0, theta, ctx = _initialize_context(psys)
    limited = []
    for index, exciter in enumerate(psys.exc):
        state_index = exciter.dif_ptr + exciter.efd_idx
        initial = float(z0[state_index])
        _set_limits(
            theta,
            exciter,
            initial - 0.01,
            initial + 0.01,
            vref_offset=0.05 if index % 2 == 0 else -0.05,
        )
        limited.append((state_index, initial - 0.01, initial + 0.01))
    ctx.set_theta(theta.copy())
    config = IntegrationConfig(
        method="beuler",
        petsc=petsc,
        steps=3,
        dt=0.005,
        ton=10.0,
        toff=11.0,
    )
    return psys, ctx, config, limited


def test_native_and_petsc_be_match_multiple_sexs_limits(data_dir):
    pytest.importorskip("petsc4py")
    native = _build_limited_ieee9(data_dir, petsc=False)
    petsc = _build_limited_ieee9(data_dir, petsc=True)

    native_result = integrate_system(native[0], native[2], native[1])
    petsc_result = integrate_system(petsc[0], petsc[2], petsc[1])

    np.testing.assert_allclose(
        petsc_result["history"], native_result["history"], atol=1e-8, rtol=1e-8
    )
    for state_index, lower, upper in petsc[3]:
        values = petsc_result["history"][state_index]
        assert np.min(values) >= lower - 1e-12
        assert np.max(values) <= upper + 1e-12


def test_native_and_petsc_be_match_off_grid_fault_boundaries(data_dir):
    pytest.importorskip("petsc4py")
    native = _biased_two_bus(data_dir, "upper", petsc=False, fault=True)
    petsc = _biased_two_bus(data_dir, "upper", petsc=True, fault=True)
    for config in (native[2], petsc[2]):
        config.steps = 4
        config.ton = 0.0075
        config.toff = 0.0135

    native_result = integrate_system(native[0], native[2], native[1])
    petsc_result = integrate_system(petsc[0], petsc[2], petsc[1])

    expected = [0.0, 0.005, 0.0075, 0.01, 0.0135, 0.015, 0.02]
    np.testing.assert_allclose(petsc_result["tvec"], expected)
    np.testing.assert_allclose(petsc_result["tvec"], native_result["tvec"])
    np.testing.assert_allclose(
        petsc_result["history"], native_result["history"], atol=1e-8, rtol=1e-8
    )
    fault = petsc[0].fault_events[0]
    residual = np.zeros(petsc_result["history"].shape[0])
    fault.apply()
    residual_function(
        residual, petsc_result["history"][:, 2], petsc[1].theta_user, petsc[0]
    )
    assert np.linalg.norm(residual[petsc[0].num_dof_dif :], np.inf) < 1e-8
    fault.remove()
    residual_function(
        residual, petsc_result["history"][:, 4], petsc[1].theta_user, petsc[0]
    )
    assert np.linalg.norm(residual[petsc[0].num_dof_dif :], np.inf) < 1e-8
    assert fault.active is False


def test_limited_petsc_be_bypasses_ts(data_dir, monkeypatch):
    pytest.importorskip("petsc4py")
    psys, ctx, config, _, _, _ = _biased_two_bus(
        data_dir, "upper", petsc=True
    )
    config.steps = 1
    monkeypatch.setattr(
        dynamics,
        "_petsc_ts_type",
        lambda *args: pytest.fail("limited PETSc BE must not configure TS"),
    )

    result = integrate_system(psys, config, ctx)

    assert result["history"].shape[1] == 2


def test_disabled_limits_keep_existing_petsc_ts_path(data_dir, monkeypatch):
    pytest.importorskip("petsc4py")
    psys = _build_two_bus(data_dir)
    monkeypatch.setattr(
        dynamics,
        "_integrate_system_petsc_beuler_limits",
        lambda *args: pytest.fail("disabled limits must keep the TS path"),
    )

    result = integrate_system(
        psys,
        IntegrationConfig(
            petsc=True,
            method="beuler",
            enforce_dynamic_limits=False,
            steps=1,
            dt=0.005,
            ton=10.0,
            toff=11.0,
        ),
    )

    assert result["history"].shape[1] == 2


def test_zero_limited_states_keep_existing_petsc_ts_path(data_dir, monkeypatch):
    pytest.importorskip("petsc4py")
    psys = load_psse(raw_filename=os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus.dyr"))
    psys.createYbusComplex()
    monkeypatch.setattr(
        dynamics,
        "_integrate_system_petsc_beuler_limits",
        lambda *args: pytest.fail("zero limited states must keep the TS path"),
    )

    result = integrate_system(
        psys,
        IntegrationConfig(
            petsc=True,
            method="beuler",
            steps=1,
            dt=0.005,
            ton=10.0,
            toff=11.0,
        ),
    )

    assert result["dynamic_limit_diagnostics"]["enabled_state_count"] == 0
    assert result["history"].shape[1] == 2


def test_petsc_snes_nonconvergence_has_structured_runtime_diagnostics():
    descriptor = SimpleNamespace(
        state_index=0,
        lower_bound=0.0,
        upper_bound=1.0,
        enabled=True,
        to_dict=lambda: {
            "state_index": 0,
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "device_type": "SEXS",
            "bus": 1,
            "device_id": "1",
            "enabled": True,
        },
    )
    prior_event = {"action": "activate", "state_index": 0}

    class FailedWorkspace:
        def solve_fixed_active_set(self, *args):
            return (
                np.asarray([0.5]),
                4,
                np.inf,
                False,
                {
                    "failure_reason": "snes_nonconvergence",
                    "snes_type": "newtonls",
                    "snes_converged_reason": -5,
                    "snes_converged_reason_name": "diverged_max_it",
                    "snes_iterations": 4,
                    "snes_function_norm": None,
                },
            )

    with pytest.raises(DynamicLimitError) as exc_info:
        dynamics._integrate_petsc_beuler_with_dynamic_limits(
            np.asarray([0.5]),
            np.asarray([]),
            0.1,
            SimpleNamespace(num_dof_dif=1),
            np.zeros(1),
            csr_matrix(np.eye(1)),
            workspace=FailedWorkspace(),
            descriptors=[descriptor],
            modes={0: DynamicLimitMode.FREE},
            state_tolerance=1e-8,
            release_tolerance=1e-10,
            max_active_set_iterations=20,
            endpoint_time=0.1,
            newton_tol=1e-10,
            newton_max_iter=4,
            prior_events=[prior_event],
        )

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["phase"] == "runtime"
    assert diagnostics["backend"] == "petsc"
    assert diagnostics["nonlinear_solver"] == "petsc_snes"
    assert diagnostics["failure_reasons"] == ["snes_nonconvergence"]
    assert diagnostics["snes_converged_reason"] == -5
    assert diagnostics["snes_converged_reason_name"] == "diverged_max_it"
    assert diagnostics["events"] == [prior_event]
    json.dumps(diagnostics, allow_nan=False)
