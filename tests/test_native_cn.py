"""Independent native Crank-Nicolson checks on a small index-1 DAE."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from uqgrid.simulation import dynamics
from uqgrid.simulation import jacobian as jacobian_module
from uqgrid.simulation import residual as residual_module
from uqgrid.simulation.config import IntegrationConfig
from uqgrid.simulation.dynamic_limits import (
    DynamicLimitError,
    DynamicLimitMode,
    LimitedStateDescriptor,
)


def _matrix():
    # Keep the complete structural pattern, including its zero entries.
    return csr_matrix(np.ones((2, 2)))


def _install_index_one(monkeypatch):
    psys = SimpleNamespace(
        num_dof_dif=1,
        num_dof_alg=1,
        nbuses=0,
        nloads=0,
        rate=1.0,
        fault_events=[],
        signal_injectors=[],
    )

    def residual(output, state, theta, system):
        output[0] = theta[0] - system.rate * state[0]
        output[1] = state[1] - system.rate * state[0] ** 2

    def jacobian(matrix, state, theta, system):
        matrix.data[:] = [-system.rate, 0.0, -2 * system.rate * state[0], 1.0]

    monkeypatch.setattr(dynamics, "residual_function", residual)
    monkeypatch.setattr(residual_module, "residual_function", residual)
    monkeypatch.setattr(dynamics, "residual_jacobian", jacobian)
    return psys


@pytest.fixture
def index_one(monkeypatch):
    return _install_index_one(monkeypatch)


def _prepare_system(monkeypatch, psys, state, theta):
    monkeypatch.setattr(dynamics, "preallocate_jacobian", lambda system: _matrix())
    monkeypatch.setattr(
        dynamics,
        "_initialize_integration_state",
        lambda *args: (
            SimpleNamespace(validation=None),
            state.copy(),
            theta.copy(),
            {"events": []},
        ),
    )
    monkeypatch.setattr(dynamics, "collect_limited_state_descriptors", lambda *args: [])


def _step(psys, state, h, *, fsolve=False):
    return dynamics.integrate(
        state,
        np.asarray([0.0]),
        h,
        psys,
        np.zeros(2),
        _matrix(),
        None,
        method="cn",
        fsolve=fsolve,
        newton_tol=1e-12,
        newton_max_iter=50,
    )[0]


@pytest.mark.parametrize("fsolve", [False, True])
def test_native_cn_uses_trapezoidal_differentials_and_endpoint_algebraics(
    index_one, fsolve
):
    initial = np.asarray([1.0, 1.0])
    h = 0.2
    endpoint = _step(index_one, initial, h, fsolve=fsolve)
    expected = (1.0 - h / 2) / (1.0 + h / 2)

    np.testing.assert_allclose(endpoint, [expected, expected**2], atol=1e-11)
    # An averaged algebraic constraint would give a different endpoint.
    assert abs(endpoint[1] - endpoint[0] ** 2) < 1e-12
    np.testing.assert_array_equal(initial, [1.0, 1.0])


def test_native_cn_is_second_order_on_an_index_one_dae(index_one):
    errors = []
    for h in (0.2, 0.1, 0.05):
        state = np.asarray([1.0, 1.0])
        for _ in range(round(1.0 / h)):
            state = _step(index_one, state, h)
            assert abs(state[1] - state[0] ** 2) < 1e-11
        errors.append(abs(state[0] - math.exp(-1.0)))

    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8


def test_native_cn_accepts_convergence_on_the_final_newton_iteration(
    index_one, monkeypatch
):
    def residual(output, state, theta, psys):
        output[:] = [-state[0], state[1] - state[0]]

    def jacobian(matrix, state, theta, psys):
        matrix.data[:] = [-1.0, 0.0, -1.0, 1.0]

    monkeypatch.setattr(dynamics, "residual_function", residual)
    monkeypatch.setattr(dynamics, "residual_jacobian", jacobian)
    endpoint = dynamics.integrate(
        np.asarray([1.0, 1.0]),
        np.asarray([0.0]),
        0.2,
        index_one,
        np.zeros(2),
        _matrix(),
        None,
        method="cn",
        newton_tol=1e-12,
        newton_max_iter=1,
    )[0]

    np.testing.assert_allclose(endpoint, [0.9 / 1.1, 0.9 / 1.1], atol=1e-12)


@pytest.mark.parametrize("jacobian_mode", ["analytical", "finite_difference"])
def test_native_cn_discrete_jacobian_matches_independent_differences(
    index_one, monkeypatch, jacobian_mode
):
    initial = np.asarray([1.0, 1.0])
    h = 0.2
    current = {}
    checked = []
    original_jacobian = dynamics.residual_jacobian
    original_solve = dynamics.spsolve

    def record_raw_jacobian(matrix, state, theta, psys):
        current["state"] = state.copy()
        if jacobian_mode == "finite_difference":
            return jacobian_module.finite_difference_residual_jacobian(
                matrix, state, theta, psys, eps=1e-7
            )
        return original_jacobian(matrix, state, theta, psys)

    def independent_step_residual(state):
        x, algebraic = state
        return np.asarray([x - 1.0 + h / 2 * (1.0 + x), algebraic - x**2])

    def check_and_solve(matrix, residual):
        state = current["state"]
        epsilon = 1e-6
        columns = []
        for column in range(2):
            offset = np.zeros(2)
            offset[column] = epsilon
            columns.append(
                (
                    independent_step_residual(state + offset)
                    - independent_step_residual(state - offset)
                )
                / (2 * epsilon)
            )
        np.testing.assert_allclose(matrix.toarray(), np.column_stack(columns), atol=2e-8)
        checked.append(True)
        return original_solve(matrix, residual)

    monkeypatch.setattr(dynamics, "residual_jacobian", record_raw_jacobian)
    monkeypatch.setattr(dynamics, "spsolve", check_and_solve)
    _step(index_one, initial, h)
    assert checked


def _descriptor():
    return LimitedStateDescriptor(
        state_index=0,
        lower_bound=0.0,
        upper_bound=1.0,
        device_type="SEXS",
        bus=1,
        device_id="1",
        enabled=True,
    )


def _limited_step(psys, state, theta, modes, *, time=0.5):
    return dynamics._integrate_cn_with_dynamic_limits(
        state,
        theta,
        0.5,
        psys,
        np.zeros(2),
        _matrix(),
        descriptors=[_descriptor()],
        modes=modes,
        state_tolerance=1e-10,
        release_tolerance=1e-12,
        max_active_set_iterations=20,
        endpoint_time=time,
        newton_tol=1e-12,
        newton_max_iter=50,
    )


@pytest.mark.parametrize(
    "side,drive,inward_drive,pinned,released",
    [("upper", 2.0, 0.0, 1.0, 0.6), ("lower", -1.0, 1.0, 0.0, 0.4)],
)
@pytest.mark.parametrize("jacobian_mode", ["analytical", "finite_difference"])
def test_native_cn_activates_then_releases_under_inward_drive(
    index_one, monkeypatch, side, drive, inward_drive, pinned, released,
    jacobian_mode,
):
    if jacobian_mode == "finite_difference":
        monkeypatch.setattr(
            dynamics,
            "residual_jacobian",
            jacobian_module.finite_difference_residual_jacobian,
        )
    state = np.asarray([0.5, 0.25])
    first, modes, first_events = _limited_step(
        index_one, state, np.asarray([drive]), {0: DynamicLimitMode.FREE}
    )
    second, modes, second_events = _limited_step(
        index_one, first, np.asarray([inward_drive]), modes, time=1.0
    )

    np.testing.assert_allclose(first, [pinned, pinned**2], atol=1e-11)
    np.testing.assert_allclose(second, [released, released**2], atol=1e-11)
    assert modes[0] == DynamicLimitMode.FREE
    assert [(event["action"], event["side"]) for event in first_events + second_events] == [
        ("activate", side),
        ("release", side),
    ]


@pytest.mark.parametrize("jacobian_mode", ["analytical", "finite_difference"])
def test_native_cn_moving_bound_jacobian_matches_independent_differences(
    index_one, monkeypatch, jacobian_mode
):
    original_jacobian = dynamics.residual_jacobian
    original_solve = dynamics.spsolve
    current = {}
    checked = []

    def record_raw_jacobian(matrix, state, theta, psys):
        current["state"] = state.copy()
        if jacobian_mode == "finite_difference":
            return jacobian_module.finite_difference_residual_jacobian(
                matrix, state, theta, psys
            )
        return original_jacobian(matrix, state, theta, psys)

    def independent_residual(state):
        x, voltage = state
        return np.asarray([x - 0.5 * voltage, voltage - x**2])

    def check_and_solve(matrix, residual):
        state = current["state"]
        epsilon = 1e-6
        offsets = np.eye(2) * epsilon
        expected = np.column_stack([
            (independent_residual(state + offset) - independent_residual(state - offset))
            / (2 * epsilon)
            for offset in offsets
        ])
        np.testing.assert_allclose(matrix.toarray(), expected, atol=2e-8)
        checked.append(True)
        return original_solve(matrix, residual)

    descriptor = LimitedStateDescriptor(
        state_index=0,
        lower_bound=0.0,
        upper_bound=0.5,
        device_type="ESDC2A",
        bus=1,
        device_id="1",
        enabled=True,
        bound_scale="terminal_voltage",
        bound_scale_indices=(1,),
    )
    monkeypatch.setattr(dynamics, "residual_jacobian", record_raw_jacobian)
    monkeypatch.setattr(dynamics, "spsolve", check_and_solve)
    endpoint, _, _, converged = dynamics._solve_native_fixed_active_set(
        np.asarray([2.0, 3.0]),
        np.asarray([0.0]),
        0.2,
        index_one,
        np.zeros(2),
        _matrix(),
        method="cn",
        start_derivative=np.asarray([-2.0, -1.0]),
        descriptors=[descriptor],
        modes={0: DynamicLimitMode.UPPER_ACTIVE},
        newton_tol=1e-12,
        newton_max_iter=50,
    )

    assert checked
    assert converged
    np.testing.assert_allclose(endpoint, [2.0, 4.0], atol=1e-11)


def test_native_cn_freezes_start_derivative_across_active_set_retries(
    index_one, monkeypatch
):
    original_solve = dynamics._solve_native_fixed_active_set
    captured = []

    def record_start(*args, **kwargs):
        start_derivative = kwargs["start_derivative"]
        captured.append(start_derivative)
        assert not start_derivative.flags.writeable
        np.testing.assert_allclose(start_derivative, [1.5, 0.0])
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(dynamics, "_solve_native_fixed_active_set", record_start)
    endpoint, modes, events = _limited_step(
        index_one,
        np.asarray([0.5, 0.25]),
        np.asarray([2.0]),
        {0: DynamicLimitMode.FREE},
    )

    assert len(captured) == 2
    assert captured[0] is captured[1]
    np.testing.assert_allclose(endpoint, [1.0, 1.0], atol=1e-11)
    assert modes[0] == DynamicLimitMode.UPPER_ACTIVE
    assert [event["action"] for event in events] == ["activate"]


def test_native_cn_nonfinite_start_has_method_and_backend_context(
    index_one, monkeypatch
):
    original_residual = dynamics.residual_function

    def nonfinite(output, *args):
        original_residual(output, *args)
        output[0] = np.nan

    monkeypatch.setattr(dynamics, "residual_function", nonfinite)
    with pytest.raises(DynamicLimitError) as error:
        _limited_step(
            index_one,
            np.asarray([1.0, 1.0]),
            np.asarray([2.0]),
            {0: DynamicLimitMode.UPPER_ACTIVE},
        )

    diagnostics = error.value.diagnostics
    assert diagnostics["phase"] == "runtime"
    assert diagnostics["method"] == "cn"
    assert diagnostics["backend"] == "native"
    assert diagnostics["failure_reasons"] == ["non_finite_derivative"]
    assert diagnostics["time"] == pytest.approx(0.0)
    assert diagnostics["stage_or_endpoint"] == "start"
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize("failure", ["nonfinite", "nonconvergence"])
def test_native_cn_unlimited_failures_name_the_method_and_backend(
    index_one, monkeypatch, failure
):
    if failure == "nonfinite":
        original_residual = dynamics.residual_function

        def nonfinite(output, *args):
            original_residual(output, *args)
            output[0] = np.nan

        monkeypatch.setattr(dynamics, "residual_function", nonfinite)

    with pytest.raises(RuntimeError) as error:
        dynamics.integrate(
            np.asarray([1.0, 1.0]),
            np.asarray([0.0]),
            0.2,
            index_one,
            np.zeros(2),
            _matrix(),
            None,
            method="cn",
            newton_tol=1e-12,
            newton_max_iter=1,
        )

    assert "method='cn'" in str(error.value)
    assert "backend='native'" in str(error.value)


def test_native_cn_refreshes_start_after_fault_algebraic_projection(
    index_one, monkeypatch
):
    class Fault:
        active = False

        def apply(self):
            self.active = True
            index_one.rate = 3.0

        def remove(self):
            self.active = False
            index_one.rate = 1.0

    fault = Fault()
    index_one.fault_events = [fault]
    _prepare_system(monkeypatch, index_one, np.asarray([1.0, 1.0]), np.asarray([0.0]))
    result = dynamics.integrate_system(
        index_one,
        IntegrationConfig(method="cn", dt=0.1, steps=3, ton=0.1, toff=0.2),
    )

    q1 = (1.0 - 0.1 / 2) / (1.0 + 0.1 / 2)
    q3 = (1.0 - 3.0 * 0.1 / 2) / (1.0 + 3.0 * 0.1 / 2)
    expected_x = np.asarray([1.0, q1, q1 * q3, q1 * q3 * q1])
    np.testing.assert_allclose(result["tvec"], [0.0, 0.1, 0.2, 0.3], atol=1e-14)
    np.testing.assert_allclose(result["history"][0], expected_x, atol=1e-11)
    np.testing.assert_allclose(
        result["history"][1], expected_x**2 * [1.0, 3.0, 1.0, 1.0], atol=1e-11
    )
    assert not fault.active


def test_native_cn_samples_injectors_once_and_holds_parameters_for_interval(
    index_one, monkeypatch
):
    calls = []

    class Injector:
        def update(self, time, theta, psys):
            calls.append(time)
            theta[0] = 1.0 + time

    index_one.signal_injectors = [Injector()]
    _prepare_system(monkeypatch, index_one, np.asarray([0.0, 0.0]), np.asarray([0.0]))
    result = dynamics.integrate_system(
        index_one, IntegrationConfig(method="cn", dt=0.1, steps=3)
    )
    expected = [0.0]
    for forcing in (1.0, 1.1, 1.2):
        expected.append((0.95 * expected[-1] + 0.1 * forcing) / 1.05)

    np.testing.assert_allclose(calls, [0.0, 0.1, 0.2], atol=1e-14)
    np.testing.assert_allclose(result["history"][0], expected, atol=1e-11)
    np.testing.assert_allclose(result["history"][1], np.square(expected), atol=1e-11)


def test_native_cn_failure_after_fault_application_restores_topology(
    index_one, monkeypatch
):
    class Fault:
        active = False

        def apply(self):
            self.active = True
            index_one.rate = 3.0

        def remove(self):
            self.active = False
            index_one.rate = 1.0

    fault = Fault()
    index_one.fault_events = [fault]
    _prepare_system(monkeypatch, index_one, np.asarray([1.0, 1.0]), np.asarray([0.0]))
    original_integrate = dynamics.integrate
    calls = []

    def fail_second_interval(state, theta, h, *args, **kwargs):
        if h > 0.0:
            calls.append(h)
            if len(calls) == 2:
                assert fault.active
                raise RuntimeError("CN test failure after fault application")
        return original_integrate(state, theta, h, *args, **kwargs)

    monkeypatch.setattr(dynamics, "integrate", fail_second_interval)
    with pytest.raises(RuntimeError, match="CN test failure"):
        dynamics.integrate_system(
            index_one,
            IntegrationConfig(method="cn", dt=0.1, steps=2, ton=0.1, toff=0.3),
        )

    assert len(calls) == 2
    assert not fault.active
    assert index_one.rate == 1.0


def test_native_cn_subprocess_does_not_require_petsc(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = f"""
import importlib.abc
import runpy
import sys

class BlockPETSc(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'petsc4py' or fullname.startswith('petsc4py.'):
            raise ImportError('PETSc deliberately unavailable for native CN')

sys.meta_path.insert(0, BlockPETSc())
namespace = runpy.run_path({str(root / 'tests' / 'test_native_cn.py')!r})
assert namespace['dynamics'].__file__ == {str(root / 'uqgrid' / 'simulation' / 'dynamics.py')!r}
monkeypatch = namespace['pytest'].MonkeyPatch()
psys = namespace['_install_index_one'](monkeypatch)
np = namespace['np']
namespace['_prepare_system'](monkeypatch, psys, np.array([1., 1.]), np.array([0.]))
config = namespace['IntegrationConfig'](method='cn', dt=0.1, steps=2)
result = namespace['dynamics'].integrate_system(psys, config)
q = 0.95 / 1.05
np.testing.assert_allclose(result['history'][0], [1., q, q*q], atol=1e-11)
assert not any(name == 'petsc4py' or name.startswith('petsc4py.') for name in sys.modules)
print('native CN completed without PETSc')
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "native CN completed without PETSc" in completed.stdout


def test_native_cn_switch_time_error_is_first_order(index_one):
    crossing_time = 0.1
    index_one.rate = 0.0
    descriptor = LimitedStateDescriptor(
        state_index=0,
        lower_bound=-1.0,
        upper_bound=crossing_time,
        device_type="SEXS",
        bus=1,
        device_id="1",
        enabled=True,
    )
    errors = []
    step_sizes = []
    for intervals_before_half_step in (4, 8, 16):
        h = crossing_time / (intervals_before_half_step + 0.5)
        state = np.asarray([0.0, 0.0])
        modes = {0: DynamicLimitMode.FREE}
        events = []
        for step in range(1, intervals_before_half_step + 2):
            state, modes, interval_events = (
                dynamics._integrate_cn_with_dynamic_limits(
                    state,
                    np.asarray([1.0]),
                    h,
                    index_one,
                    np.zeros(2),
                    _matrix(),
                    descriptors=[descriptor],
                    modes=modes,
                    state_tolerance=1e-12,
                    release_tolerance=1e-12,
                    max_active_set_iterations=20,
                    endpoint_time=step * h,
                    newton_tol=1e-12,
                    newton_max_iter=50,
                    prior_events=events,
                )
            )
            events.extend(interval_events)

        np.testing.assert_allclose(state, [crossing_time, 0.0], atol=1e-12)
        activation_time = next(
            event["time"] for event in events if event["action"] == "activate"
        )
        step_sizes.append(h)
        errors.append(activation_time - crossing_time)

    np.testing.assert_allclose(
        np.asarray(errors) / np.asarray(step_sizes), 0.5, atol=1e-10
    )
