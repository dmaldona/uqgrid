"""Shared metadata, validation, and pure helpers for hard dynamic limits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Iterable, Mapping

import numpy as np


DYNAMIC_LIMIT_EVENT_FIELDS = (
    "device_type",
    "device_id",
    "bus",
    "state_index",
    "side",
    "action",
    "time",
    "stage_or_endpoint",
    "raw_derivative",
    "state_before",
    "state_after",
    "bound",
    "active_set_iterations",
)

DYNAMIC_LIMIT_EVENT_ACTIONS = (
    "project",
    "block_outward_derivative",
    "activate",
    "release",
)


@dataclass(frozen=True)
class LimitedStateDescriptor:
    """JSON-safe identity and bounds for one limited differential state."""

    state_index: int
    lower_bound: float
    upper_bound: float
    device_type: str
    bus: int
    device_id: str
    enabled: bool
    bound_scale: str | None = None
    bound_scale_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        if self.bound_scale is None:
            values.pop("bound_scale")
            values.pop("bound_scale_indices")
        for key in ("lower_bound", "upper_bound"):
            value = float(values[key])
            values[key] = value if math.isfinite(value) else None
        return values


class DynamicLimitError(RuntimeError):
    """Raised when dynamic-limit validation or helper evaluation fails."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        reasons = diagnostics.get("failure_reasons")
        if reasons is None:
            reasons = diagnostics.get("initialization", {}).get(
                "failure_reasons", []
            )
        reasons = ", ".join(reasons)
        super().__init__(
            diagnostics.get("message", "Dynamic-limit validation failed")
            + (f": {reasons}" if reasons else "")
        )


class DynamicLimitMode(str, Enum):
    """Active-set mode for one limited differential state."""

    FREE = "free"
    LOWER_ACTIVE = "lower_active"
    UPPER_ACTIVE = "upper_active"


def collect_limited_state_descriptors(psys, theta) -> list[LimitedStateDescriptor]:
    """Resolve model-level bounded-state metadata against effective parameters."""
    theta = np.asarray(theta, dtype=float)
    descriptors = []
    for device in psys.devices:
        for metadata in getattr(device, "bounded_state_metadata", ()):
            par_ptr = int(device.par_ptr)
            bound_scale_indices = ()
            if metadata.bound_scale == "terminal_voltage":
                voltage_ptr = psys.num_dof_dif + psys.num_dof_alg + 2 * device.bus
                bound_scale_indices = (
                    (voltage_ptr,)
                    if psys.power_injection
                    else (voltage_ptr, voltage_ptr + 1)
                )
            descriptors.append(
                LimitedStateDescriptor(
                    state_index=int(device.dif_ptr + metadata.state_offset),
                    lower_bound=float(
                        theta[par_ptr + metadata.lower_parameter_offset]
                    ),
                    upper_bound=float(
                        theta[par_ptr + metadata.upper_parameter_offset]
                    ),
                    device_type=metadata.device_type,
                    bus=int(psys.buses[device.bus].id),
                    device_id=str(device.id_tag).strip(),
                    enabled=bool(
                        theta[par_ptr + metadata.enabled_parameter_offset]
                    ),
                    bound_scale=metadata.bound_scale,
                    bound_scale_indices=bound_scale_indices,
                )
            )
    return descriptors


def evaluate_limited_state_bounds(state, descriptor):
    """Return effective bounds and derivatives of their common scale."""
    if descriptor.bound_scale is None:
        return float(descriptor.lower_bound), float(descriptor.upper_bound), ()
    if descriptor.bound_scale != "terminal_voltage":
        raise ValueError(f"Unsupported dynamic-limit bound scale: {descriptor.bound_scale}")
    state = np.asarray(state, dtype=float)
    indices = descriptor.bound_scale_indices
    if len(indices) == 1:
        scale = float(state[indices[0]])
        derivatives = ((indices[0], 1.0),)
    else:
        real = float(state[indices[0]])
        imag = float(state[indices[1]])
        scale = math.hypot(real, imag)
        derivatives = (
            ((indices[0], 0.0), (indices[1], 0.0))
            if scale == 0.0
            else ((indices[0], real / scale), (indices[1], imag / scale))
        )
    return (
        float(descriptor.lower_bound) * scale,
        float(descriptor.upper_bound) * scale,
        derivatives,
    )


def _json_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _violation_record(
    descriptor: LimitedStateDescriptor,
    *,
    value: float,
    side: str,
    bound: float | None,
    violation: float | None,
) -> dict[str, Any]:
    return {
        **descriptor.to_dict(),
        "initial_value": _json_number(value),
        "side": side,
        "bound": None if bound is None else _json_number(bound),
        "violation": None if violation is None else _json_number(violation),
    }


def _validate_enabled_descriptor(
    descriptor: LimitedStateDescriptor,
    *,
    operation: str,
    time: float | None = None,
    stage_or_endpoint: str | None = None,
) -> None:
    lower = float(descriptor.lower_bound)
    upper = float(descriptor.upper_bound)
    reason = None
    if not math.isfinite(lower) or not math.isfinite(upper):
        reason = "non_finite_bounds"
    elif lower == upper:
        reason = "degenerate_bounds"
    elif lower > upper:
        reason = "inverted_bounds"
    if reason is not None:
        raise DynamicLimitError(
            {
                "message": f"Dynamic-limit {operation} failed",
                "operation": operation,
                "time": None if time is None else _json_number(time),
                "stage_or_endpoint": stage_or_endpoint,
                "failure_reasons": [reason],
                "violations": [
                    {
                        **descriptor.to_dict(),
                        "side": "invalid",
                        "bound": None,
                    }
                ],
                "events": [],
            }
        )


def _event_record(
    descriptor: LimitedStateDescriptor,
    *,
    side: str,
    action: str,
    bound: float,
    state_before: float,
    state_after: float,
    raw_derivative: float | None = None,
    free_residual: float | None = None,
    time: float | None = None,
    stage_or_endpoint: str | None = None,
    active_set_iterations: int | None = None,
) -> dict[str, Any]:
    return {
        **descriptor.to_dict(),
        "side": side,
        "action": action,
        "time": None if time is None else _json_number(time),
        "stage_or_endpoint": stage_or_endpoint,
        "raw_derivative": (
            None if raw_derivative is None else _json_number(raw_derivative)
        ),
        "free_residual": (
            None if free_residual is None else _json_number(free_residual)
        ),
        "state_before": _json_number(state_before),
        "state_after": _json_number(state_after),
        "bound": _json_number(bound),
        "active_set_iterations": active_set_iterations,
    }


def _helper_failure(
    operation: str,
    reason: str,
    descriptor: LimitedStateDescriptor,
    *,
    value: float | None = None,
    time: float | None = None,
    stage_or_endpoint: str | None = None,
) -> DynamicLimitError:
    return DynamicLimitError(
        {
            "message": f"Dynamic-limit {operation} failed",
            "operation": operation,
            "time": None if time is None else _json_number(time),
            "stage_or_endpoint": stage_or_endpoint,
            "failure_reasons": [reason],
            "violations": [
                {
                    **descriptor.to_dict(),
                    "value": None if value is None else _json_number(value),
                }
            ],
            "events": [],
        }
    )


def _contextualize_dynamic_limit_runtime_error(
    error: DynamicLimitError,
    *,
    method: str,
    backend: str,
    time: float,
    stage_or_endpoint: str,
    prior_events: Iterable[Mapping[str, Any]] = (),
) -> DynamicLimitError:
    """Return a helper failure decorated with integration runtime context."""
    diagnostics = dict(error.diagnostics)
    diagnostics["phase"] = "runtime"
    diagnostics["method"] = method
    diagnostics["backend"] = backend
    if diagnostics.get("time") is None:
        diagnostics["time"] = _json_number(time)
    if diagnostics.get("stage_or_endpoint") is None:
        diagnostics["stage_or_endpoint"] = stage_or_endpoint
    diagnostics["events"] = [
        *[dict(event) for event in prior_events],
        *[dict(event) for event in diagnostics.get("events", ())],
    ]
    return DynamicLimitError(diagnostics)


def project_limited_states(
    state,
    descriptors: Iterable[LimitedStateDescriptor],
    *,
    time: float | None = None,
    stage_or_endpoint: str | None = None,
    active_set_iterations: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Clamp enabled limited states exactly, without mutating the input."""
    projected = np.array(state, dtype=float, copy=True)
    events = []
    for descriptor in descriptors:
        if not descriptor.enabled:
            continue
        _validate_enabled_descriptor(
            descriptor,
            operation="state_projection",
            time=time,
            stage_or_endpoint=stage_or_endpoint,
        )
        value = float(projected[descriptor.state_index])
        if not math.isfinite(value):
            raise _helper_failure(
                "state_projection",
                "non_finite_state",
                descriptor,
                value=value,
                time=time,
                stage_or_endpoint=stage_or_endpoint,
            )
        lower, upper, _ = evaluate_limited_state_bounds(projected, descriptor)
        if value < lower:
            projected[descriptor.state_index] = lower
            events.append(
                _event_record(
                    descriptor,
                    side="lower",
                    action="project",
                    bound=lower,
                    state_before=value,
                    state_after=lower,
                    time=time,
                    stage_or_endpoint=stage_or_endpoint,
                    active_set_iterations=active_set_iterations,
                )
            )
        elif value > upper:
            projected[descriptor.state_index] = upper
            events.append(
                _event_record(
                    descriptor,
                    side="upper",
                    action="project",
                    bound=upper,
                    state_before=value,
                    state_after=upper,
                    time=time,
                    stage_or_endpoint=stage_or_endpoint,
                    active_set_iterations=active_set_iterations,
                )
            )
    return projected, events


def project_limited_derivatives(
    state,
    raw_derivative,
    descriptors: Iterable[LimitedStateDescriptor],
    *,
    tolerance: float,
    time: float | None = None,
    stage_or_endpoint: str | None = None,
    active_set_iterations: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Zero only derivatives that point outward at an enabled bound."""
    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError("Dynamic-limit state tolerance must be non-negative.")
    state = np.asarray(state, dtype=float)
    projected = np.array(raw_derivative, dtype=float, copy=True)
    events = []
    for descriptor in descriptors:
        if not descriptor.enabled:
            continue
        if descriptor.bound_scale is not None:
            # Moving bounds are enforced by stage/endpoint projection. Their
            # outward direction depends on the unavailable bound velocity.
            continue
        _validate_enabled_descriptor(
            descriptor,
            operation="derivative_projection",
            time=time,
            stage_or_endpoint=stage_or_endpoint,
        )
        value = float(state[descriptor.state_index])
        derivative = float(projected[descriptor.state_index])
        if not math.isfinite(value):
            raise _helper_failure(
                "derivative_projection",
                "non_finite_state",
                descriptor,
                value=value,
                time=time,
                stage_or_endpoint=stage_or_endpoint,
            )
        if not math.isfinite(derivative):
            raise _helper_failure(
                "derivative_projection",
                "non_finite_derivative",
                descriptor,
                value=derivative,
                time=time,
                stage_or_endpoint=stage_or_endpoint,
            )
        lower, upper, _ = evaluate_limited_state_bounds(state, descriptor)
        side = None
        bound = None
        if value >= upper - tolerance and derivative > 0.0:
            side = "upper"
            bound = upper
        elif value <= lower + tolerance and derivative < 0.0:
            side = "lower"
            bound = lower
        if side is not None:
            projected[descriptor.state_index] = 0.0
            events.append(
                _event_record(
                    descriptor,
                    side=side,
                    action="block_outward_derivative",
                    bound=bound,
                    state_before=value,
                    state_after=value,
                    raw_derivative=derivative,
                    time=time,
                    stage_or_endpoint=stage_or_endpoint,
                    active_set_iterations=active_set_iterations,
                )
            )
    return projected, events


def update_explicit_dynamic_limit_modes(
    state,
    raw_derivative,
    descriptors: Iterable[LimitedStateDescriptor],
    modes: Mapping[int, DynamicLimitMode | str],
    *,
    tolerance: float,
    time: float | None = None,
    stage_or_endpoint: str | None = None,
) -> tuple[dict[int, DynamicLimitMode], bool, list[dict[str, Any]]]:
    """Track explicit limiter activation and release without changing values."""
    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError("Dynamic-limit state tolerance must be non-negative.")
    state = np.asarray(state, dtype=float)
    raw_derivative = np.asarray(raw_derivative, dtype=float)
    descriptors = list(descriptors)
    updated = {
        descriptor.state_index: _mode_for(modes, descriptor)
        for descriptor in descriptors
    }
    events = []

    for descriptor in descriptors:
        state_index = descriptor.state_index
        previous_mode = updated[state_index]
        if not descriptor.enabled:
            updated[state_index] = DynamicLimitMode.FREE
            continue
        if descriptor.bound_scale is not None:
            updated[state_index] = DynamicLimitMode.FREE
            continue
        _validate_enabled_descriptor(
            descriptor,
            operation="explicit_mode_update",
            time=time,
            stage_or_endpoint=stage_or_endpoint,
        )
        value = float(state[state_index])
        derivative = float(raw_derivative[state_index])
        if not math.isfinite(value):
            raise _helper_failure(
                "explicit_mode_update",
                "non_finite_state",
                descriptor,
                value=value,
                time=time,
                stage_or_endpoint=stage_or_endpoint,
            )
        if not math.isfinite(derivative):
            raise _helper_failure(
                "explicit_mode_update",
                "non_finite_derivative",
                descriptor,
                value=derivative,
                time=time,
                stage_or_endpoint=stage_or_endpoint,
            )

        lower, upper, _ = evaluate_limited_state_bounds(state, descriptor)
        if value >= upper - tolerance and derivative > 0.0:
            next_mode = DynamicLimitMode.UPPER_ACTIVE
        elif value <= lower + tolerance and derivative < 0.0:
            next_mode = DynamicLimitMode.LOWER_ACTIVE
        elif derivative == 0.0 and (
            (
                previous_mode == DynamicLimitMode.UPPER_ACTIVE
                and value >= upper - tolerance
            )
            or (
                previous_mode == DynamicLimitMode.LOWER_ACTIVE
                and value <= lower + tolerance
            )
        ):
            next_mode = previous_mode
        else:
            next_mode = DynamicLimitMode.FREE

        if next_mode == previous_mode:
            continue
        if previous_mode != DynamicLimitMode.FREE:
            previous_side = (
                "upper"
                if previous_mode == DynamicLimitMode.UPPER_ACTIVE
                else "lower"
            )
            previous_bound = upper if previous_side == "upper" else lower
            events.append(
                _event_record(
                    descriptor,
                    side=previous_side,
                    action="release",
                    bound=previous_bound,
                    state_before=value,
                    state_after=value,
                    raw_derivative=derivative,
                    time=time,
                    stage_or_endpoint=stage_or_endpoint,
                )
            )
        if next_mode != DynamicLimitMode.FREE:
            next_side = (
                "upper"
                if next_mode == DynamicLimitMode.UPPER_ACTIVE
                else "lower"
            )
            next_bound = upper if next_side == "upper" else lower
            events.append(
                _event_record(
                    descriptor,
                    side=next_side,
                    action="activate",
                    bound=next_bound,
                    state_before=value,
                    state_after=value,
                    raw_derivative=derivative,
                    time=time,
                    stage_or_endpoint=stage_or_endpoint,
                )
            )
        updated[state_index] = next_mode

    changed = any(
        updated[descriptor.state_index] != _mode_for(modes, descriptor)
        for descriptor in descriptors
    )
    return updated, changed, events


def initialize_dynamic_limit_modes(
    descriptors: Iterable[LimitedStateDescriptor],
) -> dict[int, DynamicLimitMode]:
    """Create a free active-set mode for every discovered limited state."""
    modes = {}
    for descriptor in descriptors:
        if descriptor.state_index in modes:
            raise ValueError(
                f"Duplicate limited state index {descriptor.state_index}."
            )
        modes[descriptor.state_index] = DynamicLimitMode.FREE
    return modes


def _mode_for(
    modes: Mapping[int, DynamicLimitMode | str],
    descriptor: LimitedStateDescriptor,
) -> DynamicLimitMode:
    try:
        return DynamicLimitMode(modes[descriptor.state_index])
    except KeyError as exc:
        raise ValueError(
            f"Missing active-set mode for state {descriptor.state_index}."
        ) from exc
    except ValueError as exc:
        raise ValueError(
            f"Invalid active-set mode for state {descriptor.state_index}."
        ) from exc


def evaluate_dynamic_limit_complementarity(
    endpoint_state,
    free_residual,
    descriptors: Iterable[LimitedStateDescriptor],
    modes: Mapping[int, DynamicLimitMode | str],
    *,
    state_tolerance: float,
    release_tolerance: float,
) -> dict[str, Any]:
    """Evaluate bound feasibility and discarded-residual complementarity."""
    state_tolerance = float(state_tolerance)
    release_tolerance = float(release_tolerance)
    if state_tolerance < 0.0 or release_tolerance < 0.0:
        raise ValueError("Dynamic-limit tolerances must be non-negative.")
    endpoint_state = np.asarray(endpoint_state, dtype=float)
    free_residual = np.asarray(free_residual, dtype=float)
    records = []
    for descriptor in descriptors:
        mode = _mode_for(modes, descriptor)
        if not descriptor.enabled:
            records.append(
                {
                    **descriptor.to_dict(),
                    "mode": DynamicLimitMode.FREE.value,
                    "state_value": _json_number(
                        endpoint_state[descriptor.state_index]
                    ),
                    "free_residual": None,
                    "side": None,
                    "bound": None,
                    "consistent": mode == DynamicLimitMode.FREE,
                }
            )
            continue
        _validate_enabled_descriptor(
            descriptor, operation="complementarity_evaluation"
        )
        value = float(endpoint_state[descriptor.state_index])
        residual = float(free_residual[descriptor.state_index])
        if not math.isfinite(value):
            raise _helper_failure(
                "complementarity_evaluation",
                "non_finite_state",
                descriptor,
                value=value,
            )
        if not math.isfinite(residual):
            raise _helper_failure(
                "complementarity_evaluation",
                "non_finite_free_residual",
                descriptor,
                value=residual,
            )
        lower, upper, _ = evaluate_limited_state_bounds(endpoint_state, descriptor)
        side = None
        bound = None
        if mode == DynamicLimitMode.UPPER_ACTIVE:
            side = "upper"
            bound = upper
            consistent = (
                abs(value - upper) <= state_tolerance
                and residual <= release_tolerance
            )
        elif mode == DynamicLimitMode.LOWER_ACTIVE:
            side = "lower"
            bound = lower
            consistent = (
                abs(value - lower) <= state_tolerance
                and residual >= -release_tolerance
            )
        else:
            consistent = (
                lower - state_tolerance <= value <= upper + state_tolerance
            )
        records.append(
            {
                **descriptor.to_dict(),
                "mode": mode.value,
                "state_value": _json_number(value),
                "free_residual": _json_number(residual),
                "side": side,
                "bound": None if bound is None else _json_number(bound),
                "consistent": bool(consistent),
            }
        )
    return {
        "consistent": all(record["consistent"] for record in records),
        "records": records,
    }


def update_dynamic_limit_active_set(
    endpoint_state,
    free_residual,
    descriptors: Iterable[LimitedStateDescriptor],
    modes: Mapping[int, DynamicLimitMode | str],
    *,
    state_tolerance: float,
    release_tolerance: float,
    time: float | None = None,
    stage_or_endpoint: str | None = None,
    active_set_iterations: int | None = None,
) -> tuple[
    dict[int, DynamicLimitMode], bool, dict[str, Any], list[dict[str, Any]]
]:
    """Update implicit limiter modes from endpoint and free-row residuals."""
    descriptors = list(descriptors)
    endpoint_state = np.asarray(endpoint_state, dtype=float)
    free_residual = np.asarray(free_residual, dtype=float)
    complementarity = evaluate_dynamic_limit_complementarity(
        endpoint_state,
        free_residual,
        descriptors,
        modes,
        state_tolerance=state_tolerance,
        release_tolerance=release_tolerance,
    )
    updated = {
        descriptor.state_index: _mode_for(modes, descriptor)
        for descriptor in descriptors
    }
    events = []
    for descriptor in descriptors:
        state_index = descriptor.state_index
        mode = updated[state_index]
        if not descriptor.enabled:
            updated[state_index] = DynamicLimitMode.FREE
            continue
        value = float(endpoint_state[state_index])
        residual = float(free_residual[state_index])
        lower, upper, _ = evaluate_limited_state_bounds(endpoint_state, descriptor)
        side = None
        action = None
        bound = None
        next_mode = mode
        if mode == DynamicLimitMode.FREE:
            if value > upper + state_tolerance:
                side = "upper"
                action = "activate"
                bound = upper
                next_mode = DynamicLimitMode.UPPER_ACTIVE
            elif value < lower - state_tolerance:
                side = "lower"
                action = "activate"
                bound = lower
                next_mode = DynamicLimitMode.LOWER_ACTIVE
        elif (
            mode == DynamicLimitMode.UPPER_ACTIVE
            and residual > release_tolerance
        ):
            side = "upper"
            action = "release"
            bound = upper
            next_mode = DynamicLimitMode.FREE
        elif (
            mode == DynamicLimitMode.LOWER_ACTIVE
            and residual < -release_tolerance
        ):
            side = "lower"
            action = "release"
            bound = lower
            next_mode = DynamicLimitMode.FREE
        if action is not None:
            updated[state_index] = next_mode
            events.append(
                _event_record(
                    descriptor,
                    side=side,
                    action=action,
                    bound=bound,
                    state_before=value,
                    state_after=value,
                    free_residual=residual,
                    time=time,
                    stage_or_endpoint=stage_or_endpoint,
                    active_set_iterations=active_set_iterations,
                )
            )
    changed = any(
        updated[descriptor.state_index] != _mode_for(modes, descriptor)
        for descriptor in descriptors
    )
    return updated, changed, complementarity, events


def validate_initial_dynamic_limits(
    z0,
    descriptors: Iterable[LimitedStateDescriptor],
    *,
    enforce_dynamic_limits: bool,
    dynamic_limit_tolerance: float,
    dynamic_limit_release_tolerance: float,
    max_dynamic_limit_iterations: int,
) -> dict[str, Any]:
    """Validate enabled limited states without projecting the initial vector."""
    z0 = np.asarray(z0, dtype=float)
    descriptors = list(descriptors)
    enabled_descriptors = [item for item in descriptors if item.enabled]
    checked_descriptors = enabled_descriptors if enforce_dynamic_limits else []
    tolerance = float(dynamic_limit_tolerance)
    failure_reasons: list[str] = []
    violations: list[dict[str, Any]] = []

    def record_reason(reason: str) -> None:
        if reason not in failure_reasons:
            failure_reasons.append(reason)

    for descriptor in checked_descriptors:
        lower, upper, _ = evaluate_limited_state_bounds(z0, descriptor)
        value = float(z0[descriptor.state_index])

        if not math.isfinite(lower) or not math.isfinite(upper):
            record_reason("non_finite_bounds")
            violations.append(
                _violation_record(
                    descriptor,
                    value=value,
                    side="invalid",
                    bound=None,
                    violation=None,
                )
            )
            continue
        if lower == upper:
            record_reason("degenerate_bounds")
            violations.append(
                _violation_record(
                    descriptor,
                    value=value,
                    side="invalid",
                    bound=lower,
                    violation=0.0,
                )
            )
            continue
        if lower > upper:
            record_reason("inverted_bounds")
            violations.append(
                _violation_record(
                    descriptor,
                    value=value,
                    side="invalid",
                    bound=None,
                    violation=lower - upper,
                )
            )
            continue
        if not math.isfinite(value):
            record_reason("non_finite_initial_state")
            violations.append(
                _violation_record(
                    descriptor,
                    value=value,
                    side="invalid",
                    bound=None,
                    violation=None,
                )
            )
            continue
        if value < lower - tolerance:
            record_reason("initial_state_below_lower_bound")
            violations.append(
                _violation_record(
                    descriptor,
                    value=value,
                    side="lower",
                    bound=lower,
                    violation=lower - value,
                )
            )
        elif value > upper + tolerance:
            record_reason("initial_state_above_upper_bound")
            violations.append(
                _violation_record(
                    descriptor,
                    value=value,
                    side="upper",
                    bound=upper,
                    violation=value - upper,
                )
            )

    diagnostics = {
        "enabled": bool(enforce_dynamic_limits),
        "dynamic_limit_tolerance": tolerance,
        "dynamic_limit_release_tolerance": float(
            dynamic_limit_release_tolerance
        ),
        "max_dynamic_limit_iterations": int(max_dynamic_limit_iterations),
        "discovered_state_count": len(descriptors),
        "enabled_state_count": len(enabled_descriptors),
        "initialization": {
            "valid": not failure_reasons,
            "checked_state_count": len(checked_descriptors),
            "failure_reasons": failure_reasons,
            "violation_count": len(violations),
            "violations": violations,
        },
        "events": [],
    }
    if failure_reasons:
        raise DynamicLimitError(diagnostics)
    return diagnostics
