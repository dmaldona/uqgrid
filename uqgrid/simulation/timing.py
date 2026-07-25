"""Shared time-grid construction for dynamic integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IntegrationSchedule:
    """Stored times and fault-transition indices for one integration run."""

    times: np.ndarray
    fault_on_index: int | None
    fault_off_index: int | None


def _append_distinct(times: list[float], value: float, tolerance: float) -> None:
    if not times or abs(value - times[-1]) > tolerance:
        times.append(value)
    else:
        times[-1] = value


def build_integration_schedule(
    *,
    dt: float,
    tend: float,
    steps: int,
    ton: float,
    toff: float,
    has_fault: bool,
) -> IntegrationSchedule:
    """Build the common stored-time grid, including exact fault boundaries."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if tend < 0.0:
        raise ValueError("tend must be non-negative")
    if steps == 0 or steps < -1:
        raise ValueError("steps must be -1 or a positive integer")
    if ton < 0.0:
        raise ValueError("ton must be non-negative")
    if toff < ton:
        raise ValueError("toff must be greater than or equal to ton")
    horizon = steps * dt if steps > 0 else tend
    tolerance = 1e-12 * max(1.0, horizon, dt)
    if has_fault and ton <= tolerance:
        raise ValueError(
            "ton=0 is incompatible with storing one initialized sample at t=0"
        )

    n_full_steps = int(np.floor((horizon + tolerance) / dt))
    base_times = [index * dt for index in range(n_full_steps + 1)]
    if base_times[-1] < horizon - tolerance:
        base_times.append(horizon)
    else:
        base_times[-1] = horizon

    candidates = list(base_times)
    if has_fault:
        for event_time in (ton, toff):
            if -tolerance <= event_time <= horizon + tolerance:
                candidates.append(min(max(event_time, 0.0), horizon))

    times: list[float] = []
    for value in sorted(candidates):
        _append_distinct(times, float(value), tolerance)

    values = np.asarray(times, dtype=float)

    def event_index(event_time: float) -> int | None:
        if not has_fault or event_time < -tolerance or event_time > horizon + tolerance:
            return None
        matches = np.flatnonzero(np.abs(values - event_time) <= tolerance)
        return int(matches[0]) if matches.size else None

    return IntegrationSchedule(
        times=values,
        fault_on_index=event_index(ton),
        fault_off_index=event_index(toff),
    )
