import numpy as np
import pytest

from uqgrid.simulation.timing import build_integration_schedule


def test_explicit_steps_include_initial_sample():
    schedule = build_integration_schedule(
        dt=0.01,
        tend=99.0,
        steps=3,
        ton=10.0,
        toff=11.0,
        has_fault=False,
    )

    np.testing.assert_allclose(schedule.times, [0.0, 0.01, 0.02, 0.03])
    assert schedule.fault_on_index is None
    assert schedule.fault_off_index is None


def test_schedule_includes_exact_horizon_and_off_grid_fault_times():
    schedule = build_integration_schedule(
        dt=0.01,
        tend=0.035,
        steps=-1,
        ton=0.015,
        toff=0.027,
        has_fault=True,
    )

    np.testing.assert_allclose(
        schedule.times,
        [0.0, 0.01, 0.015, 0.02, 0.027, 0.03, 0.035],
    )
    assert schedule.fault_on_index == 2
    assert schedule.fault_off_index == 4


def test_schedule_deduplicates_aligned_fault_times():
    schedule = build_integration_schedule(
        dt=0.01,
        tend=0.03,
        steps=-1,
        ton=0.01,
        toff=0.02,
        has_fault=True,
    )

    np.testing.assert_allclose(schedule.times, [0.0, 0.01, 0.02, 0.03])
    assert schedule.fault_on_index == 1
    assert schedule.fault_off_index == 2
    assert np.all(np.diff(schedule.times) > 0.0)


def test_fault_beyond_horizon_is_not_inserted():
    schedule = build_integration_schedule(
        dt=0.01,
        tend=0.03,
        steps=-1,
        ton=0.04,
        toff=0.05,
        has_fault=True,
    )

    np.testing.assert_allclose(schedule.times, [0.0, 0.01, 0.02, 0.03])
    assert schedule.fault_on_index is None
    assert schedule.fault_off_index is None


def test_fault_active_through_horizon_has_no_clear_index():
    schedule = build_integration_schedule(
        dt=0.01,
        tend=0.03,
        steps=-1,
        ton=0.015,
        toff=0.04,
        has_fault=True,
    )

    np.testing.assert_allclose(schedule.times, [0.0, 0.01, 0.015, 0.02, 0.03])
    assert schedule.fault_on_index == 2
    assert schedule.fault_off_index is None


def test_fault_clearing_at_horizon_uses_final_sample():
    schedule = build_integration_schedule(
        dt=0.01,
        tend=0.035,
        steps=-1,
        ton=0.015,
        toff=0.035,
        has_fault=True,
    )

    assert schedule.fault_off_index == len(schedule.times) - 1
    assert schedule.times[-1] == pytest.approx(0.035)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"dt": 0.0}, "dt must be positive"),
        ({"steps": 0}, "steps must be -1 or a positive integer"),
        ({"steps": -2}, "steps must be -1 or a positive integer"),
        ({"tend": -1.0}, "tend must be non-negative"),
        ({"ton": -1.0}, "ton must be non-negative"),
        ({"ton": 0.02, "toff": 0.01}, "toff must be"),
    ],
)
def test_schedule_rejects_invalid_controls(overrides, match):
    kwargs = {
        "dt": 0.01,
        "tend": 0.03,
        "steps": -1,
        "ton": 0.01,
        "toff": 0.02,
        "has_fault": False,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=match):
        build_integration_schedule(**kwargs)


def test_schedule_rejects_fault_at_initial_time():
    with pytest.raises(ValueError, match="ton=0"):
        build_integration_schedule(
            dt=0.01,
            tend=0.03,
            steps=-1,
            ton=0.0,
            toff=0.02,
            has_fault=True,
        )


def test_schedule_rejects_fault_time_that_deduplicates_to_zero():
    with pytest.raises(ValueError, match="ton=0"):
        build_integration_schedule(
            dt=0.01,
            tend=0.03,
            steps=-1,
            ton=1e-14,
            toff=0.02,
            has_fault=True,
        )
