import types

import pytest

from uqgrid.simulation.dynamics import generate_default_partition_indices


@pytest.fixture
def simple_psys():
    psys = types.SimpleNamespace()
    psys.num_dof_dif = 5
    psys.num_dof_alg = 3
    psys.nbuses = 1
    return psys


def test_custom_slow_partition(simple_psys):
    slow = [0, 2, 4]
    slow_indices, fast_indices, fast_indices_alg, fast_indices_dif, ndiff_fast = generate_default_partition_indices(
        simple_psys,
        slow_diff_indices=slow,
    )

    expected_fast_alg = list(range(simple_psys.num_dof_dif, simple_psys.num_dof_dif + simple_psys.num_dof_alg + 2 * simple_psys.nbuses))

    assert slow_indices == sorted(slow)
    assert fast_indices_dif == [1, 3]
    assert fast_indices_alg == expected_fast_alg
    assert fast_indices == fast_indices_dif + expected_fast_alg
    assert ndiff_fast == len(fast_indices_dif)


def test_custom_slow_partition_validation(simple_psys):
    with pytest.raises(ValueError):
        generate_default_partition_indices(simple_psys, slow_diff_indices=[0, 5])

    with pytest.raises(ValueError):
        generate_default_partition_indices(simple_psys, slow_diff_indices=[0, 0, 1])


def test_custom_fast_partition(simple_psys):
    fast = [1, 3]
    slow_indices, fast_indices, fast_indices_alg, fast_indices_dif, ndiff_fast = generate_default_partition_indices(
        simple_psys,
        fast_diff_indices=fast,
    )

    expected_fast_alg = list(range(simple_psys.num_dof_dif, simple_psys.num_dof_dif + simple_psys.num_dof_alg + 2 * simple_psys.nbuses))

    assert fast_indices_dif == sorted(fast)
    assert slow_indices == [0, 2, 4]
    assert fast_indices_alg == expected_fast_alg
    assert fast_indices == fast_indices_dif + expected_fast_alg
    assert ndiff_fast == len(fast_indices_dif)


def test_conflicting_partition_inputs(simple_psys):
    with pytest.raises(ValueError):
        generate_default_partition_indices(
            simple_psys,
            slow_diff_indices=[0, 1],
            fast_diff_indices=[2, 3],
        )
