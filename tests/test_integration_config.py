import pytest

from uqgrid.simulation.config import IntegrationConfig


def test_integration_config_accepts_slow_partition():
    cfg = IntegrationConfig(arkimex=True, arkimex_slow_differential=[0, 2, 4])
    assert cfg.arkimex_slow_differential == [0, 2, 4]
    assert cfg.arkimex_fast_differential is None


def test_integration_config_accepts_fast_partition():
    cfg = IntegrationConfig(arkimex=True, arkimex_fast_differential=[1, 3])
    assert cfg.arkimex_fast_differential == [1, 3]
    assert cfg.arkimex_slow_differential is None


def test_integration_config_rejects_both_fast_and_slow_lists():
    with pytest.raises(ValueError):
        IntegrationConfig(
            arkimex=True,
            arkimex_fast_differential=[1, 3],
            arkimex_slow_differential=[0, 2],
        )
