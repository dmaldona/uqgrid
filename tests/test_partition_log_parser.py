import random

from bin.arkimex_partition_experiments import (
    extract_runtime_from_log,
    format_index_list,
    sample_slow_indices,
)


def test_extract_runtime_from_log(tmp_path):
    log_text = """
PETSc Performance Summary
Time (sec):           1.234e+00     1.000   1.234e+00
"""
    log_file = tmp_path / "log.txt"
    log_file.write_text(log_text)
    runtime = extract_runtime_from_log(log_file)
    assert runtime is not None
    assert abs(runtime - 1.234) < 1e-6


def test_extract_runtime_missing(tmp_path):
    log_file = tmp_path / "empty.txt"
    log_file.write_text("No timing info here\n")
    assert extract_runtime_from_log(log_file) is None


def test_sample_slow_indices_bounds():
    random.seed(42)
    total = 50
    indices = sample_slow_indices(total, min_frac=0.2, max_frac=0.4)
    assert 0 <= len(indices) <= total
    assert len(indices) == len(set(indices))
    assert all(0 <= idx < total for idx in indices)


def test_format_index_list():
    assert format_index_list([]) == ""
    assert format_index_list([0, 3, 5]) == "0,3,5"
