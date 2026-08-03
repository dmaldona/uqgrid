import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _load_module():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(
        repo_root,
        "scripts",
        "run",
        "validate_acopf_init_handoff.py",
    )
    spec = importlib.util.spec_from_file_location(
        "validate_acopf_init_handoff",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validator():
    return _load_module()


class _FakeBus:
    def __init__(self):
        self.vinit = None

    def set_vinit(self, magnitude, angle):
        self.vinit = (magnitude, angle)


class _FakeSystem:
    def __init__(self):
        self.buses = [_FakeBus(), _FakeBus()]
        self.loads = []
        self.gens = []
        self.load_pq = None
        self.gen_pq = None
        self.ybus_calls = 0

    def set_load_pq(self, p, q):
        self.load_pq = (np.asarray(p), np.asarray(q))

    def set_gen_pq(self, p, q):
        self.gen_pq = (np.asarray(p), np.asarray(q))

    def createYbusComplex(self):
        self.ybus_calls += 1


def test_build_validation_integration_config_requires_strict_contract(validator):
    base = {
        "integration": {
            "dt": 1 / 120,
            "petsc": True,
            "enforce_q_limits": True,
            "q_limit_tolerance": 1e-8,
            "max_q_limit_iterations": None,
            "power_flow_validation": {"enabled": True, "voltage_min": 0.9},
            "enforce_dynamic_limits": True,
            "dynamic_limit_tolerance": 2e-8,
            "dynamic_limit_release_tolerance": 3e-10,
            "max_dynamic_limit_iterations": 17,
        }
    }

    class FakeIntegrationConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    result = validator.build_validation_integration_config(
        base,
        steps=5,
        petsc=False,
        integration_config_cls=FakeIntegrationConfig,
    )

    assert result.enforce_q_limits is True
    assert result.power_flow_validation == {"enabled": True, "voltage_min": 0.9}
    assert result.enforce_dynamic_limits is True
    assert result.dynamic_limit_tolerance == pytest.approx(2e-8)
    assert result.dynamic_limit_release_tolerance == pytest.approx(3e-10)
    assert result.max_dynamic_limit_iterations == 17
    assert result.steps == 5
    assert result.tend == pytest.approx(5 / 120)
    assert result.ton == pytest.approx(0.25)
    assert result.toff == pytest.approx(0.4)
    assert result.method == "beuler"
    assert result.petsc is False

    cn_result = validator.build_validation_integration_config(
        base,
        steps=5,
        petsc=True,
        method="cn",
        integration_config_cls=FakeIntegrationConfig,
    )
    assert cn_result.method == "cn"
    assert cn_result.petsc is True

    missing_q = {"integration": {**base["integration"], "enforce_q_limits": False}}
    with pytest.raises(ValueError, match="enforce_q_limits=true"):
        validator.build_validation_integration_config(
            missing_q,
            integration_config_cls=FakeIntegrationConfig,
        )

    missing_validation = {
        "integration": {
            **base["integration"],
            "power_flow_validation": {"enabled": False},
        }
    }
    with pytest.raises(ValueError, match="power_flow_validation.enabled=true"):
        validator.build_validation_integration_config(
            missing_validation,
            integration_config_cls=FakeIntegrationConfig,
        )


def test_handoff_parser_accepts_explicit_method_override(validator, tmp_path):
    args = validator._build_parser().parse_args(
        [
            "operating-point",
            str(tmp_path / "config.json"),
            "--source",
            "uqgrid_pf",
            "--output-dir",
            str(tmp_path / "out"),
            "--no-petsc",
            "--method",
            "herk2",
        ]
    )

    assert args.petsc is False
    assert args.method == "herk2"


def test_apply_direct_initialization_context(validator):
    psys = _FakeSystem()
    added = []
    context = {
        "raw_path": "model.raw",
        "dyr_path": "model.dyr",
        "operating_point": {
            "p_load_scaled": [0.1, 0.2],
            "q_load_scaled": [-0.03, -0.04],
            "p_gen_scaled": [0.35],
            "q_gen_scaled": [0.07],
            "pf_v_magnitudes": [1.01, 0.99],
            "pf_v_angles": [0.02, -0.03],
        },
    }

    result, summary = validator.apply_initialization_context(
        context,
        "uqgrid_pf",
        load_psse_func=lambda path: psys,
        add_dyr_func=lambda system, path: added.append((system, path)),
    )

    assert result is psys
    assert added == [(psys, "model.dyr")]
    np.testing.assert_allclose(psys.load_pq[0], [0.1, 0.2])
    np.testing.assert_allclose(psys.load_pq[1], [-0.03, -0.04])
    np.testing.assert_allclose(psys.gen_pq[0], [0.35])
    np.testing.assert_allclose(psys.gen_pq[1], [0.07])
    assert psys.buses[0].vinit == pytest.approx((1.01, 0.02))
    assert psys.buses[1].vinit == pytest.approx((0.99, -0.03))
    assert psys.ybus_calls == 1
    assert summary == {
        "num_loads": 2,
        "num_generators": 1,
        "voltage_initialization_applied": True,
    }


def test_apply_acopf_initialization_context(validator):
    psys = _FakeSystem()
    parsed = object()
    calls = []
    context = {
        "raw_path": "model.raw",
        "dyr_path": "model.dyr",
        "basecase_path": "Basecase_solution.txt",
        "case_raw_path": "case.raw",
    }

    result, summary = validator.apply_initialization_context(
        context,
        "acopf",
        load_psse_func=lambda path: psys,
        add_dyr_func=lambda system, path: calls.append(("dyr", path)),
        parse_basecase_func=lambda path, raw_path: (
            calls.append(("parse", path, raw_path)) or parsed
        ),
        apply_basecase_func=lambda system, solution: (
            calls.append(("apply", system, solution))
            or {"num_loads": 2, "num_generators": 1, "adjusted_shunts": 1}
        ),
    )

    assert result is psys
    assert calls == [
        ("dyr", "model.dyr"),
        ("parse", "Basecase_solution.txt", "case.raw"),
        ("apply", psys, parsed),
    ]
    assert summary["adjusted_shunts"] == 1
    assert psys.ybus_calls == 1


def test_collect_exciter_limit_diagnostics(validator):
    exciters = [
        SimpleNamespace(id="ok", dif_ptr=0, efd_idx=1, Emin=-1.0, Emax=5.0),
        SimpleNamespace(id="high", dif_ptr=2, efd_idx=1, Emin=-1.0, Emax=5.0),
    ]
    psys = SimpleNamespace(exc=exciters)
    state = np.asarray([0.0, 3.0, 0.0, 5.1])

    result = validator.collect_exciter_limit_diagnostics(psys, state)

    assert result["applicable"] is True
    assert result["count"] == 2
    assert result["violation_count"] == 1
    assert result["efd_min"] == pytest.approx(3.0)
    assert result["efd_max"] == pytest.approx(5.1)
    assert result["violations"][0]["id"] == "high"


def test_collect_exciter_limit_diagnostics_without_sexs_is_not_applicable(validator):
    result = validator.collect_exciter_limit_diagnostics(
        SimpleNamespace(exc=[]),
        np.asarray([0.0]),
    )

    assert result["applicable"] is False
    assert result["count"] == 0
    assert result["violation_count"] == 0
    assert result["efd_min"] is None
    assert result["efd_max"] is None


def test_compute_trajectory_drift_by_state_block(validator):
    history = np.asarray(
        [
            [1.0, 1.0, 1.1],
            [2.0, 2.2, 2.0],
            [3.0, 3.0, 3.3],
            [4.0, 4.4, 4.0],
        ]
    )
    psys = SimpleNamespace(num_dof_dif=1, num_dof_alg=1)

    result = validator.compute_trajectory_drift(history, psys)

    assert result["differential"] == pytest.approx(0.1)
    assert result["algebraic"] == pytest.approx(0.2)
    assert result["voltage"] == pytest.approx(0.4)
    assert result["total"] == pytest.approx(0.4)


def test_power_flow_contract_reports_explicit_bound_failures(validator):
    diagnostics = _valid_pf_diagnostics()
    diagnostics["residual_norm"] = 2e-8
    diagnostics["voltage_min"] = 0.89
    diagnostics["branch"]["loading_max"] = 1.00002

    failures = validator._power_flow_contract_failures(diagnostics)

    assert "pf_residual" in failures
    assert "voltage_low" in failures
    assert "branch_overload" in failures


def _valid_pf_diagnostics():
    return {
        "valid": True,
        "failure_reasons": [],
        "residual_norm": 1e-12,
        "residual_tolerance": 1e-8,
        "finite_voltage": True,
        "gen_p": {"violation_count": 0},
        "gen_q": {"violation_count": 0},
        "active_set": {"violation_count": 0},
        "island_slack": {"invalid_island_count": 0},
        "branch": {
            "loading_max": 1.000002,
            "loading_limit": 1.0,
            "limit_tolerance": 1e-5,
        },
        "voltage_min": 0.98,
        "voltage_max": 1.04,
        "voltage_lower_bound": 0.9,
        "voltage_upper_bound": 1.1,
    }


def _write_dataset(tmp_path, *, sources=("acopf", "uqgrid_pf")):
    row_count = len(sources)
    X = np.arange(row_count * 12, dtype=float).reshape(row_count, 2, 6) / 100
    Y_final = np.asarray(
        [[[10.0], [5.0]], [[3.0], [-2.0]]],
        dtype=float,
    )[:row_count]
    Y_min = Y_final - 1.0
    sample_idx = np.arange(row_count, dtype=np.int64)
    fault_locations = np.asarray([142, 143], dtype=np.int64)
    fault_impedances = np.asarray([1e-4], dtype=float)
    scenario_ids = np.empty(Y_final.shape, dtype=object)
    for row in range(row_count):
        for fault in range(fault_locations.size):
            scenario_ids[row, fault, 0] = f"scenario_{row}_{fault}_0"
    feasible = np.asarray([source == "acopf" for source in sources], dtype=bool)

    basename = "tsi_probml_fullinputs_TestGrid"
    common = {
        "X": X,
        "X_flat": X.reshape(row_count, -1),
        "sample_idx": sample_idx,
        "fault_locations": fault_locations,
        "fault_impedances": fault_impedances,
        "scenario_ids": scenario_ids,
        "initialization_source": np.asarray(sources, dtype=object),
        "acopf_feasible": feasible,
    }
    np.savez_compressed(
        tmp_path / f"{basename}_final.npz",
        **common,
        Y=Y_final,
        meta=np.asarray([{"tsi_mode": "final"}], dtype=object),
    )
    np.savez_compressed(
        tmp_path / f"{basename}_min.npz",
        **common,
        Y=Y_min,
        meta=np.asarray([{"tsi_mode": "min"}], dtype=object),
    )

    scenario_metadata = {}
    simulation_log = {}
    for scenario_id in scenario_ids.reshape(-1):
        row = int(str(scenario_id).split("_")[1])
        entry = {
            "initialization_source": sources[row],
            "acopf_feasible": bool(feasible[row]),
            "tsi_final": 1.0,
            "tsi_min": 0.5,
            "file": None,
        }
        scenario_metadata[str(scenario_id)] = dict(entry)
        simulation_log[str(scenario_id)] = {
            **entry,
            "power_flow_validation": _valid_pf_diagnostics(),
        }
    source_counts = {source: sources.count(source) for source in set(sources)}
    (tmp_path / "acopf_init_progress.json").write_text(
        json.dumps(
            {
                "accepted_count": row_count,
                "next_sample_idx": row_count,
                "initialization_source_counts": source_counts,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "scenario_metadata.json").write_text(
        json.dumps(scenario_metadata),
        encoding="utf-8",
    )
    (tmp_path / "simulation_log.json").write_text(
        json.dumps(simulation_log),
        encoding="utf-8",
    )
    (tmp_path / "state_metadata.json").write_text("{}", encoding="utf-8")
    return basename


def test_validate_dataset_accepts_consistent_mixed_output(validator, tmp_path):
    basename = _write_dataset(tmp_path)

    report = validator.validate_dataset(
        tmp_path,
        basename,
        expected_sources=["acopf", "uqgrid_pf"],
        expected_fault_count=2,
    )

    assert report["valid"] is True
    assert report["shapes"] == {
        "X": (2, 2, 6),
        "X_flat": (2, 12),
        "Y_final": (2, 2, 1),
        "Y_min": (2, 2, 1),
    }
    assert report["source_counts"] == {"acopf": 1, "uqgrid_pf": 1}
    written = json.loads((tmp_path / "stage5_dataset_validation.json").read_text())
    assert written["valid"] is True


def test_validate_dataset_reports_npz_and_label_mismatches(validator, tmp_path):
    basename = _write_dataset(tmp_path)
    final_path = tmp_path / f"{basename}_final.npz"
    min_path = tmp_path / f"{basename}_min.npz"
    with np.load(final_path, allow_pickle=True) as data:
        final = {key: data[key] for key in data.files}
    with np.load(min_path, allow_pickle=True) as data:
        minimum = {key: data[key] for key in data.files}
    final["X_flat"] = final["X_flat"].copy()
    final["X_flat"][0, 0] += 1.0
    minimum["Y"] = final["Y"] + 1.0
    minimum["initialization_source"] = np.asarray(
        ["uqgrid_pf", "uqgrid_pf"],
        dtype=object,
    )
    np.savez_compressed(final_path, **final)
    np.savez_compressed(min_path, **minimum)

    report = validator.validate_dataset(
        tmp_path,
        basename,
        expected_sources=["acopf", "uqgrid_pf"],
    )

    assert report["valid"] is False
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "Final/min optional labels" in failed
    assert "X_flat layout" in failed
    assert "TSI final/min relationship" in failed


def test_validate_dataset_requires_pf_diagnostics_and_matching_counts(
    validator,
    tmp_path,
):
    basename = _write_dataset(tmp_path)
    log_path = tmp_path / "simulation_log.json"
    simulation_log = json.loads(log_path.read_text())
    first_id = next(iter(simulation_log))
    simulation_log[first_id]["power_flow_validation"]["gen_q"]["violation_count"] = 1
    log_path.write_text(json.dumps(simulation_log), encoding="utf-8")
    progress_path = tmp_path / "acopf_init_progress.json"
    progress = json.loads(progress_path.read_text())
    progress["accepted_count"] = 1
    progress_path.write_text(json.dumps(progress), encoding="utf-8")

    report = validator.validate_dataset(tmp_path, basename)

    assert report["valid"] is False
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "NPZ/restart row counts" in failed
    assert "Per-fault final PF diagnostics" in failed


def test_dataset_cli_prints_json_and_returns_nonzero(validator, tmp_path, capsys):
    exit_code = validator.main(
        [
            "dataset",
            "--output-dir",
            str(tmp_path),
            "--probml-basename",
            "missing",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert json.loads(captured.out)["valid"] is False
    assert "✗ Final NPZ exists" in captured.err
