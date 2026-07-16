import importlib.util
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from uqgrid.io.parse import load_psse


def _load_generate_scenarios_module():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(repo_root, "scripts", "run", "generate_scenarios.py")
    spec = importlib.util.spec_from_file_location("generate_scenarios", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gs():
    return _load_generate_scenarios_module()


@pytest.fixture
def data_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )


def _dummy_q_limit_psys(gs):
    return SimpleNamespace(
        buses=[
            SimpleNamespace(id=10, type=gs.Bus.SLACK),
            SimpleNamespace(id=20, type=gs.Bus.PV),
            SimpleNamespace(id=30, type=gs.Bus.PQ),
        ],
        gens=[
            SimpleNamespace(bus=0, idx="1"),
            SimpleNamespace(bus=1, idx="A"),
            SimpleNamespace(bus=2, idx="B"),
        ],
    )


def _dummy_export_psys():
    return SimpleNamespace(export_state_metadata=lambda: None)


def test_integration_config_adapter_preserves_q_limit_controls(gs):
    cfg = gs._integration_config_from_dict({
        "petsc": True,
        "enforce_q_limits": True,
        "q_limit_tolerance": 2e-7,
        "max_q_limit_iterations": 11,
        "power_flow_validation": {
            "enabled": True,
            "voltage_min": 0.9,
            "voltage_max": 1.1,
            "branch_loading_max": 1.0,
        },
    })

    assert cfg.petsc is True
    assert cfg.enforce_q_limits is True
    assert cfg.q_limit_tolerance == pytest.approx(2e-7)
    assert cfg.max_q_limit_iterations == 11
    assert cfg.power_flow_validation.enabled is True
    assert cfg.power_flow_validation.voltage_min == pytest.approx(0.9)
    assert cfg.power_flow_validation.voltage_max == pytest.approx(1.1)
    assert cfg.power_flow_validation.branch_loading_max == pytest.approx(1.0)


def test_default_scenario_config_includes_q_limit_controls(gs):
    integration = gs.get_default_config("IEEE-9")["integration"]

    assert integration["enforce_q_limits"] is False
    assert integration["q_limit_tolerance"] == pytest.approx(1e-8)
    assert integration["max_q_limit_iterations"] is None
    validation = integration["power_flow_validation"]
    assert validation["enabled"] is False
    assert validation["residual_tolerance"] == pytest.approx(1e-8)
    assert validation["generator_limit_tolerance"] == pytest.approx(1e-6)
    assert validation["voltage_min"] == pytest.approx(0.9)
    assert validation["voltage_max"] == pytest.approx(1.1)
    assert validation["branch_loading_max"] == pytest.approx(1.0)


def _fault_worker_inputs():
    scenario = {
        "sample_idx": 0,
        "fault_location": 1,
        "fault_impedance": 1e-4,
        "operating_point_id": "op-0",
        "accepted_operating_point_index": 0,
    }
    operating_point = {
        "operating_point_id": "op-0",
        "accepted_operating_point_index": 0,
        "p_load_scaled": np.array([0.5]),
        "q_load_scaled": np.array([-0.2]),
        "p_gen_scaled": np.array([0.5]),
        "q_gen_scaled": np.array([0.2]),
        "p_load_noise": np.array([0.0]),
        "q_load_noise": np.array([0.0]),
        "p_gen_noise": np.array([0.0]),
        "q_gen_noise": np.array([0.0]),
        "load_scale": 1.0,
        "load_mean_shift": 0.0,
        "diagnostics": {},
    }
    return scenario, operating_point


def _fake_fault_psys():
    class BusStub:
        def set_vinit(self, vmag, vang):
            self.vmag = vmag
            self.vang = vang

    return SimpleNamespace(
        buses=[BusStub(), BusStub()],
        set_load_pq=lambda *args: None,
        set_gen_pq=lambda *args: None,
        add_busfault=lambda *args: None,
        createYbusComplex=lambda: None,
    )


def test_fault_worker_preserves_successful_pf_validation(
        gs, tmp_path, monkeypatch):
    scenario, operating_point = _fault_worker_inputs()
    validation = {"valid": True, "failure_reasons": []}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gs, "_load_power_system", lambda *args: _fake_fault_psys())
    monkeypatch.setattr(
        gs,
        "integrate_system",
        lambda *args: {
            "history": np.zeros((2, 2)),
            "tvec": np.array([0.0, 0.1]),
            "power_flow_diagnostics": validation,
        },
    )

    result = gs._run_fault_with_operating_point_worker(
        "case.raw",
        "case.dyr",
        scenario,
        "scenario-0",
        operating_point,
        {"power_flow_validation": {"enabled": True}},
    )

    assert result["diverged"] is False
    assert result["diagnostics"]["power_flow_validation"] == validation


def test_fault_worker_preserves_failed_pf_validation(
        gs, tmp_path, monkeypatch):
    scenario, operating_point = _fault_worker_inputs()
    validation = {"valid": False, "failure_reasons": ["gen_p_limit"]}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gs, "_load_power_system", lambda *args: _fake_fault_psys())

    def fail_validation(*args):
        raise gs.PowerFlowValidationError(validation)

    monkeypatch.setattr(gs, "integrate_system", fail_validation)

    result = gs._run_fault_with_operating_point_worker(
        "case.raw",
        "case.dyr",
        scenario,
        "scenario-0",
        operating_point,
        {"power_flow_validation": {"enabled": True}},
    )

    assert result["diverged"] is True
    assert result["diagnostics"]["reject_reason"] == "power_flow_validation_failed"
    assert result["diagnostics"]["power_flow_validation"] == validation


def test_stress_load_preserves_load_power_factor_with_no_noise(gs):
    base_p = np.array([1.0, 2.0, 0.0])
    base_q = np.array([0.4, 0.8, 0.5])

    stressed_p, stressed_q, load_factor = gs._stress_load(
        base_p, base_q, load_scale=1.2, load_mean_shift=0.1
    )
    p_scaled, q_scaled = gs.generate_perturbations(
        stressed_p,
        stressed_q,
        noise_type="none",
        preserve_power_factor=True,
    )

    assert load_factor == pytest.approx(1.32)
    np.testing.assert_allclose(p_scaled[:2] / base_p[:2], [1.32, 1.32])
    np.testing.assert_allclose(q_scaled[:2] / base_q[:2], [1.32, 1.32])
    assert q_scaled[2] == pytest.approx(base_q[2] * 1.32)


def test_rebalance_active_power_can_exclude_slack_generator(gs):
    p_gen = np.array([1.0, 2.0, 3.0])
    pg_lb = np.array([0.0, 0.0, 0.0])
    pg_ub = np.array([10.0, 10.0, 10.0])
    non_slack_mask = np.array([False, True, True])

    rebalanced = gs._rebalance_active_power(
        p_gen,
        pg_lb,
        pg_ub,
        target_total=np.sum(p_gen) + 3.0,
        participation_mask=non_slack_mask,
    )

    assert rebalanced[0] == pytest.approx(p_gen[0])
    assert np.sum(rebalanced) == pytest.approx(np.sum(p_gen) + 3.0)
    assert rebalanced[1] > p_gen[1]
    assert rebalanced[2] > p_gen[2]


def test_redistribute_slack_mismatch_respects_participation_and_limits(gs):
    p_gen = np.array([5.0, 2.0, 2.0])
    pg_lb = np.array([0.0, 1.5, 1.0])
    pg_ub = np.array([10.0, 10.0, 10.0])
    non_slack_mask = np.array([False, True, True])

    redistributed = gs._redistribute_active_power_mismatch(
        p_gen,
        pg_lb,
        pg_ub,
        mismatch=-2.0,
        participation_mask=non_slack_mask,
    )

    assert redistributed[0] == pytest.approx(p_gen[0])
    assert redistributed[1] >= pg_lb[1]
    assert redistributed[2] >= pg_lb[2]
    assert np.sum(p_gen) - np.sum(redistributed) == pytest.approx(1.5)


@pytest.mark.parametrize(
    "policy",
    ["headroom", "capacity", "current_dispatch", "base_dispatch", "equal"],
)
def test_active_power_allocator_policies_respect_limits(gs, policy):
    p_gen = np.array([5.0, 2.0, 4.0, 1.0])
    base_p_gen = np.array([5.0, 3.0, 1.0, 1.0])
    pg_lb = np.array([0.0, 1.0, 3.0, 0.0])
    pg_ub = np.array([8.0, 6.0, 9.0, 2.0])
    non_slack_mask = np.array([False, True, True, True])

    allocation = gs._redistribute_active_power_mismatch(
        p_gen,
        pg_lb,
        pg_ub,
        mismatch=3.0,
        participation_mask=non_slack_mask,
        policy=policy,
        base_p_gen=base_p_gen,
        return_diagnostics=True,
    )

    redistributed = allocation["p_gen"]
    assert redistributed[0] == pytest.approx(p_gen[0])
    assert np.all(redistributed >= pg_lb)
    assert np.all(redistributed <= pg_ub)
    assert allocation["applied_mismatch"] == pytest.approx(3.0)
    assert allocation["unresolved_mismatch"] == pytest.approx(0.0)
    assert allocation["participants"] > 0


def test_equal_policy_clips_and_redistributes_remaining_mismatch(gs):
    p_gen = np.array([0.0, 0.0])
    pg_lb = np.array([0.0, 0.0])
    pg_ub = np.array([1.0, 10.0])

    allocation = gs._redistribute_active_power_mismatch(
        p_gen,
        pg_lb,
        pg_ub,
        mismatch=4.0,
        policy="equal",
        return_diagnostics=True,
    )

    np.testing.assert_allclose(allocation["p_gen"], [1.0, 3.0])
    assert allocation["applied_mismatch"] == pytest.approx(4.0)
    assert allocation["unresolved_mismatch"] == pytest.approx(0.0)


def test_bounded_allocation_reports_unresolved_mismatch(gs):
    p_gen = np.array([0.0, 0.0])
    pg_lb = np.array([0.0, 0.0])
    pg_ub = np.array([1.0, 1.0])

    allocation = gs._redistribute_active_power_mismatch(
        p_gen,
        pg_lb,
        pg_ub,
        mismatch=3.0,
        policy="headroom",
        return_diagnostics=True,
    )

    np.testing.assert_allclose(allocation["p_gen"], [1.0, 1.0])
    assert allocation["applied_mismatch"] == pytest.approx(2.0)
    assert allocation["unresolved_mismatch"] == pytest.approx(1.0)
    assert allocation["participants"] == 2


def test_loss_compensation_can_remove_slack_p_limit_burden(gs):
    op_cfg = gs._resolve_operating_point_config(
        {
            "enabled": True,
            "loss_compensation": True,
            "max_slack_p_deviation_fraction_of_load": 0.02,
            "gen_limit_tolerance": 1e-6,
        }
    )
    diagnostics = {
        "pf_converged": True,
        "pf_residual": 1e-10,
        "slack_p_deviation": 0.735,
        "voltage_min": 1.0,
        "voltage_max": 1.04,
        "gen_p_violation_max": 0.735,
        "gen_q_violation_max": 0.0,
        "branch_loading_available": False,
        "branch_loading_max": None,
    }
    accepted, reason = gs._screen_power_flow_diagnostics(
        diagnostics, total_load_p=70.0, op_cfg=op_cfg
    )
    assert not accepted
    assert reason == "gen_p_limit"

    p_gen = np.array([8.889, 2.0, 3.0])
    pg_lb = np.array([0.0, 0.0, 0.0])
    pg_ub = np.array([8.889, 10.0, 10.0])
    non_slack_mask = np.array([False, True, True])

    allocation = gs._redistribute_active_power_mismatch(
        p_gen,
        pg_lb,
        pg_ub,
        mismatch=diagnostics["slack_p_deviation"],
        participation_mask=non_slack_mask,
        policy=op_cfg["loss_compensation_policy"],
        return_diagnostics=True,
    )

    assert allocation["p_gen"][0] == pytest.approx(p_gen[0])
    assert allocation["applied_mismatch"] == pytest.approx(0.735)
    assert allocation["unresolved_mismatch"] == pytest.approx(0.0)
    slack_p_solved_after = p_gen[0] + 0.735 - allocation["applied_mismatch"]
    assert slack_p_solved_after <= pg_ub[0] + 1e-9


def test_generator_q_limit_diagnostics_reports_upper_and_lower_violations(gs):
    psys = _dummy_q_limit_psys(gs)
    q_values = np.array([1.25, -0.85, 0.0])
    qg_lb = np.array([-1.0, -0.5, -0.1])
    qg_ub = np.array([1.0, 0.5, 0.1])

    diagnostics = gs._generator_q_limit_diagnostics(psys, q_values, qg_lb, qg_ub)

    assert diagnostics["gen_q_violation_count"] == 2
    assert diagnostics["gen_q_violation_total_abs"] == pytest.approx(0.60)
    assert diagnostics["gen_q_violation_argmax"] == 1
    assert len(diagnostics["gen_q_violation_top"]) == 2

    worst = diagnostics["gen_q_violation_top"][0]
    assert worst["gen_index"] == 1
    assert worst["gen_id"] == "A"
    assert worst["bus_index"] == 1
    assert worst["bus_id"] == 20
    assert worst["bus_type"] == "PV"
    assert worst["qg"] == pytest.approx(-0.85)
    assert worst["qmin"] == pytest.approx(-0.5)
    assert worst["qmax"] == pytest.approx(0.5)
    assert worst["violation"] == pytest.approx(0.35)
    assert worst["side"] == "lower"
    assert not worst["is_slack"]

    second = diagnostics["gen_q_violation_top"][1]
    assert second["gen_index"] == 0
    assert second["side"] == "upper"
    assert second["is_slack"]


def test_generator_q_limit_diagnostics_reports_no_violations(gs):
    psys = _dummy_q_limit_psys(gs)
    q_values = np.array([0.0, -0.25, 0.05])
    qg_lb = np.array([-1.0, -0.5, -0.1])
    qg_ub = np.array([1.0, 0.5, 0.1])

    diagnostics = gs._generator_q_limit_diagnostics(psys, q_values, qg_lb, qg_ub)

    assert diagnostics["gen_q_violation_count"] == 0
    assert diagnostics["gen_q_violation_total_abs"] == pytest.approx(0.0)
    assert diagnostics["gen_q_violation_argmax"] is None
    assert diagnostics["gen_q_violation_top"] == []


def test_q_limit_summary_rolls_up_top_offenders(gs):
    psys = _dummy_q_limit_psys(gs)
    diagnostics = gs._generator_q_limit_diagnostics(
        psys,
        np.array([1.25, -0.85, 0.0]),
        np.array([-1.0, -0.5, -0.1]),
        np.array([1.0, 0.5, 0.1]),
    )

    summary = gs._summarize_diagnostics([
        {
            "accepted": False,
            "reject_reason": "gen_q_limit",
            "gen_q_violation_max": 0.35,
            "gen_q_violation_count": diagnostics["gen_q_violation_count"],
            "gen_q_violation_total_abs": diagnostics["gen_q_violation_total_abs"],
            "gen_q_violation_top": diagnostics["gen_q_violation_top"],
        },
        {
            "accepted": False,
            "reject_reason": "gen_q_limit",
            "gen_q_violation_max": 0.35,
            "gen_q_violation_count": diagnostics["gen_q_violation_count"],
            "gen_q_violation_total_abs": diagnostics["gen_q_violation_total_abs"],
            "gen_q_violation_top": diagnostics["gen_q_violation_top"],
        },
    ])

    top_offenders = summary["gen_q_violation_top_offenders"]
    assert top_offenders[0]["bus_id"] == 20
    assert top_offenders[0]["gen_id"] == "A"
    assert top_offenders[0]["count"] == 2
    assert top_offenders[0]["max_violation"] == pytest.approx(0.35)
    assert top_offenders[0]["mean_abs_violation"] == pytest.approx(0.35)


def test_q_limit_mitigation_upper_switches_pv_bus_and_clamps_qmax(gs):
    psys = _dummy_q_limit_psys(gs)
    op_cfg = gs._resolve_operating_point_config(
        {"enabled": True, "q_limit_mitigation": True}
    )
    q_gen = np.array([0.0, 0.0, 0.0])
    q_values = np.array([0.0, 0.85, 0.0])
    qg_lb = np.array([-1.0, -0.5, -0.1])
    qg_ub = np.array([1.0, 0.5, 0.1])
    q_fixed_mask = np.zeros_like(q_gen, dtype=bool)
    q_fixed_values = q_gen.copy()

    q_new, result = gs._apply_q_limit_mitigation(
        psys,
        q_gen,
        q_values,
        qg_lb,
        qg_ub,
        op_cfg,
        q_fixed_mask=q_fixed_mask,
        q_fixed_values=q_fixed_values,
        pass_idx=1,
    )

    assert result["applied"]
    assert psys.buses[1].type == gs.Bus.PQ
    assert q_new[1] == pytest.approx(qg_ub[1])
    assert q_fixed_mask[1]
    assert q_fixed_values[1] == pytest.approx(qg_ub[1])
    assert result["events"][0]["side"] == "upper"


def test_q_limit_mitigation_lower_switches_pv_bus_and_clamps_qmin(gs):
    psys = _dummy_q_limit_psys(gs)
    op_cfg = gs._resolve_operating_point_config(
        {"enabled": True, "q_limit_mitigation": True}
    )
    q_gen = np.array([0.0, 0.0, 0.0])
    q_values = np.array([0.0, -0.85, 0.0])
    qg_lb = np.array([-1.0, -0.5, -0.1])
    qg_ub = np.array([1.0, 0.5, 0.1])

    q_new, result = gs._apply_q_limit_mitigation(
        psys, q_gen, q_values, qg_lb, qg_ub, op_cfg
    )

    assert result["applied"]
    assert psys.buses[1].type == gs.Bus.PQ
    assert q_new[1] == pytest.approx(qg_lb[1])
    assert result["events"][0]["side"] == "lower"


def test_q_limit_mitigation_does_not_switch_slack_generator(gs):
    psys = _dummy_q_limit_psys(gs)
    op_cfg = gs._resolve_operating_point_config(
        {"enabled": True, "q_limit_mitigation": True}
    )
    q_gen = np.array([0.0, 0.0, 0.0])
    q_values = np.array([1.25, 0.0, 0.0])
    qg_lb = np.array([-1.0, -0.5, -0.1])
    qg_ub = np.array([1.0, 0.5, 0.1])

    q_new, result = gs._apply_q_limit_mitigation(
        psys, q_gen, q_values, qg_lb, qg_ub, op_cfg
    )

    assert not result["applied"]
    assert psys.buses[0].type == gs.Bus.SLACK
    np.testing.assert_allclose(q_new, q_gen)


def test_q_limit_mitigation_can_restore_and_repeat_bus_type_changes(gs):
    psys = _dummy_q_limit_psys(gs)
    op_cfg = gs._resolve_operating_point_config(
        {"enabled": True, "q_limit_mitigation": True}
    )
    original_bus_types = gs._capture_bus_types(psys)
    q_gen = np.array([0.0, 0.0, 0.0])
    q_values = np.array([0.0, 0.85, 0.0])
    qg_lb = np.array([-1.0, -0.5, -0.1])
    qg_ub = np.array([1.0, 0.5, 0.1])

    _, first = gs._apply_q_limit_mitigation(
        psys, q_gen, q_values, qg_lb, qg_ub, op_cfg
    )
    gs._restore_bus_types(psys, original_bus_types)
    _, second = gs._apply_q_limit_mitigation(
        psys, q_gen, q_values, qg_lb, qg_ub, op_cfg
    )

    assert first["applied"]
    assert second["applied"]
    assert psys.buses[1].type == gs.Bus.PQ
    gs._restore_bus_types(psys, original_bus_types)
    assert [bus.type for bus in psys.buses] == original_bus_types


def test_q_limit_mitigation_disabled_does_not_apply(gs):
    psys = _dummy_q_limit_psys(gs)
    op_cfg = gs._resolve_operating_point_config(
        {"enabled": True, "q_limit_mitigation": False}
    )
    q_gen = np.array([0.0, 0.0, 0.0])
    q_values = np.array([0.0, 0.85, 0.0])
    qg_lb = np.array([-1.0, -0.5, -0.1])
    qg_ub = np.array([1.0, 0.5, 0.1])

    q_new, result = gs._apply_q_limit_mitigation(
        psys, q_gen, q_values, qg_lb, qg_ub, op_cfg
    )

    assert not result["applied"]
    assert psys.buses[1].type == gs.Bus.PV
    np.testing.assert_allclose(q_new, q_gen)


def test_target_fault_metadata_covers_complete_fault_grid(gs):
    metadata = gs._build_target_fault_metadata(
        7,
        [0, 2],
        [1e-4, 2e-4],
        operating_point_id="op-1",
        accepted_operating_point_index=3,
    )

    assert len(metadata) == 4
    faults = {
        (entry["fault_location"], entry["fault_impedance"])
        for entry in metadata.values()
    }
    assert faults == {(0, 1e-4), (0, 2e-4), (2, 1e-4), (2, 2e-4)}
    assert {entry["sample_idx"] for entry in metadata.values()} == {7}
    assert {entry["operating_point_id"] for entry in metadata.values()} == {"op-1"}
    assert {
        entry["accepted_operating_point_index"] for entry in metadata.values()
    } == {3}


def test_target_mode_replaces_pf_rejected_candidates(gs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gs, "_load_power_system", lambda *_: _dummy_export_psys())

    seen_samples = []

    def fake_prepare(*args, **kwargs):
        scenario = args[2]
        seen_samples.append(scenario["sample_idx"])
        if scenario["sample_idx"] == 0:
            diagnostics = {
                "record_type": "operating_point_attempt",
                "accepted": False,
                "attempts": 1,
                "reject_reason": "pf_non_converged",
            }
            return {
                "rejected": True,
                "diverged": True,
                "diagnostics": diagnostics,
                "diagnostics_attempts": [diagnostics],
            }
        diagnostics = {
            "record_type": "operating_point_attempt",
            "accepted": True,
            "attempts": 1,
            "pf_residual": 1e-12,
        }
        return {
            "rejected": False,
            "diverged": False,
            "diagnostics": diagnostics,
            "diagnostics_attempts": [diagnostics],
            "operating_point": {
                "diagnostics": {"accepted": True, "attempts": 1},
                "operating_point_id": scenario["operating_point_id"],
                "sample_idx": scenario["sample_idx"],
                "accepted_operating_point_index": scenario[
                    "accepted_operating_point_index"
                ],
            },
        }

    def fake_fault_worker(raw, dyr, scenario, scenario_id, operating_point, integration):
        os.makedirs("simulation_data", exist_ok=True)
        path = gs._simulation_file_path(scenario_id)
        with open(path, "w") as f:
            f.write("ok")
        return {
            "file": path,
            "diverged": False,
            "rejected": False,
            "diagnostics": {
                "accepted": True,
                "sample_idx": scenario["sample_idx"],
                "operating_point_id": scenario["operating_point_id"],
            },
        }

    monkeypatch.setattr(gs, "_prepare_operating_point_candidate", fake_prepare)
    monkeypatch.setattr(gs, "_run_fault_with_operating_point_worker", fake_fault_worker)

    log = gs._run_target_accepted_driver(
        "case.raw",
        "case.dyr",
        target_accepted_scenarios=1,
        max_total_attempts=5,
        sample_idx_start=0,
        fault_locations=[10, 20],
        fault_impedances=[1e-4],
        n_jobs=1,
        operating_point_config={"enabled": True},
    )

    assert seen_samples == [0, 1]
    assert len(log) == 2
    metadata = json.load(open("scenario_metadata.json"))
    assert len(metadata) == 2
    assert {entry["sample_idx"] for entry in metadata.values()} == {1}
    assert {entry["accepted_operating_point_index"] for entry in metadata.values()} == {0}
    diagnostics = [
        json.loads(line)
        for line in open("scenario_diagnostics.jsonl")
        if line.strip()
    ]
    groups = [
        record for record in diagnostics
        if record.get("record_type") == "operating_point_group"
    ]
    attempts = [
        record for record in diagnostics
        if record.get("record_type") == "operating_point_attempt"
    ]
    assert [record["accepted"] for record in groups] == [False, True]
    assert [record["accepted"] for record in attempts] == [False, True]


def test_target_mode_incomplete_fault_group_does_not_count(gs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gs, "_load_power_system", lambda *_: _dummy_export_psys())

    seen_samples = []
    failed_group_files = []

    def fake_prepare(*args, **kwargs):
        scenario = args[2]
        seen_samples.append(scenario["sample_idx"])
        return {
            "rejected": False,
            "diverged": False,
            "diagnostics": {"accepted": True, "attempts": 1},
            "operating_point": {
                "diagnostics": {"accepted": True, "attempts": 1},
                "operating_point_id": scenario["operating_point_id"],
                "sample_idx": scenario["sample_idx"],
                "accepted_operating_point_index": scenario[
                    "accepted_operating_point_index"
                ],
            },
        }

    def fake_fault_worker(raw, dyr, scenario, scenario_id, operating_point, integration):
        if scenario["sample_idx"] == 0 and scenario["fault_location"] == 10:
            os.makedirs("simulation_data", exist_ok=True)
            path = gs._simulation_file_path(scenario_id)
            with open(path, "w") as f:
                f.write("partial")
            failed_group_files.append(path)
            return {
                "file": path,
                "diverged": False,
                "rejected": False,
                "diagnostics": {"accepted": True},
            }
        if scenario["sample_idx"] == 0:
            return {
                "file": None,
                "diverged": True,
                "rejected": False,
                "diagnostics": {
                    "accepted": False,
                    "reject_reason": "simulation_diverged",
                },
            }
        os.makedirs("simulation_data", exist_ok=True)
        path = gs._simulation_file_path(scenario_id)
        with open(path, "w") as f:
            f.write("ok")
        return {
            "file": path,
            "diverged": False,
            "rejected": False,
            "diagnostics": {"accepted": True},
        }

    monkeypatch.setattr(gs, "_prepare_operating_point_candidate", fake_prepare)
    monkeypatch.setattr(gs, "_run_fault_with_operating_point_worker", fake_fault_worker)

    log = gs._run_target_accepted_driver(
        "case.raw",
        "case.dyr",
        target_accepted_scenarios=1,
        max_total_attempts=5,
        sample_idx_start=0,
        fault_locations=[10, 20],
        fault_impedances=[1e-4],
        n_jobs=1,
        operating_point_config={"enabled": True},
    )

    assert seen_samples == [0, 1]
    assert len(log) == 2
    metadata = json.load(open("scenario_metadata.json"))
    assert {entry["sample_idx"] for entry in metadata.values()} == {1}
    assert failed_group_files
    assert not any(os.path.exists(path) for path in failed_group_files)


def test_target_mode_continue_counts_existing_complete_groups(
        gs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gs, "_load_power_system", lambda *_: _dummy_export_psys())

    os.makedirs("simulation_data", exist_ok=True)
    existing_log = {}
    existing_metadata = {}
    for idx, fault_location in enumerate([10, 20]):
        scenario_id = f"existing-{idx}"
        path = f"simulation_data/scenario_{scenario_id}.npz"
        with open(path, "w") as f:
            f.write("ok")
        row = {
            "sample_idx": 0,
            "fault_location": fault_location,
            "fault_impedance": 1e-4,
            "operating_point_id": "existing-op",
            "accepted_operating_point_index": 0,
            "file": path,
            "diverged": False,
            "rejected": False,
        }
        existing_log[scenario_id] = dict(row)
        existing_metadata[scenario_id] = {
            key: row[key]
            for key in [
                "sample_idx",
                "fault_location",
                "fault_impedance",
                "operating_point_id",
                "accepted_operating_point_index",
            ]
        }

    seen_candidates = []

    def fake_prepare(*args, **kwargs):
        scenario = args[2]
        seen_candidates.append(
            (
                scenario["sample_idx"],
                scenario["accepted_operating_point_index"],
            )
        )
        diagnostics = {
            "record_type": "operating_point_attempt",
            "accepted": True,
            "attempts": 1,
        }
        return {
            "rejected": False,
            "diverged": False,
            "diagnostics": diagnostics,
            "diagnostics_attempts": [diagnostics],
            "operating_point": {
                "diagnostics": {"accepted": True, "attempts": 1},
                "operating_point_id": scenario["operating_point_id"],
                "sample_idx": scenario["sample_idx"],
                "accepted_operating_point_index": scenario[
                    "accepted_operating_point_index"
                ],
            },
        }

    def fake_fault_worker(raw, dyr, scenario, scenario_id, operating_point, integration):
        path = gs._simulation_file_path(scenario_id)
        with open(path, "w") as f:
            f.write("ok")
        return {
            "file": path,
            "diverged": False,
            "rejected": False,
            "diagnostics": {"accepted": True},
        }

    monkeypatch.setattr(gs, "_prepare_operating_point_candidate", fake_prepare)
    monkeypatch.setattr(gs, "_run_fault_with_operating_point_worker", fake_fault_worker)

    log = gs._run_target_accepted_driver(
        "case.raw",
        "case.dyr",
        target_accepted_scenarios=2,
        max_total_attempts=5,
        sample_idx_start=5,
        fault_locations=[10, 20],
        fault_impedances=[1e-4],
        n_jobs=1,
        operating_point_config={"enabled": True},
        existing_log=existing_log,
        existing_metadata=existing_metadata,
    )

    assert seen_candidates == [(5, 1)]
    assert len(log) == 4
    metadata = json.load(open("scenario_metadata.json"))
    assert len(metadata) == 4
    assert {
        entry["accepted_operating_point_index"]
        for entry in metadata.values()
    } == {0, 1}


def test_legacy_driver_uses_predefined_grid_when_target_unset(gs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gs, "load_psse", lambda *_: _dummy_export_psys())
    monkeypatch.setattr(gs, "add_dyr", lambda *_: None)

    called_ids = []

    def fake_worker(*args):
        scenario = args[2]
        scenario_id = args[3]
        called_ids.append(scenario_id)
        return {
            "file": f"simulation_data/scenario_{scenario_id}.npz",
            "diverged": False,
            "rejected": False,
            "diagnostics": {
                "accepted": True,
                "sample_idx": scenario["sample_idx"],
            },
        }

    monkeypatch.setattr(gs, "run_single_scenario_worker", fake_worker)
    scenarios_metadata = {
        "a": {"sample_idx": 0, "fault_location": 1, "fault_impedance": 1e-4},
        "b": {"sample_idx": 0, "fault_location": 2, "fault_impedance": 1e-4},
    }

    log = gs.run_simulation_driver_batched(
        "case.raw",
        "case.dyr",
        scenarios_metadata,
        n_jobs=1,
        batch_size=2,
        checkpoint_interval=10,
        operating_point_config={"enabled": True},
    )

    assert called_ids == ["a", "b"]
    assert set(log) == {"a", "b"}


def test_legacy_driver_writes_all_attempt_diagnostics(gs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gs, "load_psse", lambda *_: _dummy_export_psys())
    monkeypatch.setattr(gs, "add_dyr", lambda *_: None)

    def fake_worker(*args):
        scenario = args[2]
        scenario_id = args[3]
        attempts = [
            {
                "record_type": "operating_point_attempt",
                "scenario_id": scenario_id,
                "sample_idx": scenario["sample_idx"],
                "accepted": False,
                "reject_reason": "voltage_low",
                "attempts": 1,
            },
            {
                "record_type": "operating_point_attempt",
                "scenario_id": scenario_id,
                "sample_idx": scenario["sample_idx"],
                "accepted": True,
                "reject_reason": None,
                "attempts": 2,
            },
        ]
        return {
            "file": f"simulation_data/scenario_{scenario_id}.npz",
            "diverged": False,
            "rejected": False,
            "diagnostics": attempts[-1],
            "diagnostics_attempts": attempts,
        }

    monkeypatch.setattr(gs, "run_single_scenario_worker", fake_worker)
    gs.run_simulation_driver_batched(
        "case.raw",
        "case.dyr",
        {"a": {"sample_idx": 0, "fault_location": 1, "fault_impedance": 1e-4}},
        n_jobs=1,
        batch_size=1,
        checkpoint_interval=10,
        operating_point_config={"enabled": True},
    )

    records = [
        json.loads(line)
        for line in open("scenario_diagnostics.jsonl")
        if line.strip()
    ]
    assert [record["accepted"] for record in records] == [False, True]
    summary = json.load(open("scenario_diagnostics_summary.json"))
    assert summary["reject_reasons"]["voltage_low"] == 1


def test_legacy_checkpoint_resume_preserves_diagnostics_summary(
    gs, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gs, "load_psse", lambda *_: _dummy_export_psys())
    monkeypatch.setattr(gs, "add_dyr", lambda *_: None)

    previous_record = {
        "record_type": "operating_point_attempt",
        "scenario_id": "a",
        "sample_idx": 0,
        "accepted": False,
        "reject_reason": "voltage_low",
        "attempts": 1,
    }
    with open("scenario_diagnostics.jsonl", "w") as f:
        f.write(json.dumps(previous_record) + "\n")
    with open("simulation_checkpoint.json", "w") as f:
        json.dump(
            {
                "last_batch": 0,
                "simulation_log": {
                    "a": {
                        "sample_idx": 0,
                        "fault_location": 1,
                        "fault_impedance": 1e-4,
                        "file": None,
                        "diverged": True,
                    }
                },
            },
            f,
        )

    called_ids = []

    def fake_worker(*args):
        scenario = args[2]
        scenario_id = args[3]
        called_ids.append(scenario_id)
        diagnostics = {
            "record_type": "operating_point_attempt",
            "scenario_id": scenario_id,
            "sample_idx": scenario["sample_idx"],
            "accepted": True,
            "reject_reason": None,
            "attempts": 1,
        }
        return {
            "file": f"simulation_data/scenario_{scenario_id}.npz",
            "diverged": False,
            "rejected": False,
            "diagnostics": diagnostics,
            "diagnostics_attempts": [diagnostics],
        }

    monkeypatch.setattr(gs, "run_single_scenario_worker", fake_worker)
    log = gs.run_simulation_driver_batched(
        "case.raw",
        "case.dyr",
        {
            "a": {"sample_idx": 0, "fault_location": 1, "fault_impedance": 1e-4},
            "b": {"sample_idx": 0, "fault_location": 2, "fault_impedance": 1e-4},
        },
        n_jobs=1,
        batch_size=1,
        checkpoint_interval=10,
        operating_point_config={"enabled": True},
    )

    assert called_ids == ["b"]
    assert set(log) == {"a", "b"}
    records = [
        json.loads(line)
        for line in open("scenario_diagnostics.jsonl")
        if line.strip()
    ]
    assert [record["scenario_id"] for record in records] == ["a", "b"]
    summary = json.load(open("scenario_diagnostics_summary.json"))
    assert summary["total_records"] == 2
    assert summary["reject_reasons"]["voltage_low"] == 1
    assert summary["accepted"] == 1


def test_screen_power_flow_diagnostics_accepts_and_rejects_synthetic_cases(gs):
    op_cfg = gs._resolve_operating_point_config({"enabled": True})
    diagnostics = {
        "pf_converged": True,
        "pf_residual": 1e-10,
        "slack_p_deviation": 0.0,
        "voltage_min": 0.95,
        "voltage_max": 1.04,
        "gen_p_violation_max": 0.0,
        "gen_q_violation_max": 0.0,
        "branch_loading_available": False,
        "branch_loading_max": None,
    }

    accepted, reason = gs._screen_power_flow_diagnostics(
        diagnostics, total_load_p=10.0, op_cfg=op_cfg
    )
    assert accepted
    assert reason is None

    diagnostics["voltage_min"] = 0.85
    accepted, reason = gs._screen_power_flow_diagnostics(
        diagnostics, total_load_p=10.0, op_cfg=op_cfg
    )
    assert not accepted
    assert reason == "voltage_low"


def test_power_flow_diagnostics_on_ieee9_are_finite(gs, data_dir):
    psys = load_psse(os.path.join(data_dir, "ieee9_v33.raw"))
    psys.createYbusComplex()
    p_gen, q_gen = psys.get_gen_pq()
    pg_lb, pg_ub = psys.get_pgen_bounds()
    qg_lb, qg_ub = psys.get_qgen_bounds()

    _, diagnostics = gs._diagnose_power_flow(
        psys,
        p_gen,
        q_gen,
        pg_lb,
        pg_ub,
        qg_lb,
        qg_ub,
        gs._resolve_operating_point_config({"enabled": True}),
    )

    assert diagnostics["pf_converged"]
    assert diagnostics["pf_residual"] < 1e-7
    assert np.isfinite(diagnostics["voltage_min"])
    assert np.isfinite(diagnostics["voltage_max"])
    assert diagnostics["voltage_min"] > 0.0
    assert diagnostics["voltage_max"] > diagnostics["voltage_min"]
    assert "gen_q_violation_count" in diagnostics
    assert "gen_q_violation_total_abs" in diagnostics
    assert "gen_q_violation_argmax" in diagnostics
    assert "gen_q_violation_top" in diagnostics
    mitigation_diagnostics = gs._q_limit_mitigation_defaults(
        gs._resolve_operating_point_config({"enabled": True})
    )
    json.dumps(gs._json_safe(mitigation_diagnostics))
    json.dumps(gs._json_safe(diagnostics))
