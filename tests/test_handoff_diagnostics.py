import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_generate_scenarios_acopf_init import _load_acopf_module
from uqgrid.io.parse import load_matpower
from uqgrid.simulation.pflow import compute_branch_loading_diagnostics

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def acopf():
    return _load_acopf_module()


@pytest.fixture
def system():
    psys = load_matpower(str(ROOT / "data" / "case9.m"))
    psys.createYbusComplex()
    return psys


def _operating_point(psys):
    return {
        "p_load_scaled": [load.pload for load in psys.loads],
        "q_load_scaled": [load.qload for load in psys.loads],
        "p_gen_scaled": [gen.psch for gen in psys.gens],
        "q_gen_scaled": [gen.qsch for gen in psys.gens],
        "pf_v_magnitudes": [bus.v0m for bus in psys.buses],
        "pf_v_angles": [bus.v0a for bus in psys.buses],
    }


def _solution(psys, **kwargs):
    return SimpleNamespace(
        v_magnitudes=np.array([b.v0m for b in psys.buses]),
        v_angles=np.array([b.v0a for b in psys.buses]),
        gen_psch=np.array([g.psch for g in psys.gens]),
        gen_qsch=np.array([g.qsch for g in psys.gens]),
        converged=True,
        residual_norm=1e-12,
    )


def _validate(acopf, system, tmp_path, monkeypatch, source="uqgrid_pf", **kwargs):
    if source == "acopf":
        monkeypatch.setattr(acopf, "parse_exajugo_basecase", lambda *a, **k: {})
        monkeypatch.setattr(
            acopf,
            "apply_exajugo_solution_to_psys",
            lambda *a: {
                "base_mva": system.basemva,
                "num_loads": len(system.loads),
                "num_generators": len(system.gens),
                "adjusted_shunts": 0,
            },
        )
    params = dict(
        raw_path="model.raw",
        dyr_path="model.dyr",
        source=source,
        basecase_path=tmp_path / "Basecase_solution.txt",
        case_raw_path=tmp_path / "case.raw",
        operating_point=_operating_point(system),
        pf_residual_tol=1e-8,
        output_dir=tmp_path,
        sample_idx=7,
        operating_point_id="op-seven",
        load_psse_func=lambda _: system,
        add_dyr_func=lambda *args: None,
        runpf_func=_solution,
    )
    params.update(kwargs)
    result = acopf._validate_replay_handoff(**params)
    report = json.loads(Path(result["handoff_report_path"]).read_text())
    return result, report


@pytest.mark.parametrize("source", ["acopf", "uqgrid_pf"])
@pytest.mark.parametrize("enabled", [True, False])
def test_shared_pf_contract_and_diagnostic_only_slacks(
    acopf, system, tmp_path, monkeypatch, source, enabled
):
    (tmp_path / "Basecase_slacks.txt").write_text(
        "--node slack section\nBus, pm, pp, qm, qp\n1, 1000, -1e-9, 0, 0\n"
    )
    calls = []
    policy = dict(
        enabled=enabled,
        residual_tolerance=2e-7,
        generator_limit_tolerance=0.03,
        voltage_min=0.9,
        voltage_max=1.1,
        branch_loading_max=1.15,
        branch_limit_tolerance=2e-5,
        active_set_voltage_tolerance=3e-6,
    )

    def runpf(psys, **settings):
        calls.append(settings)
        return _solution(psys)

    def validate(psys, solution, **settings):
        assert settings == {k: v for k, v in policy.items() if k != "enabled"}
        calls.append("limits")
        return {"valid": True, "failure_reasons": []}

    result, report = _validate(
        acopf,
        system,
        tmp_path,
        monkeypatch,
        source,
        integration_config=dict(
            enforce_q_limits=False,
            q_limit_tolerance=4e-6,
            max_q_limit_iterations=12,
            power_flow_validation=policy,
        ),
        runpf_func=runpf,
        validate_pf_func=validate,
    )
    assert result["accepted"]
    assert calls[0] == dict(
        verbose=False,
        enforce_q_limits=False,
        q_limit_tolerance=4e-6,
        max_q_limit_iterations=12,
    )
    assert len(calls) == (2 if enabled else 1)
    assert report["effective_validation_settings"]["full_limit_validation"] == (
        "enabled" if enabled else "disabled"
    )
    assert report["comparisons"]["max_abs_p_gen_change_mw"] == 0
    assert report["comparisons"]["max_terminal_power_change_mva"] == 0
    assert len(report["before"]["generator_dispatch"]["p_mw"]) == len(system.gens)
    assert (
        report["flow_definitions"]["before"] == "UQGrid evaluated at imported voltages"
    )
    if source == "acopf":
        assert (
            report["source_artifacts"]["slacks"]["families"]["p_balance"]["max_signed"]
            == 1000
        )
    else:
        assert report["source_artifacts"]["slacks"]["status"] == "not_applicable"


@pytest.mark.parametrize("source", ["acopf", "uqgrid_pf"])
@pytest.mark.parametrize("failure", ["raise", "limit", "nonfinite", "nonconvergence"])
def test_shared_gate_retains_diagnostics_on_failure(
    acopf, system, tmp_path, monkeypatch, source, failure
):
    def runpf(psys, **kwargs):
        if failure == "raise":
            raise RuntimeError("Newton failed")
        result = _solution(psys)
        if failure == "nonfinite":
            result.residual_norm = float("nan")
        if failure == "nonconvergence":
            result.converged = False
        return result

    result, report = _validate(
        acopf,
        system,
        tmp_path,
        monkeypatch,
        source,
        runpf_func=runpf,
        integration_config={"power_flow_validation": {"enabled": failure == "limit"}},
        validate_pf_func=lambda *a, **k: {
            "valid": False,
            "failure_reasons": ["branch_overload"],
        },
    )
    assert not result["accepted"]
    assert report["before"] is not None
    assert report["validation_outcome"]["accepted"] is False
    assert (report["after"] is None) == (failure == "raise")
    assert report["source_artifacts"]["slacks"]["status"] in {
        "unavailable",
        "not_applicable",
    }
    if failure == "limit":
        assert result["reject_reason"].endswith("validation_failed")


def test_voltage_dispatch_and_reversed_terminal_changes(acopf, system):
    system.buses[0].v0a = np.deg2rad(179)
    before = acopf._capture_handoff_state(system)
    pf = _solution(system)
    pf.v_angles[0] = np.deg2rad(-179)
    pf.v_magnitudes[1] += 0.02
    pf.gen_psch[0] += 0.04
    pf.gen_qsch[1] -= 0.03
    after = acopf._capture_handoff_state(system, pf)
    comparison = acopf._compare_handoff_states(before, after, {}, {})
    assert comparison["max_abs_angle_change_deg"] == pytest.approx(2)
    assert comparison["max_abs_voltage_change_pu"] == pytest.approx(0.02)
    assert comparison["max_abs_p_gen_change_mw"] == pytest.approx(4)
    assert comparison["max_abs_q_gen_change_mvar"] == pytest.approx(3)
    assert comparison["branch_change_top"][0]["terminal_change_mva"] > 0
    mismatch = after["nodal_mismatch"]
    assert mismatch["status"] == "available"
    assert mismatch["max_abs_p_mw"] > 0


def _raw():
    return SimpleNamespace(
        buses=[SimpleNamespace(busn=11, type=3), SimpleNamespace(busn=95, type=1)],
        branches=[
            SimpleNamespace(fbus=11, tbus=95, ckt=c, status=1, rateA=100)
            for c in ("'A'", "'B'")
        ],
        transformers=[SimpleNamespace(fbus=95, tbus=11, ckt="'T'", status=1, rateA=80)],
        transthree=[],
    )


def _artifacts(
    tmp_path, *, node="11, 0, 1e-10, -1e-9, 2", line="11, 95, 0, 3\n11, 95, 4, 0"
):
    (tmp_path / "Basecase_slacks.txt").write_text(
        f"--node slack section\nBus, pm, pp, qm, qp\n{node}\n"
        f"--line slack section\nFrom, To, sm, sp\n{line}\n"
        "--transformer slack section\nFrom, To, sm, sp\n95, 11, 5, -1e-8\n"
    )
    (tmp_path / "Basecase_power_flow.txt").write_text(
        "--line section\nFrom, To, pm, pp\n11, 95, 20, -19\n11, 95, 30, -29\n"
        "--transformer section\nFrom, To, pm, pp\n95, 11, 40, -39\n"
    )


def test_slack_families_units_and_parallel_circuit_association(
    acopf, tmp_path, monkeypatch
):
    from uqgrid.io import parse_psse

    monkeypatch.setattr(parse_psse, "read_raw", lambda _: _raw())
    _artifacts(tmp_path)
    result, slacks, flows = acopf._exajugo_handoff_artifacts(
        tmp_path / "Basecase_solution.txt", tmp_path / "case.raw"
    )
    families = result["slacks"]["families"]
    assert result["slacks"]["status"] == "available"
    assert families["p_balance"]["unit"] == "MW"
    assert families["p_balance"]["sum_nonnegative"] == 1e-10
    assert families["q_balance"]["unit"] == "MVAr"
    assert families["q_balance"]["min_signed"] == -1e-9
    assert families["q_balance"]["sum_nonnegative"] == 2
    assert families["line_thermal"]["sum_nonnegative"] == 7
    assert families["transformer_thermal"]["unit"] == "MVA-equivalent"
    assert families["transformer_thermal"]["min_signed"] == -1e-8
    assert slacks[("line", 11, 95, "A")]["from"] == 0
    assert slacks[("line", 11, 95, "B")]["from"] == 4
    assert flows[("line", 11, 95, "B")]["p_from_mw"] == 30
    assert "unavailable" in result["power_flow"]["q_s_status"]


@pytest.mark.parametrize(
    "failure", ["count", "endpoints", "three_winding", "missing_raw"]
)
def test_unavailable_association_keeps_aggregate_slacks(
    acopf, tmp_path, monkeypatch, failure
):
    from uqgrid.io import parse_psse

    raw = _raw()
    if failure == "count":
        raw.branches.pop()
    if failure == "endpoints":
        raw.branches[0].fbus, raw.branches[0].tbus = 95, 11
    if failure == "three_winding":
        raw.transthree = [object()]

    def read(_):
        if failure == "missing_raw":
            raise OSError("missing case.raw")
        return raw

    monkeypatch.setattr(parse_psse, "read_raw", read)
    _artifacts(tmp_path)
    result, slacks, _ = acopf._exajugo_handoff_artifacts(
        tmp_path / "Basecase_solution.txt", tmp_path / "case.raw"
    )
    family = "transformer_thermal" if failure == "three_winding" else "line_thermal"
    assert (
        result["slacks"]["families"][family]["association"]["status"] == "unavailable"
    )
    assert result["slacks"]["families"][family]["sum_nonnegative"] in (5, 7)
    if failure != "three_winding":
        assert not any(key[0] == "line" for key in slacks)


@pytest.mark.parametrize("bad", ["11, 0, nan, 0, 0", "11, 0, 0", "nonsense", ""])
def test_bad_optional_section_is_not_fabricated_zero(acopf, tmp_path, bad):
    _artifacts(tmp_path, node=bad)
    result, _, _ = acopf._exajugo_handoff_artifacts(
        tmp_path / "Basecase_solution.txt", tmp_path / "case.raw"
    )
    assert result["slacks"]["status"] == "partial"
    family = result["slacks"]["families"]["p_balance"]
    assert family["status"] == "unavailable"
    assert family["sum_nonnegative"] is None
    assert family["min_signed"] is None
    assert result["slacks"]["families"]["line_thermal"]["sum_nonnegative"] == 7


def test_missing_optional_files_are_unavailable(acopf, tmp_path):
    result, slacks, flows = acopf._exajugo_handoff_artifacts(
        tmp_path / "Basecase_solution.txt", tmp_path / "case.raw"
    )
    assert result["slacks"]["status"] == "unavailable"
    assert result["power_flow"]["status"] == "unavailable"
    assert result["slacks"]["families"]["line_thermal"]["sum_nonnegative"] is None
    assert slacks == flows == {}


def test_association_filters_inactive_raw_elements_and_isolated_buses(acopf):
    raw = _raw()
    raw.buses.append(SimpleNamespace(busn=70, type=4))
    raw.branches.insert(
        0, SimpleNamespace(fbus=11, tbus=95, ckt="off", status=0, rateA=1)
    )
    raw.branches.append(
        SimpleNamespace(fbus=11, tbus=70, ckt="isolated", status=1, rateA=1)
    )
    section = {
        "status": "available",
        "rows": [{"from_bus_id": 11, "to_bus_id": 95} for _ in range(2)],
    }
    assert acopf._associate_exajugo_rows(section, raw, "line")["status"] == "available"
    assert [r["circuit_id"] for r in section["rows"]] == ["A", "B"]


def test_ambiguous_reversed_circuit_identity_is_not_associated(acopf):
    raw = _raw()
    raw.branches[1].fbus, raw.branches[1].tbus, raw.branches[1].ckt = 95, 11, "'A'"
    section = {
        "status": "available",
        "rows": [
            {"from_bus_id": 11, "to_bus_id": 95},
            {"from_bus_id": 95, "to_bus_id": 11},
        ],
    }
    result = acopf._associate_exajugo_rows(section, raw, "line")
    assert result["status"] == "unavailable"
    assert "Duplicate" in result["reason"]


def test_missing_header_cannot_be_mistaken_for_zero_slacks(acopf, tmp_path):
    _artifacts(tmp_path)
    path = tmp_path / "Basecase_slacks.txt"
    path.write_text(path.read_text().replace("From, To, sm, sp\n", "", 1))
    result, _, _ = acopf._exajugo_handoff_artifacts(
        tmp_path / "solution.txt", tmp_path / "case.raw"
    )
    family = result["slacks"]["families"]["line_thermal"]
    assert family["status"] == "unavailable"
    assert family["sum_nonnegative"] is None


def test_empty_thermal_rows_are_zero_only_for_validated_empty_raw_section(
    acopf, tmp_path, monkeypatch
):
    from uqgrid.io import parse_psse

    raw = _raw()
    _artifacts(tmp_path, line="")
    monkeypatch.setattr(parse_psse, "read_raw", lambda _: raw)
    args = (tmp_path / "solution.txt", tmp_path / "case.raw")
    result, _, _ = acopf._exajugo_handoff_artifacts(*args)
    assert result["slacks"]["families"]["line_thermal"]["sum_nonnegative"] is None
    raw.branches = []
    result, _, _ = acopf._exajugo_handoff_artifacts(*args)
    assert result["slacks"]["families"]["line_thermal"]["sum_nonnegative"] == 0


def test_branch_external_identity_signed_powers_and_skipped_counts(system):
    system.ext2int = {11 + 10 * i: i for i in range(system.nbuses)}
    system.branches[0].raw_identity = {"element_type": "transformer", "circuit_id": "T"}
    system.branches[1].rateA = 0
    system.branches[2].r = system.branches[2].x = 0
    solution = _solution(system)
    solution.v_angles[0] = 0.2
    result = compute_branch_loading_diagnostics(system, solution, top_n=None)
    row = next(r for r in result["loading_top"] if r["branch_index"] == 0)
    assert row["from_bus_id"] == 11
    assert row["circuit_id"] == "T"
    assert row["p_from_mw"] > 0 and row["p_to_mw"] < 0
    assert row["s_from_mva"] == pytest.approx(
        np.hypot(row["p_from_mw"], row["q_from_mvar"])
    )
    assert row["loading"] == max(row["s_from_mva"], row["s_to_mva"]) / row["rateA_mva"]
    assert row["limiting_end"] in {"from", "to"}
    assert result["unrated_branch_count"] == 1
    assert result["uncomputable_branch_count"] == 1


def test_terminal_ranking_and_reverse_source_orientation(acopf, system):
    system.branches[0].raw_identity = {"element_type": "line", "circuit_id": "A"}
    before = acopf._capture_handoff_state(system)
    row = next(r for r in before["branch"]["loading_top"] if r["branch_index"] == 0)
    key = acopf._handoff_branch_key(row)
    source = dict(
        from_bus_id=row["to_bus_id"],
        to_bus_id=row["from_bus_id"],
        p_from_mw=20,
        p_to_mw=-19,
    )
    after = copy.deepcopy(before)
    changed = next(r for r in after["branch"]["loading_top"] if r["branch_index"] == 0)
    changed.update(p_from_mw=row["p_from_mw"] - 3, q_from_mvar=row["q_from_mvar"] + 4)
    comp = acopf._compare_handoff_states(before, after, {}, {key: source})
    top = comp["branch_change_top"][0]
    assert top["before"]["branch_index"] == 0
    assert top["terminal_change_mva"] == pytest.approx(5)
    assert top["exajugo_terminal_p"]["from_mw"] == -19
    assert top["solved_minus_exajugo_p_mw"]["from"] == pytest.approx(
        row["p_from_mw"] + 16
    )


def test_report_write_failure_is_fatal_and_atomic(acopf, system, tmp_path, monkeypatch):
    report = tmp_path / "handoff_diagnostics" / "op_7.json"
    report.parent.mkdir()
    report.write_text('{"previous":true}')
    monkeypatch.setattr(
        acopf.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        _validate(acopf, system, tmp_path, monkeypatch)
    assert json.loads(report.read_text()) == {"previous": True}
    assert not list(tmp_path.glob("*.npz"))


def test_report_top_lists_bounded_but_dispatch_complete(acopf, system):
    state = acopf._capture_handoff_state(system)
    state["branch"]["loading_top"] *= 4
    comp = acopf._compare_handoff_states(state, state, {}, {})
    assert len(comp["branch_loading_before_top"]) == 10
    assert len(comp["branch_change_top"]) == 10
    compact = acopf._compact_handoff_state(state)
    assert len(compact["branch"]["loading_top"]) == 10
    assert len(compact["generator_dispatch"]["p_mw"]) == len(system.gens)
