import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _load_acopf_module():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(repo_root, "scripts", "run", "generate_scenarios_acopf_init.py")
    spec = importlib.util.spec_from_file_location("generate_scenarios_acopf_init", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def acopf():
    return _load_acopf_module()


def _raw_text():
    return "\n".join(
        [
            "0, 100.0, 33, 0, 0, 60.0 / header",
            "1, 'BUS1', 138.0, 1",
            "0 / END OF BUS DATA, BEGIN LOAD DATA",
            "1, '1', 1, 1, 1, 10.0, 2.0",
            "2, '1', 1, 1, 1, 20.0, 4.0 / keep",
            "0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA",
        ]
    )


def test_patch_raw_loads_uses_uqgrid_q_sign(acopf, tmp_path):
    raw = tmp_path / "case.raw"
    patched = tmp_path / "patched.raw"
    raw.write_text(_raw_text(), encoding="utf-8")

    section = acopf.patch_raw_loads(raw, patched, [0.11, 0.22], [-0.03, -0.04])

    assert section.base_mva == pytest.approx(100.0)
    out = patched.read_text(encoding="utf-8")
    assert "1, '1', 1, 1, 1, 11.000000, 3.000000" in out
    assert "2, '1', 1, 1, 1, 22.000000, 4.000000 / keep" in out


def test_patch_raw_loads_rejects_count_mismatch(acopf, tmp_path):
    raw = tmp_path / "case.raw"
    raw.write_text(_raw_text(), encoding="utf-8")

    with pytest.raises(ValueError, match="Load vector length mismatch"):
        acopf.patch_raw_loads(raw, tmp_path / "out.raw", [0.1], [-0.1])


def _basecase_text(include_load_section=True):
    lines = [
        "--bus section",
        "i, v(p.u.), theta(deg), bcs(MVAR at v = 1 p.u.)",
        "1, 1.010000, 5.0, 2.0",
        "2, 0.990000, -3.0, 0.0",
    ]
    if include_load_section:
        lines.extend(
            [
                "--load section",
                "i, p(MW), q(MW)",
                "1, 11.0, 3.0",
                "2, 22.0, 4.0",
            ]
        )
    lines.extend(
        [
            "--generator section",
            "i, id, p(MW), q(MW)",
            "1, '1', 50.0, 10.0",
            "2, 'OFF', 0.0, 0.0",
            "2, 'A', 60.0, 12.0",
        ]
    )
    return "\n".join(lines)


def test_parse_exajugo_basecase_with_load_section(acopf, tmp_path):
    path = tmp_path / "Basecase_solution.txt"
    path.write_text(_basecase_text(include_load_section=True), encoding="utf-8")

    parsed = acopf.parse_exajugo_basecase(path)

    np.testing.assert_array_equal(parsed.bus_ids, [1, 2])
    np.testing.assert_allclose(parsed.p_load_mw, [11.0, 22.0])
    np.testing.assert_array_equal(parsed.nonzero_gen_mask, [True, False, True])
    assert parsed.gen_ids == ["1", "OFF", "A"]


def test_parse_exajugo_basecase_falls_back_to_raw_loads(acopf, tmp_path):
    path = tmp_path / "Basecase_solution.txt"
    raw = tmp_path / "case.raw"
    path.write_text(_basecase_text(include_load_section=False), encoding="utf-8")
    raw.write_text(_raw_text(), encoding="utf-8")

    parsed = acopf.parse_exajugo_basecase(path, raw_path=raw)

    np.testing.assert_array_equal(parsed.load_bus_ids, [1, 2])
    np.testing.assert_allclose(parsed.p_load_mw, [10.0, 20.0])
    np.testing.assert_allclose(parsed.q_load_mvar, [2.0, 4.0])


class _DummyBus:
    def set_vinit(self, magnitude, angle):
        self.v0m = magnitude
        self.v0a = angle


class _DummyPsys:
    def __init__(self):
        self.basemva = 100.0
        self.nbuses = 2
        self.ext2int = {1: 0, 2: 1}
        self.loads = [SimpleNamespace(bus=0), SimpleNamespace(bus=1)]
        self.gens = [SimpleNamespace(bus=0, idx="1"), SimpleNamespace(bus=1, idx="A")]
        self.buses = [_DummyBus(), _DummyBus()]
        self.shunts = [SimpleNamespace(bus=0, bsh=0.01)]
        self.load_pq = None
        self.gen_pq = None
        self.faults = []

    def set_load_pq(self, p_load, q_load):
        self.load_pq = (np.asarray(p_load), np.asarray(q_load))

    def set_gen_pq(self, p_gen, q_gen):
        self.gen_pq = (np.asarray(p_gen), np.asarray(q_gen))

    def add_shunt(self, bus, gsh, bsh):
        self.shunts.append(SimpleNamespace(bus=bus, gsh=gsh / self.basemva, bsh=bsh / self.basemva))

    def createYbusComplex(self):
        self.ybus_created = True

    def add_busfault(self, fault_location, fault_impedance):
        self.faults.append((fault_location, fault_impedance))

    def export_state_metadata(self, filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "0": {"model": "GenGENROU", "state_name": "delta"},
                    "1": {"model": "Other", "state_name": "delta"},
                    "2": {"model": "GenGENROU", "state_name": "delta"},
                },
                f,
            )


def test_apply_exajugo_solution_to_psys_checks_order_and_applies_values(acopf, tmp_path):
    path = tmp_path / "Basecase_solution.txt"
    path.write_text(_basecase_text(include_load_section=True), encoding="utf-8")
    parsed = acopf.parse_exajugo_basecase(path)
    psys = _DummyPsys()

    summary = acopf.apply_exajugo_solution_to_psys(psys, parsed)

    np.testing.assert_allclose(psys.load_pq[0], [0.11, 0.22])
    np.testing.assert_allclose(psys.load_pq[1], [-0.03, -0.04])
    np.testing.assert_allclose(psys.gen_pq[0], [0.50, 0.60])
    np.testing.assert_allclose(psys.gen_pq[1], [0.10, 0.12])
    assert summary["adjusted_shunts"] == 1
    assert psys.shunts[-1].bus == 0
    assert psys.shunts[-1].bsh == pytest.approx(0.01)
    assert psys.buses[0].v0m == pytest.approx(1.01)
    assert psys.buses[1].v0a == pytest.approx(np.deg2rad(-3.0))


def test_apply_exajugo_solution_rejects_order_mismatch(acopf, tmp_path):
    path = tmp_path / "Basecase_solution.txt"
    path.write_text(_basecase_text(include_load_section=True), encoding="utf-8")
    parsed = acopf.parse_exajugo_basecase(path)
    psys = _DummyPsys()
    psys.loads = [SimpleNamespace(bus=1), SimpleNamespace(bus=0)]

    with pytest.raises(RuntimeError, match="load ordering mismatch"):
        acopf.apply_exajugo_solution_to_psys(psys, parsed)


def test_tsi_helpers_select_delta_and_compute_final_min(acopf):
    metadata = {
        "0": {"model": "GenGENROU", "state_name": "delta"},
        "1": {"model": "Other", "state_name": "delta"},
        "2": {"model": "GenGENROU", "state_name": "delta"},
    }
    history = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [9.0, 9.0, 9.0],
            [0.0, 3.0, 1.0],
        ]
    )

    indices = acopf.select_generator_delta_indices(metadata)
    final, minimum, tsi_t = acopf.compute_tsi_final_min_from_history(history, indices)

    assert indices == [0, 2]
    assert final == pytest.approx(tsi_t[-1])
    assert minimum == pytest.approx(np.nanmin(tsi_t))
    assert minimum < final


def test_probml_writer_appends_and_validates_resume_pair(acopf, tmp_path):
    final_path = tmp_path / "tsi_final.npz"
    min_path = tmp_path / "tsi_min.npz"
    X_row = np.asarray([[0.5, 0.1, 0.2], [0.1, -0.03, -0.04]])
    fault_locations = [142, 143]
    fault_impedances = [1e-4]
    sids = np.asarray([["s142"], ["s143"]], dtype=object)

    acopf.append_probml_dataset_row(
        final_path,
        X_row=X_row,
        Y_row=np.asarray([[1.0], [2.0]]),
        sample_idx=0,
        fault_locations=fault_locations,
        fault_impedances=fault_impedances,
        scenario_ids_row=sids,
        n_gen=1,
        n_load=2,
        tsi_mode="final",
    )
    acopf.append_probml_dataset_row(
        final_path,
        X_row=X_row + 1.0,
        Y_row=np.asarray([[3.0], [4.0]]),
        sample_idx=1,
        fault_locations=fault_locations,
        fault_impedances=fault_impedances,
        scenario_ids_row=sids,
        n_gen=1,
        n_load=2,
        tsi_mode="final",
    )
    acopf.append_probml_dataset_row(
        min_path,
        X_row=X_row,
        Y_row=np.asarray([[0.5], [1.5]]),
        sample_idx=0,
        fault_locations=fault_locations,
        fault_impedances=fault_impedances,
        scenario_ids_row=sids,
        n_gen=1,
        n_load=2,
        tsi_mode="min",
    )
    acopf.append_probml_dataset_row(
        min_path,
        X_row=X_row + 1.0,
        Y_row=np.asarray([[2.5], [3.5]]),
        sample_idx=1,
        fault_locations=fault_locations,
        fault_impedances=fault_impedances,
        scenario_ids_row=sids,
        n_gen=1,
        n_load=2,
        tsi_mode="min",
    )

    with np.load(final_path, allow_pickle=True) as data:
        assert set(data.files) == {
            "X",
            "X_flat",
            "Y",
            "sample_idx",
            "fault_locations",
            "fault_impedances",
            "scenario_ids",
            "initialization_source",
            "acopf_feasible",
            "meta",
        }
        assert data["X"].shape == (2, 2, 3)
        assert data["X_flat"].shape == (2, 6)
        assert data["Y"].shape == (2, 2, 1)
        assert data["initialization_source"].tolist() == ["acopf", "acopf"]
        assert data["acopf_feasible"].tolist() == [True, True]
        meta = data["meta"][0].item() if hasattr(data["meta"][0], "item") else data["meta"][0]
        assert meta["source"] == "uqgrid_acopf_initialized"
        assert meta["tsi_mode"] == "final"

    resume = acopf.validate_probml_resume_pair(final_path, min_path)
    assert resume == {
        "accepted_count": 2,
        "next_sample_idx": 2,
        "integration_contract": None,
    }


def test_probml_resume_enforces_normalized_integration_contract(acopf, tmp_path):
    final_path = tmp_path / "tsi_final.npz"
    min_path = tmp_path / "tsi_min.npz"
    contract = acopf._integration_contract_from_config(
        {
            "integration": {
                "method": "cn",
                "petsc": True,
                "arkimex": False,
                "dt": 0.01,
                "steps": 10,
                "tend": 1.0,
                "ton": 0.2,
                "toff": 0.3,
            }
        }
    )
    kwargs = {
        "X_row": np.asarray([[0.5, 0.1], [0.1, -0.03]]),
        "Y_row": np.asarray([[1.0]]),
        "sample_idx": 0,
        "fault_locations": [1],
        "fault_impedances": [1e-4],
        "scenario_ids_row": np.asarray([["sid"]], dtype=object),
        "n_gen": 1,
        "n_load": 1,
        "integration_contract": contract,
    }
    acopf.append_probml_dataset_row(final_path, tsi_mode="final", **kwargs)
    acopf.append_probml_dataset_row(min_path, tsi_mode="min", **kwargs)

    resume = acopf.validate_probml_resume_pair(
        final_path,
        min_path,
        expected_integration_contract=contract,
    )
    assert resume["integration_contract"] == contract
    with np.load(final_path, allow_pickle=True) as data:
        assert data["meta"][0]["integration_contract"] == contract

    changed = {**contract, "method": "beuler"}
    with pytest.raises(ValueError, match="method=.*requested"):
        acopf.validate_probml_resume_pair(
            final_path,
            min_path,
            expected_integration_contract=changed,
        )


def test_integration_contract_canonicalizes_pf_settings(acopf):
    defaults = acopf._integration_contract_from_config({})
    assert defaults["enforce_q_limits"] is True
    assert defaults["q_limit_tolerance"] == pytest.approx(1e-8)
    assert defaults["max_q_limit_iterations"] is None
    assert defaults["power_flow_validation"] == {
        "enabled": False,
        "residual_tolerance": pytest.approx(1e-8),
        "generator_limit_tolerance": pytest.approx(1e-6),
        "voltage_min": None,
        "voltage_max": None,
        "branch_loading_max": None,
        "branch_limit_tolerance": pytest.approx(1e-5),
        "active_set_voltage_tolerance": pytest.approx(1e-6),
    }

    explicit = acopf._integration_contract_from_config(
        {
            "integration": {
                "enforce_q_limits": False,
                "q_limit_tolerance": 2e-8,
                "max_q_limit_iterations": 7,
                "power_flow_validation": {
                    "enabled": True,
                    "residual_tolerance": 3e-8,
                    "generator_limit_tolerance": 4e-6,
                    "voltage_min": 0.9,
                    "voltage_max": 1.1,
                    "branch_loading_max": 1.0,
                    "branch_limit_tolerance": 5e-5,
                    "active_set_voltage_tolerance": 6e-6,
                },
            }
        }
    )
    assert explicit["enforce_q_limits"] is False
    assert explicit["q_limit_tolerance"] == pytest.approx(2e-8)
    assert explicit["max_q_limit_iterations"] == 7
    assert explicit["power_flow_validation"]["enabled"] is True
    assert explicit["power_flow_validation"]["voltage_min"] == pytest.approx(0.9)
    assert explicit["power_flow_validation"]["branch_loading_max"] == pytest.approx(
        1.0
    )


def test_probml_resume_treats_legacy_pf_contract_as_q_limited_unvalidated(
    acopf,
    tmp_path,
):
    current_contract = acopf._integration_contract_from_config({})
    pf_keys = {
        "enforce_q_limits",
        "q_limit_tolerance",
        "max_q_limit_iterations",
        "power_flow_validation",
    }
    legacy_contract = {
        key: value
        for key, value in current_contract.items()
        if key not in pf_keys
    }
    normalized_legacy = acopf._normalize_integration_contract(legacy_contract)
    assert normalized_legacy["enforce_q_limits"] is True
    assert normalized_legacy["q_limit_tolerance"] == pytest.approx(1e-8)
    assert normalized_legacy["max_q_limit_iterations"] is None
    assert normalized_legacy["power_flow_validation"]["enabled"] is False

    final_path = tmp_path / "tsi_final.npz"
    min_path = tmp_path / "tsi_min.npz"
    kwargs = {
        "X_row": np.asarray([[0.5, 0.1], [0.1, -0.03]]),
        "Y_row": np.asarray([[1.0]]),
        "sample_idx": 0,
        "fault_locations": [1],
        "fault_impedances": [1e-4],
        "scenario_ids_row": np.asarray([["sid"]], dtype=object),
        "n_gen": 1,
        "n_load": 1,
        "integration_contract": legacy_contract,
    }
    acopf.append_probml_dataset_row(final_path, tsi_mode="final", **kwargs)
    acopf.append_probml_dataset_row(min_path, tsi_mode="min", **kwargs)

    resume = acopf.validate_probml_resume_pair(
        final_path,
        min_path,
        expected_integration_contract=current_contract,
    )
    assert resume["integration_contract"] == current_contract

    strict_contract = acopf._integration_contract_from_config(
        {
            "integration": {
                "power_flow_validation": {"enabled": True},
            }
        }
    )
    with pytest.raises(ValueError, match="power_flow_validation.enabled"):
        acopf.validate_probml_resume_pair(
            final_path,
            min_path,
            expected_integration_contract=strict_contract,
        )


@pytest.mark.parametrize(
    ("path", "requested"),
    [
        ("enforce_q_limits", False),
        ("q_limit_tolerance", 2e-8),
        ("max_q_limit_iterations", 7),
        ("power_flow_validation.enabled", True),
        ("power_flow_validation.residual_tolerance", 2e-8),
        ("power_flow_validation.generator_limit_tolerance", 2e-6),
        ("power_flow_validation.voltage_min", 0.9),
        ("power_flow_validation.voltage_max", 1.1),
        ("power_flow_validation.branch_loading_max", 1.0),
        ("power_flow_validation.branch_limit_tolerance", 2e-5),
        ("power_flow_validation.active_set_voltage_tolerance", 2e-6),
    ],
)
def test_probml_resume_rejects_changed_pf_contract(
    acopf,
    tmp_path,
    path,
    requested,
):
    contract = acopf._integration_contract_from_config({})
    final_path = tmp_path / "tsi_final.npz"
    min_path = tmp_path / "tsi_min.npz"
    kwargs = {
        "X_row": np.asarray([[0.5, 0.1], [0.1, -0.03]]),
        "Y_row": np.asarray([[1.0]]),
        "sample_idx": 0,
        "fault_locations": [1],
        "fault_impedances": [1e-4],
        "scenario_ids_row": np.asarray([["sid"]], dtype=object),
        "n_gen": 1,
        "n_load": 1,
        "integration_contract": contract,
    }
    acopf.append_probml_dataset_row(final_path, tsi_mode="final", **kwargs)
    acopf.append_probml_dataset_row(min_path, tsi_mode="min", **kwargs)

    changed = json.loads(json.dumps(contract))
    target = changed
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = requested

    with pytest.raises(ValueError, match=path):
        acopf.validate_probml_resume_pair(
            final_path,
            min_path,
            expected_integration_contract=changed,
        )


def test_probml_resume_treats_legacy_dynamic_limit_contract_as_disabled(
    acopf,
    tmp_path,
):
    enabled_contract = acopf._integration_contract_from_config({})
    dynamic_keys = {
        "enforce_dynamic_limits",
        "dynamic_limit_tolerance",
        "dynamic_limit_release_tolerance",
        "max_dynamic_limit_iterations",
    }
    legacy_contract = {
        key: value
        for key, value in enabled_contract.items()
        if key not in dynamic_keys
    }
    normalized_legacy = acopf._normalize_integration_contract(legacy_contract)
    assert normalized_legacy["enforce_dynamic_limits"] is False
    assert normalized_legacy["dynamic_limit_tolerance"] == pytest.approx(1e-8)
    assert normalized_legacy["dynamic_limit_release_tolerance"] == pytest.approx(
        1e-10
    )
    assert normalized_legacy["max_dynamic_limit_iterations"] == 20

    final_path = tmp_path / "tsi_final.npz"
    min_path = tmp_path / "tsi_min.npz"
    kwargs = {
        "X_row": np.asarray([[0.5, 0.1], [0.1, -0.03]]),
        "Y_row": np.asarray([[1.0]]),
        "sample_idx": 0,
        "fault_locations": [1],
        "fault_impedances": [1e-4],
        "scenario_ids_row": np.asarray([["sid"]], dtype=object),
        "n_gen": 1,
        "n_load": 1,
        "integration_contract": legacy_contract,
    }
    acopf.append_probml_dataset_row(final_path, tsi_mode="final", **kwargs)
    acopf.append_probml_dataset_row(min_path, tsi_mode="min", **kwargs)

    disabled_contract = acopf._integration_contract_from_config(
        {"integration": {"enforce_dynamic_limits": False}}
    )
    resume = acopf.validate_probml_resume_pair(
        final_path,
        min_path,
        expected_integration_contract=disabled_contract,
    )
    assert resume["integration_contract"] == disabled_contract

    with pytest.raises(ValueError, match="enforce_dynamic_limits"):
        acopf.validate_probml_resume_pair(
            final_path,
            min_path,
            expected_integration_contract=enabled_contract,
        )


def test_probml_resume_rejects_legacy_time_grid_when_contract_required(acopf, tmp_path):
    final_path = tmp_path / "tsi_final.npz"
    min_path = tmp_path / "tsi_min.npz"
    kwargs = {
        "X_row": np.asarray([[0.5, 0.1], [0.1, -0.03]]),
        "Y_row": np.asarray([[1.0]]),
        "sample_idx": 0,
        "fault_locations": [1],
        "fault_impedances": [1e-4],
        "scenario_ids_row": np.asarray([["sid"]], dtype=object),
        "n_gen": 1,
        "n_load": 1,
    }
    acopf.append_probml_dataset_row(final_path, tsi_mode="final", **kwargs)
    acopf.append_probml_dataset_row(min_path, tsi_mode="min", **kwargs)

    with pytest.raises(ValueError, match="new output directory or basename"):
        acopf.validate_probml_resume_pair(
            final_path,
            min_path,
            expected_integration_contract=acopf._integration_contract_from_config({}),
        )


def test_probml_resume_pair_rejects_row_count_mismatch(acopf, tmp_path):
    final_path = tmp_path / "tsi_final.npz"
    min_path = tmp_path / "tsi_min.npz"
    X_row = np.asarray([[0.5, 0.1], [0.1, -0.03]])
    sids = np.asarray([["sid"]], dtype=object)

    kwargs = dict(
        X_row=X_row,
        Y_row=np.asarray([[1.0]]),
        sample_idx=0,
        fault_locations=[1],
        fault_impedances=[1e-4],
        scenario_ids_row=sids,
        n_gen=1,
        n_load=1,
    )
    acopf.append_probml_dataset_row(final_path, tsi_mode="final", **kwargs)
    acopf.append_probml_dataset_row(min_path, tsi_mode="min", **kwargs)
    acopf.append_probml_dataset_row(
        min_path,
        tsi_mode="min",
        **{**kwargs, "sample_idx": 1, "Y_row": np.asarray([[2.0]])},
    )

    with pytest.raises(ValueError, match="row counts disagree"):
        acopf.validate_probml_resume_pair(final_path, min_path)


def test_probml_writer_backfills_legacy_source_labels(acopf, tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        X=np.asarray([[[0.5, 0.1, 0.2], [0.1, -0.03, -0.04]]]),
        X_flat=np.asarray([[0.5, 0.1, 0.2, 0.1, -0.03, -0.04]]),
        Y=np.asarray([[[1.0], [2.0]]]),
        sample_idx=np.asarray([0]),
        fault_locations=np.asarray([142, 143]),
        fault_impedances=np.asarray([1e-4]),
        scenario_ids=np.asarray([[["s142"], ["s143"]]], dtype=object),
        meta=np.asarray([{}], dtype=object),
    )

    acopf.append_probml_dataset_row(
        path,
        X_row=np.asarray([[1.5, 1.1, 1.2], [1.1, 0.97, 0.96]]),
        Y_row=np.asarray([[3.0], [4.0]]),
        sample_idx=1,
        fault_locations=[142, 143],
        fault_impedances=[1e-4],
        scenario_ids_row=np.asarray([["s142b"], ["s143b"]], dtype=object),
        n_gen=1,
        n_load=2,
        tsi_mode="final",
        initialization_source="uqgrid_pf",
        acopf_feasible=False,
    )

    with np.load(path, allow_pickle=True) as data:
        assert data["initialization_source"].tolist() == ["acopf", "uqgrid_pf"]
        assert data["acopf_feasible"].tolist() == [True, False]


def test_probml_resume_pair_rejects_source_label_mismatch(acopf, tmp_path):
    final_path = tmp_path / "tsi_final.npz"
    min_path = tmp_path / "tsi_min.npz"
    kwargs = dict(
        X_row=np.asarray([[0.5, 0.1, 0.2], [0.1, -0.03, -0.04]]),
        Y_row=np.asarray([[1.0]]),
        sample_idx=0,
        fault_locations=[142],
        fault_impedances=[1e-4],
        scenario_ids_row=np.asarray([["s142"]], dtype=object),
        n_gen=1,
        n_load=2,
    )
    acopf.append_probml_dataset_row(final_path, tsi_mode="final", **kwargs)
    acopf.append_probml_dataset_row(min_path, tsi_mode="min", **kwargs)
    min_payload = acopf._load_npz_dict(min_path)
    min_payload["initialization_source"] = np.asarray(["uqgrid_pf"], dtype=object)
    np.savez_compressed(min_path, **min_payload)

    with pytest.raises(ValueError, match="initialization_source"):
        acopf.validate_probml_resume_pair(final_path, min_path)


def test_resolve_acopf_initialization_config_precedence(acopf, tmp_path):
    env = {
        "JULIA": "env-julia",
        "EXAJUGO_ROOT": str(tmp_path / "env-exajugo"),
        "EXAJUGO_BASE_RAW": str(tmp_path / "env.raw"),
        "EXAJUGO_BASE_ROP": str(tmp_path / "env.rop"),
    }
    config = {
        "acopf_initialization": {
            "julia": "config-julia",
            "exajugo_root": str(tmp_path / "config-exajugo"),
            "base_raw": str(tmp_path / "config.raw"),
            "base_rop": str(tmp_path / "config.rop"),
            "acopf_timeout_s": 123,
            "acopf_probability": 0.25,
            "source_seed": 456,
        }
    }
    args = SimpleNamespace(
        julia="cli-julia",
        exajugo_root=str(tmp_path / "cli-exajugo"),
        exajugo_base_raw=str(tmp_path / "cli.raw"),
        exajugo_base_rop=str(tmp_path / "cli.rop"),
        acopf_timeout_s=11,
        acopf_probability=0.75,
        source_seed=789,
    )

    resolved = acopf.resolve_acopf_initialization_config(args, config, env=env)
    assert resolved.julia == "cli-julia"
    assert resolved.exajugo_root == tmp_path / "cli-exajugo"
    assert resolved.base_raw == tmp_path / "cli.raw"
    assert resolved.base_rop == tmp_path / "cli.rop"
    assert resolved.acopf_timeout_s == pytest.approx(11)
    assert resolved.acopf_probability == pytest.approx(0.75)
    assert resolved.source_seed == 789

    no_cli = SimpleNamespace(
        julia=None,
        exajugo_root=None,
        exajugo_base_raw=None,
        exajugo_base_rop=None,
        acopf_timeout_s=None,
        acopf_probability=None,
        source_seed=None,
    )
    resolved = acopf.resolve_acopf_initialization_config(no_cli, config, env=env)
    assert resolved.julia == "config-julia"
    assert resolved.exajugo_root == tmp_path / "config-exajugo"
    assert resolved.base_raw == tmp_path / "config.raw"
    assert resolved.base_rop == tmp_path / "config.rop"
    assert resolved.acopf_timeout_s == pytest.approx(123)
    assert resolved.acopf_probability == pytest.approx(0.25)
    assert resolved.source_seed == 456

    resolved = acopf.resolve_acopf_initialization_config(no_cli, {}, env=env)
    assert resolved.julia == "env-julia"
    assert resolved.exajugo_root == tmp_path / "env-exajugo"
    assert resolved.base_raw == tmp_path / "env.raw"
    assert resolved.base_rop == tmp_path / "env.rop"
    assert resolved.acopf_timeout_s == pytest.approx(300)
    assert resolved.acopf_probability == pytest.approx(1.0)
    assert resolved.source_seed == 1234

    direct_only = acopf.resolve_acopf_initialization_config(
        no_cli,
        {},
        env={},
        require_paths=False,
    )
    assert direct_only.exajugo_root is None
    assert direct_only.base_raw is None
    assert direct_only.base_rop is None


def _fake_import_func(missing=()):
    missing = set(missing)

    def import_func(name):
        if name in missing:
            raise ImportError(name)
        return SimpleNamespace(__version__="test")

    return import_func


def test_startup_preflight_success_for_acopf_inputs(acopf, tmp_path):
    config_path, _, _, base_raw, base_rop, _ = _production_config(tmp_path, target=1)
    config = acopf._load_json_config(config_path)
    config["integration"]["petsc"] = False
    exajugo_root = tmp_path / "exajugo"
    exajugo_root.mkdir()
    (exajugo_root / "ACOPF.jl").write_text("# acopf", encoding="utf-8")
    (exajugo_root / "Project.toml").write_text("[deps]\n", encoding="utf-8")
    julia = tmp_path / "julia"
    julia.write_text("#!/bin/sh\n", encoding="utf-8")
    julia.chmod(0o755)

    checks = acopf.collect_startup_preflight_checks(
        config_path=config_path,
        config=config,
        args=SimpleNamespace(uqgrid_root=None),
        acopf_config=acopf.AcopfInitializationConfig(
            julia=str(julia),
            exajugo_root=exajugo_root,
            base_raw=base_raw,
            base_rop=base_rop,
        ),
        require_acopf_paths=True,
        dynamic_replay=True,
        import_func=_fake_import_func(),
    )

    assert acopf._preflight_failures(checks) == []
    labels = {check.label for check in checks}
    assert {"UQGrid RAW", "UQGrid DYR", "ExaJuGO base RAW", "ExaJuGO base ROP"} <= labels


def test_startup_preflight_reports_missing_model_paths(acopf, tmp_path):
    config_path = tmp_path / "config.json"
    config = {
        "model": {
            "raw": str(tmp_path / "missing.raw"),
            "dyr": str(tmp_path / "missing.dyr"),
        },
        "integration": {"petsc": False},
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    checks = acopf.collect_startup_preflight_checks(
        config_path=config_path,
        config=config,
        args=SimpleNamespace(uqgrid_root=None),
        acopf_config=acopf.AcopfInitializationConfig(
            julia="julia",
            exajugo_root=None,
            base_raw=None,
            base_rop=None,
        ),
        require_acopf_paths=False,
        dynamic_replay=True,
        import_func=_fake_import_func(),
    )

    failures = acopf._preflight_failures(checks)
    assert {failure.label for failure in failures} == {"UQGrid RAW", "UQGrid DYR"}
    assert all("--uqgrid-root" in failure.fix for failure in failures)


def test_startup_preflight_exajugo_required_only_when_acopf_can_run(acopf, tmp_path):
    config_path, _, _, _, _, _ = _production_config(tmp_path, target=1)
    config = acopf._load_json_config(config_path)
    config["integration"]["petsc"] = False
    cfg = acopf.AcopfInitializationConfig(
        julia="julia",
        exajugo_root=None,
        base_raw=None,
        base_rop=None,
    )

    direct_checks = acopf.collect_startup_preflight_checks(
        config_path=config_path,
        config=config,
        args=SimpleNamespace(uqgrid_root=None),
        acopf_config=cfg,
        require_acopf_paths=False,
        dynamic_replay=True,
        import_func=_fake_import_func(),
    )
    assert acopf._preflight_failures(direct_checks) == []
    assert any(check.label == "ExaJuGO" and check.status == "skip" for check in direct_checks)

    acopf_checks = acopf.collect_startup_preflight_checks(
        config_path=config_path,
        config=config,
        args=SimpleNamespace(uqgrid_root=None),
        acopf_config=cfg,
        require_acopf_paths=True,
        dynamic_replay=True,
        import_func=_fake_import_func(),
        which_func=lambda executable: "/usr/bin/julia",
    )
    failure_labels = {failure.label for failure in acopf._preflight_failures(acopf_checks)}
    assert {
        "ExaJuGO root",
        "ExaJuGO ACOPF.jl",
        "ExaJuGO Project.toml",
        "ExaJuGO base RAW",
        "ExaJuGO base ROP",
    } <= failure_labels


def test_startup_preflight_petsc_checked_only_when_enabled(acopf, tmp_path):
    config_path, _, _, _, _, _ = _production_config(tmp_path, target=1)
    config = acopf._load_json_config(config_path)
    cfg = acopf.AcopfInitializationConfig(
        julia="julia",
        exajugo_root=None,
        base_raw=None,
        base_rop=None,
    )
    imported = []

    def import_func(name):
        imported.append(name)
        return SimpleNamespace(__version__="test")

    config["integration"]["petsc"] = False
    checks = acopf.collect_startup_preflight_checks(
        config_path=config_path,
        config=config,
        args=SimpleNamespace(uqgrid_root=None),
        acopf_config=cfg,
        require_acopf_paths=False,
        dynamic_replay=True,
        import_func=import_func,
    )
    assert acopf._preflight_failures(checks) == []
    assert "petsc4py" not in imported
    assert any(check.label == "PETSc" and check.status == "skip" for check in checks)

    imported.clear()
    config["integration"]["petsc"] = True
    checks = acopf.collect_startup_preflight_checks(
        config_path=config_path,
        config=config,
        args=SimpleNamespace(uqgrid_root=None),
        acopf_config=cfg,
        require_acopf_paths=False,
        dynamic_replay=True,
        import_func=import_func,
    )
    assert acopf._preflight_failures(checks) == []
    assert "petsc4py" in imported


def test_write_exajugo_smoke_case_patches_raw_and_copies_rop(acopf, tmp_path):
    base_raw = tmp_path / "base.raw"
    base_rop = tmp_path / "base.rop"
    base_raw.write_text(_raw_text(), encoding="utf-8")
    base_rop.write_text("rop", encoding="utf-8")
    cfg = acopf.AcopfInitializationConfig(
        julia="julia",
        exajugo_root=tmp_path / "exajugo",
        base_raw=base_raw,
        base_rop=base_rop,
    )
    operating_point = {
        "p_load_scaled": np.asarray([0.11, 0.22]),
        "q_load_scaled": np.asarray([-0.03, -0.04]),
        "operating_point_id": "op",
        "accepted_operating_point_index": 0,
        "diagnostics": {"accepted": True},
    }

    info = acopf.write_exajugo_smoke_case(
        tmp_path / "out",
        sample_idx=7,
        acopf_config=cfg,
        operating_point=operating_point,
    )

    case_dir = tmp_path / "out" / "acopf_smoke" / "op_7"
    assert info["case_dir"] == str(case_dir)
    assert (case_dir / "case.rop").read_text(encoding="utf-8") == "rop"
    patched = (case_dir / "case.raw").read_text(encoding="utf-8")
    assert "11.000000, 3.000000" in patched
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["n_load_rows"] == 2
    assert metadata["total_q_load_raw_pu"] == pytest.approx(0.07)


def _minimal_case_dir(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir(parents=True)
    (case_dir / "case.raw").write_text(_raw_text(), encoding="utf-8")
    (case_dir / "case.rop").write_text("rop", encoding="utf-8")
    return case_dir


def test_run_exajugo_acopf_success_nonzero_missing_and_timeout(acopf, tmp_path):
    cfg = acopf.AcopfInitializationConfig(
        julia="julia",
        exajugo_root=tmp_path / "exajugo",
        base_raw=tmp_path / "base.raw",
        base_rop=tmp_path / "base.rop",
        acopf_timeout_s=9,
    )

    case_dir = _minimal_case_dir(tmp_path)

    def success_run(cmd, timeout, capture_output, text):
        assert timeout == pytest.approx(9)
        assert capture_output is True
        assert text is True
        system_dir = os.path.abspath(cmd[-1])
        os.makedirs(system_dir, exist_ok=True)
        with open(os.path.join(system_dir, "Basecase_solution.txt"), "w", encoding="utf-8") as f:
            f.write(_basecase_text())
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    result = acopf.run_exajugo_acopf(case_dir, cfg, subprocess_run=success_run)
    assert result["success"] is True
    assert result["returncode"] == 0
    assert (case_dir / "acopf_stdout.txt").read_text(encoding="utf-8") == "ok"

    failed_dir = _minimal_case_dir(tmp_path / "failed")
    result = acopf.run_exajugo_acopf(
        failed_dir,
        cfg,
        subprocess_run=lambda cmd, timeout, capture_output, text: subprocess.CompletedProcess(
            cmd, 2, "", "bad"
        ),
    )
    assert result["success"] is False
    assert result["reject_reason"] == "acopf_nonzero_exit"

    missing_dir = _minimal_case_dir(tmp_path / "missing")
    result = acopf.run_exajugo_acopf(
        missing_dir,
        cfg,
        subprocess_run=lambda cmd, timeout, capture_output, text: subprocess.CompletedProcess(
            cmd, 0, "", ""
        ),
    )
    assert result["success"] is False
    assert result["reject_reason"] == "acopf_missing_basecase"

    timeout_dir = _minimal_case_dir(tmp_path / "timeout")

    def timeout_run(cmd, timeout, capture_output, text):
        raise subprocess.TimeoutExpired(cmd, timeout, output="partial", stderr="slow")

    result = acopf.run_exajugo_acopf(timeout_dir, cfg, subprocess_run=timeout_run)
    assert result["success"] is False
    assert result["reject_reason"] == "acopf_timeout"
    assert (timeout_dir / "acopf_stdout.txt").read_text(encoding="utf-8") == "partial"


def test_validate_acopf_power_flow_accepts_and_rejects_nonconvergence(acopf, tmp_path):
    basecase = tmp_path / "Basecase_solution.txt"
    case_raw = tmp_path / "case.raw"
    raw = tmp_path / "uqgrid.raw"
    dyr = tmp_path / "case.dyr"
    basecase.write_text(_basecase_text(include_load_section=True), encoding="utf-8")
    case_raw.write_text(_raw_text(), encoding="utf-8")
    raw.write_text(_raw_text(), encoding="utf-8")
    dyr.write_text("", encoding="utf-8")

    def add_dyr(psys, path):
        psys.dyr_path = path

    psys = _DummyPsys()
    summary = acopf.validate_acopf_power_flow(
        raw_path=raw,
        dyr_path=dyr,
        basecase_path=basecase,
        case_raw_path=case_raw,
        pf_residual_tol=1e-8,
        load_psse_func=lambda path: psys,
        add_dyr_func=add_dyr,
        runpf_func=lambda psys, verbose=False: SimpleNamespace(
            v_magnitudes=np.asarray([1.0, 1.02]),
            converged=True,
            final_residual_norm=1e-10,
            iterations=3,
        ),
    )
    assert summary["success"] is True
    assert summary["adjusted_shunts"] == 1
    assert summary["voltage_min_pu"] == pytest.approx(1.0)
    assert psys.ybus_created is True

    summary = acopf.validate_acopf_power_flow(
        raw_path=raw,
        dyr_path=dyr,
        basecase_path=basecase,
        case_raw_path=case_raw,
        pf_residual_tol=1e-8,
        load_psse_func=lambda path: _DummyPsys(),
        add_dyr_func=add_dyr,
        runpf_func=lambda psys, verbose=False: SimpleNamespace(
            v_magnitudes=np.asarray([1.0, 1.02]),
            converged=False,
            final_residual_norm=1e-10,
        ),
    )
    assert summary["success"] is False
    assert summary["reject_reason"] == "post_acopf_pf_not_converged"


def _smoke_config(tmp_path):
    raw = tmp_path / "model.raw"
    dyr = tmp_path / "model.dyr"
    base_raw = tmp_path / "base.raw"
    base_rop = tmp_path / "base.rop"
    raw.write_text(_raw_text(), encoding="utf-8")
    dyr.write_text("", encoding="utf-8")
    base_raw.write_text(_raw_text(), encoding="utf-8")
    base_rop.write_text("rop", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"raw": str(raw), "dyr": str(dyr), "n_bus": 2},
                "perturbation": {"load_noise_type": "normal", "load_noise_var": 0.1},
                "operating_point": {"pf_residual_tol": 1e-8},
            }
        ),
        encoding="utf-8",
    )
    acopf_config = None
    return config_path, raw, dyr, base_raw, base_rop, acopf_config


def _production_config(tmp_path, *, n_bus=2, target=2, max_total_attempts=10):
    raw = tmp_path / "model.raw"
    dyr = tmp_path / "model.dyr"
    base_raw = tmp_path / "base.raw"
    base_rop = tmp_path / "base.rop"
    raw.write_text(_raw_text(), encoding="utf-8")
    dyr.write_text("", encoding="utf-8")
    base_raw.write_text(_raw_text(), encoding="utf-8")
    base_rop.write_text("rop", encoding="utf-8")
    config_path = tmp_path / "config_production.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"name": "TestGrid", "raw": str(raw), "dyr": str(dyr), "n_bus": n_bus},
                "scenarios": {
                    "samples_per_fault_location": 1,
                    "fault_locations": "all",
                    "fault_impedances": [1e-4],
                    "target_accepted_scenarios": target,
                    "max_total_attempts": max_total_attempts,
                },
                "execution": {"n_jobs": 64, "batch_size": 128, "checkpoint_interval": 1},
                "perturbation": {"load_noise_type": "normal", "load_noise_var": 0.1},
                "operating_point": {"pf_residual_tol": 1e-8, "max_attempts_per_scenario": 5},
                "integration": {"toff": 0.3833333333333333},
            }
        ),
        encoding="utf-8",
    )
    cfg = None
    return config_path, raw, dyr, base_raw, base_rop, cfg


def _candidate_success(*args, **kwargs):
    return {
        "diagnostics_attempts": [
            {
                "record_type": "operating_point_attempt",
                "accepted": True,
                "attempts": 1,
                "pf_residual": 1e-10,
            }
        ],
        "diagnostics": {"accepted": True, "attempts": 1, "pf_residual": 1e-10},
        "operating_point": {
            "p_load_scaled": np.asarray([0.11, 0.22]),
            "q_load_scaled": np.asarray([-0.03, -0.04]),
            "operating_point_id": "op",
            "accepted_operating_point_index": 0,
            "diagnostics": {"accepted": True},
        },
    }


def test_run_acopf_smoke_success_writes_diagnostics(acopf, tmp_path):
    config_path, _, _, base_raw, base_rop, _ = _smoke_config(tmp_path)
    cfg = acopf.AcopfInitializationConfig(
        julia="julia",
        exajugo_root=tmp_path / "exajugo",
        base_raw=base_raw,
        base_rop=base_rop,
    )

    def acopf_runner(case_dir, acopf_config):
        basecase = tmp_path / "Basecase_solution.txt"
        basecase.write_text(_basecase_text(include_load_section=True), encoding="utf-8")
        return {"success": True, "accepted": True, "basecase_path": str(basecase)}

    progress = acopf.run_acopf_smoke(
        config_path=config_path,
        output_dir=tmp_path / "out",
        acopf_config=cfg,
        candidate_func=_candidate_success,
        op_config_resolver=lambda cfg: {"pf_residual_tol": 1e-8},
        acopf_runner_func=acopf_runner,
        pf_validator_func=lambda **kwargs: {
            "success": True,
            "accepted": True,
            "reject_reason": None,
        },
    )

    assert progress["accepted"] is True
    progress_json = json.loads(
        (tmp_path / "out" / "acopf_init_progress.json").read_text(encoding="utf-8")
    )
    assert progress_json["accepted_count"] == 1
    diagnostics = (tmp_path / "out" / "acopf_init_diagnostics.jsonl").read_text(
        encoding="utf-8"
    )
    assert "acopf_smoke" in diagnostics


@pytest.mark.parametrize(
    ("candidate_func", "acopf_runner", "pf_validator", "expected_reason"),
    [
        (
            lambda *args, **kwargs: {
                "rejected": True,
                "diagnostics": {"accepted": False, "reject_reason": "pf_non_converged"},
                "diagnostics_attempts": [],
            },
            lambda case_dir, cfg: {"success": True, "basecase_path": "unused"},
            lambda **kwargs: {"success": True},
            "pf_non_converged",
        ),
        (
            _candidate_success,
            lambda case_dir, cfg: {
                "success": False,
                "accepted": False,
                "reject_reason": "acopf_nonzero_exit",
            },
            lambda **kwargs: {"success": True},
            "acopf_nonzero_exit",
        ),
        (
            _candidate_success,
            lambda case_dir, cfg: {
                "success": True,
                "accepted": True,
                "basecase_path": "missing",
            },
            lambda **kwargs: {
                "success": False,
                "accepted": False,
                "reject_reason": "post_acopf_pf_error",
            },
            "post_acopf_pf_error",
        ),
        (
            _candidate_success,
            lambda case_dir, cfg: {
                "success": True,
                "accepted": True,
                "basecase_path": "basecase",
            },
            lambda **kwargs: {
                "success": False,
                "accepted": False,
                "reject_reason": "post_acopf_pf_not_converged",
            },
            "post_acopf_pf_not_converged",
        ),
    ],
)
def test_run_acopf_smoke_rejects_expected_stages(
    acopf,
    tmp_path,
    candidate_func,
    acopf_runner,
    pf_validator,
    expected_reason,
):
    config_path, _, _, base_raw, base_rop, _ = _smoke_config(tmp_path)
    cfg = acopf.AcopfInitializationConfig(
        julia="julia",
        exajugo_root=tmp_path / "exajugo",
        base_raw=base_raw,
        base_rop=base_rop,
    )

    progress = acopf.run_acopf_smoke(
        config_path=config_path,
        output_dir=tmp_path / "out",
        acopf_config=cfg,
        candidate_func=candidate_func,
        op_config_resolver=lambda cfg: {"pf_residual_tol": 1e-8},
        acopf_runner_func=acopf_runner,
        pf_validator_func=pf_validator,
    )

    assert progress["accepted"] is False
    assert progress["reject_reason"] == expected_reason


def test_stage3_cli_list_parsing_and_effective_jobs(acopf):
    assert acopf.parse_int_list("142,143") == [142, 143]
    assert acopf.parse_float_list("1e-4, 0.2") == [pytest.approx(1e-4), pytest.approx(0.2)]
    assert acopf.parse_int_list(None, default=[1, 2]) == [1, 2]
    assert acopf.resolve_fault_locations("all", n_bus=3) == [0, 1, 2]
    assert acopf._effective_n_jobs(64, 2) == 2
    assert acopf._effective_n_jobs(-1, 2) == 2
    assert acopf._effective_n_jobs(1, 2) == 1


def test_mixed_source_probability_parsing_and_selection(acopf):
    assert acopf.parse_probability(None) == pytest.approx(1.0)
    assert acopf.parse_probability("0.25") == pytest.approx(0.25)
    with pytest.raises(ValueError, match="between 0 and 1"):
        acopf.parse_probability(1.1)

    assert acopf.select_initialization_source(7, acopf_probability=1.0) == "acopf"
    assert acopf.select_initialization_source(7, acopf_probability=0.0) == "uqgrid_pf"
    first = acopf.select_initialization_source(
        7,
        acopf_probability=0.5,
        source_seed=99,
    )
    second = acopf.select_initialization_source(
        7,
        acopf_probability=0.5,
        source_seed=99,
    )
    assert first == second


def test_production_progress_prints_percent_rate_elapsed_and_eta(
    acopf,
    monkeypatch,
    capsys,
):
    progress = acopf._production_progress(
        target_accepted_scenarios=10,
        accepted_count=2,
        next_sample_idx=4,
        last_sample_idx=3,
        total_candidate_attempts=5,
        max_total_attempts=100,
        records_written=42,
        fault_rows_completed=1000,
        source_counts={"uqgrid_pf": 1, "acopf": 1},
    )
    monkeypatch.setattr(acopf.time, "time", lambda: 1120.0)

    acopf._print_production_progress(
        progress,
        start_time=1000.0,
        start_accepted_count=1,
    )

    out = capsys.readouterr().out.strip()
    assert "accepted OPs 2/10 (20.0%)" in out
    assert "candidate attempts 5/100" in out
    assert "OP acceptance/attempt 40.0%" in out
    assert "accepted OP rate 30.00/hr" in out
    assert "fault rows 1000" in out
    assert "diagnostics 42" in out
    assert "sources acopf=1, uqgrid_pf=1" in out
    assert "elapsed 0:02:00" in out
    assert "ETA 0:16:00" in out


def test_export_delta_state_metadata_uses_injected_uqgrid(acopf, tmp_path):
    psys = _DummyPsys()

    result = acopf.export_delta_state_metadata(
        raw_path=tmp_path / "case.raw",
        dyr_path=tmp_path / "case.dyr",
        output_path=tmp_path / "state_metadata.json",
        load_psse_func=lambda path: psys,
        add_dyr_func=lambda psys, path: None,
    )

    assert result["delta_state_indices"] == [0, 2]
    assert result["delta_state_count"] == 2
    assert os.path.exists(result["state_metadata_path"])


class _FakeIntegrationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_integration_config_adapter_preserves_pf_and_dynamic_limit_contract(acopf):
    validation = {
        "enabled": True,
        "residual_tolerance": 2e-9,
        "generator_limit_tolerance": 3e-7,
        "voltage_min": 0.92,
        "voltage_max": 1.08,
        "branch_loading_max": 0.95,
        "branch_limit_tolerance": 4e-6,
        "active_set_voltage_tolerance": 5e-7,
    }
    adapted = acopf._integration_config_from_config(
        {
            "integration": {
                "steps": 25,
                "method": "cn",
                "arkimex": False,
                "enforce_q_limits": True,
                "q_limit_tolerance": 6e-8,
                "max_q_limit_iterations": 12,
                "power_flow_validation": validation,
                "enforce_dynamic_limits": False,
                "dynamic_limit_tolerance": 7e-8,
                "dynamic_limit_release_tolerance": 8e-10,
                "max_dynamic_limit_iterations": 13,
            }
        }
    )

    assert adapted["enforce_q_limits"] is True
    assert adapted["steps"] == 25
    assert adapted["method"] == "cn"
    assert adapted["arkimex"] is False
    assert adapted["q_limit_tolerance"] == pytest.approx(6e-8)
    assert adapted["max_q_limit_iterations"] == 12
    assert adapted["power_flow_validation"] == validation
    assert adapted["power_flow_validation"] is not validation
    assert adapted["enforce_dynamic_limits"] is False
    assert adapted["dynamic_limit_tolerance"] == pytest.approx(7e-8)
    assert adapted["dynamic_limit_release_tolerance"] == pytest.approx(8e-10)
    assert adapted["max_dynamic_limit_iterations"] == 13

    defaults = acopf._integration_config_from_config({})
    assert defaults["enforce_q_limits"] is True
    assert defaults["steps"] == -1
    assert defaults["method"] == "beuler"
    assert defaults["arkimex"] is False
    assert defaults["q_limit_tolerance"] == pytest.approx(1e-8)
    assert defaults["max_q_limit_iterations"] is None
    assert defaults["power_flow_validation"] == {}
    assert defaults["enforce_dynamic_limits"] is True
    assert defaults["dynamic_limit_tolerance"] == pytest.approx(1e-8)
    assert defaults["dynamic_limit_release_tolerance"] == pytest.approx(1e-10)
    assert defaults["max_dynamic_limit_iterations"] == 20

    disabled = acopf._integration_config_from_config(
        {
            "integration": {
                "enforce_q_limits": False,
                "enforce_dynamic_limits": False,
            }
        }
    )
    assert disabled["enforce_q_limits"] is False
    assert disabled["enforce_dynamic_limits"] is False


def _replay_context(tmp_path, keep_fault_histories=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    basecase = tmp_path / "Basecase_solution.txt"
    case_raw = tmp_path / "case.raw"
    raw = tmp_path / "uqgrid.raw"
    dyr = tmp_path / "case.dyr"
    basecase.write_text(_basecase_text(include_load_section=True), encoding="utf-8")
    case_raw.write_text(_raw_text(), encoding="utf-8")
    raw.write_text(_raw_text(), encoding="utf-8")
    dyr.write_text("", encoding="utf-8")
    return {
        "raw_path": str(raw),
        "dyr_path": str(dyr),
        "basecase_path": str(basecase),
        "case_raw_path": str(case_raw),
        "integration_config": {
            "tend": 10.0,
            "dt": 1 / 120,
            "power_injection": False,
            "ton": 0.25,
            "toff": 0.3833333333333333,
            "verbose": False,
            "petsc": True,
        },
        "delta_state_indices": [0, 2],
        "keep_fault_histories": keep_fault_histories,
        "history_dir": str(tmp_path / "fault_histories"),
        "debug_tracebacks": False,
    }


def _direct_replay_context(tmp_path, keep_fault_histories=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = tmp_path / "uqgrid.raw"
    dyr = tmp_path / "case.dyr"
    raw.write_text(_raw_text(), encoding="utf-8")
    dyr.write_text("", encoding="utf-8")
    return {
        "raw_path": str(raw),
        "dyr_path": str(dyr),
        "operating_point": {
            "p_load_scaled": np.asarray([0.11, 0.22]),
            "q_load_scaled": np.asarray([-0.03, -0.04]),
            "p_gen_scaled": np.asarray([0.50, 0.60]),
            "q_gen_scaled": np.asarray([0.10, 0.12]),
            "pf_v_magnitudes": np.asarray([1.01, 0.99]),
            "pf_v_angles": np.asarray([0.1, -0.2]),
        },
        "integration_config": {
            "tend": 10.0,
            "dt": 1 / 120,
            "power_injection": False,
            "ton": 0.25,
            "toff": 0.3833333333333333,
            "verbose": False,
            "petsc": True,
        },
        "delta_state_indices": [0, 2],
        "keep_fault_histories": keep_fault_histories,
        "history_dir": str(tmp_path / "fault_histories"),
        "debug_tracebacks": False,
    }


def _fault_task(acopf, scenario_id="sid"):
    return acopf.FaultReplayTask(
        sample_idx=0,
        operating_point_id="op",
        accepted_operating_point_index=0,
        fault_location=142,
        fault_impedance=1e-4,
        fault_location_index=0,
        fault_impedance_index=0,
        scenario_id=scenario_id,
    )


def test_replay_workers_share_pf_contract_and_success_diagnostics(acopf, tmp_path):
    validation = {
        "valid": True,
        "failure_reasons": [],
        "residual_norm": 1e-12,
    }
    dynamic_limits = {
        "enabled": True,
        "enabled_state_count": 3,
        "initialization": {"valid": True, "failure_reasons": []},
        "events": [{"action": "activate", "side": "upper"}],
    }
    integration_config = {
        "tend": 10.0,
        "dt": 1 / 120,
        "steps": -1,
        "method": "cn",
        "arkimex": False,
        "power_injection": False,
        "ton": 0.25,
        "toff": 0.4,
        "verbose": False,
        "petsc": True,
        "enforce_q_limits": True,
        "q_limit_tolerance": 2e-8,
        "max_q_limit_iterations": 9,
        "power_flow_validation": {"enabled": True},
        "enforce_dynamic_limits": True,
        "dynamic_limit_tolerance": 3e-8,
        "dynamic_limit_release_tolerance": 4e-10,
        "max_dynamic_limit_iterations": 11,
    }
    worker_contexts = [
        (
            acopf.replay_acopf_fault_task,
            _replay_context(tmp_path / "acopf"),
        ),
        (
            acopf.replay_uqgrid_fault_task,
            _direct_replay_context(tmp_path / "direct"),
        ),
    ]

    for index, (worker, context) in enumerate(worker_contexts):
        context["integration_config"] = dict(integration_config)
        captured = []

        def integrate_system(psys, cfg):
            captured.append(cfg.kwargs)
            return {
                "history": np.asarray(
                    [[0.0, 0.0], [9.0, 9.0], [0.0, 1.0]]
                ),
                "tvec": np.asarray([0.0, 1.0]),
                "power_flow_diagnostics": validation,
                "dynamic_limit_diagnostics": dynamic_limits,
            }

        result = worker(
            _fault_task(acopf, scenario_id=f"sid-{index}"),
            context,
            load_psse_func=lambda path: _DummyPsys(),
            add_dyr_func=lambda psys, path: None,
            integration_config_cls=_FakeIntegrationConfig,
            integrate_system_func=integrate_system,
        )

        assert result["accepted"] is True
        assert result["power_flow_validation"] == validation
        assert result["dynamic_limit_diagnostics"] == dynamic_limits
        assert captured == [integration_config]


def test_replay_workers_preserve_power_flow_validation_failure(acopf, tmp_path):
    class PowerFlowValidationError(RuntimeError):
        def __init__(self, diagnostics):
            self.diagnostics = diagnostics
            super().__init__("Power-flow operating-point validation failed")

    diagnostics = {
        "valid": False,
        "failure_reasons": ["gen_q_limit"],
        "gen_q": {"violation_count": 1},
    }
    worker_contexts = [
        (
            acopf.replay_acopf_fault_task,
            _replay_context(tmp_path / "acopf"),
        ),
        (
            acopf.replay_uqgrid_fault_task,
            _direct_replay_context(tmp_path / "direct"),
        ),
    ]

    def fail_validation(psys, cfg):
        raise PowerFlowValidationError(diagnostics)

    for index, (worker, context) in enumerate(worker_contexts):
        result = worker(
            _fault_task(acopf, scenario_id=f"failed-{index}"),
            context,
            load_psse_func=lambda path: _DummyPsys(),
            add_dyr_func=lambda psys, path: None,
            integration_config_cls=_FakeIntegrationConfig,
            integrate_system_func=fail_validation,
        )

        assert result["accepted"] is False
        assert result["reject_reason"] == "power_flow_validation_failed"
        assert result["reject_stage"] == "power_flow_validation"
        assert result["power_flow_validation"] == diagnostics
        assert result["simulation_diverged"] is False
        assert "tsi_final" not in result
        assert not os.path.exists(context["history_dir"])


@pytest.mark.parametrize(
    ("phase", "reject_reason", "reject_stage", "simulation_diverged"),
    [
        (
            None,
            "dynamic_limit_initialization_failed",
            "dynamic_limit_initialization",
            False,
        ),
        (
            "runtime",
            "dynamic_limit_runtime_failed",
            "dynamic_limit_runtime",
            True,
        ),
    ],
)
def test_replay_workers_preserve_dynamic_limit_failures(
    acopf,
    tmp_path,
    phase,
    reject_reason,
    reject_stage,
    simulation_diverged,
):
    class DynamicLimitError(RuntimeError):
        def __init__(self, diagnostics):
            self.diagnostics = diagnostics
            super().__init__("Dynamic-limit failure")

    diagnostics = {
        "enabled": True,
        "failure_reasons": ["active_set_cycle"],
        "events": [],
    }
    if phase is not None:
        diagnostics["phase"] = phase

    def fail_dynamic_limits(psys, cfg):
        raise DynamicLimitError(diagnostics)

    workers = [
        (acopf.replay_acopf_fault_task, _replay_context(tmp_path / "acopf")),
        (
            acopf.replay_uqgrid_fault_task,
            _direct_replay_context(tmp_path / "direct"),
        ),
    ]
    for index, (worker, context) in enumerate(workers):
        result = worker(
            _fault_task(acopf, scenario_id=f"dynamic-limit-{index}"),
            context,
            load_psse_func=lambda path: _DummyPsys(),
            add_dyr_func=lambda psys, path: None,
            integration_config_cls=_FakeIntegrationConfig,
            integrate_system_func=fail_dynamic_limits,
        )

        assert result["accepted"] is False
        assert result["reject_reason"] == reject_reason
        assert result["reject_stage"] == reject_stage
        assert result["dynamic_limit_diagnostics"] == diagnostics
        assert result["simulation_diverged"] is simulation_diverged
        assert "tsi_final" not in result
        assert not os.path.exists(context["history_dir"])


def test_fault_result_log_preserves_successful_power_flow_validation(acopf):
    validation = {"valid": True, "failure_reasons": [], "residual_norm": 1e-12}
    dynamic_limits = {
        "enabled": True,
        "enabled_state_count": 3,
        "initialization": {"valid": True},
        "events": [],
    }
    fault_result = {
        "scenario_id": "sid",
        "fault_location": 142,
        "fault_impedance": 1e-4,
        "initialization_source": "acopf",
        "acopf_feasible": True,
        "tsi_final": 10.0,
        "tsi_min": 5.0,
        "history_file": None,
        "final_simulation_time_s": 10.0,
        "power_flow_validation": validation,
        "dynamic_limit_diagnostics": dynamic_limits,
    }

    metadata, simulation_log = acopf._fault_result_entries(
        [fault_result],
        sample_idx=0,
        operating_point_id="op",
        accepted_operating_point_index=0,
    )

    assert "power_flow_validation" not in metadata["sid"]
    assert "dynamic_limit_diagnostics" not in metadata["sid"]
    assert simulation_log["sid"]["power_flow_validation"] == validation
    assert simulation_log["sid"]["dynamic_limit_diagnostics"] == dynamic_limits


def test_replay_acopf_fault_task_success_and_history_default(acopf, tmp_path):
    psys = _DummyPsys()
    task = acopf.FaultReplayTask(
        sample_idx=0,
        operating_point_id="op",
        accepted_operating_point_index=0,
        fault_location=142,
        fault_impedance=1e-4,
        fault_location_index=0,
        fault_impedance_index=0,
        scenario_id="sid",
    )

    result = acopf.replay_acopf_fault_task(
        task,
        _replay_context(tmp_path),
        load_psse_func=lambda path: psys,
        add_dyr_func=lambda psys, path: None,
        integration_config_cls=_FakeIntegrationConfig,
        integrate_system_func=lambda psys, cfg: {
            "history": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [9.0, 9.0, 9.0],
                    [0.0, 3.0, 1.0],
                ]
            ),
            "tvec": np.asarray([0.0, 0.5, 1.0]),
        },
    )

    assert result["accepted"] is True
    assert result["reject_reason"] is None
    assert result["tsi_min"] < result["tsi_final"]
    assert result["history_file"] is None
    assert psys.faults == [(142, 1e-4)]
    assert not (tmp_path / "fault_histories").exists()


def test_replay_uqgrid_fault_task_applies_operating_point_and_no_history(acopf, tmp_path):
    psys = _DummyPsys()
    task = acopf.FaultReplayTask(
        sample_idx=0,
        operating_point_id="op",
        accepted_operating_point_index=0,
        fault_location=142,
        fault_impedance=1e-4,
        fault_location_index=0,
        fault_impedance_index=0,
        scenario_id="sid",
    )

    result = acopf.replay_uqgrid_fault_task(
        task,
        _direct_replay_context(tmp_path),
        load_psse_func=lambda path: psys,
        add_dyr_func=lambda psys, path: None,
        integration_config_cls=_FakeIntegrationConfig,
        integrate_system_func=lambda psys, cfg: {
            "history": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [9.0, 9.0, 9.0],
                    [0.0, 3.0, 1.0],
                ]
            ),
            "tvec": np.asarray([0.0, 0.5, 1.0]),
        },
    )

    assert result["accepted"] is True
    assert result["initialization_source"] == "uqgrid_pf"
    assert result["acopf_feasible"] is False
    np.testing.assert_allclose(psys.load_pq[0], [0.11, 0.22])
    np.testing.assert_allclose(psys.gen_pq[1], [0.10, 0.12])
    assert psys.buses[0].v0m == pytest.approx(1.01)
    assert psys.buses[1].v0a == pytest.approx(-0.2)
    assert psys.faults == [(142, 1e-4)]
    assert result["history_file"] is None
    assert not (tmp_path / "fault_histories").exists()


def test_replay_acopf_fault_task_debug_history_and_failure(acopf, tmp_path):
    task = acopf.FaultReplayTask(
        sample_idx=0,
        operating_point_id="op",
        accepted_operating_point_index=0,
        fault_location=143,
        fault_impedance=1e-4,
        fault_location_index=1,
        fault_impedance_index=0,
        scenario_id="sid_debug",
    )

    result = acopf.replay_acopf_fault_task(
        task,
        _replay_context(tmp_path, keep_fault_histories=True),
        load_psse_func=lambda path: _DummyPsys(),
        add_dyr_func=lambda psys, path: None,
        integration_config_cls=_FakeIntegrationConfig,
        integrate_system_func=lambda psys, cfg: {
            "history": np.asarray([[0.0, 0.0], [9.0, 9.0], [0.0, 1.0]]),
            "tvec": np.asarray([0.0, 1.0]),
        },
    )
    assert result["accepted"] is True
    assert result["history_file"] is not None
    assert os.path.exists(result["history_file"])

    failure = acopf.replay_acopf_fault_task(
        task,
        _replay_context(tmp_path / "failed"),
        load_psse_func=lambda path: _DummyPsys(),
        add_dyr_func=lambda psys, path: None,
        integration_config_cls=_FakeIntegrationConfig,
        integrate_system_func=lambda psys, cfg: {"history": None, "tvec": None},
    )
    assert failure["accepted"] is False
    assert failure["reject_reason"] == "dynamic_fault_failed"


def _large_parsed_basecase(acopf):
    n_gen = 56
    n_load = 206
    return acopf.ParsedExaJuGOBasecase(
        bus_ids=np.arange(1, n_load + 1, dtype=np.int64),
        bus_v_pu=np.ones(n_load),
        bus_theta_deg=np.zeros(n_load),
        bus_bcs_mvar=np.zeros(n_load),
        load_bus_ids=np.arange(1, n_load + 1, dtype=np.int64),
        p_load_mw=np.linspace(10.0, 20.0, n_load),
        q_load_mvar=np.linspace(1.0, 2.0, n_load),
        gen_bus_ids=np.arange(1, n_gen + 1, dtype=np.int64),
        gen_ids=[str(i) for i in range(n_gen)],
        p_gen_mw=np.linspace(50.0, 60.0, n_gen),
        q_gen_mvar=np.linspace(5.0, 6.0, n_gen),
        nonzero_gen_mask=np.ones(n_gen, dtype=bool),
    )


def test_run_acopf_replay_smoke_writes_one_row_probml(acopf, tmp_path):
    config_path, raw, dyr, _, _, _ = _smoke_config(tmp_path)
    parsed = _large_parsed_basecase(acopf)

    def context_func(**kwargs):
        return {
            "accepted": True,
            "records": [{"record_type": "acopf_smoke", "accepted": True}],
            "raw_path": raw,
            "dyr_path": dyr,
            "case_raw_path": tmp_path / "case.raw",
            "basecase_path": tmp_path / "Basecase_solution.txt",
            "parsed_basecase": parsed,
            "sample_idx": 0,
            "operating_point_id": "op",
            "accepted_operating_point_index": 0,
            "base_mva": 100.0,
        }

    def fault_runner(tasks, context, n_jobs, parallel_timeout_s):
        assert n_jobs == 2
        assert parallel_timeout_s == pytest.approx(600.0)
        return [
            {
                "record_type": "fault_scenario",
                "accepted": True,
                "reject_reason": None,
                "scenario_id": task.scenario_id,
                "fault_location_index": task.fault_location_index,
                "fault_impedance_index": task.fault_impedance_index,
                "fault_location": task.fault_location,
                "fault_impedance": task.fault_impedance,
                "tsi_final": 10.0 + task.fault_location_index,
                "tsi_min": 5.0 + task.fault_location_index,
            }
            for task in tasks
        ]

    progress = acopf.run_acopf_replay_smoke(
        config_path=config_path,
        output_dir=tmp_path / "out",
        acopf_config=acopf.AcopfInitializationConfig(
            julia="julia",
            exajugo_root=tmp_path / "exajugo",
            base_raw=tmp_path / "base.raw",
            base_rop=tmp_path / "base.rop",
        ),
        fault_locations="142,143",
        fault_impedances="1e-4",
        n_jobs=64,
        probml_basename="stage3_probml",
        acopf_context_func=context_func,
        state_metadata_func=lambda **kwargs: {"delta_state_indices": [0, 2]},
        fault_runner_func=fault_runner,
    )

    assert progress["accepted"] is True
    final_path = tmp_path / "out" / "stage3_probml_final.npz"
    min_path = tmp_path / "out" / "stage3_probml_min.npz"
    with np.load(final_path, allow_pickle=True) as data:
        assert data["X"].shape == (1, 2, 262)
        assert data["X_flat"].shape == (1, 524)
        assert data["Y"].shape == (1, 2, 1)
        assert data["scenario_ids"].shape == (1, 2, 1)
    with np.load(min_path, allow_pickle=True) as data:
        assert data["Y"].shape == (1, 2, 1)
        assert data["meta"][0]["tsi_mode"] == "min"


def test_replay_smoke_preserves_power_flow_validation_rejection(acopf, tmp_path):
    config_path, raw, dyr, base_raw, base_rop, _ = _smoke_config(tmp_path)
    parsed = _large_parsed_basecase(acopf)

    def context_func(**kwargs):
        return {
            "accepted": True,
            "records": [{"record_type": "acopf_smoke", "accepted": True}],
            "raw_path": raw,
            "dyr_path": dyr,
            "case_raw_path": tmp_path / "case.raw",
            "basecase_path": tmp_path / "Basecase_solution.txt",
            "parsed_basecase": parsed,
            "sample_idx": 0,
            "operating_point_id": "op",
            "accepted_operating_point_index": 0,
            "base_mva": 100.0,
        }

    def failed_fault_runner(tasks, context, n_jobs, parallel_timeout_s):
        return [
            {
                "record_type": "fault_scenario",
                "accepted": False,
                "reject_reason": "power_flow_validation_failed",
                "reject_stage": "power_flow_validation",
                "power_flow_validation": {
                    "valid": False,
                    "failure_reasons": ["gen_q_limit"],
                },
                "scenario_id": task.scenario_id,
                "fault_location_index": task.fault_location_index,
                "fault_impedance_index": task.fault_impedance_index,
                "fault_location": task.fault_location,
                "fault_impedance": task.fault_impedance,
            }
            for task in tasks
        ]

    output_dir = tmp_path / "out"
    progress = acopf.run_acopf_replay_smoke(
        config_path=config_path,
        output_dir=output_dir,
        acopf_config=acopf.AcopfInitializationConfig(
            julia="julia",
            exajugo_root=tmp_path / "exajugo",
            base_raw=base_raw,
            base_rop=base_rop,
        ),
        fault_locations="142,143",
        fault_impedances="1e-4",
        probml_basename="stage3_probml",
        acopf_context_func=context_func,
        state_metadata_func=lambda **kwargs: {"delta_state_indices": [0, 2]},
        fault_runner_func=failed_fault_runner,
    )

    assert progress["accepted"] is False
    assert progress["reject_reason"] == "power_flow_validation_failed"
    assert not (output_dir / "stage3_probml_final.npz").exists()
    records = acopf._read_jsonl(output_dir / "acopf_init_diagnostics.jsonl")
    group = records[-1]
    assert group["reject_reason"] == "power_flow_validation_failed"
    assert group["fault_failure_reasons"] == {
        "power_flow_validation_failed": 2
    }


def test_replay_smoke_preserves_dynamic_limit_rejection(acopf, tmp_path):
    config_path, raw, dyr, base_raw, base_rop, _ = _smoke_config(tmp_path)
    parsed = _large_parsed_basecase(acopf)

    def context_func(**kwargs):
        return {
            "accepted": True,
            "records": [{"record_type": "acopf_smoke", "accepted": True}],
            "raw_path": raw,
            "dyr_path": dyr,
            "case_raw_path": tmp_path / "case.raw",
            "basecase_path": tmp_path / "Basecase_solution.txt",
            "parsed_basecase": parsed,
            "sample_idx": 0,
            "operating_point_id": "op",
            "accepted_operating_point_index": 0,
            "base_mva": 100.0,
        }

    diagnostics = {
        "phase": "runtime",
        "failure_reasons": ["active_set_cycle"],
        "events": [],
    }

    def failed_fault_runner(tasks, context, n_jobs, parallel_timeout_s):
        return [
            {
                "record_type": "fault_scenario",
                "accepted": False,
                "reject_reason": "dynamic_limit_runtime_failed",
                "reject_stage": "dynamic_limit_runtime",
                "dynamic_limit_diagnostics": diagnostics,
                "scenario_id": task.scenario_id,
                "fault_location_index": task.fault_location_index,
                "fault_impedance_index": task.fault_impedance_index,
                "fault_location": task.fault_location,
                "fault_impedance": task.fault_impedance,
            }
            for task in tasks
        ]

    output_dir = tmp_path / "out"
    progress = acopf.run_acopf_replay_smoke(
        config_path=config_path,
        output_dir=output_dir,
        acopf_config=acopf.AcopfInitializationConfig(
            julia="julia",
            exajugo_root=tmp_path / "exajugo",
            base_raw=base_raw,
            base_rop=base_rop,
        ),
        fault_locations="142,143",
        fault_impedances="1e-4",
        probml_basename="stage3_probml",
        acopf_context_func=context_func,
        state_metadata_func=lambda **kwargs: {"delta_state_indices": [0, 2]},
        fault_runner_func=failed_fault_runner,
    )

    assert progress["accepted"] is False
    assert progress["reject_reason"] == "dynamic_limit_runtime_failed"
    assert not (output_dir / "stage3_probml_final.npz").exists()
    records = acopf._read_jsonl(output_dir / "acopf_init_diagnostics.jsonl")
    group = records[-1]
    assert group["reject_reason"] == "dynamic_limit_runtime_failed"
    assert group["fault_failure_reasons"] == {
        "dynamic_limit_runtime_failed": 2
    }


def test_run_acopf_replay_smoke_rejects_existing_npz_before_work(acopf, tmp_path):
    config_path, _, _, base_raw, base_rop, _ = _smoke_config(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "existing_final.npz").write_text("present", encoding="utf-8")

    def context_func(**kwargs):
        raise AssertionError("context should not be prepared when NPZ already exists")

    with pytest.raises(FileExistsError):
        acopf.run_acopf_replay_smoke(
            config_path=config_path,
            output_dir=output_dir,
            acopf_config=acopf.AcopfInitializationConfig(
                julia="julia",
                exajugo_root=tmp_path / "exajugo",
                base_raw=base_raw,
                base_rop=base_rop,
            ),
            probml_basename="existing",
            acopf_context_func=context_func,
        )


def _production_acopf_config(acopf, tmp_path, base_raw, base_rop):
    return acopf.AcopfInitializationConfig(
        julia="julia",
        exajugo_root=tmp_path / "exajugo",
        base_raw=base_raw,
        base_rop=base_rop,
    )


def _accepted_production_context(acopf, tmp_path, parsed):
    def context_func(**kwargs):
        sample_idx = int(kwargs["sample_idx"])
        accepted_index = int(kwargs["accepted_operating_point_index"])
        case_dir = tmp_path / f"case_dir_{sample_idx}"
        case_dir.mkdir(exist_ok=True)
        return {
            "accepted": True,
            "success": True,
            "records": [{"record_type": "acopf_run", "accepted": True}],
            "raw_path": tmp_path / "model.raw",
            "dyr_path": tmp_path / "model.dyr",
            "case_raw_path": tmp_path / f"case_{sample_idx}.raw",
            "basecase_path": tmp_path / f"Basecase_{sample_idx}.txt",
            "parsed_basecase": parsed,
            "sample_idx": sample_idx,
            "operating_point_id": f"op-{sample_idx}",
            "accepted_operating_point_index": accepted_index,
            "base_mva": 100.0,
            "candidate_attempts": 1,
            "case_dir": case_dir,
        }

    return context_func


def _accepted_direct_production_context(tmp_path, *, n_gen=56, n_load=206):
    def context_func(**kwargs):
        sample_idx = int(kwargs["sample_idx"])
        accepted_index = int(kwargs["accepted_operating_point_index"])
        return {
            "accepted": True,
            "success": True,
            "records": [{"record_type": "uqgrid_pf_initialization", "accepted": True}],
            "raw_path": tmp_path / "model.raw",
            "dyr_path": tmp_path / "model.dyr",
            "operating_point": {
                "p_load_scaled": np.linspace(0.1, 0.2, n_load),
                "q_load_scaled": np.linspace(-0.01, -0.02, n_load),
                "p_gen_scaled": np.linspace(0.3, 0.4, n_gen),
                "q_gen_scaled": np.linspace(0.03, 0.04, n_gen),
                "pf_v_magnitudes": np.ones(2),
                "pf_v_angles": np.zeros(2),
            },
            "sample_idx": sample_idx,
            "operating_point_id": f"direct-op-{sample_idx}",
            "accepted_operating_point_index": accepted_index,
            "candidate_attempts": 1,
            "case_dir": None,
            "initialization_source": "uqgrid_pf",
            "acopf_feasible": False,
        }

    return context_func


def _successful_fault_runner(calls):
    def fault_runner(tasks, context, n_jobs, parallel_timeout_s):
        calls.append(
            {
                "fault_locations": [task.fault_location for task in tasks],
                "n_jobs": n_jobs,
                "parallel_timeout_s": parallel_timeout_s,
                "toff": context["integration_config"]["toff"],
            }
        )
        return [
            {
                "record_type": "fault_scenario",
                "accepted": True,
                "reject_reason": None,
                "scenario_id": task.scenario_id,
                "fault_location_index": task.fault_location_index,
                "fault_impedance_index": task.fault_impedance_index,
                "fault_location": task.fault_location,
                "fault_impedance": task.fault_impedance,
                "tsi_final": 10.0 + task.sample_idx + task.fault_location_index,
                "tsi_min": 5.0 + task.sample_idx + task.fault_location_index,
                "file": None,
                "history_file": None,
            }
            for task in tasks
        ]

    return fault_runner


def test_production_defaults_all_fault_locations_and_two_row_append(acopf, tmp_path):
    config_path, _, _, base_raw, base_rop, _ = _production_config(tmp_path, n_bus=2, target=2)
    parsed = _large_parsed_basecase(acopf)
    calls = []

    progress = acopf.run_acopf_production(
        config_path=config_path,
        output_dir=tmp_path / "out",
        acopf_config=_production_acopf_config(acopf, tmp_path, base_raw, base_rop),
        probml_basename="prod",
        acopf_context_func=_accepted_production_context(acopf, tmp_path, parsed),
        state_metadata_func=lambda **kwargs: {"delta_state_indices": [0, 2]},
        fault_runner_func=_successful_fault_runner(calls),
    )

    assert progress["completed"] is True
    assert progress["accepted_count"] == 2
    assert progress["next_sample_idx"] == 2
    assert progress["initialization_source_counts"] == {"acopf": 2}
    assert calls[0]["fault_locations"] == [0, 1]
    assert calls[0]["n_jobs"] == 2
    assert calls[0]["toff"] == pytest.approx(0.3833333333333333)

    final_path = tmp_path / "out" / "prod_final.npz"
    min_path = tmp_path / "out" / "prod_min.npz"
    with np.load(final_path, allow_pickle=True) as data:
        assert data["X"].shape == (2, 2, 262)
        assert data["X_flat"].shape == (2, 524)
        assert data["Y"].shape == (2, 2, 1)
        assert data["sample_idx"].tolist() == [0, 1]
        assert data["initialization_source"].tolist() == ["acopf", "acopf"]
        assert data["acopf_feasible"].tolist() == [True, True]
    with np.load(min_path, allow_pickle=True) as data:
        assert data["Y"].shape == (2, 2, 1)

    metadata = json.loads((tmp_path / "out" / "scenario_metadata.json").read_text())
    log = json.loads((tmp_path / "out" / "simulation_log.json").read_text())
    assert len(metadata) == 4
    assert len(log) == 4
    first_log = next(iter(log.values()))
    assert first_log["file"] is None
    assert first_log["initialization_source"] == "acopf"
    assert first_log["acopf_feasible"] is True
    assert "tsi_final" in first_log
    assert "tsi_min" in first_log


def test_production_acopf_probability_zero_uses_uqgrid_direct_only(acopf, tmp_path):
    config_path, _, _, base_raw, base_rop, _ = _production_config(tmp_path, n_bus=2, target=1)
    calls = []

    progress = acopf.run_acopf_production(
        config_path=config_path,
        output_dir=tmp_path / "out",
        acopf_config=_production_acopf_config(acopf, tmp_path, base_raw, base_rop),
        acopf_probability=0.0,
        probml_basename="prod",
        acopf_context_func=lambda **kwargs: pytest.fail("ACOPF should be skipped"),
        direct_context_func=_accepted_direct_production_context(tmp_path),
        state_metadata_func=lambda **kwargs: {"delta_state_indices": [0, 2]},
        fault_runner_func=_successful_fault_runner(calls),
    )

    assert progress["accepted_count"] == 1
    assert progress["initialization_source_counts"] == {"uqgrid_pf": 1}
    assert progress["integration_contract"]["method"] == "beuler"
    assert progress["integration_contract"]["enforce_q_limits"] is True
    assert progress["integration_contract"]["power_flow_validation"]["enabled"] is False
    for mode in ("final", "min"):
        with np.load(
            tmp_path / "out" / f"prod_{mode}.npz", allow_pickle=True
        ) as data:
            assert data["initialization_source"].tolist() == ["uqgrid_pf"]
            assert data["acopf_feasible"].tolist() == [False]
            assert data["X"].shape == (1, 2, 262)
            assert (
                data["meta"][0]["integration_contract"]
                == progress["integration_contract"]
            )

    log = json.loads((tmp_path / "out" / "simulation_log.json").read_text())
    assert {row["initialization_source"] for row in log.values()} == {"uqgrid_pf"}
    assert {row["acopf_feasible"] for row in log.values()} == {False}


def test_production_mixed_two_rows_writes_source_labels(acopf, tmp_path):
    config_path, _, _, base_raw, base_rop, _ = _production_config(tmp_path, n_bus=2, target=2)
    parsed = _large_parsed_basecase(acopf)

    progress = acopf.run_acopf_production(
        config_path=config_path,
        output_dir=tmp_path / "out",
        acopf_config=_production_acopf_config(acopf, tmp_path, base_raw, base_rop),
        acopf_probability=0.5,
        source_seed=2,
        probml_basename="prod",
        acopf_context_func=_accepted_production_context(acopf, tmp_path, parsed),
        direct_context_func=_accepted_direct_production_context(tmp_path),
        state_metadata_func=lambda **kwargs: {"delta_state_indices": [0, 2]},
        fault_runner_func=_successful_fault_runner([]),
    )

    assert progress["accepted_count"] == 2
    assert progress["initialization_source_counts"] == {"acopf": 1, "uqgrid_pf": 1}
    with np.load(tmp_path / "out" / "prod_final.npz", allow_pickle=True) as data:
        assert data["initialization_source"].tolist() == ["acopf", "uqgrid_pf"]
        assert data["acopf_feasible"].tolist() == [True, False]
        assert data["X"].shape == (2, 2, 262)


def test_production_no_continue_rejects_existing_outputs(acopf, tmp_path):
    config_path, _, _, base_raw, base_rop, _ = _production_config(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "prod_final.npz").write_text("present", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--continue"):
        acopf.run_acopf_production(
            config_path=config_path,
            output_dir=output_dir,
            acopf_config=_production_acopf_config(acopf, tmp_path, base_raw, base_rop),
            probml_basename="prod",
            acopf_context_func=lambda **kwargs: pytest.fail("should fail before work"),
        )


def test_production_resume_success_and_mismatch_rejection(acopf, tmp_path):
    config_path, _, _, base_raw, base_rop, _ = _production_config(tmp_path, target=1)
    parsed = _large_parsed_basecase(acopf)
    output_dir = tmp_path / "out"

    first = acopf.run_acopf_production(
        config_path=config_path,
        output_dir=output_dir,
        acopf_config=_production_acopf_config(acopf, tmp_path, base_raw, base_rop),
        probml_basename="prod",
        acopf_context_func=_accepted_production_context(acopf, tmp_path, parsed),
        state_metadata_func=lambda **kwargs: {"delta_state_indices": [0, 2]},
        fault_runner_func=_successful_fault_runner([]),
    )
    assert first["accepted_count"] == 1

    resumed = acopf.run_acopf_production(
        config_path=config_path,
        output_dir=output_dir,
        acopf_config=_production_acopf_config(acopf, tmp_path, base_raw, base_rop),
        target_accepted_scenarios=1,
        continue_run=True,
        probml_basename="prod",
        acopf_context_func=lambda **kwargs: pytest.fail("resume should already be complete"),
        state_metadata_func=lambda **kwargs: pytest.fail("resume should not export state"),
        fault_runner_func=lambda *args, **kwargs: pytest.fail("resume should not replay"),
    )
    assert resumed["accepted_count"] == 1

    progress_path = output_dir / "acopf_init_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["next_sample_idx"] = 7
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    with pytest.raises(ValueError, match="next_sample_idx"):
        acopf.run_acopf_production(
            config_path=config_path,
            output_dir=output_dir,
            acopf_config=_production_acopf_config(acopf, tmp_path, base_raw, base_rop),
            target_accepted_scenarios=1,
            continue_run=True,
            probml_basename="prod",
        )


def test_production_resume_rejects_changed_integration_method(acopf, tmp_path):
    config_path, _, _, base_raw, base_rop, _ = _production_config(tmp_path, target=1)
    parsed = _large_parsed_basecase(acopf)
    output_dir = tmp_path / "out"
    acopf.run_acopf_production(
        config_path=config_path,
        output_dir=output_dir,
        acopf_config=_production_acopf_config(acopf, tmp_path, base_raw, base_rop),
        probml_basename="prod",
        acopf_context_func=_accepted_production_context(acopf, tmp_path, parsed),
        state_metadata_func=lambda **kwargs: {"delta_state_indices": [0, 2]},
        fault_runner_func=_successful_fault_runner([]),
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["integration"]["method"] = "cn"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="method=.*requested"):
        acopf.run_acopf_production(
            config_path=config_path,
            output_dir=output_dir,
            acopf_config=_production_acopf_config(acopf, tmp_path, base_raw, base_rop),
            target_accepted_scenarios=1,
            continue_run=True,
            probml_basename="prod",
        )


@pytest.mark.parametrize(
    "context",
    [
        {
            "accepted": False,
            "records": [
                {
                    "record_type": "operating_point_group",
                    "accepted": False,
                    "reject_reason": "pf_non_converged",
                }
            ],
            "reject_reason": "pf_non_converged",
            "candidate_attempts": 1,
            "sample_idx": 0,
        },
        {
            "accepted": False,
            "records": [
                {
                    "record_type": "operating_point_group",
                    "accepted": False,
                    "reject_reason": "acopf_nonzero_exit",
                }
            ],
            "reject_reason": "acopf_nonzero_exit",
            "candidate_attempts": 1,
            "sample_idx": 0,
            "case_dir": None,
        },
        {
            "accepted": False,
            "records": [
                {
                    "record_type": "operating_point_group",
                    "accepted": False,
                    "reject_reason": "post_acopf_pf_not_converged",
                }
            ],
            "reject_reason": "post_acopf_pf_not_converged",
            "candidate_attempts": 1,
            "sample_idx": 0,
            "case_dir": None,
        },
    ],
)
def test_production_rejected_candidate_paths_do_not_append_npz(acopf, tmp_path, context):
    config_path, _, _, base_raw, base_rop, _ = _production_config(
        tmp_path,
        target=1,
        max_total_attempts=1,
    )

    progress = acopf.run_acopf_production(
        config_path=config_path,
        output_dir=tmp_path / "out",
        acopf_config=_production_acopf_config(acopf, tmp_path, base_raw, base_rop),
        probml_basename="prod",
        acopf_context_func=lambda **kwargs: dict(context),
        state_metadata_func=lambda **kwargs: {"delta_state_indices": [0, 2]},
        fault_runner_func=lambda *args, **kwargs: pytest.fail("rejected candidate should not replay"),
    )

    assert progress["accepted_count"] == 0
    assert not (tmp_path / "out" / "prod_final.npz").exists()
    assert progress["reject_reason"] == context["reject_reason"]


def test_production_dynamic_fault_rejection_does_not_append_npz(acopf, tmp_path):
    config_path, _, _, base_raw, base_rop, _ = _production_config(
        tmp_path,
        target=1,
        max_total_attempts=1,
    )
    parsed = _large_parsed_basecase(acopf)

    def failed_fault_runner(tasks, context, n_jobs, parallel_timeout_s):
        return [
            {
                "record_type": "fault_scenario",
                "accepted": False,
                "reject_reason": "dynamic_limit_runtime_failed",
                "reject_stage": "dynamic_limit_runtime",
                "dynamic_limit_diagnostics": {
                    "phase": "runtime",
                    "failure_reasons": ["active_set_cycle"],
                    "events": [],
                },
                "scenario_id": task.scenario_id,
                "fault_location_index": task.fault_location_index,
                "fault_impedance_index": task.fault_impedance_index,
                "fault_location": task.fault_location,
                "fault_impedance": task.fault_impedance,
            }
            for task in tasks
        ]

    progress = acopf.run_acopf_production(
        config_path=config_path,
        output_dir=tmp_path / "out",
        acopf_config=_production_acopf_config(acopf, tmp_path, base_raw, base_rop),
        probml_basename="prod",
        acopf_context_func=_accepted_production_context(acopf, tmp_path, parsed),
        state_metadata_func=lambda **kwargs: {"delta_state_indices": [0, 2]},
        fault_runner_func=failed_fault_runner,
    )

    assert progress["accepted_count"] == 0
    assert progress["reject_reason"] == "dynamic_limit_runtime_failed"
    assert not (tmp_path / "out" / "prod_final.npz").exists()
    records = acopf._read_jsonl(
        tmp_path / "out" / "acopf_init_diagnostics.jsonl"
    )
    assert records[-1]["reject_reason"] == "dynamic_limit_runtime_failed"
    assert records[-1]["fault_failure_reasons"] == {
        "dynamic_limit_runtime_failed": 2
    }


@pytest.mark.parametrize(
    ("acopf_probability", "initialization_source"),
    [(1.0, "acopf"), (0.0, "uqgrid_pf")],
)
def test_production_preserves_power_flow_validation_rejection(
    acopf,
    tmp_path,
    acopf_probability,
    initialization_source,
):
    config_path, _, _, base_raw, base_rop, _ = _production_config(
        tmp_path,
        n_bus=2,
        target=1,
        max_total_attempts=1,
    )
    parsed = _large_parsed_basecase(acopf)

    def failed_fault_runner(tasks, context, n_jobs, parallel_timeout_s):
        return [
            {
                "record_type": "fault_scenario",
                "accepted": False,
                "reject_reason": "power_flow_validation_failed",
                "reject_stage": "power_flow_validation",
                "power_flow_validation": {
                    "valid": False,
                    "failure_reasons": ["gen_q_limit"],
                },
                "scenario_id": task.scenario_id,
                "fault_location_index": task.fault_location_index,
                "fault_impedance_index": task.fault_impedance_index,
                "fault_location": task.fault_location,
                "fault_impedance": task.fault_impedance,
            }
            for task in tasks
        ]

    def unexpected_acopf_context(**kwargs):
        pytest.fail("ACOPF context should not be used for direct-only production")

    output_dir = tmp_path / "out"
    progress = acopf.run_acopf_production(
        config_path=config_path,
        output_dir=output_dir,
        acopf_config=_production_acopf_config(acopf, tmp_path, base_raw, base_rop),
        acopf_probability=acopf_probability,
        probml_basename="prod",
        acopf_context_func=(
            _accepted_production_context(acopf, tmp_path, parsed)
            if initialization_source == "acopf"
            else unexpected_acopf_context
        ),
        direct_context_func=_accepted_direct_production_context(tmp_path),
        state_metadata_func=lambda **kwargs: {"delta_state_indices": [0, 2]},
        fault_runner_func=failed_fault_runner,
    )

    assert progress["accepted_count"] == 0
    assert progress["reject_reason"] == "power_flow_validation_failed"
    assert not (output_dir / "prod_final.npz").exists()
    records = acopf._read_jsonl(output_dir / "acopf_init_diagnostics.jsonl")
    group = records[-1]
    assert group["initialization_source"] == initialization_source
    assert group["reject_reason"] == "power_flow_validation_failed"
    assert group["fault_failure_reasons"] == {
        "power_flow_validation_failed": 2
    }


def test_status_reads_existing_state_without_acopf_config(acopf, tmp_path, capsys):
    config_path, _, _, base_raw, base_rop, _ = _production_config(tmp_path, target=1)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["integration"].update(
        {
            "enforce_q_limits": False,
            "q_limit_tolerance": 2e-8,
            "max_q_limit_iterations": 7,
            "power_flow_validation": {
                "enabled": True,
                "residual_tolerance": 3e-8,
                "generator_limit_tolerance": 4e-6,
                "voltage_min": 0.9,
                "voltage_max": 1.1,
                "branch_loading_max": 1.2,
                "branch_limit_tolerance": 5e-5,
                "active_set_voltage_tolerance": 6e-6,
            },
        }
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    parsed = _large_parsed_basecase(acopf)
    output_dir = tmp_path / "out"
    acopf.run_acopf_production(
        config_path=config_path,
        output_dir=output_dir,
        acopf_config=_production_acopf_config(acopf, tmp_path, base_raw, base_rop),
        probml_basename="prod",
        acopf_context_func=_accepted_production_context(acopf, tmp_path, parsed),
        state_metadata_func=lambda **kwargs: {"delta_state_indices": [0, 2]},
        fault_runner_func=_successful_fault_runner([]),
    )

    rc = acopf.main(["--status", "--output-dir", str(output_dir), "--probml-basename", "prod"])

    assert rc == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["accepted_count"] == 1
    assert printed["fault_rows_completed"] == 2
    assert printed["initialization_source_counts"] == {"acopf": 1}
    assert printed["npz_shapes"]["final"]["Y"] == [1, 2, 1]
    assert printed["integration_contract"]["method"] == "beuler"
    assert printed["integration_contract"]["enforce_q_limits"] is False
    assert printed["integration_contract"]["q_limit_tolerance"] == pytest.approx(
        2e-8
    )
    assert printed["integration_contract"]["max_q_limit_iterations"] == 7
    assert printed["integration_contract"]["power_flow_validation"] == {
        "enabled": True,
        "residual_tolerance": pytest.approx(3e-8),
        "generator_limit_tolerance": pytest.approx(4e-6),
        "voltage_min": pytest.approx(0.9),
        "voltage_max": pytest.approx(1.1),
        "branch_loading_max": pytest.approx(1.2),
        "branch_limit_tolerance": pytest.approx(5e-5),
        "active_set_voltage_tolerance": pytest.approx(6e-6),
    }


def test_main_preflight_prints_to_stderr_and_preserves_stdout_json(
    acopf,
    tmp_path,
    monkeypatch,
    capsys,
):
    config_path, _, _, _, _, _ = _production_config(tmp_path, target=1)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["integration"]["petsc"] = False
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def fake_run(**kwargs):
        return {"stage": "stage_4_production", "completed": True, "accepted_count": 0}

    monkeypatch.setattr(acopf, "run_acopf_production", fake_run)

    rc = acopf.main(
        [
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--acopf-probability",
            "0.0",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    printed = json.loads(captured.out.strip())
    assert printed["completed"] is True
    assert "Startup preflight:" in captured.err
    assert "\u2713 UQGrid RAW:" in captured.err
    assert "- ExaJuGO: skipped" in captured.err


def test_main_preflight_failure_returns_2_before_generation(
    acopf,
    tmp_path,
    monkeypatch,
    capsys,
):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "raw": str(tmp_path / "missing.raw"),
                    "dyr": str(tmp_path / "missing.dyr"),
                    "n_bus": 2,
                },
                "integration": {"petsc": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        acopf,
        "run_acopf_production",
        lambda **kwargs: pytest.fail("generation should not run after failed preflight"),
    )

    rc = acopf.main(
        [
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--acopf-probability",
            "0.0",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "\u2717 UQGrid RAW:" in captured.err
    assert "\u2717 UQGrid DYR:" in captured.err
    assert "Preflight failed" in captured.err
    assert not (tmp_path / "out").exists()


def test_status_skips_startup_preflight(acopf, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        acopf,
        "collect_startup_preflight_checks",
        lambda **kwargs: pytest.fail("--status should not run startup preflight"),
    )
    monkeypatch.setattr(
        acopf,
        "read_acopf_production_status",
        lambda **kwargs: {"accepted_count": 0, "next_sample_idx": 0},
    )

    rc = acopf.main(["--status", "--output-dir", str(tmp_path), "--probml-basename", "prod"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Startup preflight" not in captured.err
    assert json.loads(captured.out)["accepted_count"] == 0
