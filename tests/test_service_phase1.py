import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from uqgrid.service import (
    ArtifactNotFoundError,
    CaseNotFoundError,
    CaseValidationError,
    DynamicsJobRequest,
    LocalArtifactStore,
    LocalCaseRepository,
    LocalResultRepository,
    PowerFlowJobRequest,
    ResultNotFoundError,
    ResultQuery,
    SimulationService,
)
from uqgrid.service.schemas import ArtifactKind, CaseFormat


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def service(tmp_path):
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    cases = LocalCaseRepository(tmp_path / "cases", artifacts)
    results = LocalResultRepository(tmp_path / "results", artifacts)
    return SimulationService(cases, results)


def test_case_repository_copies_inputs_and_inspects_psse(service, tmp_path):
    raw = tmp_path / "case.raw"
    dyr = tmp_path / "case.dyr"
    raw.write_bytes((ROOT / "data/ieee9_v33.raw").read_bytes())
    dyr.write_bytes((ROOT / "data/ieee9bus.dyr").read_bytes())

    manifest = service.cases.import_files("alice", "ieee9", [raw, dyr])
    raw.write_text("changed after import", encoding="utf-8")
    inspection = service.cases.inspect("alice", manifest.case_id)
    stored_raw, stored_dyr = service.cases.resolve_files("alice", manifest.case_id)

    assert manifest.format == CaseFormat.PSSE
    assert manifest.sha256 == hashlib.sha256(
        b"case.dyr" + bytes.fromhex(manifest.files[1].sha256)
        + b"case.raw" + bytes.fromhex(manifest.files[0].sha256)
    ).hexdigest()
    assert stored_raw.read_bytes() == (ROOT / "data/ieee9_v33.raw").read_bytes()
    assert stored_dyr.read_bytes() == (ROOT / "data/ieee9bus.dyr").read_bytes()
    assert inspection.bus_count == 9
    assert inspection.generator_count == 3
    assert inspection.dynamic_model_count == 3
    assert inspection.dynamic_models["GenGENROU"] == 3


def test_case_repository_rejects_invalid_bundle(service, tmp_path):
    bad = tmp_path / "case.py"
    bad.write_text("print('not a case')", encoding="utf-8")

    with pytest.raises(ValueError, match="case bundle"):
        service.cases.import_files("alice", "invalid", [bad])


def test_case_inspection_rejects_malformed_input(service, tmp_path):
    raw = tmp_path / "malformed.raw"
    raw.write_text("not a PSS/E case", encoding="utf-8")
    case = service.cases.import_files("alice", "malformed", [raw])

    with pytest.raises(CaseValidationError, match="could not be parsed"):
        service.cases.inspect("alice", case.case_id)


def test_case_and_artifact_ownership_is_hidden(service):
    manifest = service.cases.import_files(
        "alice", "case9", [ROOT / "data/case9.m"]
    )
    artifact_id = manifest.files[0].artifact_id

    with pytest.raises(CaseNotFoundError):
        service.cases.get("bob", manifest.case_id)
    with pytest.raises(ArtifactNotFoundError):
        service.cases.artifacts.read_bytes("bob", artifact_id)


def test_power_flow_creates_versioned_artifacts_and_queryable_signals(service):
    case = service.cases.import_files(
        "alice", "ieee9", [ROOT / "data/ieee9_v33.raw"]
    )

    summary = service.run_power_flow(
        "alice", PowerFlowJobRequest(case_id=case.case_id)
    )
    signals = service.results.list_signals("alice", summary.result_id)
    query = service.results.query(
        "alice",
        ResultQuery(
            result_id=summary.result_id,
            signals=["bus.0.voltage_magnitude", "bus.8.voltage_angle"],
        ),
    )

    assert summary.converged
    assert summary.bus_count == 9
    assert summary.voltage_min_pu == pytest.approx(0.9956308580698596)
    assert {item.kind for item in summary.artifacts} == {
        "summary", "results", "state_metadata", "manifest", "log"
    }
    assert len(signals) == 18
    assert query.time_s == [0.0]
    assert query.signals[0].values == [pytest.approx(1.04)]
    for artifact in summary.artifacts:
        data = service.results.artifacts.read_bytes("alice", artifact.artifact_id)
        assert hashlib.sha256(data).hexdigest() == artifact.sha256


def test_dynamics_creates_named_signals_and_bounded_queries(service):
    case = service.cases.import_files(
        "alice",
        "ieee9-dynamics",
        [ROOT / "data/ieee9_v33.raw", ROOT / "data/ieee9bus.dyr"],
    )
    request = DynamicsJobRequest(
        case_id=case.case_id,
        scenario={
            "events": [
                {
                    "bus_id": 1,
                    "impedance_pu": 1.0,
                    "start_s": 0.02,
                    "clear_s": 0.03,
                }
            ]
        },
        integration={"dt_s": 0.01, "end_s": 0.05},
    )

    summary = service.run_dynamics("alice", request)
    signals = service.results.list_signals("alice", summary.result_id)
    speed_signal = next(item.name for item in signals if item.name.endswith(".speed"))
    trace = service.results.query(
        "alice",
        ResultQuery(result_id=summary.result_id, signals=[speed_signal], max_points=3),
    )
    maximum = service.results.query(
        "alice",
        ResultQuery(
            result_id=summary.result_id,
            signals=[speed_signal],
            aggregate="max",
        ),
    )
    state_artifact = next(
        item for item in summary.artifacts if item.kind == ArtifactKind.STATE_METADATA
    )
    metadata = json.loads(
        service.results.artifacts.read_bytes("alice", state_artifact.artifact_id)
    )

    assert summary.assessment == "not_evaluated"
    assert summary.step_count == 6
    assert summary.state_count == 60
    assert len(signals) == 12
    assert trace.downsampled
    assert trace.time_s == [0.0, 0.02, 0.05]
    assert len(trace.signals[0].values) == 3
    assert maximum.aggregate == "max"
    assert maximum.time_s is None
    assert len(metadata) == summary.state_count


def test_result_queries_enforce_owner_and_signal_names(service):
    case = service.cases.import_files(
        "alice", "case9", [ROOT / "data/case9.m"]
    )
    summary = service.run_power_flow(
        "alice", PowerFlowJobRequest(case_id=case.case_id)
    )

    with pytest.raises(ResultNotFoundError):
        service.results.get_summary("bob", summary.result_id)
    with pytest.raises(ValueError, match="unknown signals"):
        service.results.query(
            "alice",
            ResultQuery(result_id=summary.result_id, signals=["unknown"]),
        )


def test_dynamics_requires_dyr_file(service):
    case = service.cases.import_files(
        "alice", "case9", [ROOT / "data/ieee9_v33.raw"]
    )
    request = DynamicsJobRequest(
        case_id=case.case_id,
        scenario={
            "events": [
                {
                    "bus_id": 1,
                    "impedance_pu": 1.0,
                    "start_s": 0.01,
                    "clear_s": 0.02,
                }
            ]
        },
        integration={"dt_s": 0.01, "end_s": 0.03},
    )

    with pytest.raises(ValueError, match="requires a PSS/E case with a .dyr"):
        service.run_dynamics("alice", request)


def test_dynamics_rejects_unknown_external_bus(service):
    case = service.cases.import_files(
        "alice",
        "ieee9-dynamics",
        [ROOT / "data/ieee9_v33.raw", ROOT / "data/ieee9bus.dyr"],
    )
    request = DynamicsJobRequest(
        case_id=case.case_id,
        scenario={
            "events": [
                {
                    "bus_id": 99999,
                    "impedance_pu": 1.0,
                    "start_s": 0.01,
                    "clear_s": 0.02,
                }
            ]
        },
        integration={"dt_s": 0.01, "end_s": 0.03},
    )

    with pytest.raises(ValueError, match="unknown external bus ID"):
        service.run_dynamics("alice", request)
