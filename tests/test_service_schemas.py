from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from uqgrid.service.schemas import (
    BusFaultEvent,
    CreateCaseUploadRequest,
    DynamicsJobRequest,
    Job,
    JobError,
    JobKind,
    JobStatus,
    ResultQuery,
    UploadFileRequest,
    public_json_schemas,
)


SHA256 = "a" * 64


def test_case_upload_accepts_psse_bundle():
    request = CreateCaseUploadRequest(
        name="ieee9",
        files=[
            UploadFileRequest(name="ieee9.raw", size_bytes=10, sha256=SHA256),
            UploadFileRequest(name="ieee9.dyr", size_bytes=20, sha256=SHA256),
        ],
    )

    assert [item.name for item in request.files] == ["ieee9.raw", "ieee9.dyr"]


@pytest.mark.parametrize("name", ["../case.raw", "folder/case.raw", "folder\\case.raw"])
def test_upload_file_rejects_paths(name):
    with pytest.raises(ValidationError, match="plain filename"):
        UploadFileRequest(name=name, size_bytes=1, sha256=SHA256)


@pytest.mark.parametrize(
    "filenames",
    [
        ["case.py"],
        ["case.raw", "second.raw"],
        ["case.m", "case.dyr"],
        ["case.zip"],
        ["case.raw", "case.dyr", "extra.dyr"],
    ],
)
def test_case_upload_rejects_invalid_bundles(filenames):
    with pytest.raises(ValidationError):
        CreateCaseUploadRequest(
            name="invalid",
            files=[
                UploadFileRequest(name=name, size_bytes=1, sha256=SHA256)
                for name in filenames
            ],
        )


def test_dynamics_request_rejects_unsupported_petsc_and_event_count():
    event = BusFaultEvent(bus_id=1, impedance_pu=0.01, start_s=0.1, clear_s=0.2)

    with pytest.raises(ValidationError):
        DynamicsJobRequest(
            case_id="case_1",
            scenario={"events": [event]},
            integration={"petsc": True},
        )

    with pytest.raises(ValidationError):
        DynamicsJobRequest(
            case_id="case_1",
            scenario={"events": [event, event]},
        )


def test_dynamics_request_requires_event_within_simulation_window():
    with pytest.raises(ValidationError, match="clear_s"):
        DynamicsJobRequest(
            case_id="case_1",
            scenario={
                "events": [
                    {
                        "bus_id": 1,
                        "impedance_pu": 0.01,
                        "start_s": 0.1,
                        "clear_s": 1.1,
                    }
                ]
            },
            integration={"end_s": 1.0},
        )


def test_terminal_job_invariants_are_enforced():
    now = datetime.now(timezone.utc)

    with pytest.raises(ValidationError, match="result_id"):
        Job(
            job_id="job_1",
            case_id="case_1",
            kind=JobKind.POWER_FLOW,
            status=JobStatus.SUCCEEDED,
            progress=1.0,
            created_at=now,
            finished_at=now,
        )

    failed = Job(
        job_id="job_2",
        case_id="case_1",
        kind=JobKind.POWER_FLOW,
        status=JobStatus.FAILED,
        progress=1.0,
        created_at=now,
        finished_at=now,
        error=JobError(code="non_convergence", message="Power flow did not converge"),
    )
    assert failed.error.code == "non_convergence"


def test_result_query_is_bounded():
    with pytest.raises(ValidationError):
        ResultQuery(result_id="result_1", signals=["bus.1.vm"], max_points=1001)

    with pytest.raises(ValidationError, match="end_s"):
        ResultQuery(
            result_id="result_1",
            signals=["bus.1.vm"],
            start_s=2.0,
            end_s=1.0,
        )


def test_public_contracts_generate_json_schemas():
    schemas = public_json_schemas()

    assert "CaseManifest" in schemas
    assert "DynamicsJobRequest" in schemas
    assert "PowerFlowResultSummary" in schemas
    assert all(schema["type"] == "object" for schema in schemas.values())
    assert all(schema["additionalProperties"] is False for schema in schemas.values())
