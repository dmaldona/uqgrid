import multiprocessing
import time
from pathlib import Path

import pytest

from uqgrid.service import (
    InMemoryJobRepository,
    JobConflictError,
    JobManager,
    JobNotFoundError,
    LocalArtifactStore,
    LocalCaseRepository,
    LocalResultRepository,
    PowerFlowJobRequest,
    SimulationService,
)
from uqgrid.service.schemas import JobStatus


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def jobs(tmp_path):
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    cases = LocalCaseRepository(tmp_path / "cases", artifacts)
    results = LocalResultRepository(tmp_path / "results", artifacts)
    service = SimulationService(cases, results)
    return JobManager(
        service,
        repository=InMemoryJobRepository(),
        mp_context=multiprocessing.get_context("spawn"),
    )


def test_power_flow_job_runs_in_child_process(jobs):
    case = jobs.service.cases.import_files(
        "alice", "ieee9", [ROOT / "data/ieee9_v33.raw"]
    )

    submitted = jobs.submit_power_flow(
        "alice", PowerFlowJobRequest(case_id=case.case_id)
    )
    completed = jobs.wait("alice", submitted.job_id, timeout=30)

    assert submitted.status == JobStatus.RUNNING
    assert completed.status == JobStatus.SUCCEEDED
    assert completed.result_id is not None
    assert jobs.service.results.get_summary("alice", completed.result_id).converged


def test_idempotent_submission_returns_same_job(jobs):
    case = jobs.service.cases.import_files(
        "alice", "case9", [ROOT / "data/case9.m"]
    )
    request = PowerFlowJobRequest(case_id=case.case_id, idempotency_key="same-request")

    first = jobs.submit_power_flow("alice", request)
    second = jobs.submit_power_flow("alice", request)
    jobs.wait("alice", first.job_id, timeout=30)

    assert second.job_id == first.job_id
    assert len(jobs.list("alice")) == 1


def test_idempotency_key_rejects_different_request(jobs):
    first_case = jobs.service.cases.import_files(
        "alice", "case9", [ROOT / "data/case9.m"]
    )
    second_case = jobs.service.cases.import_files(
        "alice", "case14", [ROOT / "data/case14.m"]
    )
    jobs.submit_power_flow(
        "alice",
        PowerFlowJobRequest(case_id=first_case.case_id, idempotency_key="duplicate"),
    )

    with pytest.raises(JobConflictError, match="already used"):
        jobs.submit_power_flow(
            "alice",
            PowerFlowJobRequest(case_id=second_case.case_id, idempotency_key="duplicate"),
        )


def test_jobs_are_owner_scoped(jobs):
    case = jobs.service.cases.import_files(
        "alice", "case9", [ROOT / "data/case9.m"]
    )
    submitted = jobs.submit_power_flow(
        "alice", PowerFlowJobRequest(case_id=case.case_id)
    )

    with pytest.raises(JobNotFoundError):
        jobs.get("bob", submitted.job_id)
    jobs.wait("alice", submitted.job_id, timeout=30)


def test_cancel_running_job(monkeypatch, jobs):
    case = jobs.service.cases.import_files(
        "alice", "case9", [ROOT / "data/case9.m"]
    )

    class SlowProcess:
        exitcode = None

        def start(self):
            pass

        def is_alive(self):
            return True

        def terminate(self):
            self.exitcode = -15

        def join(self, timeout=None):
            time.sleep(0.01)

    monkeypatch.setattr(jobs._context, "Process", lambda **kwargs: SlowProcess())
    submitted = jobs.submit_power_flow(
        "alice", PowerFlowJobRequest(case_id=case.case_id)
    )
    cancelled = jobs.cancel("alice", submitted.job_id)

    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.finished_at is not None
