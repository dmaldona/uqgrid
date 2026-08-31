import multiprocessing
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
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


class BlockingProcess:
    def __init__(self, output=None):
        self.output = output
        self.exitcode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self._alive = False
        self._done = threading.Event()

    def start(self):
        self._alive = True

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminate_calls += 1
        self.exitcode = -15
        self._alive = False
        self._done.set()

    def kill(self):
        self.kill_calls += 1
        self.exitcode = -9
        self._alive = False
        self._done.set()

    def join(self, timeout=None):
        self._done.wait(timeout)


class SuccessfulProcess:
    def __init__(self, output):
        self.output = output
        self.exitcode = None
        self.started = False

    def start(self):
        self.started = True
        self.exitcode = 0
        self.output.put({"result_id": "result_fake"})

    def is_alive(self):
        return not self.output.empty()

    def terminate(self):
        raise AssertionError("completed worker must not be terminated")

    def kill(self):
        raise AssertionError("completed worker must not be killed")

    def join(self, timeout=None):
        pass


class StubbornProcess(BlockingProcess):
    def __init__(self, output=None):
        super().__init__(output)
        self.join_timeouts = []

    def terminate(self):
        self.terminate_calls += 1

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)


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

    monkeypatch.setattr(jobs._context, "Queue", lambda **kwargs: queue.Queue())
    monkeypatch.setattr(
        jobs._context,
        "Process",
        lambda **kwargs: BlockingProcess(kwargs["args"][-1]),
    )
    submitted = jobs.submit_power_flow(
        "alice", PowerFlowJobRequest(case_id=case.case_id)
    )
    cancelled = jobs.cancel("alice", submitted.job_id)

    assert cancelled.status == JobStatus.CANCELLED
    assert cancelled.finished_at is not None


def test_concurrent_admission_reserves_exactly_one_worker(monkeypatch, jobs):
    case = jobs.service.cases.import_files(
        "alice", "case9", [ROOT / "data/case9.m"]
    )
    jobs.max_concurrent_jobs = 1
    created_processes = []

    def process_factory(**kwargs):
        process = BlockingProcess(kwargs["args"][-1])
        created_processes.append(process)
        return process

    monkeypatch.setattr(jobs._context, "Queue", lambda **kwargs: queue.Queue())
    monkeypatch.setattr(jobs._context, "Process", process_factory)
    barrier = threading.Barrier(6)

    def submit(_):
        barrier.wait()
        return jobs.submit_power_flow(
            "alice", PowerFlowJobRequest(case_id=case.case_id)
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        submitted = list(executor.map(submit, range(6)))

    running = [job for job in submitted if job.status == JobStatus.RUNNING]
    rejected = [job for job in submitted if job.status == JobStatus.FAILED]
    assert len(running) == 1
    assert len(rejected) == 5
    assert len(created_processes) == 1
    assert len(jobs._processes) == 1
    assert all(job.error.code == "concurrency_limit" for job in rejected)

    jobs.cancel("alice", running[0].job_id)
    jobs.wait("alice", running[0].job_id, timeout=2)


def test_capacity_failure_releases_idempotency_binding(monkeypatch, jobs):
    case = jobs.service.cases.import_files(
        "alice", "case9", [ROOT / "data/case9.m"]
    )
    jobs.max_concurrent_jobs = 1
    monkeypatch.setattr(jobs._context, "Queue", lambda **kwargs: queue.Queue())
    monkeypatch.setattr(
        jobs._context,
        "Process",
        lambda **kwargs: BlockingProcess(kwargs["args"][-1]),
    )

    occupying = jobs.submit_power_flow(
        "alice", PowerFlowJobRequest(case_id=case.case_id)
    )
    request = PowerFlowJobRequest(
        case_id=case.case_id, idempotency_key="retry-after-capacity"
    )
    rejected = jobs.submit_power_flow("alice", request)

    assert rejected.status == JobStatus.FAILED
    assert rejected.error.code == "concurrency_limit"
    assert rejected.error.retryable is True
    jobs.cancel("alice", occupying.job_id)
    jobs.wait("alice", occupying.job_id, timeout=2)

    retried = jobs.submit_power_flow("alice", request)
    assert retried.status == JobStatus.RUNNING
    assert retried.job_id != rejected.job_id
    assert jobs.get("alice", rejected.job_id).status == JobStatus.FAILED

    with pytest.raises(JobConflictError, match="already used"):
        jobs.submit_power_flow(
            "alice",
            PowerFlowJobRequest(
                case_id=case.case_id,
                options={"q_limit_tolerance": 1e-7},
                idempotency_key="retry-after-capacity",
            ),
        )

    jobs.cancel("alice", retried.job_id)
    jobs.wait("alice", retried.job_id, timeout=2)


def test_cancellation_wins_over_late_worker_result(monkeypatch, jobs):
    case = jobs.service.cases.import_files(
        "alice", "case9", [ROOT / "data/case9.m"]
    )
    processes = []

    def process_factory(**kwargs):
        process = BlockingProcess(kwargs["args"][-1])
        processes.append(process)
        return process

    monkeypatch.setattr(jobs._context, "Queue", lambda **kwargs: queue.Queue())
    monkeypatch.setattr(jobs._context, "Process", process_factory)
    submitted = jobs.submit_power_flow(
        "alice", PowerFlowJobRequest(case_id=case.case_id)
    )
    processes[0].output.put({"result_id": "result_too_late"})

    jobs.cancel("alice", submitted.job_id)
    completed = jobs.wait("alice", submitted.job_id, timeout=2)

    assert completed.status == JobStatus.CANCELLED
    assert completed.result_id is None


def test_completion_wins_over_late_cancellation(monkeypatch, jobs):
    case = jobs.service.cases.import_files(
        "alice", "case9", [ROOT / "data/case9.m"]
    )
    monkeypatch.setattr(jobs._context, "Queue", lambda **kwargs: queue.Queue())
    monkeypatch.setattr(
        jobs._context,
        "Process",
        lambda **kwargs: SuccessfulProcess(kwargs["args"][-1]),
    )
    submitted = jobs.submit_power_flow(
        "alice", PowerFlowJobRequest(case_id=case.case_id)
    )
    completed = jobs.wait("alice", submitted.job_id, timeout=2)

    assert completed.status == JobStatus.SUCCEEDED
    with pytest.raises(JobConflictError, match="already succeeded"):
        jobs.cancel("alice", submitted.job_id)
    assert jobs.get("alice", submitted.job_id).status == JobStatus.SUCCEEDED


def test_worker_timeout_kills_process_and_releases_slot(monkeypatch, jobs):
    case = jobs.service.cases.import_files(
        "alice", "case9", [ROOT / "data/case9.m"]
    )
    jobs.max_concurrent_jobs = 1
    jobs.max_job_runtime_seconds = 0.01
    processes = []

    def stubborn_factory(**kwargs):
        process = StubbornProcess(kwargs["args"][-1])
        processes.append(process)
        return process

    monkeypatch.setattr(jobs._context, "Queue", lambda **kwargs: queue.Queue())
    monkeypatch.setattr(jobs._context, "Process", stubborn_factory)
    submitted = jobs.submit_power_flow(
        "alice", PowerFlowJobRequest(case_id=case.case_id)
    )
    failed = jobs.wait("alice", submitted.job_id, timeout=2)

    assert failed.status == JobStatus.FAILED
    assert failed.error.code == "worker_timeout"
    assert failed.error.retryable is False
    assert failed.error.details == {"max_job_runtime_seconds": 0.01}
    assert processes[0].terminate_calls == 1
    assert processes[0].kill_calls == 1
    assert processes[0].join_timeouts == [5, 5]
    assert submitted.job_id not in jobs._processes
    assert jobs._completion_events[submitted.job_id].is_set()

    monkeypatch.setattr(
        jobs._context,
        "Process",
        lambda **kwargs: SuccessfulProcess(kwargs["args"][-1]),
    )
    next_job = jobs.submit_power_flow(
        "alice", PowerFlowJobRequest(case_id=case.case_id)
    )
    assert jobs.wait("alice", next_job.job_id, timeout=2).status == JobStatus.SUCCEEDED


@pytest.mark.parametrize("value", [0, float("inf"), float("nan")])
def test_job_runtime_limit_must_be_positive_and_finite(jobs, value):
    with pytest.raises(ValueError, match="positive and finite"):
        JobManager(jobs.service, max_job_runtime_seconds=value)
