"""In-process job management with process-isolated UQGrid execution."""

import math
import multiprocessing
import threading
import time
import traceback
from datetime import datetime, timezone
from queue import Empty
from typing import Dict, Iterable, Optional, Tuple, Union
from uuid import uuid4

from .artifacts import LocalArtifactStore
from .cases import LocalCaseRepository
from .results import LocalResultRepository
from .schemas import (
    DynamicsJobRequest,
    Job,
    JobError,
    JobKind,
    JobStatus,
    PowerFlowJobRequest,
)
from .simulations import SimulationService


JobRequest = Union[PowerFlowJobRequest, DynamicsJobRequest]


class JobNotFoundError(KeyError):
    pass


class JobConflictError(ValueError):
    pass


def _run_job_process(
    artifact_root: str,
    case_root: str,
    result_root: str,
    owner_id: str,
    kind: str,
    request_data: dict,
    output,
):
    try:
        artifacts = LocalArtifactStore(artifact_root)
        cases = LocalCaseRepository(case_root, artifacts)
        results = LocalResultRepository(result_root, artifacts)
        service = SimulationService(cases, results)
        if kind == JobKind.POWER_FLOW.value:
            request = PowerFlowJobRequest.model_validate(request_data)
            summary = service.run_power_flow(owner_id, request)
        else:
            request = DynamicsJobRequest.model_validate(request_data)
            summary = service.run_dynamics(owner_id, request)
        output.put({"result_id": summary.result_id})
    except Exception as exc:
        output.put({
            "error": {
                "code": "simulation_failed",
                "message": str(exc) or type(exc).__name__,
                "details": {"exception_type": type(exc).__name__},
                "traceback": traceback.format_exc(),
            }
        })


class InMemoryJobRepository:
    """Thread-safe job records for the initial single-process server."""

    def __init__(self):
        self._jobs: Dict[str, Tuple[str, Job, dict]] = {}
        self._idempotency: Dict[
            Tuple[str, str, str], Tuple[Optional[str], dict]
        ] = {}
        self._lock = threading.RLock()

    def create(self, owner_id: str, kind: JobKind, request: JobRequest) -> Tuple[Job, bool]:
        request_data = request.model_dump(mode="json")
        key = request.idempotency_key
        identity = (owner_id, kind.value, key) if key is not None else None
        with self._lock:
            if identity is not None and identity in self._idempotency:
                job_id, original = self._idempotency[identity]
                if original != request_data:
                    raise JobConflictError("idempotency key was already used for another request")
                if job_id is not None:
                    return self._jobs[job_id][1], False
            job = Job(
                job_id=f"job_{uuid4().hex}",
                case_id=request.case_id,
                kind=kind,
                status=JobStatus.QUEUED,
                progress=0.0,
                created_at=datetime.now(timezone.utc),
            )
            self._jobs[job.job_id] = (owner_id, job, request_data)
            if identity is not None:
                self._idempotency[identity] = (job.job_id, request_data)
            return job, True

    def release_idempotency(
        self,
        owner_id: str,
        kind: JobKind,
        request: JobRequest,
        job_id: str,
    ) -> None:
        """Release one job binding while retaining its canonical request payload."""
        key = request.idempotency_key
        if key is None:
            return
        identity = (owner_id, kind.value, key)
        request_data = request.model_dump(mode="json")
        with self._lock:
            current = self._idempotency.get(identity)
            if current == (job_id, request_data):
                self._idempotency[identity] = (None, request_data)

    def get(self, owner_id: str, job_id: str) -> Job:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record[0] != owner_id:
                raise JobNotFoundError(job_id)
            return record[1].model_copy(deep=True)

    def list(self, owner_id: str):
        with self._lock:
            return [
                job.model_copy(deep=True)
                for record_owner, job, _ in self._jobs.values()
                if record_owner == owner_id
            ]

    def update(self, owner_id: str, job_id: str, **changes) -> Job:
        with self._lock:
            current = self.get(owner_id, job_id)
            return self._replace(owner_id, current, **changes)

    def compare_and_set(
        self,
        owner_id: str,
        job_id: str,
        expected_statuses: Iterable[JobStatus],
        **changes,
    ) -> Tuple[Job, bool]:
        """Apply a state transition only while the job has an expected status."""
        expected = {
            status.value if isinstance(status, JobStatus) else status
            for status in expected_statuses
        }
        with self._lock:
            current = self.get(owner_id, job_id)
            if current.status not in expected:
                return current, False
            return self._replace(owner_id, current, **changes), True

    def _replace(self, owner_id: str, current: Job, **changes) -> Job:
        data = current.model_dump()
        data.update(changes)
        updated = Job.model_validate(data)
        self._jobs[current.job_id] = (
            owner_id,
            updated,
            self._jobs[current.job_id][2],
        )
        return updated.model_copy(deep=True)


class JobManager:
    """Submit simulations to isolated child processes and track their state."""

    def __init__(
        self,
        service: SimulationService,
        repository=None,
        mp_context=None,
        max_concurrent_jobs: int = 2,
        max_job_runtime_seconds: float = 300,
    ):
        if max_concurrent_jobs < 1:
            raise ValueError("max_concurrent_jobs must be positive")
        if not math.isfinite(max_job_runtime_seconds) or max_job_runtime_seconds <= 0:
            raise ValueError("max_job_runtime_seconds must be positive and finite")
        self.service = service
        self.repository = repository or InMemoryJobRepository()
        self._context = mp_context or multiprocessing.get_context("spawn")
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_job_runtime_seconds = max_job_runtime_seconds
        self._processes = {}
        self._completion_events = {}
        self._lock = threading.RLock()

    def submit_power_flow(self, owner_id: str, request: PowerFlowJobRequest) -> Job:
        return self._submit(owner_id, JobKind.POWER_FLOW, request)

    def submit_dynamics(self, owner_id: str, request: DynamicsJobRequest) -> Job:
        return self._submit(owner_id, JobKind.DYNAMICS, request)

    def get(self, owner_id: str, job_id: str) -> Job:
        return self.repository.get(owner_id, job_id)

    def list(self, owner_id: str):
        return self.repository.list(owner_id)

    def cancel(self, owner_id: str, job_id: str) -> Job:
        with self._lock:
            cancelled, transitioned = self.repository.compare_and_set(
                owner_id,
                job_id,
                {JobStatus.QUEUED, JobStatus.RUNNING},
                status=JobStatus.CANCELLED,
                progress=1.0,
                finished_at=datetime.now(timezone.utc),
            )
            if not transitioned:
                raise JobConflictError(f"job is already {cancelled.status}")
            process = self._processes.get(job_id)
            completion = self._completion_events.get(job_id)
        if process is not None and process.is_alive():
            process.terminate()
        if completion is not None:
            completion.wait(timeout=5)
        return cancelled

    def wait(self, owner_id: str, job_id: str, timeout: Optional[float] = None) -> Job:
        with self._lock:
            completion = self._completion_events.get(job_id)
        if completion is not None:
            completed = completion.wait(timeout=timeout)
            if not completed:
                raise TimeoutError(f"job did not finish within {timeout} seconds")
        return self.repository.get(owner_id, job_id)

    def _submit(self, owner_id: str, kind: JobKind, request: JobRequest) -> Job:
        self.service.cases.get(owner_id, request.case_id)
        with self._lock:
            job, created = self.repository.create(owner_id, kind, request)
            if not created:
                return job
            if len(self._processes) >= self.max_concurrent_jobs:
                failed = self.repository.update(
                    owner_id,
                    job.job_id,
                    status=JobStatus.FAILED,
                    progress=1.0,
                    error=JobError(
                        code="concurrency_limit",
                        message="maximum concurrent simulation jobs reached",
                        retryable=True,
                    ),
                    finished_at=datetime.now(timezone.utc),
                )
                self.repository.release_idempotency(owner_id, kind, request, job.job_id)
                return failed
            output = self._context.Queue(maxsize=1)
            process = self._context.Process(
                target=_run_job_process,
                args=(
                    str(self.service.results.artifacts.root),
                    str(self.service.cases.root),
                    str(self.service.results.root),
                    owner_id,
                    kind.value,
                    request.model_dump(mode="json"),
                    output,
                ),
                daemon=True,
            )
            self._processes[job.job_id] = process
            self._completion_events[job.job_id] = threading.Event()
            try:
                process.start()
            except Exception as exc:
                self._processes.pop(job.job_id, None)
                self._completion_events[job.job_id].set()
                failed, _ = self.repository.compare_and_set(
                    owner_id,
                    job.job_id,
                    {JobStatus.QUEUED},
                    status=JobStatus.FAILED,
                    progress=1.0,
                    error=JobError(
                        code="worker_exited",
                        message=str(exc) or "simulation worker could not start",
                        retryable=True,
                    ),
                    finished_at=datetime.now(timezone.utc),
                )
                self.repository.release_idempotency(
                    owner_id, kind, request, job.job_id
                )
                return failed
            running, transitioned = self.repository.compare_and_set(
                owner_id,
                job.job_id,
                {JobStatus.QUEUED},
                status=JobStatus.RUNNING,
                progress=0.05,
                started_at=datetime.now(timezone.utc),
            )
            threading.Thread(
                target=self._monitor,
                args=(owner_id, job.job_id, process, output),
                daemon=True,
            ).start()
            if not transitioned and process.is_alive():
                process.terminate()
            return running

    def _monitor(self, owner_id, job_id, process, output):
        deadline = time.monotonic() + self.max_job_runtime_seconds
        payload = None
        try:
            while payload is None:
                current = self.repository.get(owner_id, job_id)
                if current.status == JobStatus.CANCELLED:
                    self._stop_process(process)
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    payload = output.get(timeout=min(0.05, remaining))
                except Empty:
                    if not process.is_alive():
                        try:
                            payload = output.get(timeout=min(0.1, remaining))
                        except Empty:
                            payload = {
                                "error": {
                                    "code": "worker_exited",
                                    "message": (
                                        "simulation worker exited with code "
                                        f"{process.exitcode}"
                                    ),
                                    "details": {},
                                }
                            }
                        break
            if payload is None:
                self._stop_process(process)
                self._publish_timeout(owner_id, job_id)
                return

            process.join(timeout=max(0.0, deadline - time.monotonic()))
            current = self.repository.get(owner_id, job_id)
            if current.status == JobStatus.CANCELLED:
                self._stop_process(process)
                return
            if process.is_alive():
                self._stop_process(process)
                self._publish_timeout(owner_id, job_id)
                return
            if "result_id" in payload:
                self.repository.compare_and_set(
                    owner_id,
                    job_id,
                    {JobStatus.QUEUED, JobStatus.RUNNING},
                    status=JobStatus.SUCCEEDED,
                    progress=1.0,
                    result_id=payload["result_id"],
                    finished_at=datetime.now(timezone.utc),
                )
            else:
                error = payload["error"]
                error.pop("traceback", None)
                self.repository.compare_and_set(
                    owner_id,
                    job_id,
                    {JobStatus.QUEUED, JobStatus.RUNNING},
                    status=JobStatus.FAILED,
                    progress=1.0,
                    error=JobError.model_validate(error),
                    finished_at=datetime.now(timezone.utc),
                )
        finally:
            self._finish_monitor(job_id)

    @staticmethod
    def _stop_process(process):
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)

    def _publish_timeout(self, owner_id: str, job_id: str):
        self.repository.compare_and_set(
            owner_id,
            job_id,
            {JobStatus.QUEUED, JobStatus.RUNNING},
            status=JobStatus.FAILED,
            progress=1.0,
            error=JobError(
                code="worker_timeout",
                message="simulation worker exceeded its runtime limit",
                retryable=False,
                details={"max_job_runtime_seconds": self.max_job_runtime_seconds},
            ),
            finished_at=datetime.now(timezone.utc),
        )

    def _finish_monitor(self, job_id):
        with self._lock:
            self._processes.pop(job_id, None)
            completion = self._completion_events.get(job_id)
        if completion is not None:
            completion.set()
