"""Versioned public contracts for the UQGrid remote service."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


SCHEMA_VERSION = "1.0"


class ServiceModel(BaseModel):
    """Base model for stable service inputs and outputs."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class CaseFormat(str, Enum):
    PSSE = "psse"
    MATPOWER = "matpower"


class CaseStatus(str, Enum):
    UPLOADING = "uploading"
    READY = "ready"
    INVALID = "invalid"


class JobKind(str, Enum):
    POWER_FLOW = "power_flow"
    DYNAMICS = "dynamics"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactKind(str, Enum):
    CASE_INPUT = "case_input"
    MANIFEST = "manifest"
    SUMMARY = "summary"
    RESULTS = "results"
    STATE_METADATA = "state_metadata"
    LOG = "log"


class UploadFileRequest(ServiceModel):
    name: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
            raise ValueError("name must be a plain filename")
        return value


class CreateCaseUploadRequest(ServiceModel):
    name: str = Field(min_length=1, max_length=200)
    files: List[UploadFileRequest] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def validate_bundle(self):
        names = [item.name for item in self.files]
        if len(names) != len(set(names)):
            raise ValueError("case bundle filenames must be unique")

        suffixes = sorted(name.lower().rsplit(".", 1)[-1] for name in names if "." in name)
        valid = suffixes in (["m"], ["raw"], ["dyr", "raw"])
        if not valid:
            raise ValueError(
                "case bundle must contain one .m or one .raw with optional .dyr"
            )
        return self


class UploadTarget(ServiceModel):
    name: str
    method: Literal["PUT"] = "PUT"
    url: HttpUrl
    headers: Dict[str, str] = Field(default_factory=dict)
    expires_at: datetime


class UploadSession(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    upload_id: str = Field(min_length=1)
    targets: List[UploadTarget]
    expires_at: datetime


class DownloadTarget(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    artifact_id: str = Field(min_length=1)
    method: Literal["GET"] = "GET"
    url: HttpUrl
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expires_at: datetime


class CompleteCaseUploadRequest(ServiceModel):
    upload_id: str = Field(min_length=1)


class CaseFile(ServiceModel):
    artifact_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str = Field(min_length=1)


class CaseManifest(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    owner_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    format: CaseFormat
    status: CaseStatus
    files: List[CaseFile] = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime


class CaseInspection(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    format: CaseFormat
    base_mva: float = Field(gt=0.0)
    bus_count: int = Field(ge=0)
    branch_count: int = Field(ge=0)
    generator_count: int = Field(ge=0)
    load_count: int = Field(ge=0)
    shunt_count: int = Field(ge=0)
    dynamic_model_count: int = Field(ge=0)
    dynamic_models: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class CaseList(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    cases: List[CaseManifest]


class PowerFlowOptions(ServiceModel):
    enforce_q_limits: bool = True
    q_limit_tolerance: float = Field(1e-8, ge=0.0)
    max_q_limit_iterations: Optional[int] = Field(None, ge=1)


class PowerFlowJobRequest(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    options: PowerFlowOptions = Field(default_factory=PowerFlowOptions)
    idempotency_key: Optional[str] = Field(None, min_length=1, max_length=200)


class BusFaultEvent(ServiceModel):
    type: Literal["bus_fault"] = "bus_fault"
    bus_id: int
    impedance_pu: float = Field(gt=0.0)
    start_s: float = Field(ge=0.0)
    clear_s: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_times(self):
        if self.clear_s < self.start_s:
            raise ValueError("clear_s must be greater than or equal to start_s")
        return self


class DynamicsScenario(ServiceModel):
    events: List[BusFaultEvent] = Field(min_length=1, max_length=1)


class DynamicsIntegrationOptions(ServiceModel):
    method: Literal["beuler"] = "beuler"
    dt_s: float = Field(1.0 / 120.0, gt=0.0)
    end_s: float = Field(5.0, gt=0.0)
    petsc: Literal[False] = False
    enforce_q_limits: bool = True
    enforce_dynamic_limits: bool = True


class DynamicsOutputOptions(ServiceModel):
    signals: List[str] = Field(
        default_factory=lambda: ["bus.voltage_magnitude", "generator.speed"],
        min_length=1,
        max_length=20,
    )
    summary: bool = True


class DynamicsJobRequest(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    scenario: DynamicsScenario
    integration: DynamicsIntegrationOptions = Field(default_factory=DynamicsIntegrationOptions)
    outputs: DynamicsOutputOptions = Field(default_factory=DynamicsOutputOptions)
    idempotency_key: Optional[str] = Field(None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_event_window(self):
        event = self.scenario.events[0]
        if event.start_s > self.integration.end_s:
            raise ValueError("fault start_s must not exceed integration end_s")
        if event.clear_s > self.integration.end_s:
            raise ValueError("fault clear_s must not exceed integration end_s")
        return self


class JobError(ServiceModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class Artifact(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    artifact_id: str = Field(min_length=1)
    kind: ArtifactKind
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resource_uri: str = Field(pattern=r"^uqgrid://artifacts/")
    expires_at: Optional[datetime] = None


class Job(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    job_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    kind: JobKind
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    poll_interval_ms: int = Field(1000, ge=100)
    error: Optional[JobError] = None
    result_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_terminal_state(self):
        terminal = self.status in {
            JobStatus.SUCCEEDED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        }
        if terminal and self.finished_at is None:
            raise ValueError("terminal jobs require finished_at")
        if self.status == JobStatus.SUCCEEDED.value and self.result_id is None:
            raise ValueError("succeeded jobs require result_id")
        if self.status == JobStatus.FAILED.value and self.error is None:
            raise ValueError("failed jobs require error")
        return self


class JobList(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    jobs: List[Job]


class PowerFlowResultSummary(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    result_id: str = Field(min_length=1)
    converged: bool
    residual_norm: float = Field(ge=0.0)
    bus_count: int = Field(ge=0)
    generator_count: int = Field(ge=0)
    voltage_min_pu: float
    voltage_max_pu: float
    q_limit_iterations: int = Field(ge=0)
    q_limit_event_count: int = Field(ge=0)
    warnings: List[str] = Field(default_factory=list)
    artifacts: List[Artifact] = Field(default_factory=list)


class DynamicsResultSummary(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    result_id: str = Field(min_length=1)
    assessment: Literal["not_evaluated"] = "not_evaluated"
    step_count: int = Field(ge=1)
    state_count: int = Field(ge=1)
    end_s: float = Field(ge=0.0)
    minimum_bus_voltage_pu: float
    maximum_abs_generator_speed_pu: float = Field(ge=0.0)
    warnings: List[str] = Field(default_factory=list)
    artifacts: List[Artifact] = Field(default_factory=list)


class SignalDescriptor(ServiceModel):
    name: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    entity_type: Literal["bus", "generator", "system"]
    entity_id: str = Field(min_length=1)
    source_state_indices: List[int] = Field(min_length=1)


class SignalList(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    result_id: str = Field(min_length=1)
    signals: List[SignalDescriptor]


class ServiceCapabilities(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    service: Literal["uqgrid"] = "uqgrid"
    uqgrid_version: str
    case_formats: List[str]
    simulation_kinds: List[str]
    integration_methods: List[str]
    supports_petsc: bool
    max_query_points: int = Field(gt=0)


class SignalValues(ServiceModel):
    signal: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    values: List[float]


class ResultQueryResponse(ServiceModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    result_id: str = Field(min_length=1)
    time_s: Optional[List[float]] = None
    signals: List[SignalValues] = Field(min_length=1)
    aggregate: Optional[str] = None
    downsampled: bool = False


class ResultQuery(ServiceModel):
    result_id: str = Field(min_length=1)
    signals: List[str] = Field(min_length=1, max_length=20)
    start_s: Optional[float] = Field(None, ge=0.0)
    end_s: Optional[float] = Field(None, ge=0.0)
    max_points: int = Field(200, ge=2, le=1000)
    aggregate: Optional[Literal["min", "max", "mean", "final", "time_of_min", "time_of_max"]] = None

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.start_s is not None and self.end_s is not None and self.end_s < self.start_s:
            raise ValueError("end_s must be greater than or equal to start_s")
        return self


PUBLIC_SCHEMA_MODELS = {
    model.__name__: model
    for model in (
        Artifact,
        CaseInspection,
        CaseList,
        CaseManifest,
        CompleteCaseUploadRequest,
        CreateCaseUploadRequest,
        DynamicsJobRequest,
        DynamicsResultSummary,
        DownloadTarget,
        Job,
        JobList,
        PowerFlowJobRequest,
        PowerFlowResultSummary,
        ResultQuery,
        ResultQueryResponse,
        ServiceCapabilities,
        SignalDescriptor,
        SignalList,
        UploadSession,
    )
}


def public_json_schemas() -> Dict[str, Dict[str, Any]]:
    """Return JSON Schemas for every externally visible contract."""

    return {name: model.model_json_schema() for name, model in PUBLIC_SCHEMA_MODELS.items()}
