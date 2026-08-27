"""MCP tools and resources for the UQGrid remote service."""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import uqgrid
from mcp.server import MCPServer
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import AnyHttpUrl
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

from uqgrid.service import (
    CaseList,
    CaseInspection,
    CaseManifest,
    CapabilitySigner,
    CreateCaseUploadRequest,
    DownloadManager,
    DownloadTarget,
    DynamicsJobRequest,
    InMemoryJobRepository,
    Job,
    JobList,
    LocalArtifactStore,
    LocalCaseRepository,
    LocalResultRepository,
    PowerFlowJobRequest,
    ResultQuery,
    ResultQueryResponse,
    ServiceCapabilities,
    SignalList,
    SimulationService,
    TransferAuthorizationError,
    UploadConflictError,
    UploadManager,
    UploadNotFoundError,
    UploadSession,
    UploadValidationError,
)
from .security import StaticTokenVerifier


logger = logging.getLogger(__name__)


def _csv_environment(name: str, default: str):
    return [item.strip() for item in os.environ.get(name, default).split(",") if item.strip()]


def create_server(
    data_root=None,
    owner_id=None,
    public_base_url=None,
    signing_secret=None,
    api_token=None,
    max_concurrent_jobs=None,
    max_simulation_seconds=None,
    max_simulation_steps=None,
):
    """Create an MCP server backed by an isolated service data directory."""

    root = Path(data_root or os.environ.get("UQGRID_SERVICE_DATA", ".uqgrid-service"))
    owner = owner_id or os.environ.get("UQGRID_OWNER_ID", "local-user")
    base_url = public_base_url or os.environ.get("UQGRID_PUBLIC_BASE_URL", "http://127.0.0.1:8000")
    configured_secret = signing_secret or os.environ.get("UQGRID_ARTIFACT_SIGNING_SECRET")
    secret = (
        configured_secret.encode("utf-8")
        if isinstance(configured_secret, str)
        else configured_secret
    )
    if secret is None:
        secret = os.urandom(32)
    token = api_token or os.environ.get("UQGRID_API_TOKEN")
    concurrency = int(max_concurrent_jobs or os.environ.get("UQGRID_MAX_CONCURRENT_JOBS", "2"))
    maximum_seconds = float(
        max_simulation_seconds or os.environ.get("UQGRID_MAX_SIMULATION_SECONDS", "60")
    )
    maximum_steps = int(
        max_simulation_steps or os.environ.get("UQGRID_MAX_SIMULATION_STEPS", "100000")
    )
    artifacts = LocalArtifactStore(root / "artifacts")
    cases = LocalCaseRepository(root / "cases", artifacts)
    results = LocalResultRepository(root / "results", artifacts)
    simulations = SimulationService(cases, results)
    signer = CapabilitySigner(secret)
    uploads = UploadManager(root / "uploads", cases, signer, base_url)
    downloads = DownloadManager(artifacts, signer, base_url)
    jobs = InMemoryJobRepository()

    from uqgrid.service import JobManager

    manager = JobManager(
        simulations,
        repository=jobs,
        max_concurrent_jobs=concurrency,
    )
    server_options = {}
    if token:
        server_options.update(
            token_verifier=StaticTokenVerifier(token, f"{base_url}/mcp", owner),
            auth=AuthSettings(
                issuer_url=AnyHttpUrl(f"{base_url}/"),
                resource_server_url=AnyHttpUrl(f"{base_url}/mcp"),
                required_scopes=["uqgrid"],
            ),
        )
    server = MCPServer(
        "uqgrid",
        version=uqgrid.__version__,
        instructions=(
            "Use case IDs rather than filesystem paths. Submit simulations as jobs, "
            "poll get_job, then request summaries or bounded signal data."
        ),
        **server_options,
    )

    @server.tool()
    def list_cases() -> CaseList:
        """List cases available to the current UQGrid user."""

        return CaseList(cases=cases.list(owner))

    @server.tool()
    def create_case_upload(
        name: str,
        files: list[dict[str, object]],
    ) -> UploadSession:
        """Create signed HTTP PUT targets for a RAW/DYR or MATPOWER case bundle."""

        return uploads.create(owner, CreateCaseUploadRequest(name=name, files=files))

    @server.tool()
    def complete_case_upload(upload_id: str) -> CaseManifest:
        """Verify all uploaded files and create an immutable case manifest."""

        return uploads.complete(owner, upload_id)

    @server.tool()
    def inspect_case(case_id: str) -> CaseInspection:
        """Parse a case and report its topology and dynamic model coverage."""

        return cases.inspect(owner, case_id)

    @server.tool()
    def submit_power_flow(
        case_id: str,
        enforce_q_limits: bool = True,
        q_limit_tolerance: float = 1e-8,
        max_q_limit_iterations: Optional[int] = None,
        idempotency_key: Optional[str] = None,
    ) -> Job:
        """Submit an AC power-flow job and return immediately with its job ID."""

        request = PowerFlowJobRequest(
            case_id=case_id,
            options={
                "enforce_q_limits": enforce_q_limits,
                "q_limit_tolerance": q_limit_tolerance,
                "max_q_limit_iterations": max_q_limit_iterations,
            },
            idempotency_key=idempotency_key,
        )
        return manager.submit_power_flow(owner, request)

    @server.tool()
    def submit_dynamics(
        case_id: str,
        fault_bus_id: int,
        fault_impedance_pu: float = 0.01,
        fault_start_s: float = 0.1,
        fault_clear_s: float = 0.2,
        end_s: float = 5.0,
        dt_s: float = 1.0 / 120.0,
        idempotency_key: Optional[str] = None,
    ) -> Job:
        """Submit a non-PETSc backward Euler bus-fault simulation."""

        if end_s > maximum_seconds:
            raise ValueError(f"end_s exceeds server maximum of {maximum_seconds} seconds")
        if end_s / dt_s > maximum_steps:
            raise ValueError(f"simulation exceeds server maximum of {maximum_steps} steps")

        request = DynamicsJobRequest(
            case_id=case_id,
            scenario={
                "events": [{
                    "bus_id": fault_bus_id,
                    "impedance_pu": fault_impedance_pu,
                    "start_s": fault_start_s,
                    "clear_s": fault_clear_s,
                }]
            },
            integration={"dt_s": dt_s, "end_s": end_s},
            idempotency_key=idempotency_key,
        )
        return manager.submit_dynamics(owner, request)

    @server.tool()
    def get_job(job_id: str) -> Job:
        """Get the current state of a submitted simulation job."""

        return manager.get(owner, job_id)

    @server.tool()
    def list_jobs() -> JobList:
        """List simulation jobs belonging to the current UQGrid user."""

        return JobList(jobs=manager.list(owner))

    @server.tool()
    def cancel_job(job_id: str) -> Job:
        """Cancel a queued or running simulation job."""

        return manager.cancel(owner, job_id)

    @server.tool()
    def get_result_summary(result_id: str) -> dict[str, object]:
        """Get the compact summary for a completed simulation result."""

        return results.get_summary(owner, result_id).model_dump(mode="json")

    @server.tool()
    def list_result_signals(result_id: str) -> SignalList:
        """List semantic signals available in a completed result."""

        return SignalList(result_id=result_id, signals=results.list_signals(owner, result_id))

    @server.tool()
    def query_result(
        result_id: str,
        signals: list[str],
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
        max_points: int = 200,
        aggregate: Optional[str] = None,
    ) -> ResultQueryResponse:
        """Retrieve bounded traces or aggregate values for named result signals."""

        query = ResultQuery(
            result_id=result_id,
            signals=signals,
            start_s=start_s,
            end_s=end_s,
            max_points=max_points,
            aggregate=aggregate,
        )
        return results.query(owner, query)

    @server.tool()
    def get_artifact_download(artifact_id: str) -> DownloadTarget:
        """Create a short-lived HTTP download URL for one result artifact."""

        return downloads.create(owner, artifact_id)

    @server.custom_route(
        "/api/v1/artifact-uploads/{upload_id}/{slot_id}", methods=["PUT"]
    )
    async def upload_route(request: Request):
        token = request.query_params.get("token", "")
        try:
            result = await uploads.receive(
                request.path_params["upload_id"],
                request.path_params["slot_id"],
                token,
                request.stream(),
            )
            logger.info(
                "artifact upload completed owner=%s upload_id=%s slot_id=%s bytes=%s",
                owner,
                result["upload_id"],
                result["slot_id"],
                result["size_bytes"],
            )
            return JSONResponse(result, status_code=201)
        except TransferAuthorizationError as exc:
            return JSONResponse({"error": str(exc)}, status_code=401)
        except UploadNotFoundError:
            return JSONResponse({"error": "upload not found"}, status_code=404)
        except UploadConflictError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        except UploadValidationError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)

    @server.custom_route(
        "/api/v1/artifacts/{artifact_id}", methods=["GET", "HEAD"]
    )
    async def download_route(request: Request):
        token = request.query_params.get("token", "")
        try:
            owner_id, artifact, path = downloads.authorize(
                request.path_params["artifact_id"], token
            )
            response = FileResponse(
                path,
                media_type=artifact.media_type,
                filename=artifacts.filename(owner_id, artifact.artifact_id),
            )
            response.headers["Cache-Control"] = "private, no-store"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["ETag"] = f'"{artifact.sha256}"'
            return response
        except (TransferAuthorizationError, KeyError):
            return JSONResponse({"error": "artifact not found"}, status_code=404)

    @server.custom_route("/health/live", methods=["GET"])
    async def liveness(request: Request):
        return JSONResponse({"status": "ok"})

    @server.custom_route("/health/ready", methods=["GET"])
    async def readiness(request: Request):
        ready = all(path.exists() and path.is_dir() for path in (artifacts.root, cases.root, results.root))
        return JSONResponse({"status": "ready" if ready else "not_ready"}, status_code=200 if ready else 503)

    @server.resource("uqgrid://service/capabilities", mime_type="application/json")
    def capabilities() -> str:
        """Describe this server's supported case and simulation features."""

        value = ServiceCapabilities(
            uqgrid_version=uqgrid.__version__,
            case_formats=["psse-raw-dyr", "matpower-m"],
            simulation_kinds=["power_flow", "dynamics"],
            integration_methods=["beuler"],
            supports_petsc=False,
            max_query_points=1000,
        )
        return value.model_dump_json(indent=2)

    @server.resource("uqgrid://cases/{case_id}/manifest", mime_type="application/json")
    def case_manifest(case_id: str) -> str:
        """Return an immutable case manifest."""

        return cases.get(owner, case_id).model_dump_json(indent=2)

    @server.resource("uqgrid://results/{result_id}/summary", mime_type="application/json")
    def result_summary(result_id: str) -> str:
        """Return a completed result summary."""

        return results.get_summary(owner, result_id).model_dump_json(indent=2)

    server._uqgrid_services = {
        "artifacts": artifacts,
        "cases": cases,
        "results": results,
        "simulations": simulations,
        "jobs": manager,
        "uploads": uploads,
        "downloads": downloads,
        "owner_id": owner,
        "transport_security": TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=_csv_environment(
                "UQGRID_ALLOWED_HOSTS", "127.0.0.1:*,localhost:*"
            ),
            allowed_origins=_csv_environment(
                "UQGRID_ALLOWED_ORIGINS", "http://127.0.0.1:*,http://localhost:*"
            ),
        ),
    }
    return server


mcp = create_server()


def main():
    parser = argparse.ArgumentParser(description="Run the UQGrid MCP server")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default=os.environ.get("UQGRID_MCP_TRANSPORT", "stdio"),
    )
    parser.add_argument("--host", default=os.environ.get("UQGRID_MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("UQGRID_MCP_PORT", "8000")))
    args = parser.parse_args()
    if args.transport == "streamable-http":
        if "UQGRID_ARTIFACT_SIGNING_SECRET" not in os.environ:
            parser.error("streamable-http requires UQGRID_ARTIFACT_SIGNING_SECRET")
        if "UQGRID_API_TOKEN" not in os.environ:
            parser.error("streamable-http requires UQGRID_API_TOKEN")
    kwargs = {}
    if args.transport == "streamable-http":
        kwargs.update(
            host=args.host,
            port=args.port,
            streamable_http_path="/mcp",
            transport_security=mcp._uqgrid_services["transport_security"],
        )
    mcp.run(transport=args.transport, **kwargs)


if __name__ == "__main__":
    main()
