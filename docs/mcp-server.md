# Remote MCP Server

UQGrid can run as a remote Model Context Protocol server. MCP carries case and
job metadata; signed HTTP URLs carry case files and complete result artifacts.

## Install

The server requires Python 3.10 or newer. Core UQGrid retains its lower Python
version support.

```bash
pip install -e ".[mcp]"
```

## Local stdio

```bash
export UQGRID_SERVICE_DATA="$PWD/.uqgrid-service"
uqgrid-mcp --transport stdio
```

An MCP host can launch the same command directly. Stdio is intended for local,
single-user use and obtains trust from the process environment.

The stdio server exposes `import_local_case`, which accepts a case name and
absolute paths to one `.m` file or one `.raw` file with an optional `.dyr`
file. The server process reads those paths directly. This tool is available
only when the CLI runs with `--transport stdio`; it is deliberately absent
from HTTP and default programmatic servers so remote clients cannot request
arbitrary server-side files. The `uqgrid-remote upload` command remains the
case-ingestion path for HTTP deployments.

## Remote HTTP

Set secrets and network allowlists before starting HTTP:

```bash
export UQGRID_SERVICE_DATA=/var/lib/uqgrid
export UQGRID_PUBLIC_BASE_URL=https://uqgrid.example
export UQGRID_OWNER_ID=initial-user
export UQGRID_API_TOKEN='replace-with-at-least-32-random-characters'
export UQGRID_ARTIFACT_SIGNING_SECRET='replace-with-an-independent-32-byte-secret'
export UQGRID_ALLOWED_HOSTS='uqgrid.example'
export UQGRID_ALLOWED_ORIGINS='https://agent.example'
uqgrid-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

Terminate TLS at a trusted reverse proxy. The MCP endpoint is `/mcp`; liveness
and readiness are `/health/live` and `/health/ready`.

The initial authentication mechanism is a pre-shared bearer token:

```http
Authorization: Bearer replace-with-at-least-32-random-characters
```

This is appropriate for a controlled single-user deployment. Before enabling
organization-wide access, replace it with an external OAuth/OIDC token verifier
with audience validation and per-user ownership.

## Upload A Case

```bash
uqgrid-remote upload \
  --server https://uqgrid.example/mcp \
  --token "$UQGRID_API_TOKEN" \
  --name ieee9 \
  --raw data/ieee9_v33.raw \
  --dyr data/ieee9bus.dyr
```

The token can be omitted from the command when `UQGRID_API_TOKEN` is exported.
Upload and download URLs contain their own short-lived HMAC capability and
never contain the MCP token.

## Limits

The following environment variables protect shared compute and storage:

| Variable | Default |
| --- | --- |
| `UQGRID_MAX_CONCURRENT_JOBS` | `2` |
| `UQGRID_MAX_SIMULATION_SECONDS` | `60` |
| `UQGRID_MAX_SIMULATION_STEPS` | `100000` |
| `UQGRID_MAX_JOB_RUNTIME_SECONDS` | `300` |
| Upload size | 512 MiB per declared file |
| Trace query points | 1000 maximum |

`UQGRID_MAX_SIMULATION_SECONDS` limits the simulated `end_s` requested by a
dynamics job. `UQGRID_MAX_JOB_RUNTIME_SECONDS` is a separate wall-clock
deadline for every power-flow or dynamics worker. A worker that exceeds the
deadline is terminated and the job fails with `worker_timeout`.

## Current Durability Boundary

Jobs are process-isolated, but their registry is in memory. Server restart
recovery, database-backed job ownership, transactional result publication, and
automatic orphan cleanup are deferred. Do not treat this release as a durable
multi-user service until those controls are implemented.

## Container

Build the dedicated server image with:

```bash
docker build -f Dockerfile.mcp -t uqgrid-mcp .
docker run --rm -p 8000:8000 \
  -v uqgrid-data:/var/lib/uqgrid \
  -e UQGRID_PUBLIC_BASE_URL=https://uqgrid.example \
  -e UQGRID_API_TOKEN="$UQGRID_API_TOKEN" \
  -e UQGRID_ARTIFACT_SIGNING_SECRET="$UQGRID_ARTIFACT_SIGNING_SECRET" \
  -e UQGRID_ALLOWED_HOSTS=uqgrid.example \
  -e UQGRID_ALLOWED_ORIGINS=https://agent.example \
  uqgrid-mcp
```

Do not place secrets in the image or source-controlled environment files.
