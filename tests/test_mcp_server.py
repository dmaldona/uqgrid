import asyncio
import sys
from pathlib import Path

import pytest

mcp_package = pytest.importorskip("mcp")
from mcp import Client

from uqgrid.mcp import server as server_module
from uqgrid.mcp.server import create_server


ROOT = Path(__file__).resolve().parents[1]


def run(coro):
    return asyncio.run(coro)


def test_mcp_discovers_tools_and_capability_resource(tmp_path):
    server = create_server(tmp_path, owner_id="alice")

    async def exercise():
        async with Client(server, raise_exceptions=True) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {"list_cases", "submit_power_flow", "get_job", "query_result"} <= names
            assert "import_local_case" not in names
            resource = await client.read_resource("uqgrid://service/capabilities")
            assert '"service": "uqgrid"' in resource.contents[0].text

    run(exercise())


def test_stdio_server_imports_absolute_local_case_paths(tmp_path):
    server = create_server(
        tmp_path,
        owner_id="alice",
        enable_local_case_import=True,
    )

    async def exercise():
        async with Client(server) as client:
            tools = await client.list_tools()
            assert "import_local_case" in {tool.name for tool in tools.tools}
            relative = await client.call_tool(
                "import_local_case",
                {"name": "invalid", "paths": ["data/ieee9_v33.raw"]},
            )
            assert relative.is_error
            imported = await client.call_tool(
                "import_local_case",
                {
                    "name": "ieee9",
                    "paths": [
                        str(ROOT / "data/ieee9_v33.raw"),
                        str(ROOT / "data/ieee9bus.dyr"),
                    ],
                },
            )
            assert not imported.is_error
            assert imported.structured_content["status"] == "ready"
            listed = await client.call_tool("list_cases", {})
            assert [item["name"] for item in listed.structured_content["cases"]] == [
                "ieee9"
            ]

    run(exercise())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"api_token": ""}, "at least 32 characters"),
        ({"signing_secret": ""}, "at least 32 bytes"),
    ],
)
def test_configured_empty_credentials_fail_closed(monkeypatch, tmp_path, kwargs, message):
    monkeypatch.delenv("UQGRID_API_TOKEN", raising=False)
    monkeypatch.delenv("UQGRID_ARTIFACT_SIGNING_SECRET", raising=False)

    with pytest.raises(ValueError, match=message):
        create_server(tmp_path, owner_id="alice", **kwargs)


@pytest.mark.parametrize(
    "empty_name",
    ["UQGRID_API_TOKEN", "UQGRID_ARTIFACT_SIGNING_SECRET"],
)
def test_http_main_rejects_empty_credentials(monkeypatch, capsys, empty_name):
    monkeypatch.setenv("UQGRID_API_TOKEN", "t" * 32)
    monkeypatch.setenv("UQGRID_ARTIFACT_SIGNING_SECRET", "s" * 32)
    monkeypatch.setenv(empty_name, "")
    monkeypatch.setattr(
        sys,
        "argv",
        ["uqgrid-mcp", "--transport", "streamable-http"],
    )

    with pytest.raises(SystemExit):
        server_module.main()

    assert empty_name in capsys.readouterr().err


def test_job_runtime_limit_configuration(monkeypatch, tmp_path):
    monkeypatch.delenv("UQGRID_MAX_JOB_RUNTIME_SECONDS", raising=False)
    default_server = create_server(tmp_path / "default", owner_id="alice")
    assert default_server._uqgrid_services["jobs"].max_job_runtime_seconds == 300

    monkeypatch.setenv("UQGRID_MAX_JOB_RUNTIME_SECONDS", "12.5")
    environment_server = create_server(tmp_path / "environment", owner_id="alice")
    assert environment_server._uqgrid_services["jobs"].max_job_runtime_seconds == 12.5

    explicit_server = create_server(
        tmp_path / "explicit",
        owner_id="alice",
        max_job_runtime_seconds=7,
    )
    assert explicit_server._uqgrid_services["jobs"].max_job_runtime_seconds == 7


def test_mcp_runs_power_flow_job_end_to_end(tmp_path):
    server = create_server(tmp_path, owner_id="alice")
    services = server._uqgrid_services
    case = services["cases"].import_files(
        "alice", "ieee9", [ROOT / "data/ieee9_v33.raw"]
    )

    async def exercise():
        async with Client(server, raise_exceptions=True) as client:
            submitted = await client.call_tool("submit_power_flow", {"case_id": case.case_id})
            job_id = submitted.structured_content["job_id"]
            services["jobs"].wait("alice", job_id, timeout=30)
            completed = await client.call_tool("get_job", {"job_id": job_id})
            assert completed.structured_content["status"] == "succeeded"
            result_id = completed.structured_content["result_id"]
            summary = await client.call_tool("get_result_summary", {"result_id": result_id})
            assert summary.structured_content["converged"] is True
            signals = await client.call_tool("list_result_signals", {"result_id": result_id})
            assert len(signals.structured_content["signals"]) == 18

    run(exercise())
