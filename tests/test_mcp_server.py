import asyncio
from pathlib import Path

import pytest

mcp_package = pytest.importorskip("mcp")
from mcp import Client

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
            resource = await client.read_resource("uqgrid://service/capabilities")
            assert '"service": "uqgrid"' in resource.contents[0].text

    run(exercise())


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
