"""Command-line client for uploading case bundles to a remote UQGrid server."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def _file_description(path: Path):
    data = path.read_bytes()
    return data, {
        "name": path.name,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _put(url: str, data: bytes, headers: dict):
    request = Request(url, data=data, headers=headers, method="PUT")
    with urlopen(request, timeout=300) as response:
        if response.status != 201:
            raise RuntimeError(f"upload failed with HTTP {response.status}")


async def upload_case(server_url: str, name: str, paths, token=None):
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    transport = server_url
    http_client = None
    if token:
        import httpx2

        http_client = httpx2.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx2.Timeout(30.0, read=300.0),
        )
        transport = streamable_http_client(server_url, http_client=http_client)

    described = [_file_description(Path(path)) for path in paths]
    try:
        async with Client(transport) as client:
            created = await client.call_tool(
                "create_case_upload",
                {"name": name, "files": [description for _, description in described]},
            )
            if created.is_error:
                raise RuntimeError(created.content[0].text)
            session = created.structured_content
            by_name = {description["name"]: data for data, description in described}
            for target in session["targets"]:
                await asyncio.to_thread(
                    _put,
                    target["url"],
                    by_name[target["name"]],
                    target.get("headers", {}),
                )
            completed = await client.call_tool(
                "complete_case_upload", {"upload_id": session["upload_id"]}
            )
            if completed.is_error:
                raise RuntimeError(completed.content[0].text)
            return completed.structured_content
    finally:
        if http_client is not None:
            await http_client.aclose()


def main():
    parser = argparse.ArgumentParser(description="UQGrid remote service client")
    subparsers = parser.add_subparsers(dest="command", required=True)
    upload = subparsers.add_parser("upload", help="upload a case bundle")
    upload.add_argument("--server", required=True, help="MCP endpoint, such as http://host/mcp")
    upload.add_argument("--token", default=os.environ.get("UQGRID_API_TOKEN"))
    upload.add_argument("--name", required=True)
    upload.add_argument("--raw")
    upload.add_argument("--dyr")
    upload.add_argument("--matpower")
    args = parser.parse_args()
    paths = [value for value in (args.raw, args.dyr, args.matpower) if value]
    if not paths:
        parser.error("upload requires --raw or --matpower")
    result = asyncio.run(upload_case(args.server, args.name, paths, token=args.token))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
