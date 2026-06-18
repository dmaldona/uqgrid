"""MCP server entrypoint for UQGrid."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Sequence

from .tools import (
    generate_osl_dataset_tool,
    get_uqgrid_info as get_uqgrid_info_tool,
    inspect_osl_dataset_tool,
)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without optional extra
    FastMCP = None
    _MCP_IMPORT_ERROR = exc
else:
    _MCP_IMPORT_ERROR = None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options for the UQGrid MCP server."""

    parser = argparse.ArgumentParser(description="Run the UQGrid MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport to use. Defaults to stdio.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for streamable HTTP transport. Defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for streamable HTTP transport. Defaults to 8000.",
    )
    parser.add_argument(
        "--mcp-path",
        default="/mcp",
        help="Path for streamable HTTP MCP endpoint. Defaults to /mcp.",
    )
    return parser.parse_args(argv)


def create_server(
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    mcp_path: str = "/mcp",
):
    """Create the UQGrid MCP server."""

    if FastMCP is None:
        raise RuntimeError(
            "The MCP SDK is not installed. Install UQGrid with the MCP extra, "
            "for example: pip install -e '.[mcp]'"
        ) from _MCP_IMPORT_ERROR

    use_http = transport == "streamable-http"
    server = FastMCP(
        "uqgrid",
        host=host,
        port=port,
        streamable_http_path=mcp_path,
        stateless_http=use_http,
        json_response=use_http,
    )

    @server.tool()
    def get_uqgrid_info() -> Dict[str, Any]:
        """Return UQGrid version, optional dependency status, and API capabilities."""

        return get_uqgrid_info_tool()

    @server.tool()
    def generate_osl_dataset(
        config_path: Optional[str] = None,
        raw: Optional[str] = None,
        dyr: Optional[str] = None,
        outdir: Optional[str] = None,
        tend: Optional[float] = None,
        dt: Optional[float] = None,
        fo_start: Optional[float] = None,
        fo_buses: Optional[List[int]] = None,
        freqs: Optional[List[float]] = None,
        amplitudes: Optional[List[float]] = None,
        seed_start: Optional[int] = None,
        limit: Optional[int] = None,
        observed_buses: Optional[Any] = None,
        pmu_rate_hz: Optional[float] = None,
        p_class_fraction: Optional[float] = None,
        missing_rate: Optional[float] = None,
        colored_noise: Optional[bool] = None,
        noise_sigma_lf: Optional[float] = None,
        noise_sigma_hf: Optional[float] = None,
        noise_tau_lf_range: Optional[List[float]] = None,
        overwrite: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Generate an OSL-style PMU dataset.

        RAW and DYR paths must be supplied here or by config_path.
        """

        return generate_osl_dataset_tool(
            config_path=config_path,
            raw=raw,
            dyr=dyr,
            outdir=outdir,
            tend=tend,
            dt=dt,
            fo_start=fo_start,
            fo_buses=fo_buses,
            freqs=freqs,
            amplitudes=amplitudes,
            seed_start=seed_start,
            limit=limit,
            observed_buses=observed_buses,
            pmu_rate_hz=pmu_rate_hz,
            p_class_fraction=p_class_fraction,
            missing_rate=missing_rate,
            colored_noise=colored_noise,
            noise_sigma_lf=noise_sigma_lf,
            noise_sigma_hf=noise_sigma_hf,
            noise_tau_lf_range=noise_tau_lf_range,
            overwrite=overwrite,
        )

    @server.tool()
    def inspect_osl_dataset(outdir: str, include_rows: bool = True) -> Dict[str, Any]:
        """Inspect an OSL dataset manifest by output directory."""

        return inspect_osl_dataset_tool(outdir=outdir, include_rows=include_rows)

    return server


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the UQGrid MCP server."""

    args = parse_args(argv)
    server = create_server(
        transport=args.transport,
        host=args.host,
        port=args.port,
        mcp_path=args.mcp_path,
    )
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
