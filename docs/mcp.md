# UQGrid MCP Server

UQGrid includes an optional local Model Context Protocol (MCP) server. The MCP
layer is intentionally thin: it exposes stable tools backed by `uqgrid.api`
rather than calling scripts or depending on repository layout.

## Install

From the UQGrid repository:

```bash
cd /Users/emconsta/Research/REPO/uqgrid
python -m pip install -e '.[mcp]'
```

This installs the optional MCP SDK dependency and the `uqgrid-mcp` console
script.

## Local Client Configuration

Most MCP clients need a JSON entry that tells them how to start the server. Use
absolute paths to the Python environment that has UQGrid installed.

```json
{
  "mcpServers": {
    "uqgrid": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "uqgrid_mcp.server"]
    }
  }
}
```

If the console script is available, this is equivalent:

```json
{
  "mcpServers": {
    "uqgrid": {
      "command": "/path/to/venv/bin/uqgrid-mcp",
      "args": []
    }
  }
}
```

The current server runs over stdio, so it is meant to be launched by the MCP
client as a local process.

## Tools

The server currently exposes:

- `get_uqgrid_info`: Return UQGrid version, optional dependency status, and API
  capabilities.
- `generate_osl_dataset`: Generate an OSL-style PMU dataset.
- `inspect_osl_dataset`: Inspect an existing OSL dataset manifest without
  loading dense case arrays.

`generate_osl_dataset` requires RAW and DYR paths after config merging. Provide
them either in `config_path` or directly as `raw` and `dyr` tool arguments.

## Example Dataset Request

Use absolute paths in MCP calls so results do not depend on the client's working
directory.

```json
{
  "config_path": "/Users/emconsta/Research/REPO/uqgrid/scripts/osl/configs/activsg200_small.json",
  "outdir": "/Users/emconsta/Research/REPO/uqgrid/outputs/osl_dataset/location_sweep_sparse_obs",
  "fo_buses": [49, 50, 51, 52],
  "freqs": [0.8],
  "amplitudes": [0.2],
  "fo_start": 2.0,
  "tend": 8.0,
  "observed_buses": [49, 50, 51, 52, 60, 70],
  "p_class_fraction": 1.0,
  "missing_rate": 0.0,
  "colored_noise": false,
  "overwrite": true
}
```

Because the config file above includes `raw` and `dyr`, they are not repeated.
Without `config_path`, include them explicitly:

```json
{
  "raw": "/Users/emconsta/Research/REPO/uqgrid/data/ACTIVSg200.raw",
  "dyr": "/Users/emconsta/Research/REPO/uqgrid/data/ACTIVSg200.dyr",
  "outdir": "/Users/emconsta/Research/REPO/uqgrid/outputs/osl_dataset/example",
  "fo_buses": [49],
  "freqs": [0.8],
  "amplitudes": [0.2],
  "overwrite": true
}
```

## Debugging

Use MCP Inspector to test the server outside a chat client:

```bash
npx @modelcontextprotocol/inspector \
  /path/to/venv/bin/python \
  -m uqgrid_mcp.server
```

In the Inspector, connect to the server, open the tools tab, run
`get_uqgrid_info`, then test `generate_osl_dataset` with a small request.

If a client cannot connect:

- Confirm `python -m uqgrid_mcp.server` starts without import errors.
- Confirm the client config uses absolute paths.
- Check the MCP client's server logs.
- Reinstall with `python -m pip install -e '.[mcp]'` if the MCP SDK is missing.
