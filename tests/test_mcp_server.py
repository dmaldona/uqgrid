from uqgrid_mcp import server as server_module


class DummyFastMCP:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.tool_names = []
        self.run_calls = []
        DummyFastMCP.instances.append(self)

    def tool(self):
        def register(func):
            self.tool_names.append(func.__name__)
            return func

        return register

    def run(self, transport="stdio", mount_path=None):
        self.run_calls.append({"transport": transport, "mount_path": mount_path})


def _patch_fastmcp(monkeypatch):
    DummyFastMCP.instances.clear()
    monkeypatch.setattr(server_module, "FastMCP", DummyFastMCP)
    monkeypatch.setattr(server_module, "_MCP_IMPORT_ERROR", None)


def test_parse_args_defaults_to_stdio():
    args = server_module.parse_args([])

    assert args.transport == "stdio"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.mcp_path == "/mcp"


def test_parse_args_accepts_streamable_http_options():
    args = server_module.parse_args([
        "--transport",
        "streamable-http",
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
        "--mcp-path",
        "/custom-mcp",
    ])

    assert args.transport == "streamable-http"
    assert args.host == "127.0.0.1"
    assert args.port == 8765
    assert args.mcp_path == "/custom-mcp"


def test_main_runs_stdio_by_default(monkeypatch):
    _patch_fastmcp(monkeypatch)

    server_module.main([])

    instance = DummyFastMCP.instances[0]
    assert instance.args == ("uqgrid",)
    assert instance.kwargs["host"] == "127.0.0.1"
    assert instance.kwargs["port"] == 8000
    assert instance.kwargs["streamable_http_path"] == "/mcp"
    assert instance.kwargs["stateless_http"] is False
    assert instance.kwargs["json_response"] is False
    assert instance.run_calls == [{"transport": "stdio", "mount_path": None}]
    assert set(instance.tool_names) == {
        "get_uqgrid_info",
        "generate_osl_dataset",
        "inspect_osl_dataset",
    }


def test_main_runs_streamable_http_with_local_defaults(monkeypatch):
    _patch_fastmcp(monkeypatch)

    server_module.main(["--transport", "streamable-http"])

    instance = DummyFastMCP.instances[0]
    assert instance.kwargs["host"] == "127.0.0.1"
    assert instance.kwargs["port"] == 8000
    assert instance.kwargs["streamable_http_path"] == "/mcp"
    assert instance.kwargs["stateless_http"] is True
    assert instance.kwargs["json_response"] is True
    assert instance.run_calls == [{"transport": "streamable-http", "mount_path": None}]


def test_main_passes_custom_streamable_http_settings(monkeypatch):
    _patch_fastmcp(monkeypatch)

    server_module.main([
        "--transport",
        "streamable-http",
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
        "--mcp-path",
        "/custom-mcp",
    ])

    instance = DummyFastMCP.instances[0]
    assert instance.kwargs["host"] == "127.0.0.1"
    assert instance.kwargs["port"] == 8765
    assert instance.kwargs["streamable_http_path"] == "/custom-mcp"
    assert instance.kwargs["stateless_http"] is True
    assert instance.kwargs["json_response"] is True
    assert instance.run_calls == [{"transport": "streamable-http", "mount_path": None}]
