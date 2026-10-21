import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from uqgrid.simulation import dynamics
from uqgrid.simulation.config import IntegrationConfig


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_import_dynamics_does_not_call_petsc_init(tmp_path):
    fake_pkg = tmp_path / "petsc4py"
    fake_pkg.mkdir()
    (fake_pkg / "__init__.py").write_text(
        "\n".join(
            [
                "init_calls = []",
                "class _PETSc:",
                "    pass",
                "PETSc = _PETSc()",
                "def init(args=None):",
                "    init_calls.append(list(args or []))",
                "    raise RuntimeError('petsc4py.init should not run during import')",
            ]
        )
    )

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(fake_pkg.parent), str(repo_root), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import uqgrid.simulation.dynamics; "
            "import petsc4py; "
            "assert petsc4py.init_calls == []",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_petsc_config_initializes_with_only_petsc_args(monkeypatch):
    calls = []
    fake_petsc = object()
    fake_petsc4py = types.ModuleType("petsc4py")
    fake_petsc4py.PETSc = fake_petsc

    def init(args=None):
        calls.append(list(args or []))

    fake_petsc4py.init = init
    monkeypatch.setitem(sys.modules, "petsc4py", fake_petsc4py)
    monkeypatch.setattr(dynamics, "petsc4py", None)
    monkeypatch.setattr(dynamics, "PETSc", None)

    disabled = IntegrationConfig(petsc=False, petsc_args=["-ignored"])
    assert dynamics._get_petsc_for_config(disabled) is None
    assert calls == []

    enabled = IntegrationConfig(petsc=True, petsc_args=["-ksp_type", "gmres"])
    assert dynamics._get_petsc_for_config(enabled) is fake_petsc
    assert calls == [["-ksp_type", "gmres"]]


def test_dynamics_driver_preserves_sys_argv_and_stores_petsc_args(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / "bin" / "dynamics_driver.py", "test_dynamics_driver")
    argv = [
        "dynamics_driver.py",
        "--raw",
        "case.raw",
        "--dyr",
        "case.dyr",
        "--petsc",
        "--method",
        "cn",
        "--",
        "-ts_monitor",
        "-ksp_type",
        "gmres",
    ]
    monkeypatch.setattr(sys, "argv", argv.copy())

    raw, dyr, _, config, _ = module.parse_args()

    assert sys.argv == argv
    assert raw == "case.raw"
    assert dyr == "case.dyr"
    assert config.petsc is True
    assert config.method == "cn"
    assert config.petsc_args == ["-ts_monitor", "-ksp_type", "gmres"]


@pytest.mark.parametrize(
    "arguments,exit_code,message",
    [
        (["--arkimex"], 2, "unrecognized arguments: --arkimex"),
        (["--slow-diff", "0,1"], 2, "unrecognized arguments: --slow-diff"),
        (["--fast-diff", "0,1"], 2, "unrecognized arguments: --fast-diff"),
        (["--method", "arkimex"], 2, "invalid choice"),
        (["--", "-ts_arkimex_type", "a2"], 1, "ARKIMEX has been removed"),
        (["--", "-ts_arkimex_type=a2"], 1, "ARKIMEX has been removed"),
        (["--", "-ts_type", "arkimex"], 1, "cannot override"),
        (["--", "-ts_type=arkimex"], 1, "cannot override"),
    ],
)
def test_dynamics_driver_rejects_removed_arkimex_requests(
    monkeypatch, capsys, arguments, exit_code, message,
):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / "bin" / "dynamics_driver.py", "test_dynamics_driver")
    argv = ["dynamics_driver.py", "--raw", "case.raw", "--dyr", "case.dyr", "--petsc"]
    monkeypatch.setattr(sys, "argv", argv + arguments)

    def unexpected_initialization(*args, **kwargs):
        raise AssertionError("Rejected requests must not initialize PETSc")

    monkeypatch.setattr(dynamics, "_initialize_petsc", unexpected_initialization)
    with pytest.raises(SystemExit) as error:
        module.parse_args()

    assert error.value.code == exit_code
    captured = capsys.readouterr()
    assert message in captured.out + captured.err


@pytest.mark.parametrize(
    "arguments,method,petsc,petsc_args",
    [
        ([], "cn", False, []),
        (["--method", "beuler"], "beuler", False, []),
        (["--petsc", "--", "-ksp_type", "gmres"], "cn", True, ["-ksp_type", "gmres"]),
    ],
)
def test_dynamics_driver_defaults_and_explicit_method_without_petsc_init(
    monkeypatch, arguments, method, petsc, petsc_args,
):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / "bin" / "dynamics_driver.py", "test_dynamics_driver")
    argv = ["dynamics_driver.py", "--raw", "case.raw", "--dyr", "case.dyr"] + arguments
    monkeypatch.setattr(sys, "argv", argv.copy())

    def unexpected_initialization(*args, **kwargs):
        pytest.fail("Parsing integration options must not initialize PETSc")

    monkeypatch.setattr(dynamics, "_initialize_petsc", unexpected_initialization)

    _, _, _, config, _ = module.parse_args()

    assert config.method == method
    assert config.petsc is petsc
    assert config.petsc_args == petsc_args
    assert sys.argv == argv
