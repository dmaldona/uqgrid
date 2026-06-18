import json
from pathlib import Path

import numpy as np
import pytest

from uqgrid.api.osl import (
    OSLDatasetConfig,
    generate_osl_dataset,
    inspect_osl_dataset,
    merge_osl_dataset_config,
)
from uqgrid_mcp.tools import generate_osl_dataset_tool, get_uqgrid_info, inspect_osl_dataset_tool


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "2bus_33.raw"
DYR = REPO_ROOT / "data" / "2bus_TGOV1.dyr"


def test_merge_osl_dataset_config_applies_json_then_overrides(tmp_path):
    config_path = tmp_path / "osl.json"
    config_path.write_text(json.dumps({
        "raw": str(RAW),
        "dyr": str(DYR),
        "outdir": str(tmp_path / "from_json"),
        "fo_buses": [1],
        "freqs": [0.6],
        "amplitudes": [0.01],
        "colored_noise": True,
    }))

    config = merge_osl_dataset_config(
        config_path=config_path,
        overrides={
            "outdir": str(tmp_path / "from_override"),
            "freqs": [0.8],
            "colored_noise": False,
        },
    )

    assert config.raw == str(RAW)
    assert config.outdir == str(tmp_path / "from_override")
    assert config.freqs == [0.8]
    assert config.colored_noise is False
    assert config.config_path == str(config_path)


def test_merge_osl_dataset_config_requires_raw_and_dyr(tmp_path):
    with pytest.raises(ValueError, match="raw and dyr must be provided"):
        merge_osl_dataset_config(overrides={"outdir": str(tmp_path / "dataset")})


def test_generate_osl_dataset_writes_manifest_and_cases(tmp_path):
    config = OSLDatasetConfig(
        raw=RAW,
        dyr=DYR,
        outdir=tmp_path / "dataset",
        tend=0.5,
        dt=1.0 / 240.0,
        fo_start=0.05,
        fo_buses=[1],
        freqs=[0.8],
        amplitudes=[0.01],
        observed_buses=[1, 2],
        p_class_fraction=1.0,
        missing_rate=0.0,
        colored_noise=False,
        overwrite=True,
    )

    result = generate_osl_dataset(config)

    assert result.case_count == 1
    assert result.manifest_path.exists()
    row = result.rows[0]
    assert row["target"] == ["gov", 1]
    assert row["n_observed_buses"] == 2
    assert (result.outdir / row["npz"]).exists()
    assert (result.outdir / row["json"]).exists()

    loaded = np.load(result.outdir / row["npz"])
    assert "V_mag" in loaded.files

    inspection = inspect_osl_dataset(result.outdir)
    assert inspection.case_count == 1
    assert inspection.observed_bus_counts == [2]


def test_mcp_tools_wrap_api_without_mcp_sdk(tmp_path):
    info = get_uqgrid_info()
    assert info["api_capabilities"]["osl_dataset_generation"] is True

    empty = generate_osl_dataset_tool(
        raw=str(RAW),
        dyr=str(DYR),
        outdir=str(tmp_path / "empty_dataset"),
        fo_buses=[1],
        freqs=[0.8],
        amplitudes=[0.01],
        limit=0,
        overwrite=True,
    )
    assert empty["case_count"] == 0

    inspection = inspect_osl_dataset_tool(empty["outdir"], include_rows=False)
    assert inspection["case_count"] == 0
    assert "rows" not in inspection
