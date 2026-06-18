"""Tool implementations for the UQGrid MCP server.

These functions deliberately depend on ``uqgrid.api`` instead of scripts or
internal simulation modules, so the MCP layer remains a thin adapter.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from uqgrid import get_info as _get_uqgrid_info
from uqgrid.api.osl import generate_osl_dataset, inspect_osl_dataset, merge_osl_dataset_config


def get_uqgrid_info() -> Dict[str, Any]:
    """Return UQGrid runtime and API capability information."""

    info = dict(_get_uqgrid_info())
    info["api_capabilities"] = {
        "osl_dataset_generation": True,
        "osl_dataset_inspection": True,
    }
    return info


def generate_osl_dataset_tool(
    *,
    config_path: Optional[str] = None,
    raw: Optional[str] = None,
    dyr: Optional[str] = None,
    outdir: Optional[str] = None,
    tend: Optional[float] = None,
    dt: Optional[float] = None,
    fo_start: Optional[float] = None,
    fo_buses: Optional[Any] = None,
    freqs: Optional[Any] = None,
    amplitudes: Optional[Any] = None,
    seed_start: Optional[int] = None,
    limit: Optional[int] = None,
    observed_buses: Optional[Any] = None,
    pmu_rate_hz: Optional[float] = None,
    p_class_fraction: Optional[float] = None,
    missing_rate: Optional[float] = None,
    colored_noise: Optional[bool] = None,
    noise_sigma_lf: Optional[float] = None,
    noise_sigma_hf: Optional[float] = None,
    noise_tau_lf_range: Optional[Any] = None,
    overwrite: Optional[bool] = None,
) -> Dict[str, Any]:
    """Generate an OSL dataset and return a JSON-serializable summary."""

    overrides: Mapping[str, Any] = {
        "raw": raw,
        "dyr": dyr,
        "outdir": outdir,
        "tend": tend,
        "dt": dt,
        "fo_start": fo_start,
        "fo_buses": fo_buses,
        "freqs": freqs,
        "amplitudes": amplitudes,
        "seed_start": seed_start,
        "limit": limit,
        "observed_buses": observed_buses,
        "pmu_rate_hz": pmu_rate_hz,
        "p_class_fraction": p_class_fraction,
        "missing_rate": missing_rate,
        "colored_noise": colored_noise,
        "noise_sigma_lf": noise_sigma_lf,
        "noise_sigma_hf": noise_sigma_hf,
        "noise_tau_lf_range": noise_tau_lf_range,
        "overwrite": overwrite,
    }
    config = merge_osl_dataset_config(config_path=config_path, overrides=overrides)
    result = generate_osl_dataset(config)
    return result.to_dict(include_rows=True)


def inspect_osl_dataset_tool(outdir: str, include_rows: bool = True) -> Dict[str, Any]:
    """Inspect an OSL dataset manifest without loading dense case arrays."""

    inspection = inspect_osl_dataset(outdir)
    return inspection.to_dict(include_rows=include_rows)
