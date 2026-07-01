#!/usr/bin/env python
"""ACOPF-initialized UQGrid scenario generation.

This script keeps the original generator independent of ExaJuGO while adding
ACOPF smoke, replay smoke, and production ProbML final/min dataset modes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

for _thread_env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_var, "1")

import numpy as np
from joblib import Parallel, delayed


@dataclass(frozen=True)
class RawLoadRow:
    line_index: int
    parts: list[str]
    suffix: str


@dataclass(frozen=True)
class RawLoadSection:
    base_mva: float
    load_start: int
    load_end: int
    load_rows: list[RawLoadRow]


@dataclass(frozen=True)
class ParsedExaJuGOBasecase:
    bus_ids: np.ndarray
    bus_v_pu: np.ndarray
    bus_theta_deg: np.ndarray
    bus_bcs_mvar: np.ndarray
    load_bus_ids: np.ndarray
    p_load_mw: np.ndarray
    q_load_mvar: np.ndarray
    gen_bus_ids: np.ndarray
    gen_ids: list[str]
    p_gen_mw: np.ndarray
    q_gen_mvar: np.ndarray
    nonzero_gen_mask: np.ndarray


def _parse_base_mva(first_line: str) -> float:
    parts = first_line.split(",", maxsplit=2)
    if len(parts) < 2:
        raise ValueError(f"Could not parse base MVA from RAW first line: {first_line!r}")
    return float(parts[1].strip())


def _split_raw_row(line: str) -> tuple[list[str], str]:
    data, sep, comment = line.partition("/")
    suffix = f"{sep}{comment}" if sep else ""
    return [part.strip() for part in data.rstrip().split(",")], suffix


def _format_raw_row(parts: Sequence[str], suffix: str) -> str:
    return ", ".join(parts) + (f" {suffix}" if suffix else "")


def read_raw_load_section(raw_path: str | Path) -> RawLoadSection:
    """Read the PSS/E RAW load section and base MVA."""
    path = Path(raw_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"RAW file is empty: {path}")

    load_start = load_end = None
    for idx, line in enumerate(lines):
        upper = line.upper()
        if "BEGIN LOAD DATA" in upper:
            load_start = idx + 1
        elif "END OF LOAD DATA" in upper and load_start is not None:
            load_end = idx
            break

    if load_start is None or load_end is None:
        raise ValueError(f"Could not locate RAW load section in {path}")

    load_rows: list[RawLoadRow] = []
    for line_index in range(load_start, load_end):
        raw_line = lines[line_index]
        if not raw_line.strip():
            continue
        parts, suffix = _split_raw_row(raw_line)
        if len(parts) < 7:
            raise ValueError(f"Malformed load row in {path}: {raw_line!r}")
        load_rows.append(RawLoadRow(line_index=line_index, parts=parts, suffix=suffix))

    return RawLoadSection(
        base_mva=_parse_base_mva(lines[0]),
        load_start=load_start,
        load_end=load_end,
        load_rows=load_rows,
    )


def patch_raw_loads(
    raw_path: str | Path,
    output_path: str | Path,
    p_load_pu: Sequence[float],
    q_load_pu: Sequence[float],
) -> RawLoadSection:
    """Patch a RAW load section from UQGrid per-unit load vectors."""
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    sections = read_raw_load_section(raw_path)
    p_load = np.asarray(p_load_pu, dtype=float).reshape(-1)
    q_load = np.asarray(q_load_pu, dtype=float).reshape(-1)
    if p_load.size != len(sections.load_rows) or q_load.size != len(sections.load_rows):
        raise ValueError(
            "Load vector length mismatch: "
            f"RAW rows={len(sections.load_rows)}, p_load={p_load.size}, q_load={q_load.size}"
        )

    lines = raw_path.read_text(encoding="utf-8", errors="replace").splitlines()
    patched = list(lines)
    for row, p_pu, q_pu in zip(sections.load_rows, p_load, q_load):
        parts = list(row.parts)
        parts[5] = f"{float(p_pu) * sections.base_mva:.6f}"
        parts[6] = f"{-float(q_pu) * sections.base_mva:.6f}"
        patched[row.line_index] = _format_raw_row(parts, row.suffix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(patched) + "\n", encoding="utf-8")
    return sections


def _parse_raw_load_rows(raw_path: str | Path) -> list[tuple[int, float, float]]:
    rows = []
    for row in read_raw_load_section(raw_path).load_rows:
        rows.append((int(row.parts[0]), float(row.parts[5]), float(row.parts[6])))
    return rows


def _csv_row(line: str) -> list[str]:
    return next(csv.reader([line], skipinitialspace=True))


def _nonempty_section_rows(lines: Sequence[str], start: int, end: int) -> list[list[str]]:
    rows = []
    for raw_line in lines[start:end]:
        line = raw_line.strip()
        if line:
            rows.append(_csv_row(line))
    return rows


def parse_exajugo_basecase(
    path: str | Path,
    raw_path: str | Path | None = None,
) -> ParsedExaJuGOBasecase:
    """Parse ExaJuGO's sectioned Basecase_solution.txt output."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        bus_start = lines.index("--bus section")
        gen_start = lines.index("--generator section")
    except ValueError as exc:
        raise ValueError(f"{path} is missing a required ExaJuGO section") from exc

    try:
        load_start = lines.index("--load section")
    except ValueError:
        load_start = None

    bus_end = load_start if load_start is not None else gen_start
    bus_rows = _nonempty_section_rows(lines, bus_start + 2, bus_end)

    if load_start is None:
        if raw_path is None:
            raise ValueError(
                f"{path} does not contain a --load section; provide raw_path for fallback"
            )
        load_rows = _parse_raw_load_rows(raw_path)
    else:
        load_rows = [
            (int(row[0]), float(row[1]), float(row[2]))
            for row in _nonempty_section_rows(lines, load_start + 2, gen_start)
        ]

    gen_rows = _nonempty_section_rows(lines, gen_start + 2, len(lines))
    p_gen_mw = np.asarray([float(row[2]) for row in gen_rows], dtype=np.float64)
    q_gen_mvar = np.asarray([float(row[3]) for row in gen_rows], dtype=np.float64)
    nonzero_gen_mask = (np.abs(p_gen_mw) > 0.0) | (np.abs(q_gen_mvar) > 0.0)

    return ParsedExaJuGOBasecase(
        bus_ids=np.asarray([int(row[0]) for row in bus_rows], dtype=np.int64),
        bus_v_pu=np.asarray([float(row[1]) for row in bus_rows], dtype=np.float64),
        bus_theta_deg=np.asarray([float(row[2]) for row in bus_rows], dtype=np.float64),
        bus_bcs_mvar=np.asarray([float(row[3]) for row in bus_rows], dtype=np.float64),
        load_bus_ids=np.asarray([int(row[0]) for row in load_rows], dtype=np.int64),
        p_load_mw=np.asarray([float(row[1]) for row in load_rows], dtype=np.float64),
        q_load_mvar=np.asarray([float(row[2]) for row in load_rows], dtype=np.float64),
        gen_bus_ids=np.asarray([int(row[0]) for row in gen_rows], dtype=np.int64),
        gen_ids=[normalize_unit_id(row[1]) for row in gen_rows],
        p_gen_mw=p_gen_mw,
        q_gen_mvar=q_gen_mvar,
        nonzero_gen_mask=nonzero_gen_mask,
    )


def normalize_unit_id(value: Any) -> str:
    return str(value).strip().strip("'").strip()


def _compare_sequence(label: str, parsed: Sequence[Any], uqgrid: Sequence[Any]) -> None:
    if len(parsed) != len(uqgrid):
        raise RuntimeError(f"{label} count mismatch: parsed={len(parsed)} uqgrid={len(uqgrid)}")
    for idx, (lhs, rhs) in enumerate(zip(parsed, uqgrid)):
        if lhs != rhs:
            raise RuntimeError(f"{label} mismatch at position {idx}: parsed={lhs} uqgrid={rhs}")


def apply_exajugo_solution_to_psys(psys: Any, parsed: ParsedExaJuGOBasecase) -> dict[str, Any]:
    """Apply a parsed ExaJuGO basecase to a fresh UQGrid power-system object."""
    load_bus_ids = parsed.load_bus_ids.astype(int).tolist()
    uqgrid_load_bus_ids = [int(load.bus) + 1 for load in psys.loads]
    _compare_sequence("load ordering", load_bus_ids, uqgrid_load_bus_ids)

    active_gen_bus_ids = parsed.gen_bus_ids[parsed.nonzero_gen_mask].astype(int).tolist()
    active_gen_ids = [
        normalize_unit_id(gen_id)
        for gen_id, keep in zip(parsed.gen_ids, parsed.nonzero_gen_mask)
        if keep
    ]
    parsed_gen_pairs = list(zip(active_gen_bus_ids, active_gen_ids))
    uqgrid_gen_pairs = [
        (int(gen.bus) + 1, normalize_unit_id(getattr(gen, "idx", "")))
        for gen in psys.gens
    ]
    _compare_sequence("generator ordering", parsed_gen_pairs, uqgrid_gen_pairs)

    base_mva = float(psys.basemva)
    p_load = parsed.p_load_mw / base_mva
    q_load = -parsed.q_load_mvar / base_mva
    p_gen = parsed.p_gen_mw[parsed.nonzero_gen_mask] / base_mva
    q_gen = parsed.q_gen_mvar[parsed.nonzero_gen_mask] / base_mva
    psys.set_load_pq(p_load, q_load)
    psys.set_gen_pq(p_gen, q_gen)

    bsh_total = np.zeros(int(psys.nbuses), dtype=np.float64)
    for shunt in getattr(psys, "shunts", []):
        bsh_total[int(shunt.bus)] += float(shunt.bsh)

    adjusted_shunts = 0
    for ext_bus, bcs_mvar in zip(parsed.bus_ids, parsed.bus_bcs_mvar):
        if abs(float(bcs_mvar)) <= 1e-10:
            continue
        bus_int = psys.ext2int[int(ext_bus)]
        desired_bsh = float(bcs_mvar) / base_mva
        delta_bsh = desired_bsh - bsh_total[bus_int]
        if abs(delta_bsh) <= 1e-10:
            continue
        psys.add_shunt(bus_int, 0.0, delta_bsh * base_mva)
        bsh_total[bus_int] += delta_bsh
        adjusted_shunts += 1

    for ext_bus, vm_pu, theta_deg in zip(
        parsed.bus_ids,
        parsed.bus_v_pu,
        parsed.bus_theta_deg,
    ):
        bus_int = psys.ext2int[int(ext_bus)]
        psys.buses[bus_int].v0m = float(vm_pu)
        psys.buses[bus_int].v0a = float(np.deg2rad(float(theta_deg)))

    return {
        "base_mva": base_mva,
        "num_loads": len(uqgrid_load_bus_ids),
        "num_generators": len(uqgrid_gen_pairs),
        "num_exajugo_generators": int(parsed.gen_bus_ids.size),
        "num_active_exajugo_generators": int(np.count_nonzero(parsed.nonzero_gen_mask)),
        "adjusted_shunts": adjusted_shunts,
        "load_order_checked": True,
        "generator_order_checked": True,
    }


def select_generator_delta_indices(
    state_metadata: Mapping[str, Mapping[str, Any]],
    *,
    model_name: str = "GenGENROU",
    state_name: str = "delta",
) -> list[int]:
    """Select generator rotor-angle state indices from UQGrid state metadata."""
    indices = [
        int(idx)
        for idx, meta in state_metadata.items()
        if meta.get("model") == model_name and meta.get("state_name") == state_name
    ]
    if not indices:
        raise RuntimeError(
            f"No states matched model={model_name!r} and state_name={state_name!r}"
        )
    return indices


def compute_tsi_time_series(delta_history: np.ndarray) -> np.ndarray:
    deltas = np.asarray(delta_history, dtype=np.float64)
    if deltas.ndim != 2:
        raise ValueError(f"delta_history must be 2D, got shape {deltas.shape}")
    if deltas.shape[0] < 1 or deltas.shape[1] < 1:
        raise ValueError("delta_history must contain at least one generator and one time step")
    spread = np.nanmax(deltas, axis=0) - np.nanmin(deltas, axis=0)
    return (2.0 * np.pi - spread) / (2.0 * np.pi + spread) * 100.0


def extract_tsi_scalar(tsi_time_series: np.ndarray, mode: str = "final") -> float:
    tsi = np.asarray(tsi_time_series, dtype=np.float64).reshape(-1)
    if tsi.size == 0:
        raise ValueError("tsi_time_series must be non-empty")
    if mode == "final":
        return float(tsi[-1])
    if mode == "min":
        return float(np.nanmin(tsi))
    raise ValueError(f"Unsupported tsi mode {mode!r}; expected 'final' or 'min'")


def compute_tsi_final_min_from_history(
    history: np.ndarray,
    delta_state_indices: Sequence[int],
) -> tuple[float, float, np.ndarray]:
    H = np.asarray(history, dtype=np.float64)
    idx = np.asarray(delta_state_indices, dtype=np.int64)
    if H.ndim != 2:
        raise ValueError(f"history must be 2D, got shape {H.shape}")
    if idx.size == 0:
        raise ValueError("delta_state_indices must be non-empty")
    tsi_t = compute_tsi_time_series(H[idx, :])
    return extract_tsi_scalar(tsi_t, "final"), extract_tsi_scalar(tsi_t, "min"), tsi_t


def build_acopf_probml_x(parsed: ParsedExaJuGOBasecase, base_mva: float) -> np.ndarray:
    """Build one ProbML input row from the ACOPF operating point."""
    scale = float(base_mva)
    pg = parsed.p_gen_mw[parsed.nonzero_gen_mask] / scale
    qg = parsed.q_gen_mvar[parsed.nonzero_gen_mask] / scale
    pl = parsed.p_load_mw / scale
    ql = -parsed.q_load_mvar / scale
    return np.stack([np.concatenate([pg, pl]), np.concatenate([qg, ql])], axis=0)


def _probml_meta(n_gen: int, n_load: int, tsi_mode: str) -> dict[str, Any]:
    if tsi_mode not in {"final", "min"}:
        raise ValueError("tsi_mode must be 'final' or 'min'")
    meaning = (
        "TSI at last time step for each (fault_location, fault_impedance)"
        if tsi_mode == "final"
        else "Minimum TSI across all time steps for each (fault_location, fault_impedance)"
    )
    return {
        "inputs": "full_per_unit",
        "channels": ["P", "Q"],
        "unit_axis_order": "generators_then_loads",
        "Ngen": int(n_gen),
        "Nload": int(n_load),
        "require_complete_grid": True,
        "concat_generators_and_loads": True,
        "return_X_flat": True,
        "tsi_mode": tsi_mode,
        "meaning_Y": meaning,
        "axes_Y": {"axis0": "fault_location", "axis1": "fault_impedance"},
        "source": "uqgrid_acopf_initialized",
    }


def _load_npz_dict(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _atomic_save_npz(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npz":
        tmp_path = path.with_suffix(".tmp.npz")
    else:
        tmp_path = path.with_name(f"{path.name}.tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, path)


def _validate_probml_shapes(
    X: np.ndarray,
    Y: np.ndarray,
    sample_idx: np.ndarray,
    fault_locations: np.ndarray,
    fault_impedances: np.ndarray,
    scenario_ids: np.ndarray,
) -> None:
    if X.ndim != 3 or X.shape[1] != 2:
        raise ValueError(f"X must have shape (N, 2, units), got {X.shape}")
    if Y.shape != (X.shape[0], fault_locations.size, fault_impedances.size):
        raise ValueError(
            "Y shape must be (N, len(fault_locations), len(fault_impedances)); "
            f"got {Y.shape}"
        )
    if scenario_ids.shape != Y.shape:
        raise ValueError(f"scenario_ids shape must match Y shape; got {scenario_ids.shape}")
    if sample_idx.shape != (X.shape[0],):
        raise ValueError(f"sample_idx must have shape (N,), got {sample_idx.shape}")


def append_probml_dataset_row(
    out_path: str | Path,
    *,
    X_row: np.ndarray,
    Y_row: np.ndarray,
    sample_idx: int,
    fault_locations: Sequence[int],
    fault_impedances: Sequence[float],
    scenario_ids_row: np.ndarray,
    n_gen: int,
    n_load: int,
    tsi_mode: str,
) -> dict[str, Any]:
    """Append one accepted operating-point row to a ProbML NPZ dataset."""
    out_path = Path(out_path)
    X_new = np.asarray(X_row, dtype=np.float64)
    Y_new = np.asarray(Y_row, dtype=np.float64)
    fault_locations_arr = np.asarray(fault_locations, dtype=np.int64)
    fault_impedances_arr = np.asarray(fault_impedances, dtype=np.float64)
    scenario_ids_new = np.asarray(scenario_ids_row, dtype=object)

    expected_y_shape = (fault_locations_arr.size, fault_impedances_arr.size)
    if X_new.shape != (2, int(n_gen) + int(n_load)):
        raise ValueError(
            f"X_row must have shape (2, {int(n_gen) + int(n_load)}), got {X_new.shape}"
        )
    if Y_new.shape != expected_y_shape:
        raise ValueError(f"Y_row must have shape {expected_y_shape}, got {Y_new.shape}")
    if scenario_ids_new.shape != expected_y_shape:
        raise ValueError(
            f"scenario_ids_row must have shape {expected_y_shape}, got {scenario_ids_new.shape}"
        )

    if out_path.exists():
        existing = _load_npz_dict(out_path)
        required = {
            "X",
            "X_flat",
            "Y",
            "sample_idx",
            "fault_locations",
            "fault_impedances",
            "scenario_ids",
            "meta",
        }
        missing = required.difference(existing)
        if missing:
            raise ValueError(f"Existing NPZ is missing keys: {sorted(missing)}")
        if not np.array_equal(existing["fault_locations"], fault_locations_arr):
            raise ValueError("Existing fault_locations do not match requested append")
        if not np.allclose(existing["fault_impedances"], fault_impedances_arr):
            raise ValueError("Existing fault_impedances do not match requested append")
        X = np.concatenate([existing["X"], X_new[np.newaxis, :, :]], axis=0)
        Y = np.concatenate([existing["Y"], Y_new[np.newaxis, :, :]], axis=0)
        sample_idx_arr = np.concatenate(
            [np.asarray(existing["sample_idx"], dtype=np.int64), np.asarray([sample_idx])]
        )
        scenario_ids = np.concatenate(
            [existing["scenario_ids"], scenario_ids_new[np.newaxis, :, :]], axis=0
        )
    else:
        X = X_new[np.newaxis, :, :]
        Y = Y_new[np.newaxis, :, :]
        sample_idx_arr = np.asarray([sample_idx], dtype=np.int64)
        scenario_ids = scenario_ids_new[np.newaxis, :, :]

    _validate_probml_shapes(
        X,
        Y,
        sample_idx_arr,
        fault_locations_arr,
        fault_impedances_arr,
        scenario_ids,
    )
    payload = {
        "X": X,
        "X_flat": X.reshape(X.shape[0], 2 * X.shape[2]),
        "Y": Y,
        "sample_idx": sample_idx_arr,
        "fault_locations": fault_locations_arr,
        "fault_impedances": fault_impedances_arr,
        "scenario_ids": scenario_ids,
        "meta": np.asarray([_probml_meta(n_gen, n_load, tsi_mode)], dtype=object),
    }
    _atomic_save_npz(out_path, payload)
    return payload


def validate_probml_resume_pair(
    final_path: str | Path,
    min_path: str | Path,
) -> dict[str, int]:
    """Validate final/min ProbML files agree before resuming generation."""
    final_path = Path(final_path)
    min_path = Path(min_path)
    if not final_path.exists() and not min_path.exists():
        return {"accepted_count": 0, "next_sample_idx": 0}
    if final_path.exists() != min_path.exists():
        raise ValueError("final and min NPZ files must both exist to resume")

    final = _load_npz_dict(final_path)
    min_data = _load_npz_dict(min_path)
    for key in ["X", "X_flat", "Y", "sample_idx", "fault_locations", "fault_impedances", "scenario_ids"]:
        if key not in final or key not in min_data:
            raise ValueError(f"Missing required resume key {key!r}")

    if final["Y"].shape[0] != min_data["Y"].shape[0]:
        raise ValueError("final/min NPZ row counts disagree")
    for key in ["X", "X_flat", "sample_idx", "fault_locations", "scenario_ids"]:
        if not np.array_equal(final[key], min_data[key]):
            raise ValueError(f"final/min NPZ key {key!r} disagrees")
    if not np.allclose(final["fault_impedances"], min_data["fault_impedances"]):
        raise ValueError("final/min NPZ fault_impedances disagree")

    accepted_count = int(final["Y"].shape[0])
    sample_idx = np.asarray(final["sample_idx"], dtype=np.int64)
    next_sample_idx = int(np.max(sample_idx) + 1) if sample_idx.size else 0
    return {"accepted_count": accepted_count, "next_sample_idx": next_sample_idx}


@dataclass(frozen=True)
class AcopfInitializationConfig:
    julia: str
    exajugo_root: Path
    base_raw: Path
    base_rop: Path
    acopf_timeout_s: float = 300.0


@dataclass(frozen=True)
class FaultReplayTask:
    sample_idx: int
    operating_point_id: str
    accepted_operating_point_index: int
    fault_location: int
    fault_impedance: float
    fault_location_index: int
    fault_impedance_index: int
    scenario_id: str


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


def _write_jsonl(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(_json_safe(record), sort_keys=True) + "\n" for record in records)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _get_optional_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _text_payload(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _pick_setting(
    args: Any,
    config_section: Mapping[str, Any],
    env: Mapping[str, str],
    *,
    arg_name: str,
    config_name: str,
    env_name: str | None = None,
    default: Any = None,
) -> Any:
    for source in (
        _get_optional_attr(args, arg_name),
        config_section.get(config_name),
        env.get(env_name) if env_name else None,
        default,
    ):
        if source is not None and str(source) != "":
            return source
    return None


def resolve_acopf_initialization_config(
    args: Any,
    config: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
) -> AcopfInitializationConfig:
    """Resolve ACOPF paths from CLI, config, then environment variables."""
    env = env if env is not None else os.environ
    section = config.get("acopf_initialization", {}) or {}

    julia = _pick_setting(
        args,
        section,
        env,
        arg_name="julia",
        config_name="julia",
        env_name="JULIA",
        default="julia",
    )
    exajugo_root = _pick_setting(
        args,
        section,
        env,
        arg_name="exajugo_root",
        config_name="exajugo_root",
        env_name="EXAJUGO_ROOT",
    )
    base_raw = _pick_setting(
        args,
        section,
        env,
        arg_name="exajugo_base_raw",
        config_name="base_raw",
        env_name="EXAJUGO_BASE_RAW",
    )
    base_rop = _pick_setting(
        args,
        section,
        env,
        arg_name="exajugo_base_rop",
        config_name="base_rop",
        env_name="EXAJUGO_BASE_ROP",
    )
    timeout = _pick_setting(
        args,
        section,
        env,
        arg_name="acopf_timeout_s",
        config_name="acopf_timeout_s",
        default=300.0,
    )

    missing = [
        name
        for name, value in {
            "exajugo_root": exajugo_root,
            "exajugo_base_raw": base_raw,
            "exajugo_base_rop": base_rop,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "Missing ACOPF path setting(s): "
            + ", ".join(missing)
            + ". Provide CLI options, acopf_initialization config, or environment variables."
        )

    return AcopfInitializationConfig(
        julia=str(julia),
        exajugo_root=Path(exajugo_root).expanduser(),
        base_raw=Path(base_raw).expanduser(),
        base_rop=Path(base_rop).expanduser(),
        acopf_timeout_s=float(timeout),
    )


def _load_json_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _script_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _installed_uqgrid_root() -> Path | None:
    try:
        import uqgrid  # type: ignore
    except Exception:
        return None
    package_file = getattr(uqgrid, "__file__", None)
    if not package_file:
        return None
    return Path(package_file).resolve().parent.parent


def _resolve_model_path(
    value: str | Path,
    *,
    config_path: str | Path,
    uqgrid_root: str | Path | None = None,
) -> Path:
    """Resolve model paths from config, cwd, configured UQGrid root, or install root."""
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path]

    if not path.is_absolute():
        config_dir = Path(config_path).expanduser().resolve().parent
        candidates.append(config_dir / path)
        if uqgrid_root:
            candidates.append(Path(uqgrid_root).expanduser() / path)
        env_root = os.environ.get("UQGRID_ROOT")
        if env_root:
            candidates.append(Path(env_root).expanduser() / path)
        installed_root = _installed_uqgrid_root()
        if installed_root is not None:
            candidates.append(installed_root / path)
        candidates.append(_script_repo_root() / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve model path {value!r}; checked: {checked}")


def _load_generate_scenarios_module() -> Any:
    module_name = "_uqgrid_generate_scenarios_for_acopf_init"
    if module_name in sys.modules:
        return sys.modules[module_name]

    script_path = Path(__file__).resolve().with_name("generate_scenarios.py")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load generate_scenarios.py from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_powerflow_functions() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    from uqgrid.io.parse import add_dyr, load_psse
    from uqgrid.simulation.pflow import runpf

    return load_psse, add_dyr, runpf


def _load_dynamic_functions() -> tuple[
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
]:
    from uqgrid.io.parse import add_dyr, load_psse
    from uqgrid.simulation.config import IntegrationConfig
    from uqgrid.simulation.dynamics import integrate_system

    return load_psse, add_dyr, IntegrationConfig, integrate_system


def parse_int_list(value: Any, *, default: Sequence[int] | None = None) -> list[int]:
    if value is None:
        if default is None:
            return []
        return [int(item) for item in default]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        return [int(part.strip()) for part in stripped.split(",") if part.strip()]
    return [int(item) for item in value]


def parse_float_list(value: Any, *, default: Sequence[float] | None = None) -> list[float]:
    if value is None:
        if default is None:
            return []
        return [float(item) for item in default]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        return [float(part.strip()) for part in stripped.split(",") if part.strip()]
    return [float(item) for item in value]


def _integration_config_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    integration_cfg = config.get("integration", {}) or {}
    return {
        "tend": integration_cfg.get("tend", 10.0),
        "dt": integration_cfg.get("dt", 1 / 120.0),
        "power_injection": integration_cfg.get("power_injection", False),
        "ton": integration_cfg.get("ton", 0.25),
        "toff": integration_cfg.get("toff", 0.4),
        "verbose": integration_cfg.get("verbose", False),
        "petsc": integration_cfg.get("petsc", True),
    }


def _default_probml_basename(config: Mapping[str, Any], *, smoke: bool = False) -> str:
    storage = config.get("storage", {}) or {}
    configured = storage.get("probml_basename")
    if configured:
        return str(configured)
    model_name = str((config.get("model", {}) or {}).get("name", "uqgrid"))
    suffix = "_stage3_smoke" if smoke else ""
    return f"tsi_probml_fullinputs_{model_name}{suffix}"


def resolve_fault_locations(value: Any, *, n_bus: int) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str) and value.strip().lower() == "all":
        return list(range(int(n_bus)))
    return parse_int_list(value)


def _effective_n_jobs(requested_n_jobs: int | None, task_count: int) -> int:
    if task_count <= 0:
        return 0
    if requested_n_jobs is None:
        requested = task_count
    else:
        requested = int(requested_n_jobs)
    if requested == 0:
        raise ValueError("n_jobs must be nonzero")
    if requested < 0:
        requested = task_count
    return max(1, min(requested, task_count))


def export_delta_state_metadata(
    *,
    raw_path: str | Path,
    dyr_path: str | Path,
    output_path: str | Path,
    load_psse_func: Callable[..., Any] | None = None,
    add_dyr_func: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Export UQGrid state metadata once and return selected delta indices."""
    psys = None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if load_psse_func is None or add_dyr_func is None:
            load_psse_func, add_dyr_func, _ = _load_powerflow_functions()
        psys = load_psse_func(str(raw_path))
        add_dyr_func(psys, str(dyr_path))
        try:
            psys.export_state_metadata(filename=str(output_path))
        except TypeError:
            psys.export_state_metadata(str(output_path))
        metadata = json.loads(output_path.read_text(encoding="utf-8"))
        indices = select_generator_delta_indices(metadata)
        return {
            "state_metadata_path": str(output_path),
            "delta_state_indices": indices,
            "delta_state_count": len(indices),
        }
    finally:
        if psys is not None:
            del psys
        gc.collect()


def replay_acopf_fault_task(
    task: FaultReplayTask,
    context: Mapping[str, Any],
    *,
    load_psse_func: Callable[..., Any] | None = None,
    add_dyr_func: Callable[..., Any] | None = None,
    integration_config_cls: Callable[..., Any] | None = None,
    integrate_system_func: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run one ACOPF-initialized dynamic fault replay task."""
    psys = None
    sim = None
    history = None
    tvec = None
    record = {
        "record_type": "fault_scenario",
        "scenario_id": task.scenario_id,
        "sample_idx": int(task.sample_idx),
        "operating_point_id": task.operating_point_id,
        "accepted_operating_point_index": int(task.accepted_operating_point_index),
        "fault_location": int(task.fault_location),
        "fault_impedance": float(task.fault_impedance),
        "fault_location_index": int(task.fault_location_index),
        "fault_impedance_index": int(task.fault_impedance_index),
        "accepted": False,
        "reject_reason": "dynamic_fault_failed",
    }
    try:
        if (
            load_psse_func is None
            or add_dyr_func is None
            or integration_config_cls is None
            or integrate_system_func is None
        ):
            (
                load_psse_func,
                add_dyr_func,
                integration_config_cls,
                integrate_system_func,
            ) = _load_dynamic_functions()

        parsed = parse_exajugo_basecase(
            context["basecase_path"],
            raw_path=context["case_raw_path"],
        )
        psys = load_psse_func(str(context["raw_path"]))
        add_dyr_func(psys, str(context["dyr_path"]))
        apply_summary = apply_exajugo_solution_to_psys(psys, parsed)
        psys.add_busfault(int(task.fault_location), float(task.fault_impedance))
        psys.createYbusComplex()
        cfg = integration_config_cls(**dict(context["integration_config"]))
        sim = integrate_system_func(psys, cfg)
        history = sim.get("history") if isinstance(sim, Mapping) else None
        tvec = sim.get("tvec") if isinstance(sim, Mapping) else None
        if history is None:
            raise RuntimeError("UQGrid returned no history")

        tsi_final, tsi_min, tsi_t = compute_tsi_final_min_from_history(
            history,
            context["delta_state_indices"],
        )
        history_file = None
        if context.get("keep_fault_histories", False):
            history_dir = Path(context["history_dir"])
            history_dir.mkdir(parents=True, exist_ok=True)
            history_file = (
                history_dir
                / f"{task.scenario_id}_history.npz"
            )
            np.savez_compressed(
                history_file,
                history=history,
                tvec=np.asarray([]) if tvec is None else tvec,
                tsi=tsi_t,
            )

        final_time = None
        if tvec is not None:
            tvec_arr = np.asarray(tvec).reshape(-1)
            if tvec_arr.size:
                final_time = float(tvec_arr[-1])

        record.update(
            {
                "accepted": True,
                "reject_reason": None,
                "success": True,
                "simulation_diverged": False,
                "file": None,
                "history_file": None if history_file is None else str(history_file),
                "tsi_final": float(tsi_final),
                "tsi_min": float(tsi_min),
                "final_simulation_time_s": final_time,
                "base_mva": float(apply_summary["base_mva"]),
                "adjusted_shunts": int(apply_summary["adjusted_shunts"]),
                "num_loads": int(apply_summary["num_loads"]),
                "num_generators": int(apply_summary["num_generators"]),
            }
        )
        return record
    except Exception as exc:
        if context.get("debug_tracebacks", False):
            traceback.print_exc()
        record.update(
            {
                "success": False,
                "accepted": False,
                "reject_reason": "dynamic_fault_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "simulation_diverged": True,
                "file": None,
            }
        )
        return record
    finally:
        tvec = None
        history = None
        sim = None
        if psys is not None:
            del psys
        gc.collect()


def write_exajugo_smoke_case(
    output_dir: str | Path,
    *,
    sample_idx: int,
    acopf_config: AcopfInitializationConfig,
    operating_point: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    case_root: str = "acopf_smoke",
) -> dict[str, Any]:
    """Write a load-patched ExaJuGO case for one PF-screened UQGrid candidate."""
    case_dir = Path(output_dir).expanduser() / str(case_root) / f"op_{int(sample_idx)}"
    case_raw = case_dir / "case.raw"
    case_rop = case_dir / "case.rop"

    p_load = np.asarray(operating_point["p_load_scaled"], dtype=np.float64).reshape(-1)
    q_load = np.asarray(operating_point["q_load_scaled"], dtype=np.float64).reshape(-1)
    raw_section = patch_raw_loads(acopf_config.base_raw, case_raw, p_load, q_load)
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(acopf_config.base_rop, case_rop)

    case_metadata = {
        "sample_idx": int(sample_idx),
        "case_dir": str(case_dir),
        "case_raw": str(case_raw),
        "case_rop": str(case_rop),
        "base_raw": str(acopf_config.base_raw),
        "base_rop": str(acopf_config.base_rop),
        "base_mva": float(raw_section.base_mva),
        "n_load_rows": len(raw_section.load_rows),
        "total_p_load_pu": float(np.sum(p_load)),
        "total_q_load_uqgrid_pu": float(np.sum(q_load)),
        "total_q_load_raw_pu": float(-np.sum(q_load)),
        "operating_point_id": operating_point.get("operating_point_id"),
        "accepted_operating_point_index": operating_point.get(
            "accepted_operating_point_index"
        ),
        "diagnostics": operating_point.get("diagnostics", {}),
    }
    if metadata:
        case_metadata.update(metadata)
    _write_json(case_dir / "metadata.json", case_metadata)

    return {
        "case_dir": str(case_dir),
        "case_raw": str(case_raw),
        "case_rop": str(case_rop),
        "metadata": case_metadata,
    }


def run_exajugo_acopf(
    case_dir: str | Path,
    acopf_config: AcopfInitializationConfig,
    *,
    subprocess_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run ExaJuGO ACOPF and return a compact success/failure diagnostic."""
    case_dir = Path(case_dir).expanduser()
    solution_dir = case_dir / "acopf_solution"
    system_dir = case_dir / "acopf_system"
    stdout_path = case_dir / "acopf_stdout.txt"
    stderr_path = case_dir / "acopf_stderr.txt"
    basecase_path = system_dir / "Basecase_solution.txt"
    command = [
        acopf_config.julia,
        f"--project={acopf_config.exajugo_root}",
        str(acopf_config.exajugo_root / "ACOPF.jl"),
        str(case_dir / "case.raw"),
        str(case_dir / "case.rop"),
        str(solution_dir),
        str(system_dir),
    ]

    start = time.monotonic()
    result: dict[str, Any] = {
        "record_type": "acopf_run",
        "case_dir": str(case_dir),
        "command": command,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "basecase_path": str(basecase_path),
        "success": False,
        "accepted": False,
    }
    try:
        completed = subprocess_run(
            command,
            timeout=float(acopf_config.acopf_timeout_s),
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_text_payload(exc.stdout), encoding="utf-8")
        stderr_path.write_text(_text_payload(exc.stderr), encoding="utf-8")
        result.update(
            {
                "duration_s": float(time.monotonic() - start),
                "returncode": None,
                "timeout_s": float(acopf_config.acopf_timeout_s),
                "reject_stage": "acopf",
                "reject_reason": "acopf_timeout",
                "error": str(exc),
            }
        )
        return result

    stdout_path.write_text(_text_payload(completed.stdout), encoding="utf-8")
    stderr_path.write_text(_text_payload(completed.stderr), encoding="utf-8")
    result.update(
        {
            "duration_s": float(time.monotonic() - start),
            "returncode": int(completed.returncode),
        }
    )
    if completed.returncode != 0:
        result.update(
            {
                "reject_stage": "acopf",
                "reject_reason": "acopf_nonzero_exit",
            }
        )
        return result
    if not basecase_path.exists():
        result.update(
            {
                "reject_stage": "acopf",
                "reject_reason": "acopf_missing_basecase",
            }
        )
        return result

    result.update({"success": True, "accepted": True, "reject_reason": None})
    return result


def _pf_solution_attrs(pf_solution: Any) -> dict[str, Any]:
    attrs = {}
    for name in (
        "converged",
        "iterations",
        "initial_residual_norm",
        "final_residual_norm",
        "residual_norm",
    ):
        if hasattr(pf_solution, name):
            attrs[name] = getattr(pf_solution, name)
    return attrs


def _residual_from_attrs(attrs: Mapping[str, Any]) -> float | None:
    for name in ("final_residual_norm", "residual_norm", "initial_residual_norm"):
        value = attrs.get(name)
        if value is not None and np.isfinite(value):
            return float(value)
    return None


def validate_acopf_power_flow(
    *,
    raw_path: str | Path,
    dyr_path: str | Path,
    basecase_path: str | Path,
    case_raw_path: str | Path,
    pf_residual_tol: float,
    verbose: bool = False,
    load_psse_func: Callable[..., Any] | None = None,
    add_dyr_func: Callable[..., Any] | None = None,
    runpf_func: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Import an ExaJuGO basecase into UQGrid and run a PF validation check."""
    psys = None
    try:
        if load_psse_func is None or add_dyr_func is None or runpf_func is None:
            load_psse_func, add_dyr_func, runpf_func = _load_powerflow_functions()

        parsed = parse_exajugo_basecase(basecase_path, raw_path=case_raw_path)
        psys = load_psse_func(str(raw_path))
        add_dyr_func(psys, str(dyr_path))
        apply_summary = apply_exajugo_solution_to_psys(psys, parsed)
        psys.createYbusComplex()
        pf_solution = runpf_func(psys, verbose=verbose)

        attrs = _pf_solution_attrs(pf_solution)
        voltages = np.asarray(pf_solution.v_magnitudes, dtype=np.float64)
        summary = {
            "record_type": "post_acopf_pf_validation",
            "success": True,
            "accepted": True,
            "reject_reason": None,
            "raw_path": str(raw_path),
            "dyr_path": str(dyr_path),
            "basecase_path": str(basecase_path),
            "case_raw_path": str(case_raw_path),
            "base_mva": float(apply_summary["base_mva"]),
            "num_buses": int(psys.nbuses),
            "num_loads": int(apply_summary["num_loads"]),
            "num_generators": int(apply_summary["num_generators"]),
            "num_exajugo_generators": int(apply_summary["num_exajugo_generators"]),
            "num_active_exajugo_generators": int(
                apply_summary["num_active_exajugo_generators"]
            ),
            "adjusted_shunts": int(apply_summary["adjusted_shunts"]),
            "voltage_min_pu": float(np.nanmin(voltages)),
            "voltage_max_pu": float(np.nanmax(voltages)),
            "pf_solution_attrs": attrs,
        }

        converged = attrs.get("converged")
        if converged is not None and not bool(converged):
            summary.update(
                {
                    "success": False,
                    "accepted": False,
                    "reject_stage": "post_acopf_pf",
                    "reject_reason": "post_acopf_pf_not_converged",
                }
            )
            return summary

        residual = _residual_from_attrs(attrs)
        if residual is not None and residual > float(pf_residual_tol):
            summary.update(
                {
                    "success": False,
                    "accepted": False,
                    "reject_stage": "post_acopf_pf",
                    "reject_reason": "post_acopf_pf_residual_too_large",
                    "pf_residual": residual,
                    "pf_residual_tol": float(pf_residual_tol),
                }
            )
            return summary

        if residual is not None:
            summary["pf_residual"] = residual
            summary["pf_residual_tol"] = float(pf_residual_tol)
        return summary
    except Exception as exc:
        return {
            "record_type": "post_acopf_pf_validation",
            "success": False,
            "accepted": False,
            "reject_stage": "post_acopf_pf",
            "reject_reason": "post_acopf_pf_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "raw_path": str(raw_path),
            "dyr_path": str(dyr_path),
            "basecase_path": str(basecase_path),
            "case_raw_path": str(case_raw_path),
        }
    finally:
        if psys is not None:
            del psys
        gc.collect()


def _summarize_smoke_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reject_counts: dict[str, int] = {}
    for record in records:
        reason = record.get("reject_reason")
        if reason:
            reject_counts[str(reason)] = reject_counts.get(str(reason), 0) + 1
    return {
        "total_records": len(records),
        "accepted_records": sum(1 for record in records if record.get("accepted")),
        "rejected_records": sum(1 for record in records if record.get("accepted") is False),
        "reject_reason_counts": reject_counts,
    }


def _write_smoke_diagnostics(
    output_dir: str | Path,
    *,
    progress: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    output_dir = Path(output_dir).expanduser()
    _write_json(output_dir / "acopf_init_progress.json", progress)
    _write_jsonl(output_dir / "acopf_init_diagnostics.jsonl", records)
    _write_json(
        output_dir / "acopf_init_diagnostics_summary.json",
        _summarize_smoke_records(records),
    )


def _smoke_progress(
    *,
    accepted: bool,
    sample_idx: int,
    reject_reason: str | None = None,
    records_written: int = 0,
) -> dict[str, Any]:
    return {
        "stage": "stage_2_acopf_smoke",
        "target_accepted_scenarios": 1,
        "accepted_count": 1 if accepted else 0,
        "next_sample_idx": int(sample_idx) + (1 if accepted else 0),
        "last_sample_idx": int(sample_idx),
        "accepted": bool(accepted),
        "reject_reason": reject_reason,
        "diagnostic_records": int(records_written),
    }


def run_acopf_smoke(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    acopf_config: AcopfInitializationConfig,
    target_accepted_scenarios: int = 1,
    sample_idx_start: int = 0,
    uqgrid_root: str | Path | None = None,
    pf_verbose: bool = False,
    candidate_func: Callable[..., Mapping[str, Any]] | None = None,
    op_config_resolver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    acopf_runner_func: Callable[..., Mapping[str, Any]] = run_exajugo_acopf,
    pf_validator_func: Callable[..., Mapping[str, Any]] = validate_acopf_power_flow,
) -> dict[str, Any]:
    """Run the Stage 2 ACOPF smoke path for one PF-screened operating point."""
    if int(target_accepted_scenarios) != 1:
        raise ValueError("Stage 2 smoke mode supports exactly one accepted scenario")

    config_path = Path(config_path).expanduser()
    config = _load_json_config(config_path)
    if candidate_func is None or op_config_resolver is None:
        gs = _load_generate_scenarios_module()
        candidate_func = candidate_func or gs._prepare_operating_point_candidate
        op_config_resolver = op_config_resolver or gs._resolve_operating_point_config

    model_cfg = config["model"]
    raw_path = _resolve_model_path(
        model_cfg["raw"],
        config_path=config_path,
        uqgrid_root=uqgrid_root,
    )
    dyr_path = _resolve_model_path(
        model_cfg["dyr"],
        config_path=config_path,
        uqgrid_root=uqgrid_root,
    )
    pert_cfg = config.get("perturbation", {})
    op_cfg = dict(op_config_resolver(config.get("operating_point", {})))
    scenario = {
        "sample_idx": int(sample_idx_start),
        "operating_point_id": f"acopf-smoke-op-{int(sample_idx_start)}",
        "accepted_operating_point_index": 0,
    }

    records: list[Mapping[str, Any]] = []
    prep = candidate_func(
        str(raw_path),
        str(dyr_path),
        scenario,
        scenario["operating_point_id"],
        noise_type=pert_cfg.get("load_noise_type", config.get("noise_type", "normal")),
        noise_var=pert_cfg.get("load_noise_var", config.get("noise_var", 0.1)),
        global_seed=1234,
        balance_generation=pert_cfg.get("balance_generation", True),
        perturb_loads=pert_cfg.get("perturb_loads", True),
        perturb_gens=pert_cfg.get("perturb_gens", True),
        load_noise_type=pert_cfg.get("load_noise_type"),
        gen_noise_type=pert_cfg.get("gen_noise_type"),
        load_noise_var=pert_cfg.get("load_noise_var"),
        gen_noise_var=pert_cfg.get("gen_noise_var"),
        keep_power_factor=pert_cfg.get("keep_power_factor", True),
        clamp_gens=pert_cfg.get("clamp_gens", True),
        load_scale=pert_cfg.get("load_scale", 1.0),
        load_mean_shift=pert_cfg.get("load_mean_shift", 0.0),
        generation_dispatch_init=pert_cfg.get("generation_dispatch_init", "perturbed"),
        operating_point_config=op_cfg,
    )
    records.extend(prep.get("diagnostics_attempts") or [])
    prep_diag = prep.get("diagnostics") or {}
    if prep.get("rejected") or not prep.get("operating_point"):
        record = {
            **prep_diag,
            "record_type": "acopf_smoke",
            "accepted": False,
            "reject_stage": "pre_pf",
            "reject_reason": prep_diag.get("reject_reason", "operating_point_rejected"),
        }
        records.append(record)
        progress = _smoke_progress(
            accepted=False,
            sample_idx=int(sample_idx_start),
            reject_reason=str(record["reject_reason"]),
            records_written=len(records),
        )
        _write_smoke_diagnostics(output_dir, progress=progress, records=records)
        return progress

    case_info = write_exajugo_smoke_case(
        output_dir,
        sample_idx=int(sample_idx_start),
        acopf_config=acopf_config,
        operating_point=prep["operating_point"],
        metadata={"raw_path": str(raw_path), "dyr_path": str(dyr_path)},
    )
    acopf_result = dict(acopf_runner_func(case_info["case_dir"], acopf_config))
    records.append(acopf_result)
    if not acopf_result.get("success"):
        progress = _smoke_progress(
            accepted=False,
            sample_idx=int(sample_idx_start),
            reject_reason=str(acopf_result.get("reject_reason", "acopf_failed")),
            records_written=len(records),
        )
        _write_smoke_diagnostics(output_dir, progress=progress, records=records)
        return progress

    pf_summary = dict(
        pf_validator_func(
            raw_path=raw_path,
            dyr_path=dyr_path,
            basecase_path=acopf_result["basecase_path"],
            case_raw_path=case_info["case_raw"],
            pf_residual_tol=float(op_cfg.get("pf_residual_tol", 1e-8)),
            verbose=pf_verbose,
        )
    )
    records.append(pf_summary)
    if not pf_summary.get("success"):
        progress = _smoke_progress(
            accepted=False,
            sample_idx=int(sample_idx_start),
            reject_reason=str(pf_summary.get("reject_reason", "post_acopf_pf_failed")),
            records_written=len(records),
        )
        _write_smoke_diagnostics(output_dir, progress=progress, records=records)
        return progress

    smoke_record = {
        "record_type": "acopf_smoke",
        "sample_idx": int(sample_idx_start),
        "operating_point_id": scenario["operating_point_id"],
        "accepted_operating_point_index": 0,
        "accepted": True,
        "reject_reason": None,
        "case_dir": case_info["case_dir"],
        "basecase_path": acopf_result["basecase_path"],
        "pre_acopf_attempts": prep_diag.get("attempts"),
        "pre_acopf_pf_residual": prep_diag.get("pf_residual"),
        "post_acopf_pf_validation": pf_summary,
    }
    records.append(smoke_record)
    progress = _smoke_progress(
        accepted=True,
        sample_idx=int(sample_idx_start),
        records_written=len(records),
    )
    _write_smoke_diagnostics(output_dir, progress=progress, records=records)
    return progress


def prepare_acopf_replay_context(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    acopf_config: AcopfInitializationConfig,
    target_accepted_scenarios: int = 1,
    sample_idx_start: int = 0,
    uqgrid_root: str | Path | None = None,
    pf_verbose: bool = False,
    smoke_func: Callable[..., Mapping[str, Any]] = run_acopf_smoke,
) -> dict[str, Any]:
    """Run Stage 2 smoke and reconstruct the accepted context for replay."""
    output_dir = Path(output_dir).expanduser()
    progress = dict(
        smoke_func(
            config_path=config_path,
            output_dir=output_dir,
            acopf_config=acopf_config,
            target_accepted_scenarios=target_accepted_scenarios,
            sample_idx_start=sample_idx_start,
            uqgrid_root=uqgrid_root,
            pf_verbose=pf_verbose,
        )
    )
    records = _read_jsonl(output_dir / "acopf_init_diagnostics.jsonl")
    if not progress.get("accepted"):
        return {
            "success": False,
            "accepted": False,
            "progress": progress,
            "records": records,
            "reject_reason": progress.get("reject_reason", "acopf_smoke_failed"),
        }

    config = _load_json_config(config_path)
    model_cfg = config["model"]
    raw_path = _resolve_model_path(
        model_cfg["raw"],
        config_path=config_path,
        uqgrid_root=uqgrid_root,
    )
    dyr_path = _resolve_model_path(
        model_cfg["dyr"],
        config_path=config_path,
        uqgrid_root=uqgrid_root,
    )
    sample_idx = int(sample_idx_start)
    case_dir = output_dir / "acopf_smoke" / f"op_{sample_idx}"
    case_raw = case_dir / "case.raw"
    basecase_path = case_dir / "acopf_system" / "Basecase_solution.txt"
    parsed = parse_exajugo_basecase(basecase_path, raw_path=case_raw)
    pf_summary = next(
        (
            record
            for record in records
            if record.get("record_type") == "post_acopf_pf_validation"
        ),
        {},
    )
    smoke_record = next(
        (
            record
            for record in records
            if record.get("record_type") == "acopf_smoke" and record.get("accepted")
        ),
        {},
    )
    return {
        "success": True,
        "accepted": True,
        "progress": progress,
        "records": records,
        "config": config,
        "raw_path": raw_path,
        "dyr_path": dyr_path,
        "case_dir": case_dir,
        "case_raw_path": case_raw,
        "basecase_path": basecase_path,
        "parsed_basecase": parsed,
        "post_acopf_pf_validation": pf_summary,
        "sample_idx": sample_idx,
        "operating_point_id": smoke_record.get(
            "operating_point_id",
            f"acopf-smoke-op-{sample_idx}",
        ),
        "accepted_operating_point_index": int(
            smoke_record.get("accepted_operating_point_index", 0)
        ),
        "pre_acopf_attempts": smoke_record.get("pre_acopf_attempts"),
        "pre_acopf_pf_residual": smoke_record.get("pre_acopf_pf_residual"),
        "base_mva": float(pf_summary.get("base_mva", 100.0)),
    }


def prepare_acopf_candidate_context(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    acopf_config: AcopfInitializationConfig,
    sample_idx: int,
    accepted_operating_point_index: int,
    target_accepted_scenarios: int,
    max_total_attempts: int,
    remaining_attempts: int,
    uqgrid_root: str | Path | None = None,
    pf_verbose: bool = False,
    case_root: str = "acopf_cases",
    candidate_func: Callable[..., Mapping[str, Any]] | None = None,
    op_config_resolver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    acopf_runner_func: Callable[..., Mapping[str, Any]] = run_exajugo_acopf,
    pf_validator_func: Callable[..., Mapping[str, Any]] = validate_acopf_power_flow,
) -> dict[str, Any]:
    """Prepare one PF-screened candidate and ACOPF-initialized UQGrid context."""
    config_path = Path(config_path).expanduser()
    config = _load_json_config(config_path)
    if candidate_func is None or op_config_resolver is None:
        gs = _load_generate_scenarios_module()
        candidate_func = candidate_func or gs._prepare_operating_point_candidate
        op_config_resolver = op_config_resolver or gs._resolve_operating_point_config

    model_cfg = config["model"]
    raw_path = _resolve_model_path(
        model_cfg["raw"],
        config_path=config_path,
        uqgrid_root=uqgrid_root,
    )
    dyr_path = _resolve_model_path(
        model_cfg["dyr"],
        config_path=config_path,
        uqgrid_root=uqgrid_root,
    )
    pert_cfg = config.get("perturbation", {}) or {}
    op_cfg = dict(op_config_resolver(config.get("operating_point", {})))
    if "max_attempts_per_scenario" in op_cfg:
        op_cfg["max_attempts_per_scenario"] = min(
            int(op_cfg["max_attempts_per_scenario"]),
            max(1, int(remaining_attempts)),
        )

    operating_point_id = str(uuid.uuid4())
    scenario = {
        "sample_idx": int(sample_idx),
        "operating_point_id": operating_point_id,
        "accepted_operating_point_index": int(accepted_operating_point_index),
    }
    records: list[Mapping[str, Any]] = []
    prep = candidate_func(
        str(raw_path),
        str(dyr_path),
        scenario,
        operating_point_id,
        noise_type=pert_cfg.get("load_noise_type", config.get("noise_type", "normal")),
        noise_var=pert_cfg.get("load_noise_var", config.get("noise_var", 0.1)),
        global_seed=1234,
        balance_generation=pert_cfg.get("balance_generation", True),
        perturb_loads=pert_cfg.get("perturb_loads", True),
        perturb_gens=pert_cfg.get("perturb_gens", True),
        load_noise_type=pert_cfg.get("load_noise_type"),
        gen_noise_type=pert_cfg.get("gen_noise_type"),
        load_noise_var=pert_cfg.get("load_noise_var"),
        gen_noise_var=pert_cfg.get("gen_noise_var"),
        keep_power_factor=pert_cfg.get("keep_power_factor", True),
        clamp_gens=pert_cfg.get("clamp_gens", True),
        load_scale=pert_cfg.get("load_scale", 1.0),
        load_mean_shift=pert_cfg.get("load_mean_shift", 0.0),
        generation_dispatch_init=pert_cfg.get("generation_dispatch_init", "perturbed"),
        operating_point_config=op_cfg,
    )
    records.extend(prep.get("diagnostics_attempts") or [])
    prep_diag = prep.get("diagnostics") or {}
    candidate_attempts = int(prep_diag.get("attempts") or max(1, len(records)))
    group_base = {
        "record_type": "operating_point_group",
        "sample_idx": int(sample_idx),
        "operating_point_id": operating_point_id,
        "accepted_operating_point_index": int(accepted_operating_point_index),
        "target_accepted_scenarios": int(target_accepted_scenarios),
        "max_total_attempts": int(max_total_attempts),
        "candidate_attempts": candidate_attempts,
        "pre_acopf_attempts": prep_diag.get("attempts"),
        "pre_acopf_pf_residual": prep_diag.get("pf_residual"),
    }

    if prep.get("rejected") or not prep.get("operating_point"):
        record = {
            **prep_diag,
            **group_base,
            "accepted": False,
            "reject_stage": "pre_pf",
            "reject_reason": prep_diag.get("reject_reason", "operating_point_rejected"),
            "faults_required": 0,
            "faults_successful": 0,
        }
        records.append(record)
        return {
            "success": False,
            "accepted": False,
            "records": records,
            "sample_idx": int(sample_idx),
            "operating_point_id": operating_point_id,
            "accepted_operating_point_index": int(accepted_operating_point_index),
            "candidate_attempts": candidate_attempts,
            "reject_stage": "pre_pf",
            "reject_reason": record["reject_reason"],
            "case_dir": None,
        }

    operating_point = dict(prep["operating_point"])
    operating_point["operating_point_id"] = operating_point_id
    operating_point["accepted_operating_point_index"] = int(accepted_operating_point_index)
    case_info = write_exajugo_smoke_case(
        output_dir,
        sample_idx=int(sample_idx),
        acopf_config=acopf_config,
        operating_point=operating_point,
        metadata={"raw_path": str(raw_path), "dyr_path": str(dyr_path)},
        case_root=case_root,
    )
    acopf_result = dict(acopf_runner_func(case_info["case_dir"], acopf_config))
    records.append(acopf_result)
    if not acopf_result.get("success"):
        records.append(
            {
                **group_base,
                "accepted": False,
                "reject_stage": "acopf",
                "reject_reason": acopf_result.get("reject_reason", "acopf_failed"),
                "case_dir": case_info["case_dir"],
                "faults_required": 0,
                "faults_successful": 0,
            }
        )
        return {
            "success": False,
            "accepted": False,
            "records": records,
            "sample_idx": int(sample_idx),
            "operating_point_id": operating_point_id,
            "accepted_operating_point_index": int(accepted_operating_point_index),
            "candidate_attempts": candidate_attempts,
            "reject_stage": "acopf",
            "reject_reason": acopf_result.get("reject_reason", "acopf_failed"),
            "case_dir": case_info["case_dir"],
        }

    pf_summary = dict(
        pf_validator_func(
            raw_path=raw_path,
            dyr_path=dyr_path,
            basecase_path=acopf_result["basecase_path"],
            case_raw_path=case_info["case_raw"],
            pf_residual_tol=float(op_cfg.get("pf_residual_tol", 1e-8)),
            verbose=pf_verbose,
        )
    )
    records.append(pf_summary)
    if not pf_summary.get("success"):
        records.append(
            {
                **group_base,
                "accepted": False,
                "reject_stage": "post_acopf_pf",
                "reject_reason": pf_summary.get("reject_reason", "post_acopf_pf_failed"),
                "case_dir": case_info["case_dir"],
                "basecase_path": acopf_result.get("basecase_path"),
                "faults_required": 0,
                "faults_successful": 0,
            }
        )
        return {
            "success": False,
            "accepted": False,
            "records": records,
            "sample_idx": int(sample_idx),
            "operating_point_id": operating_point_id,
            "accepted_operating_point_index": int(accepted_operating_point_index),
            "candidate_attempts": candidate_attempts,
            "reject_stage": "post_acopf_pf",
            "reject_reason": pf_summary.get("reject_reason", "post_acopf_pf_failed"),
            "case_dir": case_info["case_dir"],
        }

    try:
        parsed = parse_exajugo_basecase(acopf_result["basecase_path"], raw_path=case_info["case_raw"])
    except Exception as exc:
        records.append(
            {
                **group_base,
                "accepted": False,
                "reject_stage": "basecase_parse",
                "reject_reason": "basecase_parse_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "case_dir": case_info["case_dir"],
                "basecase_path": acopf_result.get("basecase_path"),
                "faults_required": 0,
                "faults_successful": 0,
            }
        )
        return {
            "success": False,
            "accepted": False,
            "records": records,
            "sample_idx": int(sample_idx),
            "operating_point_id": operating_point_id,
            "accepted_operating_point_index": int(accepted_operating_point_index),
            "candidate_attempts": candidate_attempts,
            "reject_stage": "basecase_parse",
            "reject_reason": "basecase_parse_failed",
            "case_dir": case_info["case_dir"],
        }

    return {
        "success": True,
        "accepted": True,
        "records": records,
        "config": config,
        "raw_path": raw_path,
        "dyr_path": dyr_path,
        "case_dir": Path(case_info["case_dir"]),
        "case_raw_path": Path(case_info["case_raw"]),
        "basecase_path": Path(acopf_result["basecase_path"]),
        "parsed_basecase": parsed,
        "post_acopf_pf_validation": pf_summary,
        "sample_idx": int(sample_idx),
        "operating_point_id": operating_point_id,
        "accepted_operating_point_index": int(accepted_operating_point_index),
        "candidate_attempts": candidate_attempts,
        "pre_acopf_attempts": prep_diag.get("attempts"),
        "pre_acopf_pf_residual": prep_diag.get("pf_residual"),
        "base_mva": float(pf_summary.get("base_mva", 100.0)),
    }


def _probml_output_paths(output_dir: str | Path, basename: str) -> tuple[Path, Path]:
    base = Path(basename).expanduser()
    if not base.is_absolute():
        base = Path(output_dir).expanduser() / base
    return (
        base.with_name(f"{base.name}_final.npz"),
        base.with_name(f"{base.name}_min.npz"),
    )


def build_fault_replay_tasks(
    *,
    sample_idx: int,
    operating_point_id: str,
    accepted_operating_point_index: int,
    fault_locations: Sequence[int],
    fault_impedances: Sequence[float],
) -> list[FaultReplayTask]:
    tasks = []
    for loc_idx, fault_location in enumerate(fault_locations):
        for imp_idx, fault_impedance in enumerate(fault_impedances):
            scenario_id = (
                f"sample_{int(sample_idx)}_fault_{int(fault_location)}_imp_{imp_idx}"
            )
            tasks.append(
                FaultReplayTask(
                    sample_idx=int(sample_idx),
                    operating_point_id=str(operating_point_id),
                    accepted_operating_point_index=int(accepted_operating_point_index),
                    fault_location=int(fault_location),
                    fault_impedance=float(fault_impedance),
                    fault_location_index=int(loc_idx),
                    fault_impedance_index=int(imp_idx),
                    scenario_id=scenario_id,
                )
            )
    return tasks


def run_fault_replay_tasks(
    tasks: Sequence[FaultReplayTask],
    context: Mapping[str, Any],
    *,
    n_jobs: int,
    parallel_timeout_s: float = 600.0,
    worker_func: Callable[[FaultReplayTask, Mapping[str, Any]], dict[str, Any]] = replay_acopf_fault_task,
) -> list[dict[str, Any]]:
    effective_n_jobs = _effective_n_jobs(n_jobs, len(tasks))
    if effective_n_jobs <= 1:
        return [worker_func(task, context) for task in tasks]
    return Parallel(n_jobs=effective_n_jobs, backend="loky", timeout=parallel_timeout_s)(
        delayed(worker_func)(task, context) for task in tasks
    )


def append_acopf_probml_outputs(
    *,
    output_dir: str | Path,
    probml_basename: str,
    parsed_basecase: ParsedExaJuGOBasecase,
    base_mva: float,
    sample_idx: int,
    fault_locations: Sequence[int],
    fault_impedances: Sequence[float],
    fault_results: Sequence[Mapping[str, Any]],
    refuse_existing: bool = False,
) -> dict[str, Any]:
    final_path, min_path = _probml_output_paths(output_dir, probml_basename)
    if refuse_existing and (final_path.exists() or min_path.exists()):
        raise FileExistsError(
            "Refusing to append; remove existing NPZ files, use a fresh basename, or resume explicitly"
        )

    X_row = build_acopf_probml_x(parsed_basecase, base_mva)
    n_gen = int(np.count_nonzero(parsed_basecase.nonzero_gen_mask))
    n_load = int(parsed_basecase.load_bus_ids.size)
    y_shape = (len(fault_locations), len(fault_impedances))
    Y_final = np.empty(y_shape, dtype=np.float64)
    Y_min = np.empty(y_shape, dtype=np.float64)
    scenario_ids = np.empty(y_shape, dtype=object)

    for result in fault_results:
        if not result.get("accepted"):
            raise RuntimeError("Cannot write ProbML outputs with failed fault results")
        loc_idx = int(result["fault_location_index"])
        imp_idx = int(result["fault_impedance_index"])
        Y_final[loc_idx, imp_idx] = float(result["tsi_final"])
        Y_min[loc_idx, imp_idx] = float(result["tsi_min"])
        scenario_ids[loc_idx, imp_idx] = str(result["scenario_id"])

    final_payload = append_probml_dataset_row(
        final_path,
        X_row=X_row,
        Y_row=Y_final,
        sample_idx=int(sample_idx),
        fault_locations=fault_locations,
        fault_impedances=fault_impedances,
        scenario_ids_row=scenario_ids,
        n_gen=n_gen,
        n_load=n_load,
        tsi_mode="final",
    )
    min_payload = append_probml_dataset_row(
        min_path,
        X_row=X_row,
        Y_row=Y_min,
        sample_idx=int(sample_idx),
        fault_locations=fault_locations,
        fault_impedances=fault_impedances,
        scenario_ids_row=scenario_ids,
        n_gen=n_gen,
        n_load=n_load,
        tsi_mode="min",
    )
    return {
        "final_path": str(final_path),
        "min_path": str(min_path),
        "X_shape": tuple(int(v) for v in final_payload["X"].shape),
        "X_flat_shape": tuple(int(v) for v in final_payload["X_flat"].shape),
        "Y_final_shape": tuple(int(v) for v in final_payload["Y"].shape),
        "Y_min_shape": tuple(int(v) for v in min_payload["Y"].shape),
        "n_gen": n_gen,
        "n_load": n_load,
    }


def write_stage3_probml_outputs(
    *,
    output_dir: str | Path,
    probml_basename: str,
    parsed_basecase: ParsedExaJuGOBasecase,
    base_mva: float,
    sample_idx: int,
    fault_locations: Sequence[int],
    fault_impedances: Sequence[float],
    fault_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return append_acopf_probml_outputs(
        output_dir=output_dir,
        probml_basename=probml_basename,
        parsed_basecase=parsed_basecase,
        base_mva=base_mva,
        sample_idx=sample_idx,
        fault_locations=fault_locations,
        fault_impedances=fault_impedances,
        fault_results=fault_results,
        refuse_existing=True,
    )


def _stage3_progress(
    *,
    accepted: bool,
    sample_idx: int,
    reject_reason: str | None = None,
    records_written: int = 0,
    fault_samples_completed: int = 0,
    probml_outputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    progress = {
        "stage": "stage_3_replay_smoke",
        "target_accepted_scenarios": 1,
        "accepted_count": 1 if accepted else 0,
        "next_sample_idx": int(sample_idx) + (1 if accepted else 0),
        "last_sample_idx": int(sample_idx),
        "accepted": bool(accepted),
        "reject_reason": reject_reason,
        "diagnostic_records": int(records_written),
        "fault_samples_completed": int(fault_samples_completed),
    }
    if probml_outputs:
        progress["probml_outputs"] = dict(probml_outputs)
    return progress


def run_acopf_replay_smoke(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    acopf_config: AcopfInitializationConfig,
    target_accepted_scenarios: int = 1,
    sample_idx_start: int = 0,
    uqgrid_root: str | Path | None = None,
    pf_verbose: bool = False,
    fault_locations: Sequence[int] | str | None = None,
    fault_impedances: Sequence[float] | str | None = None,
    n_jobs: int | None = None,
    parallel_timeout_s: float = 600.0,
    probml_basename: str | None = None,
    keep_fault_histories: bool = False,
    debug_tracebacks: bool = False,
    acopf_context_func: Callable[..., Mapping[str, Any]] = prepare_acopf_replay_context,
    state_metadata_func: Callable[..., Mapping[str, Any]] = export_delta_state_metadata,
    fault_runner_func: Callable[..., Sequence[Mapping[str, Any]]] = run_fault_replay_tasks,
) -> dict[str, Any]:
    """Run Stage 3: one ACOPF-initialized OP, tiny parallel replay set, one NPZ row."""
    if int(target_accepted_scenarios) != 1:
        raise ValueError("Stage 3 smoke mode supports exactly one accepted scenario")
    output_dir = Path(output_dir).expanduser()
    config = _load_json_config(config_path)
    fault_locations_list = parse_int_list(fault_locations, default=[142, 143])
    fault_impedances_list = parse_float_list(
        fault_impedances,
        default=(config.get("scenarios", {}) or {}).get("fault_impedances", [1e-4]),
    )
    if not fault_locations_list or not fault_impedances_list:
        raise ValueError("Stage 3 smoke mode requires at least one fault location and impedance")

    basename = probml_basename or _default_probml_basename(config, smoke=True)
    final_path, min_path = _probml_output_paths(output_dir, basename)
    if final_path.exists() or min_path.exists():
        raise FileExistsError(
            f"Stage 3 smoke refuses to append to existing NPZ files: {final_path}, {min_path}"
        )

    requested_n_jobs = (
        int(n_jobs)
        if n_jobs is not None
        else int((config.get("execution", {}) or {}).get("n_jobs", len(fault_locations_list)))
    )
    task_count = len(fault_locations_list) * len(fault_impedances_list)
    effective_n_jobs = _effective_n_jobs(requested_n_jobs, task_count)

    context = dict(
        acopf_context_func(
            config_path=config_path,
            output_dir=output_dir,
            acopf_config=acopf_config,
            target_accepted_scenarios=target_accepted_scenarios,
            sample_idx_start=sample_idx_start,
            uqgrid_root=uqgrid_root,
            pf_verbose=pf_verbose,
        )
    )
    records = list(context.get("records", []))
    sample_idx = int(context.get("sample_idx", sample_idx_start))
    if not context.get("accepted"):
        progress = _stage3_progress(
            accepted=False,
            sample_idx=sample_idx,
            reject_reason=str(context.get("reject_reason", "acopf_smoke_failed")),
            records_written=len(records),
        )
        _write_smoke_diagnostics(output_dir, progress=progress, records=records)
        return progress

    try:
        state_info = dict(
            state_metadata_func(
                raw_path=context["raw_path"],
                dyr_path=context["dyr_path"],
                output_path=output_dir / "state_metadata.json",
            )
        )
    except Exception as exc:
        record = {
            "record_type": "state_metadata",
            "accepted": False,
            "reject_reason": "state_metadata_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        records.append(record)
        progress = _stage3_progress(
            accepted=False,
            sample_idx=sample_idx,
            reject_reason="state_metadata_failed",
            records_written=len(records),
        )
        _write_smoke_diagnostics(output_dir, progress=progress, records=records)
        return progress

    tasks = build_fault_replay_tasks(
        sample_idx=sample_idx,
        operating_point_id=str(context["operating_point_id"]),
        accepted_operating_point_index=int(context["accepted_operating_point_index"]),
        fault_locations=fault_locations_list,
        fault_impedances=fault_impedances_list,
    )
    replay_context = {
        "raw_path": str(context["raw_path"]),
        "dyr_path": str(context["dyr_path"]),
        "basecase_path": str(context["basecase_path"]),
        "case_raw_path": str(context["case_raw_path"]),
        "integration_config": _integration_config_from_config(config),
        "delta_state_indices": list(state_info["delta_state_indices"]),
        "keep_fault_histories": bool(keep_fault_histories),
        "history_dir": str(output_dir / "fault_histories"),
        "debug_tracebacks": bool(debug_tracebacks),
    }
    try:
        fault_results = list(
            fault_runner_func(
                tasks,
                replay_context,
                n_jobs=effective_n_jobs,
                parallel_timeout_s=float(parallel_timeout_s),
            )
        )
    except Exception as exc:
        record = {
            "record_type": "fault_scenario_group",
            "accepted": False,
            "reject_reason": "dynamic_fault_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        records.append(record)
        progress = _stage3_progress(
            accepted=False,
            sample_idx=sample_idx,
            reject_reason="dynamic_fault_failed",
            records_written=len(records),
        )
        _write_smoke_diagnostics(output_dir, progress=progress, records=records)
        return progress

    records.extend(fault_results)
    failed = [result for result in fault_results if not result.get("accepted")]
    if failed:
        group_record = {
            "record_type": "operating_point_group",
            "sample_idx": sample_idx,
            "accepted": False,
            "reject_reason": "dynamic_fault_failed",
            "faults_required": task_count,
            "faults_successful": task_count - len(failed),
        }
        records.append(group_record)
        progress = _stage3_progress(
            accepted=False,
            sample_idx=sample_idx,
            reject_reason="dynamic_fault_failed",
            records_written=len(records),
            fault_samples_completed=task_count - len(failed),
        )
        _write_smoke_diagnostics(output_dir, progress=progress, records=records)
        return progress

    probml_outputs = write_stage3_probml_outputs(
        output_dir=output_dir,
        probml_basename=basename,
        parsed_basecase=context["parsed_basecase"],
        base_mva=float(context["base_mva"]),
        sample_idx=sample_idx,
        fault_locations=fault_locations_list,
        fault_impedances=fault_impedances_list,
        fault_results=fault_results,
    )
    final_tsi_values = np.asarray([result["tsi_final"] for result in fault_results], dtype=float)
    min_tsi_values = np.asarray([result["tsi_min"] for result in fault_results], dtype=float)
    group_record = {
        "record_type": "operating_point_group",
        "sample_idx": sample_idx,
        "operating_point_id": context["operating_point_id"],
        "accepted_operating_point_index": context["accepted_operating_point_index"],
        "accepted": True,
        "reject_reason": None,
        "faults_required": task_count,
        "faults_successful": task_count,
        "effective_n_jobs": effective_n_jobs,
        "final_tsi_min": float(np.nanmin(final_tsi_values)),
        "final_tsi_num_unstable": int(np.sum(final_tsi_values <= 0.0)),
        "min_tsi_min": float(np.nanmin(min_tsi_values)),
        "min_tsi_num_unstable": int(np.sum(min_tsi_values <= 0.0)),
        "probml_outputs": probml_outputs,
    }
    records.append(group_record)
    progress = _stage3_progress(
        accepted=True,
        sample_idx=sample_idx,
        records_written=len(records),
        fault_samples_completed=task_count,
        probml_outputs=probml_outputs,
    )
    _write_smoke_diagnostics(output_dir, progress=progress, records=records)
    return progress


def _read_json_file(path: str | Path, default: Any) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _production_state_paths(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser()
    return {
        "progress": output_dir / "acopf_init_progress.json",
        "diagnostics": output_dir / "acopf_init_diagnostics.jsonl",
        "diagnostics_summary": output_dir / "acopf_init_diagnostics_summary.json",
        "scenario_metadata": output_dir / "scenario_metadata.json",
        "simulation_log": output_dir / "simulation_log.json",
        "state_metadata": output_dir / "state_metadata.json",
    }


def _npz_shapes(path: Path) -> dict[str, tuple[int, ...]] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return {key: tuple(int(v) for v in data[key].shape) for key in data.files}


def _infer_status_paths(
    output_dir: str | Path,
    *,
    probml_basename: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[Path | None, Path | None]:
    output_dir = Path(output_dir).expanduser()
    if probml_basename is not None:
        return _probml_output_paths(output_dir, probml_basename)
    if config is not None:
        return _probml_output_paths(output_dir, _default_probml_basename(config))
    final_candidates = sorted(output_dir.glob("*_final.npz"))
    if len(final_candidates) != 1:
        return None, None
    final_path = final_candidates[0]
    suffix = "_final.npz"
    min_path = final_path.with_name(final_path.name[: -len(suffix)] + "_min.npz")
    return final_path, min_path


def read_acopf_production_status(
    *,
    output_dir: str | Path,
    probml_basename: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read Stage 4 output state without running candidate, ACOPF, or replay code."""
    output_dir = Path(output_dir).expanduser()
    state_paths = _production_state_paths(output_dir)
    final_path, min_path = _infer_status_paths(
        output_dir,
        probml_basename=probml_basename,
        config=config,
    )
    progress = _read_json_file(state_paths["progress"], {})
    summary = _read_json_file(state_paths["diagnostics_summary"], {})
    simulation_log = _read_json_file(state_paths["simulation_log"], {})
    scenario_metadata = _read_json_file(state_paths["scenario_metadata"], {})

    resume: dict[str, Any] = {}
    resume_error = None
    if final_path is not None and min_path is not None:
        try:
            resume = validate_probml_resume_pair(final_path, min_path)
        except Exception as exc:
            resume_error = f"{type(exc).__name__}: {exc}"

    return {
        "stage": progress.get("stage", "stage_4_production"),
        "accepted_count": int(progress.get("accepted_count", resume.get("accepted_count", 0))),
        "next_sample_idx": int(
            progress.get("next_sample_idx", resume.get("next_sample_idx", 0))
        ),
        "target_accepted_scenarios": progress.get("target_accepted_scenarios"),
        "fault_rows_completed": len(simulation_log),
        "scenario_metadata_rows": len(scenario_metadata),
        "reject_reason_counts": summary.get("reject_reason_counts", {}),
        "latest_row_paths": {
            "final": None if final_path is None else str(final_path),
            "min": None if min_path is None else str(min_path),
        },
        "npz_shapes": {
            "final": None if final_path is None else _npz_shapes(final_path),
            "min": None if min_path is None else _npz_shapes(min_path),
        },
        "resume_error": resume_error,
        "progress_path": str(state_paths["progress"]),
        "diagnostics_path": str(state_paths["diagnostics"]),
    }


def _production_progress(
    *,
    target_accepted_scenarios: int,
    accepted_count: int,
    next_sample_idx: int,
    last_sample_idx: int | None,
    total_candidate_attempts: int,
    max_total_attempts: int,
    records_written: int,
    fault_rows_completed: int,
    probml_outputs: Mapping[str, Any] | None = None,
    reject_reason: str | None = None,
) -> dict[str, Any]:
    progress = {
        "stage": "stage_4_production",
        "target_accepted_scenarios": int(target_accepted_scenarios),
        "accepted_count": int(accepted_count),
        "next_sample_idx": int(next_sample_idx),
        "last_sample_idx": None if last_sample_idx is None else int(last_sample_idx),
        "total_candidate_attempts": int(total_candidate_attempts),
        "max_total_attempts": int(max_total_attempts),
        "diagnostic_records": int(records_written),
        "fault_rows_completed": int(fault_rows_completed),
        "completed": int(accepted_count) >= int(target_accepted_scenarios),
        "attempts_exhausted": int(total_candidate_attempts) >= int(max_total_attempts),
        "reject_reason": reject_reason,
    }
    if probml_outputs:
        progress["probml_outputs"] = dict(probml_outputs)
    return progress


def _write_production_state(
    output_dir: str | Path,
    *,
    progress: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    scenario_metadata: Mapping[str, Any],
    simulation_log: Mapping[str, Any],
) -> None:
    state_paths = _production_state_paths(output_dir)
    _write_json(state_paths["progress"], progress)
    _write_jsonl(state_paths["diagnostics"], records)
    _write_json(state_paths["diagnostics_summary"], _summarize_smoke_records(records))
    _write_json(state_paths["scenario_metadata"], scenario_metadata)
    _write_json(state_paths["simulation_log"], simulation_log)


def _load_production_resume_state(
    *,
    output_dir: str | Path,
    final_path: Path,
    min_path: Path,
    continue_run: bool,
    fault_locations: Sequence[int],
    fault_impedances: Sequence[float],
) -> dict[str, Any]:
    state_paths = _production_state_paths(output_dir)
    managed_paths = [final_path, min_path, *state_paths.values()]
    if not continue_run:
        existing = [str(path) for path in managed_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Stage 4 output files already exist. Use --continue after verifying "
                "the run state, or choose a fresh --output-dir/--probml-basename. "
                f"Existing paths: {existing}"
            )
        return {
            "accepted_count": 0,
            "next_sample_idx": 0,
            "total_candidate_attempts": 0,
            "records": [],
            "scenario_metadata": {},
            "simulation_log": {},
            "progress": {},
        }

    resume = validate_probml_resume_pair(final_path, min_path)
    progress = _read_json_file(state_paths["progress"], {})
    if progress:
        progress_accepted = int(progress.get("accepted_count", -1))
        progress_next_sample = int(progress.get("next_sample_idx", -1))
        if progress_accepted != int(resume["accepted_count"]):
            raise ValueError(
                "Resume rejected: progress accepted_count disagrees with final/min NPZ "
                f"({progress_accepted} != {resume['accepted_count']}). Repair the "
                "output directory by restoring a matching progress file or removing "
                "partial outputs before rerunning."
            )
        if progress_next_sample != int(resume["next_sample_idx"]):
            raise ValueError(
                "Resume rejected: progress next_sample_idx disagrees with final/min NPZ "
                f"({progress_next_sample} != {resume['next_sample_idx']}). Repair the "
                "output directory by restoring a matching progress file or removing "
                "partial outputs before rerunning."
            )
    elif resume["accepted_count"]:
        raise ValueError(
            "Resume rejected: NPZ rows exist but acopf_init_progress.json is missing. "
            "Restore the matching progress file or start from a clean output directory."
        )

    scenario_metadata = _read_json_file(state_paths["scenario_metadata"], {})
    simulation_log = _read_json_file(state_paths["simulation_log"], {})
    expected_fault_rows = (
        int(resume["accepted_count"]) * len(fault_locations) * len(fault_impedances)
    )
    if len(simulation_log) != expected_fault_rows:
        raise ValueError(
            "Resume rejected: simulation_log.json row count disagrees with NPZ rows "
            f"({len(simulation_log)} != {expected_fault_rows}). Repair the output "
            "directory by restoring matching log/metadata files or removing partial outputs."
        )
    if len(scenario_metadata) != expected_fault_rows:
        raise ValueError(
            "Resume rejected: scenario_metadata.json row count disagrees with NPZ rows "
            f"({len(scenario_metadata)} != {expected_fault_rows}). Repair the output "
            "directory by restoring matching metadata or removing partial outputs."
        )

    return {
        "accepted_count": int(resume["accepted_count"]),
        "next_sample_idx": int(resume["next_sample_idx"]),
        "total_candidate_attempts": int(progress.get("total_candidate_attempts", 0)),
        "records": _read_jsonl(state_paths["diagnostics"]),
        "scenario_metadata": dict(scenario_metadata),
        "simulation_log": dict(simulation_log),
        "progress": progress,
    }


def _fault_result_entries(
    fault_results: Sequence[Mapping[str, Any]],
    *,
    sample_idx: int,
    operating_point_id: str,
    accepted_operating_point_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata: dict[str, Any] = {}
    simulation_log: dict[str, Any] = {}
    for result in fault_results:
        sid = str(result["scenario_id"])
        base = {
            "sample_idx": int(sample_idx),
            "fault_location": int(result["fault_location"]),
            "fault_impedance": float(result["fault_impedance"]),
            "operating_point_id": str(operating_point_id),
            "accepted_operating_point_index": int(accepted_operating_point_index),
            "tsi_final": float(result["tsi_final"]),
            "tsi_min": float(result["tsi_min"]),
            "file": None,
        }
        metadata[sid] = dict(base)
        simulation_log[sid] = {
            **base,
            "scenario_id": sid,
            "accepted": True,
            "reject_reason": None,
            "simulation_diverged": False,
            "history_file": result.get("history_file"),
            "final_simulation_time_s": result.get("final_simulation_time_s"),
        }
    return metadata, simulation_log


def _delete_case_dir(case_dir: Any) -> None:
    if not case_dir:
        return
    path = Path(case_dir)
    if path.exists():
        shutil.rmtree(path)


def _print_production_progress(progress: Mapping[str, Any]) -> None:
    print(
        "ACOPF production | "
        f"accepted OPs {progress['accepted_count']}/{progress['target_accepted_scenarios']} | "
        f"candidate attempts {progress['total_candidate_attempts']}/"
        f"{progress['max_total_attempts']} | "
        f"fault rows {progress['fault_rows_completed']} | "
        f"next sample {progress['next_sample_idx']} | "
        f"last: {progress.get('reject_reason') or 'accepted'}"
    )


def run_acopf_production(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    acopf_config: AcopfInitializationConfig,
    target_accepted_scenarios: int | None = None,
    max_total_attempts: int | None = None,
    continue_run: bool = False,
    sample_idx_start: int = 0,
    uqgrid_root: str | Path | None = None,
    pf_verbose: bool = False,
    fault_locations: Sequence[int] | str | None = None,
    fault_impedances: Sequence[float] | str | None = None,
    n_jobs: int | None = None,
    parallel_timeout_s: float = 600.0,
    probml_basename: str | None = None,
    keep_intermediate_acopf_cases: bool = False,
    keep_failed_acopf_cases: bool = True,
    keep_fault_histories: bool = False,
    debug_tracebacks: bool = False,
    acopf_context_func: Callable[..., Mapping[str, Any]] = prepare_acopf_candidate_context,
    state_metadata_func: Callable[..., Mapping[str, Any]] = export_delta_state_metadata,
    fault_runner_func: Callable[..., Sequence[Mapping[str, Any]]] = run_fault_replay_tasks,
    candidate_func: Callable[..., Mapping[str, Any]] | None = None,
    op_config_resolver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    acopf_runner_func: Callable[..., Mapping[str, Any]] = run_exajugo_acopf,
    pf_validator_func: Callable[..., Mapping[str, Any]] = validate_acopf_power_flow,
) -> dict[str, Any]:
    """Run Stage 4 production ACOPF-initialized scenario generation."""
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path).expanduser()
    config = _load_json_config(config_path)
    model_cfg = config["model"]
    scenario_cfg = config.get("scenarios", {}) or {}
    exec_cfg = config.get("execution", {}) or {}

    n_bus = int(model_cfg.get("n_bus", 0))
    if n_bus <= 0:
        raise ValueError("Production mode requires model.n_bus in the config")
    target = int(
        target_accepted_scenarios
        if target_accepted_scenarios is not None
        else scenario_cfg.get("target_accepted_scenarios", 1000)
    )
    if target <= 0:
        raise ValueError("target_accepted_scenarios must be positive")

    if max_total_attempts is None:
        max_total_attempts = scenario_cfg.get("max_total_attempts")
    if max_total_attempts is None:
        max_attempts_per_scenario = int(
            (config.get("operating_point", {}) or {}).get("max_attempts_per_scenario", 100)
        )
        max_total_attempts = target * max_attempts_per_scenario
    max_attempts = int(max_total_attempts)
    if max_attempts <= 0:
        raise ValueError("max_total_attempts must be positive")

    fault_locations_value = (
        fault_locations if fault_locations is not None else scenario_cfg.get("fault_locations", "all")
    )
    fault_locations_list = resolve_fault_locations(fault_locations_value, n_bus=n_bus)
    fault_impedances_list = parse_float_list(
        fault_impedances,
        default=scenario_cfg.get("fault_impedances", [1e-4]),
    )
    if not fault_locations_list or not fault_impedances_list:
        raise ValueError("Production mode requires at least one fault location and impedance")
    task_count = len(fault_locations_list) * len(fault_impedances_list)
    requested_n_jobs = (
        int(n_jobs)
        if n_jobs is not None
        else int(exec_cfg.get("n_jobs", task_count))
    )
    effective_n_jobs = _effective_n_jobs(requested_n_jobs, task_count)

    basename = probml_basename or _default_probml_basename(config)
    final_path, min_path = _probml_output_paths(output_dir, basename)
    resume_state = _load_production_resume_state(
        output_dir=output_dir,
        final_path=final_path,
        min_path=min_path,
        continue_run=continue_run,
        fault_locations=fault_locations_list,
        fault_impedances=fault_impedances_list,
    )
    records = list(resume_state["records"])
    scenario_metadata = dict(resume_state["scenario_metadata"])
    simulation_log = dict(resume_state["simulation_log"])
    accepted_count = int(resume_state["accepted_count"])
    accepted_index = accepted_count
    sample_idx = int(
        resume_state["next_sample_idx"] if continue_run else int(sample_idx_start)
    )
    total_candidate_attempts = int(resume_state["total_candidate_attempts"])
    latest_probml_outputs = None
    last_sample_idx = sample_idx - 1 if accepted_count else None
    last_reject_reason = None

    raw_path = _resolve_model_path(
        model_cfg["raw"],
        config_path=config_path,
        uqgrid_root=uqgrid_root,
    )
    dyr_path = _resolve_model_path(
        model_cfg["dyr"],
        config_path=config_path,
        uqgrid_root=uqgrid_root,
    )

    if accepted_count >= target or total_candidate_attempts >= max_attempts:
        progress = _production_progress(
            target_accepted_scenarios=target,
            accepted_count=accepted_count,
            next_sample_idx=sample_idx,
            last_sample_idx=last_sample_idx,
            total_candidate_attempts=total_candidate_attempts,
            max_total_attempts=max_attempts,
            records_written=len(records),
            fault_rows_completed=len(simulation_log),
            probml_outputs=latest_probml_outputs,
        )
        _write_production_state(
            output_dir,
            progress=progress,
            records=records,
            scenario_metadata=scenario_metadata,
            simulation_log=simulation_log,
        )
        return progress

    state_info = dict(
        state_metadata_func(
            raw_path=raw_path,
            dyr_path=dyr_path,
            output_path=output_dir / "state_metadata.json",
        )
    )

    while accepted_count < target and total_candidate_attempts < max_attempts:
        remaining_attempts = max_attempts - total_candidate_attempts
        context_kwargs: dict[str, Any] = {
            "config_path": config_path,
            "output_dir": output_dir,
            "acopf_config": acopf_config,
            "sample_idx": sample_idx,
            "accepted_operating_point_index": accepted_index,
            "target_accepted_scenarios": target,
            "max_total_attempts": max_attempts,
            "remaining_attempts": remaining_attempts,
            "uqgrid_root": uqgrid_root,
            "pf_verbose": pf_verbose,
            "case_root": "acopf_cases",
            "acopf_runner_func": acopf_runner_func,
            "pf_validator_func": pf_validator_func,
        }
        if candidate_func is not None:
            context_kwargs["candidate_func"] = candidate_func
        if op_config_resolver is not None:
            context_kwargs["op_config_resolver"] = op_config_resolver
        context = dict(acopf_context_func(**context_kwargs))
        last_sample_idx = int(context.get("sample_idx", sample_idx))
        candidate_attempts = int(context.get("candidate_attempts", 1))
        total_candidate_attempts += candidate_attempts
        context_records = list(context.get("records", []))
        for record in context_records:
            if record.get("record_type") == "operating_point_group":
                record.setdefault("total_candidate_attempts", total_candidate_attempts)
                record.setdefault("faults_required", task_count)
        records.extend(context_records)

        if not context.get("accepted"):
            last_reject_reason = str(context.get("reject_reason", "candidate_rejected"))
            if context.get("case_dir") and not keep_failed_acopf_cases:
                _delete_case_dir(context.get("case_dir"))
            sample_idx += 1
            progress = _production_progress(
                target_accepted_scenarios=target,
                accepted_count=accepted_count,
                next_sample_idx=sample_idx,
                last_sample_idx=last_sample_idx,
                total_candidate_attempts=total_candidate_attempts,
                max_total_attempts=max_attempts,
                records_written=len(records),
                fault_rows_completed=len(simulation_log),
                probml_outputs=latest_probml_outputs,
                reject_reason=last_reject_reason,
            )
            _write_production_state(
                output_dir,
                progress=progress,
                records=records,
                scenario_metadata=scenario_metadata,
                simulation_log=simulation_log,
            )
            _print_production_progress(progress)
            continue

        tasks = build_fault_replay_tasks(
            sample_idx=int(context["sample_idx"]),
            operating_point_id=str(context["operating_point_id"]),
            accepted_operating_point_index=int(context["accepted_operating_point_index"]),
            fault_locations=fault_locations_list,
            fault_impedances=fault_impedances_list,
        )
        replay_context = {
            "raw_path": str(context["raw_path"]),
            "dyr_path": str(context["dyr_path"]),
            "basecase_path": str(context["basecase_path"]),
            "case_raw_path": str(context["case_raw_path"]),
            "integration_config": _integration_config_from_config(config),
            "delta_state_indices": list(state_info["delta_state_indices"]),
            "keep_fault_histories": bool(keep_fault_histories),
            "history_dir": str(output_dir / "fault_histories"),
            "debug_tracebacks": bool(debug_tracebacks),
        }
        try:
            fault_results = list(
                fault_runner_func(
                    tasks,
                    replay_context,
                    n_jobs=effective_n_jobs,
                    parallel_timeout_s=float(parallel_timeout_s),
                )
            )
        except Exception as exc:
            fault_results = []
            records.append(
                {
                    "record_type": "fault_scenario_group",
                    "sample_idx": int(context["sample_idx"]),
                    "operating_point_id": str(context["operating_point_id"]),
                    "accepted_operating_point_index": int(
                        context["accepted_operating_point_index"]
                    ),
                    "accepted": False,
                    "reject_reason": "dynamic_fault_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

        records.extend(fault_results)
        failed = [result for result in fault_results if not result.get("accepted")]
        if len(fault_results) != task_count or failed:
            last_reject_reason = "dynamic_fault_failed"
            failure_counts: dict[str, int] = {}
            for result in failed:
                reason = str(result.get("reject_reason", "dynamic_fault_failed"))
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
            if len(fault_results) != task_count:
                failure_counts["dynamic_fault_missing_result"] = (
                    task_count - len(fault_results)
                )
            records.append(
                {
                    "record_type": "operating_point_group",
                    "sample_idx": int(context["sample_idx"]),
                    "operating_point_id": str(context["operating_point_id"]),
                    "accepted_operating_point_index": int(
                        context["accepted_operating_point_index"]
                    ),
                    "accepted": False,
                    "reject_reason": "dynamic_fault_failed",
                    "total_candidate_attempts": total_candidate_attempts,
                    "max_total_attempts": max_attempts,
                    "faults_required": task_count,
                    "faults_successful": sum(
                        1 for result in fault_results if result.get("accepted")
                    ),
                    "fault_failure_reasons": failure_counts,
                }
            )
            if not keep_failed_acopf_cases:
                _delete_case_dir(context.get("case_dir"))
            sample_idx += 1
            progress = _production_progress(
                target_accepted_scenarios=target,
                accepted_count=accepted_count,
                next_sample_idx=sample_idx,
                last_sample_idx=last_sample_idx,
                total_candidate_attempts=total_candidate_attempts,
                max_total_attempts=max_attempts,
                records_written=len(records),
                fault_rows_completed=len(simulation_log),
                probml_outputs=latest_probml_outputs,
                reject_reason=last_reject_reason,
            )
            _write_production_state(
                output_dir,
                progress=progress,
                records=records,
                scenario_metadata=scenario_metadata,
                simulation_log=simulation_log,
            )
            _print_production_progress(progress)
            continue

        probml_outputs = append_acopf_probml_outputs(
            output_dir=output_dir,
            probml_basename=basename,
            parsed_basecase=context["parsed_basecase"],
            base_mva=float(context["base_mva"]),
            sample_idx=int(context["sample_idx"]),
            fault_locations=fault_locations_list,
            fault_impedances=fault_impedances_list,
            fault_results=fault_results,
        )
        final_tsi_values = np.asarray(
            [result["tsi_final"] for result in fault_results],
            dtype=float,
        )
        min_tsi_values = np.asarray(
            [result["tsi_min"] for result in fault_results],
            dtype=float,
        )
        records.append(
            {
                "record_type": "operating_point_group",
                "sample_idx": int(context["sample_idx"]),
                "operating_point_id": str(context["operating_point_id"]),
                "accepted_operating_point_index": int(
                    context["accepted_operating_point_index"]
                ),
                "accepted": True,
                "reject_reason": None,
                "total_candidate_attempts": total_candidate_attempts,
                "max_total_attempts": max_attempts,
                "faults_required": task_count,
                "faults_successful": task_count,
                "effective_n_jobs": effective_n_jobs,
                "final_tsi_min": float(np.nanmin(final_tsi_values)),
                "final_tsi_num_unstable": int(np.sum(final_tsi_values <= 0.0)),
                "min_tsi_min": float(np.nanmin(min_tsi_values)),
                "min_tsi_num_unstable": int(np.sum(min_tsi_values <= 0.0)),
                "probml_outputs": probml_outputs,
            }
        )
        new_metadata, new_log = _fault_result_entries(
            fault_results,
            sample_idx=int(context["sample_idx"]),
            operating_point_id=str(context["operating_point_id"]),
            accepted_operating_point_index=int(context["accepted_operating_point_index"]),
        )
        scenario_metadata.update(new_metadata)
        simulation_log.update(new_log)
        last_reject_reason = None
        next_accepted_count = accepted_count + 1
        next_sample_idx = sample_idx + 1
        latest_probml_outputs = probml_outputs
        progress = _production_progress(
            target_accepted_scenarios=target,
            accepted_count=next_accepted_count,
            next_sample_idx=next_sample_idx,
            last_sample_idx=last_sample_idx,
            total_candidate_attempts=total_candidate_attempts,
            max_total_attempts=max_attempts,
            records_written=len(records),
            fault_rows_completed=len(simulation_log),
            probml_outputs=latest_probml_outputs,
        )
        _write_production_state(
            output_dir,
            progress=progress,
            records=records,
            scenario_metadata=scenario_metadata,
            simulation_log=simulation_log,
        )
        accepted_count = next_accepted_count
        accepted_index += 1
        sample_idx = next_sample_idx
        if not keep_intermediate_acopf_cases:
            _delete_case_dir(context.get("case_dir"))
        _print_production_progress(progress)
        gc.collect()

    final_progress = _production_progress(
        target_accepted_scenarios=target,
        accepted_count=accepted_count,
        next_sample_idx=sample_idx,
        last_sample_idx=last_sample_idx,
        total_candidate_attempts=total_candidate_attempts,
        max_total_attempts=max_attempts,
        records_written=len(records),
        fault_rows_completed=len(simulation_log),
        probml_outputs=latest_probml_outputs,
        reject_reason=last_reject_reason,
    )
    _write_production_state(
        output_dir,
        progress=final_progress,
        records=records,
        scenario_metadata=scenario_metadata,
        simulation_log=simulation_log,
    )
    return final_progress


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ACOPF-initialized UQGrid scenario generation",
    )
    parser.add_argument("config", nargs="?", help="Scenario generator JSON config")
    parser.add_argument(
        "--smoke-acopf",
        action="store_true",
        help="Run Stage 2 ACOPF smoke integration only",
    )
    parser.add_argument(
        "--smoke-replay",
        action="store_true",
        help="Run Stage 3 ACOPF plus tiny dynamic replay smoke integration",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Read Stage 4 output status without running generation",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--julia", default=None, help="Julia executable")
    parser.add_argument("--exajugo-root", default=None, help="ExaJuGO repository root")
    parser.add_argument("--exajugo-base-raw", default=None, help="Base ExaJuGO RAW file")
    parser.add_argument("--exajugo-base-rop", default=None, help="Base ExaJuGO ROP file")
    parser.add_argument(
        "--acopf-timeout-s",
        type=float,
        default=None,
        help="ExaJuGO ACOPF subprocess timeout in seconds",
    )
    parser.add_argument(
        "--target-accepted-scenarios",
        type=int,
        default=None,
        help="Accepted operating-point target; smoke modes default to 1",
    )
    parser.add_argument(
        "--max-total-attempts",
        type=int,
        default=None,
        help="Maximum top-level PF-screened candidate attempts in production mode",
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Resume a Stage 4 production output directory after validation",
    )
    parser.add_argument("--sample-idx-start", type=int, default=0)
    parser.add_argument("--uqgrid-root", default=None, help="Root for relative model paths")
    parser.add_argument("--pf-verbose", action="store_true", help="Verbose UQGrid PF check")
    parser.add_argument(
        "--fault-locations",
        default=None,
        help="all or comma-separated zero-based UQGrid fault bus indices",
    )
    parser.add_argument(
        "--fault-impedances",
        default=None,
        help="Comma-separated fault impedances for Stage 3 smoke",
    )
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel fault jobs")
    parser.add_argument(
        "--parallel-timeout-s",
        type=float,
        default=600.0,
        help="Timeout for parallel fault replay tasks",
    )
    parser.add_argument("--probml-basename", default=None, help="ProbML output basename")
    parser.add_argument(
        "--keep-intermediate-acopf-cases",
        action="store_true",
        help="Keep accepted ACOPF case directories after successful row writes",
    )
    parser.add_argument(
        "--keep-failed-acopf-cases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep failed ACOPF case directories for debugging",
    )
    parser.add_argument(
        "--keep-fault-histories",
        action="store_true",
        help="Write dense per-fault histories for debugging",
    )
    parser.add_argument(
        "--debug-tracebacks",
        action="store_true",
        help="Print dynamic replay tracebacks from workers",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.smoke_acopf and args.smoke_replay:
        parser.error("Use only one of --smoke-acopf or --smoke-replay")
    if not args.output_dir:
        parser.error("--output-dir is required")

    if args.status:
        config = _load_json_config(args.config) if args.config else None
        status = read_acopf_production_status(
            output_dir=args.output_dir,
            probml_basename=args.probml_basename,
            config=config,
        )
        print(json.dumps(_json_safe(status), sort_keys=True))
        return 0

    if args.continue_run and (args.smoke_acopf or args.smoke_replay):
        parser.error("--continue is only supported in Stage 4 production mode")
    if not args.config:
        parser.error("generation modes require a config file")

    config = _load_json_config(args.config)
    acopf_config = resolve_acopf_initialization_config(args, config)
    if args.smoke_replay:
        progress = run_acopf_replay_smoke(
            config_path=args.config,
            output_dir=args.output_dir,
            acopf_config=acopf_config,
            target_accepted_scenarios=args.target_accepted_scenarios or 1,
            sample_idx_start=args.sample_idx_start,
            uqgrid_root=args.uqgrid_root,
            pf_verbose=args.pf_verbose,
            fault_locations=args.fault_locations,
            fault_impedances=args.fault_impedances,
            n_jobs=args.n_jobs,
            parallel_timeout_s=args.parallel_timeout_s,
            probml_basename=args.probml_basename,
            keep_fault_histories=args.keep_fault_histories,
            debug_tracebacks=args.debug_tracebacks,
        )
    elif args.smoke_acopf:
        progress = run_acopf_smoke(
            config_path=args.config,
            output_dir=args.output_dir,
            acopf_config=acopf_config,
            target_accepted_scenarios=args.target_accepted_scenarios or 1,
            sample_idx_start=args.sample_idx_start,
            uqgrid_root=args.uqgrid_root,
            pf_verbose=args.pf_verbose,
        )
    else:
        progress = run_acopf_production(
            config_path=args.config,
            output_dir=args.output_dir,
            acopf_config=acopf_config,
            target_accepted_scenarios=args.target_accepted_scenarios,
            max_total_attempts=args.max_total_attempts,
            continue_run=args.continue_run,
            sample_idx_start=args.sample_idx_start,
            uqgrid_root=args.uqgrid_root,
            pf_verbose=args.pf_verbose,
            fault_locations=args.fault_locations,
            fault_impedances=args.fault_impedances,
            n_jobs=args.n_jobs,
            parallel_timeout_s=args.parallel_timeout_s,
            probml_basename=args.probml_basename,
            keep_intermediate_acopf_cases=args.keep_intermediate_acopf_cases,
            keep_failed_acopf_cases=args.keep_failed_acopf_cases,
            keep_fault_histories=args.keep_fault_histories,
            debug_tracebacks=args.debug_tracebacks,
        )
    print(json.dumps(_json_safe(progress), sort_keys=True))
    if progress.get("stage") == "stage_4_production":
        return 0 if progress.get("completed") else 1
    return 0 if progress.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
