#!/usr/bin/env python3
"""Validate ACOPF/direct-UQGrid initialization handoffs and output datasets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_scenarios_acopf_init as acopf_init  # noqa: E402


REQUIRED_NPZ_KEYS = {
    "X",
    "X_flat",
    "Y",
    "sample_idx",
    "fault_locations",
    "fault_impedances",
    "scenario_ids",
    "meta",
}
OPTIONAL_LABEL_KEYS = {"initialization_source", "acopf_feasible"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
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
    tmp_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "joblib": _package_version("joblib"),
        "petsc4py": _package_version("petsc4py"),
        "uqgrid": _package_version("uqgrid"),
    }


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    details: Any = None,
) -> None:
    record = {"name": str(name), "passed": bool(passed)}
    if details is not None:
        record["details"] = _json_safe(details)
    checks.append(record)


def _emit_checks(report: Mapping[str, Any]) -> None:
    for check in report.get("checks", []):
        marker = "✓" if check.get("passed") else "✗"
        details = check.get("details")
        suffix = "" if details is None else f": {details}"
        print(f"{marker} {check.get('name')}{suffix}", file=sys.stderr)


def collect_exciter_limit_diagnostics(
    psys: Any,
    initial_state: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Collect initialized SEXS Efd values and compare them with EMIN/EMAX."""
    if tolerance < 0.0:
        raise ValueError("Efd limit tolerance must be non-negative")

    state = np.asarray(initial_state, dtype=float).reshape(-1)
    records: list[dict[str, Any]] = []
    for exciter_index, exciter in enumerate(getattr(psys, "exc", [])):
        if not all(hasattr(exciter, name) for name in ("Emin", "Emax", "dif_ptr")):
            continue
        efd_offset = int(getattr(exciter, "efd_idx", 1))
        state_index = int(exciter.dif_ptr) + efd_offset
        if state_index < 0 or state_index >= state.size:
            records.append(
                {
                    "exciter_index": exciter_index,
                    "id": str(getattr(exciter, "id", getattr(exciter, "id_tag", exciter_index))),
                    "state_index": state_index,
                    "finite": False,
                    "within_limits": False,
                    "error": "Efd state index is outside the initialized state vector",
                }
            )
            continue

        efd = float(state[state_index])
        emin = float(exciter.Emin)
        emax = float(exciter.Emax)
        finite = bool(np.isfinite(efd) and np.isfinite(emin) and np.isfinite(emax))
        within = bool(finite and efd >= emin - tolerance and efd <= emax + tolerance)
        records.append(
            {
                "exciter_index": exciter_index,
                "id": str(getattr(exciter, "id", getattr(exciter, "id_tag", exciter_index))),
                "state_index": state_index,
                "efd": efd if np.isfinite(efd) else None,
                "emin": emin if np.isfinite(emin) else None,
                "emax": emax if np.isfinite(emax) else None,
                "lower_margin": efd - emin if finite else None,
                "upper_margin": emax - efd if finite else None,
                "finite": finite,
                "within_limits": within,
            }
        )

    finite_efd = [record["efd"] for record in records if record.get("efd") is not None]
    finite_margins = [
        min(record["lower_margin"], record["upper_margin"])
        for record in records
        if record.get("lower_margin") is not None and record.get("upper_margin") is not None
    ]
    violations = [record for record in records if not record.get("within_limits")]
    return {
        "applicable": bool(records),
        "count": len(records),
        "violation_count": len(violations),
        "finite_count": sum(bool(record.get("finite")) for record in records),
        "efd_min": min(finite_efd) if finite_efd else None,
        "efd_max": max(finite_efd) if finite_efd else None,
        "minimum_limit_margin": min(finite_margins) if finite_margins else None,
        "tolerance": float(tolerance),
        "violations": violations,
        "records": records,
    }


def compute_trajectory_drift(history: np.ndarray, psys: Any) -> dict[str, float]:
    """Return maximum no-disturbance drift by DAE state block."""
    values = np.asarray(history, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("history must have shape (state, time) with at least one time point")
    if not np.all(np.isfinite(values)):
        raise ValueError("history contains non-finite values")

    dif_size = int(psys.num_dof_dif)
    alg_size = int(psys.num_dof_alg)
    if dif_size < 0 or alg_size < 0 or dif_size + alg_size > values.shape[0]:
        raise ValueError("Power-system DAE dimensions do not match history")

    delta = np.abs(values - values[:, :1])

    def block_max(block: np.ndarray) -> float:
        return float(np.max(block)) if block.size else 0.0

    return {
        "differential": block_max(delta[:dif_size]),
        "algebraic": block_max(delta[dif_size : dif_size + alg_size]),
        "voltage": block_max(delta[dif_size + alg_size :]),
        "total": block_max(delta),
    }


def build_validation_integration_config(
    config: Mapping[str, Any],
    *,
    steps: int = 5,
    petsc: bool | None = None,
    integration_config_cls: Callable[..., Any] | None = None,
) -> Any:
    """Build the strict, undisturbed IntegrationConfig used by the validator."""
    if steps < 1:
        raise ValueError("steps must be at least one")
    values = acopf_init._integration_config_from_config(config)
    validation = values.get("power_flow_validation", {}) or {}
    if not values.get("enforce_q_limits", False):
        raise ValueError("Stage 5 requires integration.enforce_q_limits=true")
    if not validation.get("enabled", False):
        raise ValueError(
            "Stage 5 requires integration.power_flow_validation.enabled=true"
        )
    dt = float(values["dt"])
    if dt <= 0.0:
        raise ValueError("integration.dt must be positive")

    tend = float(steps) * dt
    values.update(
        {
            "steps": int(steps),
            "tend": tend,
            "ton": tend,
            "toff": tend + dt,
            "method": "beuler",
            "verbose": False,
        }
    )
    if petsc is not None:
        values["petsc"] = bool(petsc)
    if integration_config_cls is None:
        from uqgrid.simulation.config import IntegrationConfig

        integration_config_cls = IntegrationConfig
    return integration_config_cls(**values)


def apply_initialization_context(
    context: Mapping[str, Any],
    source: str,
    *,
    load_psse_func: Callable[..., Any] | None = None,
    add_dyr_func: Callable[..., Any] | None = None,
    parse_basecase_func: Callable[..., Any] | None = None,
    apply_basecase_func: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Apply one production replay context to a fresh UQGrid system."""
    if source not in {"acopf", "uqgrid_pf"}:
        raise ValueError("source must be 'acopf' or 'uqgrid_pf'")
    if load_psse_func is None or add_dyr_func is None:
        from uqgrid.io.parse import add_dyr, load_psse

        load_psse_func = load_psse_func or load_psse
        add_dyr_func = add_dyr_func or add_dyr

    psys = load_psse_func(str(context["raw_path"]))
    add_dyr_func(psys, str(context["dyr_path"]))
    if source == "acopf":
        parse_basecase_func = parse_basecase_func or acopf_init.parse_exajugo_basecase
        apply_basecase_func = apply_basecase_func or acopf_init.apply_exajugo_solution_to_psys
        parsed = parse_basecase_func(
            context["basecase_path"],
            raw_path=context["case_raw_path"],
        )
        summary = dict(apply_basecase_func(psys, parsed))
    else:
        operating_point = context["operating_point"]
        p_load = np.asarray(operating_point["p_load_scaled"], dtype=float).reshape(-1)
        q_load = np.asarray(operating_point["q_load_scaled"], dtype=float).reshape(-1)
        p_gen = np.asarray(operating_point["p_gen_scaled"], dtype=float).reshape(-1)
        q_gen = np.asarray(operating_point["q_gen_scaled"], dtype=float).reshape(-1)
        psys.set_load_pq(p_load, q_load)
        psys.set_gen_pq(p_gen, q_gen)
        v_magnitudes = operating_point.get("pf_v_magnitudes")
        v_angles = operating_point.get("pf_v_angles")
        if v_magnitudes is not None and v_angles is not None:
            if len(v_magnitudes) != len(psys.buses) or len(v_angles) != len(psys.buses):
                raise ValueError("Saved PF voltage initialization does not match bus count")
            for bus_index, bus in enumerate(psys.buses):
                bus.set_vinit(
                    float(v_magnitudes[bus_index]),
                    float(v_angles[bus_index]),
                )
        summary = {
            "num_loads": int(p_load.size),
            "num_generators": int(p_gen.size),
            "voltage_initialization_applied": bool(
                v_magnitudes is not None and v_angles is not None
            ),
        }
    psys.createYbusComplex()
    return psys, summary


def _prepare_context(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    common = {
        "config_path": args.config,
        "output_dir": args.output_dir,
        "sample_idx": int(args.sample_idx),
        "accepted_operating_point_index": 0,
        "target_accepted_scenarios": 1,
        "max_total_attempts": int(args.max_total_attempts),
        "remaining_attempts": int(args.max_total_attempts),
        "uqgrid_root": args.uqgrid_root,
        "pf_verbose": bool(args.pf_verbose),
    }
    if args.source == "acopf":
        acopf_config = acopf_init.resolve_acopf_initialization_config(
            args,
            config,
            require_paths=True,
        )
        return dict(
            acopf_init.prepare_acopf_candidate_context(
                **common,
                acopf_config=acopf_config,
                case_root="handoff_validation_cases",
            )
        )
    return dict(acopf_init.prepare_uqgrid_candidate_context(**common))


def _power_flow_contract_failures(diagnostics: Any) -> list[str]:
    if not isinstance(diagnostics, Mapping):
        return ["missing_power_flow_validation"]
    failures = []
    if not diagnostics.get("valid", False):
        failures.append("power_flow_validation_invalid")
    residual = diagnostics.get("residual_norm")
    residual_tolerance = diagnostics.get("residual_tolerance")
    if (
        residual is None
        or residual_tolerance is None
        or not np.isfinite(residual)
        or float(residual) > float(residual_tolerance)
    ):
        failures.append("pf_residual")
    if not diagnostics.get("finite_voltage", False):
        failures.append("nonfinite_voltage")
    if (diagnostics.get("gen_p") or {}).get("violation_count", 0):
        failures.append("generator_p_limit")
    if (diagnostics.get("gen_q") or {}).get("violation_count", 0):
        failures.append("generator_q_limit")
    voltage_min = diagnostics.get("voltage_min")
    voltage_max = diagnostics.get("voltage_max")
    voltage_lower_bound = diagnostics.get("voltage_lower_bound")
    voltage_upper_bound = diagnostics.get("voltage_upper_bound")
    if (
        voltage_lower_bound is not None
        and (voltage_min is None or float(voltage_min) < float(voltage_lower_bound))
    ):
        failures.append("voltage_low")
    if (
        voltage_upper_bound is not None
        and (voltage_max is None or float(voltage_max) > float(voltage_upper_bound))
    ):
        failures.append("voltage_high")
    branch = diagnostics.get("branch") or {}
    loading_max = branch.get("loading_max")
    loading_limit = branch.get("loading_limit")
    loading_tolerance = float(branch.get("limit_tolerance", 0.0))
    if (
        loading_limit is not None
        and loading_max is not None
        and float(loading_max) > float(loading_limit) + loading_tolerance
    ):
        failures.append("branch_overload")
    if (diagnostics.get("active_set") or {}).get("violation_count", 0):
        failures.append("active_set_inconsistent")
    if (diagnostics.get("island_slack") or {}).get("invalid_island_count", 0):
        failures.append("invalid_slack_topology")
    return failures


def validate_operating_point(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full source-to-initialized-DAE validation and return its report."""
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).expanduser()
    report_path = Path(args.report_path) if args.report_path else (
        output_dir
        / f"handoff_validation_{args.source}_{'petsc' if args.petsc else 'beuler'}.json"
    )
    checks: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "mode": "operating_point",
        "source": args.source,
        "sample_idx": int(args.sample_idx),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "git_commit": _git_commit(),
        "environment": _environment_info(),
        "checks": checks,
    }
    try:
        config = acopf_init._load_json_config(config_path)
        integration_config = build_validation_integration_config(
            config,
            steps=int(args.steps),
            petsc=args.petsc,
        )
        report["solver"] = "petsc" if integration_config.petsc else "beuler"
        context = _prepare_context(args, config)
        _check(checks, "PF-screened source context accepted", context.get("accepted"), {
            "reject_reason": context.get("reject_reason"),
            "sample_idx": context.get("sample_idx"),
        })
        if not context.get("accepted"):
            raise RuntimeError(
                f"Source context was rejected: {context.get('reject_reason', 'unknown')}"
            )

        from uqgrid.simulation.dynamics import (
            _initialize_system_from_config,
            integrate_system,
        )
        from uqgrid.simulation.residual import residual_function

        psys, application = apply_initialization_context(context, args.source)
        psys.power_injection = bool(integration_config.power_injection)
        pf_solution, initial_state, theta = _initialize_system_from_config(
            psys,
            integration_config,
        )
        pf_diagnostics = pf_solution.validation
        pf_failures = _power_flow_contract_failures(pf_diagnostics)
        _check(checks, "Final PF contract", not pf_failures, pf_failures)

        residual = np.zeros_like(initial_state)
        residual_function(residual, initial_state, theta, psys)
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        _check(
            checks,
            "Initialized DAE residual",
            np.isfinite(residual_norm) and residual_norm <= args.residual_tolerance,
            {"norm_inf": residual_norm, "tolerance": args.residual_tolerance},
        )

        efd = collect_exciter_limit_diagnostics(
            psys,
            initial_state,
            tolerance=float(args.efd_limit_tolerance),
        )
        _check(
            checks,
            "Initialized SEXS Efd limits",
            efd["violation_count"] == 0,
            {
                "applicable": efd["applicable"],
                "count": efd["count"],
                "violation_count": efd["violation_count"],
                "efd_min": efd["efd_min"],
                "efd_max": efd["efd_max"],
            },
        )

        replay_psys, replay_application = apply_initialization_context(context, args.source)
        replay_psys.add_busfault(0, float(args.dormant_fault_impedance))
        replay_psys.createYbusComplex()
        simulation = integrate_system(replay_psys, integration_config)
        replay_pf_diagnostics = simulation.get("power_flow_diagnostics")
        replay_pf_failures = _power_flow_contract_failures(replay_pf_diagnostics)
        _check(checks, "Replay final PF contract", not replay_pf_failures, replay_pf_failures)
        history = simulation.get("history")
        if history is None:
            raise RuntimeError("Undisturbed replay returned no history")
        drift = compute_trajectory_drift(history, replay_psys)
        _check(
            checks,
            "Undisturbed trajectory drift",
            drift["total"] <= args.trajectory_tolerance,
            {**drift, "tolerance": args.trajectory_tolerance},
        )

        report.update(
            {
                "operating_point_id": context.get("operating_point_id"),
                "prepared_sample_idx": context.get("sample_idx"),
                "application": application,
                "replay_application": replay_application,
                "power_flow_validation": pf_diagnostics,
                "replay_power_flow_validation": replay_pf_diagnostics,
                "initial_dae_residual_norm_inf": residual_norm,
                "exciter_limits": efd,
                "trajectory_drift": drift,
                "integration": {
                    "dt": float(integration_config.dt),
                    "steps": int(integration_config.steps),
                    "tend": float(integration_config.tend),
                    "ton": float(integration_config.ton),
                    "toff": float(integration_config.toff),
                    "petsc": bool(integration_config.petsc),
                    "method": str(integration_config.method),
                },
            }
        )
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        diagnostics = getattr(exc, "diagnostics", None)
        if isinstance(diagnostics, Mapping):
            report["power_flow_validation"] = diagnostics
            _check(
                checks,
                "Final PF contract",
                False,
                _power_flow_contract_failures(diagnostics),
            )
        _check(checks, "Operating-point validation completed", False, str(exc))

    report["valid"] = bool(checks) and all(check["passed"] for check in checks)
    report["report_path"] = str(report_path)
    _write_json(report_path, report)
    return report


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _meta_value(data: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
    value = np.asarray(data["meta"], dtype=object).reshape(-1)[0]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if not isinstance(value, Mapping):
        raise ValueError("NPZ meta must contain one mapping")
    return value


def _dataset_pf_failures(simulation_log: Mapping[str, Any]) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {}
    for scenario_id, entry in simulation_log.items():
        reasons = _power_flow_contract_failures(entry.get("power_flow_validation"))
        if reasons:
            failures[str(scenario_id)] = reasons
    return failures


def validate_dataset(
    output_dir: str | Path,
    probml_basename: str,
    *,
    expected_sources: Sequence[str] | None = None,
    expected_fault_count: int | None = None,
    tsi_tolerance: float = 1e-12,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a final/min ProbML pair and the matching restart state."""
    output_dir = Path(output_dir).expanduser()
    final_path, min_path = acopf_init._probml_output_paths(output_dir, probml_basename)
    report_path = Path(report_path) if report_path else output_dir / "stage5_dataset_validation.json"
    checks: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "mode": "dataset",
        "output_dir": str(output_dir.resolve()),
        "probml_basename": probml_basename,
        "final_path": str(final_path),
        "min_path": str(min_path),
        "git_commit": _git_commit(),
        "checks": checks,
    }
    try:
        _check(checks, "Final NPZ exists", final_path.is_file(), str(final_path))
        _check(checks, "Minimum NPZ exists", min_path.is_file(), str(min_path))
        if not final_path.is_file() or not min_path.is_file():
            raise FileNotFoundError("Both final and minimum NPZ files are required")

        final = _load_npz(final_path)
        minimum = _load_npz(min_path)
        missing_final = sorted(REQUIRED_NPZ_KEYS.difference(final))
        missing_min = sorted(REQUIRED_NPZ_KEYS.difference(minimum))
        _check(checks, "Required NPZ keys", not missing_final and not missing_min, {
            "missing_final": missing_final,
            "missing_min": missing_min,
        })
        if missing_final or missing_min:
            raise ValueError("Required NPZ keys are missing")

        pair_keys = [
            "X",
            "X_flat",
            "sample_idx",
            "fault_locations",
            "fault_impedances",
            "scenario_ids",
        ]
        pair_mismatches = [
            key for key in pair_keys if not np.array_equal(final[key], minimum[key])
        ]
        _check(checks, "Final/min shared arrays", not pair_mismatches, pair_mismatches)

        final_labels = OPTIONAL_LABEL_KEYS.intersection(final)
        min_labels = OPTIONAL_LABEL_KEYS.intersection(minimum)
        label_pair_valid = (
            final_labels in (set(), OPTIONAL_LABEL_KEYS)
            and final_labels == min_labels
            and all(np.array_equal(final[key], minimum[key]) for key in final_labels)
        )
        _check(checks, "Final/min optional labels", label_pair_valid, {
            "final": sorted(final_labels),
            "minimum": sorted(min_labels),
        })

        X = np.asarray(final["X"])
        X_flat = np.asarray(final["X_flat"])
        Y_final = np.asarray(final["Y"], dtype=float)
        Y_min = np.asarray(minimum["Y"], dtype=float)
        sample_idx = np.asarray(final["sample_idx"], dtype=np.int64)
        fault_locations = np.asarray(final["fault_locations"])
        fault_impedances = np.asarray(final["fault_impedances"])
        scenario_ids = np.asarray(final["scenario_ids"], dtype=object)
        expected_y_shape = (X.shape[0], fault_locations.size, fault_impedances.size)
        shape_valid = bool(
            X.ndim == 3
            and X.shape[1] == 2
            and X_flat.shape == (X.shape[0], 2 * X.shape[2])
            and Y_final.shape == expected_y_shape
            and Y_min.shape == expected_y_shape
            and sample_idx.shape == (X.shape[0],)
            and scenario_ids.shape == expected_y_shape
        )
        _check(checks, "ProbML shapes", shape_valid, {
            "X": X.shape,
            "X_flat": X_flat.shape,
            "Y_final": Y_final.shape,
            "Y_min": Y_min.shape,
        })
        _check(
            checks,
            "X_flat layout",
            np.array_equal(X_flat, X.reshape(X.shape[0], -1)),
        )
        tsi_valid = bool(
            np.all(np.isfinite(Y_final))
            and np.all(np.isfinite(Y_min))
            and np.all(Y_min <= Y_final + float(tsi_tolerance))
        )
        _check(checks, "TSI final/min relationship", tsi_valid)
        _check(
            checks,
            "Unique scenario IDs",
            np.unique(scenario_ids.astype(str)).size == scenario_ids.size,
        )

        if expected_fault_count is not None:
            _check(
                checks,
                "Expected fault-location count",
                fault_locations.size == int(expected_fault_count),
                {"expected": int(expected_fault_count), "actual": fault_locations.size},
            )

        if "initialization_source" in final and "acopf_feasible" in final:
            sources = np.asarray(final["initialization_source"], dtype=object).astype(str)
            feasible = np.asarray(final["acopf_feasible"], dtype=bool)
            labels_valid = sources.shape == (X.shape[0],) and feasible.shape == (X.shape[0],)
            labels_valid = labels_valid and np.array_equal(feasible, sources == "acopf")
            _check(checks, "Initialization labels", labels_valid, {
                "initialization_source": sources.tolist(),
                "acopf_feasible": feasible.tolist(),
            })
        else:
            sources = np.full(X.shape[0], "acopf", dtype=str)
            feasible = np.ones(X.shape[0], dtype=bool)
            _check(
                checks,
                "Legacy initialization labels",
                expected_sources is None,
                "labels absent; interpreted as ACOPF/feasible for backward compatibility",
            )
        if expected_sources is not None:
            expected = [str(source) for source in expected_sources]
            _check(
                checks,
                "Expected initialization sources",
                sources.tolist() == expected,
                {"expected": expected, "actual": sources.tolist()},
            )

        final_meta = _meta_value(final)
        min_meta = _meta_value(minimum)
        _check(checks, "TSI metadata modes", (
            final_meta.get("tsi_mode") == "final" and min_meta.get("tsi_mode") == "min"
        ), {"final": final_meta.get("tsi_mode"), "minimum": min_meta.get("tsi_mode")})

        progress_path = output_dir / "acopf_init_progress.json"
        metadata_path = output_dir / "scenario_metadata.json"
        simulation_log_path = output_dir / "simulation_log.json"
        state_metadata_path = output_dir / "state_metadata.json"
        state_paths = [progress_path, metadata_path, simulation_log_path, state_metadata_path]
        _check(
            checks,
            "Restart state files",
            all(path.is_file() for path in state_paths),
            [str(path) for path in state_paths if not path.is_file()],
        )
        if not all(path.is_file() for path in state_paths):
            raise FileNotFoundError("One or more restart state files are missing")

        progress = _load_json(progress_path)
        scenario_metadata = _load_json(metadata_path)
        simulation_log = _load_json(simulation_log_path)
        _load_json(state_metadata_path)
        expected_fault_rows = int(X.shape[0] * fault_locations.size * fault_impedances.size)
        state_counts_valid = bool(
            int(progress.get("accepted_count", -1)) == X.shape[0]
            and len(scenario_metadata) == expected_fault_rows
            and len(simulation_log) == expected_fault_rows
        )
        _check(checks, "NPZ/restart row counts", state_counts_valid, {
            "npz_rows": X.shape[0],
            "scenario_metadata_rows": len(scenario_metadata),
            "simulation_log_rows": len(simulation_log),
            "expected_fault_rows": expected_fault_rows,
        })
        expected_next_sample_idx = int(np.max(sample_idx) + 1) if sample_idx.size else 0
        _check(
            checks,
            "Restart next sample index",
            int(progress.get("next_sample_idx", -1)) == expected_next_sample_idx,
            {
                "expected": expected_next_sample_idx,
                "actual": progress.get("next_sample_idx"),
            },
        )
        log_ids = set(str(key) for key in simulation_log)
        metadata_ids = set(str(key) for key in scenario_metadata)
        npz_ids = set(scenario_ids.astype(str).reshape(-1).tolist())
        _check(
            checks,
            "Scenario IDs agree with logs",
            log_ids == metadata_ids == npz_ids,
            {
                "npz": len(npz_ids),
                "scenario_metadata": len(metadata_ids),
                "simulation_log": len(log_ids),
            },
        )
        metadata_labels_valid = all(
            scenario_metadata[scenario_id].get("initialization_source")
            == simulation_log[scenario_id].get("initialization_source")
            and bool(scenario_metadata[scenario_id].get("acopf_feasible"))
            == bool(simulation_log[scenario_id].get("acopf_feasible"))
            for scenario_id in log_ids.intersection(metadata_ids)
        )
        _check(
            checks,
            "Scenario metadata source labels",
            metadata_labels_valid and log_ids == metadata_ids,
        )
        pf_failures = _dataset_pf_failures(simulation_log)
        _check(checks, "Per-fault final PF diagnostics", not pf_failures, pf_failures)

        source_counts: dict[str, int] = {}
        for source in sources:
            source_counts[str(source)] = source_counts.get(str(source), 0) + 1
        progress_counts = {
            str(key): int(value)
            for key, value in (progress.get("initialization_source_counts") or {}).items()
        }
        _check(
            checks,
            "Progress source counts",
            progress_counts == source_counts,
            {"expected": source_counts, "actual": progress_counts},
        )
        report.update(
            {
                "shapes": {
                    "X": X.shape,
                    "X_flat": X_flat.shape,
                    "Y_final": Y_final.shape,
                    "Y_min": Y_min.shape,
                },
                "sample_idx": sample_idx,
                "initialization_source": sources,
                "acopf_feasible": feasible,
                "source_counts": source_counts,
                "fault_rows": expected_fault_rows,
                "sha256": {
                    "final": _sha256(final_path),
                    "minimum": _sha256(min_path),
                },
            }
        )
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        _check(checks, "Dataset validation completed", False, str(exc))

    report["valid"] = bool(checks) and all(check["passed"] for check in checks)
    report["report_path"] = str(report_path)
    _write_json(report_path, report)
    return report


def _parse_sources(value: str | None) -> list[str] | None:
    if value is None:
        return None
    sources = [part.strip() for part in value.split(",") if part.strip()]
    invalid = [source for source in sources if source not in {"acopf", "uqgrid_pf"}]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid initialization source(s): {', '.join(invalid)}"
        )
    return sources


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate ACOPF/direct-UQGrid PF-to-replay handoffs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    operating = subparsers.add_parser(
        "operating-point",
        help="Validate one initialized operating point and undisturbed replay",
    )
    operating.add_argument("config", help="Strict Stage 5 scenario config")
    operating.add_argument("--source", choices=["acopf", "uqgrid_pf"], required=True)
    operating.add_argument("--output-dir", required=True)
    operating.add_argument("--sample-idx", type=int, default=0)
    operating.add_argument("--max-total-attempts", type=int, default=500)
    operating.add_argument("--uqgrid-root", default=None)
    operating.add_argument("--pf-verbose", action="store_true")
    operating.add_argument("--julia", default=None)
    operating.add_argument("--exajugo-root", default=None)
    operating.add_argument("--exajugo-base-raw", default=None)
    operating.add_argument("--exajugo-base-rop", default=None)
    operating.add_argument("--acopf-timeout-s", type=float, default=None)
    operating.add_argument(
        "--petsc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override integration.petsc for the undisturbed replay",
    )
    operating.add_argument("--steps", type=int, default=5)
    operating.add_argument("--residual-tolerance", type=float, default=1e-8)
    operating.add_argument("--trajectory-tolerance", type=float, default=1e-8)
    operating.add_argument("--efd-limit-tolerance", type=float, default=1e-8)
    operating.add_argument("--dormant-fault-impedance", type=float, default=1e-4)
    operating.add_argument("--report-path", default=None)

    dataset = subparsers.add_parser(
        "dataset",
        help="Validate final/min NPZ files and matching restart state",
    )
    dataset.add_argument("--output-dir", required=True)
    dataset.add_argument("--probml-basename", required=True)
    dataset.add_argument(
        "--expected-sources",
        default=None,
        help="Comma-separated ordered row labels",
    )
    dataset.add_argument("--expected-fault-count", type=int, default=None)
    dataset.add_argument("--tsi-tolerance", type=float, default=1e-12)
    dataset.add_argument("--report-path", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "operating-point":
        report = validate_operating_point(args)
    else:
        try:
            expected_sources = _parse_sources(args.expected_sources)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        report = validate_dataset(
            args.output_dir,
            args.probml_basename,
            expected_sources=expected_sources,
            expected_fault_count=args.expected_fault_count,
            tsi_tolerance=args.tsi_tolerance,
            report_path=args.report_path,
        )
    _emit_checks(report)
    print(json.dumps(_json_safe(report), sort_keys=True))
    return 0 if report.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
