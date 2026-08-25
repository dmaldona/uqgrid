"""Local end-to-end acceptance checks for the ACTIVSg dynamic cases."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from scripts.validation.dyr_coverage import (
    ACTIVSG_TARGET_NATIVE_MODELS,
    ACTIVSG_TARGET_REDIRECTS,
    analyze_dyr_coverage,
)
from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.simulation.config import IntegrationConfig
from uqgrid.simulation.dynamics import initialize_system, integrate_system, preallocate_jacobian
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.jacobian_check import compare_jacobian_columns
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


_CLASS_TO_SOURCE = {
    "GenGENROU": "GENROU",
    "GenGENSAL": "GENSAL",
    "GovGAST": "GAST",
    "GovHYGOV": "HYGOV",
    "GovIEEEG1": "IEEEG1",
    "GovIEESGO": "IEESGO",
    "GovTGOV1": "TGOV1",
    "ExcESAC1A": "ESAC1A",
    "ExcESDC1A": "ESDC1A",
    "ExcESDC2A": "ESDC2A",
    "ExcESST4B": "ESST4B",
    "ExcEXAC1": "EXAC1",
    "ExcEXAC2": "EXAC2",
    "ExcIEEET1": "IEEET1",
    "ExcSEXS": "SEXS",
    "PssIEEEST": "IEEEST",
}

TARGET_EXPECTATIONS = {
    "ACTIVSg200": {
        "counts": {"active": 114, "inactive": 33, "native": 114, "redirected": 0},
        "models": {"GENROU": 38, "SEXS": 38, "TGOV1": 38},
        "machine_less": 0,
        "fault_bus": 1,
    },
    "ACTIVSg500": {
        "counts": {"active": 170, "inactive": 102, "native": 170, "redirected": 0},
        "models": {
            "GAST": 6, "GENROU": 56, "HYGOV": 35, "IEEEST": 2,
            "SEXS": 56, "TGOV1": 15,
        },
        "machine_less": 0,
        "fault_bus": 1,
    },
    "ACTIVSg2000": {
        "counts": {"active": 1335, "inactive": 404, "native": 989, "redirected": 346},
        "models": {
            "ESAC1A": 2, "ESAC6A": 2, "ESDC1A": 10, "ESDC2A": 1,
            "ESST4B": 212, "EXAC1": 4, "EXAC2": 31, "EXPIC1": 52,
            "GENROU": 314, "GENSAL": 20, "GGOV1": 288, "HYGOV": 20,
            "IEEEG1": 26, "IEEEST": 333, "IEEET1": 16, "SCRX": 4,
        },
        "machine_less": 98,
        "fault_bus": 1001,
    },
}


def _external_bus_map(psys):
    return {internal: external for external, internal in psys.ext2int.items()}


def runtime_attachment_counts(psys):
    """Return exact source-model attachment identities from production objects."""
    int2ext = _external_bus_map(psys)
    attachments = Counter()
    for device in (*psys.gendyn, *psys.gov, *psys.exc, *psys.pss):
        source = getattr(device, "source_model", None)
        if source is None:
            source = _CLASS_TO_SOURCE[type(device).__name__]
        key = (str(source).upper(), int(int2ext[device.bus]), str(device.id_tag).strip())
        attachments[key] += 1
    return attachments


def reconcile_runtime_attachments(report, psys):
    """Require one runtime object for every active accepted DYR record."""
    expected = Counter(
        (record.source_model, record.bus, record.device_id)
        for record in report.records
        if record.status in {"native", "redirected"}
    )
    actual = runtime_attachment_counts(psys)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise AssertionError(
            f"Runtime DYR attachments differ: missing={dict(missing)}, "
            f"unexpected={dict(unexpected)}"
        )
    return {
        "attached": sum(actual.values()),
        "by_source_model": dict(sorted(Counter(key[0] for key in actual).items())),
        "redirects": dict(sorted(Counter(
            f"{item['source_model']}->{item['effective_model']}"
            for item in psys.dynamic_model_redirects
        ).items())),
    }


def _residual_by_type(psys, residual):
    values = {}
    for device in psys.devices:
        rows = list(range(device.dif_ptr, device.dif_ptr + device.dif_dim))
        rows.extend(
            range(
                psys.num_dof_dif + device.alg_ptr,
                psys.num_dof_dif + device.alg_ptr + device.alg_dim,
            )
        )
        if rows:
            name = type(device).__name__
            values[name] = max(values.get(name, 0.0), float(np.max(np.abs(residual[rows]))))
    return dict(sorted(values.items()))


def sampled_device_columns(psys):
    """Select every local column of one representative per device class."""
    columns = []
    seen = set()
    for device in psys.devices:
        name = type(device).__name__
        if name in seen:
            continue
        seen.add(name)
        columns.extend(range(device.dif_ptr, device.dif_ptr + device.dif_dim))
        columns.extend(
            range(
                psys.num_dof_dif + device.alg_ptr,
                psys.num_dof_dif + device.alg_ptr + device.alg_dim,
            )
        )
    columns.extend(
        range(
            psys.num_dof_dif + psys.num_dof_alg,
            psys.num_dof_dif + psys.num_dof_alg + min(2 * psys.nbuses, 4),
        )
    )
    return columns


def _sample_jacobian(psys, state, theta):
    jacobian = preallocate_jacobian(psys)
    residual_jacobian(jacobian, state, theta, psys)
    samples = compare_jacobian_columns(
        psys, state, theta, jacobian, sampled_device_columns(psys), eps=1e-7
    )
    return {
        "finite": bool(np.all(np.isfinite(jacobian.data))),
        "sample_count": len(samples),
        "maximum_absolute_error": max(
            (item["maximum_absolute_error"] for item in samples), default=0.0
        ),
        "samples": samples,
    }


def _require_target_inventory(case, report):
    try:
        expected = TARGET_EXPECTATIONS[case]
    except KeyError as exc:
        raise ValueError(f"No acceptance inventory is defined for {case!r}.") from exc
    for name, value in expected["counts"].items():
        if report.counts[name] != value:
            raise AssertionError(
                f"{case} coverage {name}={report.counts[name]}, expected {value}."
            )
    actual_models = {
        name: item.active for name, item in report.by_source_model.items() if item.active
    }
    if actual_models != expected["models"]:
        raise AssertionError(
            f"{case} active model inventory differs: {actual_models}."
        )
    if len(report.active_generators_without_machine) != expected["machine_less"]:
        raise AssertionError(f"{case} machine-less generator count differs.")
    return expected


def run_local_acceptance(
    raw_path, dyr_path, *, enforce_q_limits=True, fault_bus=None,
    tend=0.05, dt=1.0 / 120.0, residual_tolerance=1e-8,
    jacobian_tolerance=1e-4, flat_tolerance=1e-7,
    trajectory_tolerance=1e-6,
):
    """Run local coverage, initialization, Jacobian, and trajectory gates."""
    raw_path = Path(raw_path)
    dyr_path = Path(dyr_path)
    case = raw_path.stem
    coverage = analyze_dyr_coverage(
        raw_path,
        dyr_path,
        native_models=ACTIVSG_TARGET_NATIVE_MODELS,
        redirects=ACTIVSG_TARGET_REDIRECTS,
        strict=True,
    )
    expected = _require_target_inventory(case, coverage)
    if fault_bus is None:
        fault_bus = expected["fault_bus"]
    psys = load_psse(str(raw_path))
    add_dyr(psys, str(dyr_path))
    attachments = reconcile_runtime_attachments(coverage, psys)
    psys.createYbusComplex()
    solution = runpf(psys, verbose=False, enforce_q_limits=enforce_q_limits)
    state, theta = initialize_system(psys, solution)
    residual = np.zeros_like(state)
    residual_function(residual, state, theta, psys)
    jacobian_result = _sample_jacobian(psys, state, theta)

    config = IntegrationConfig(
        tend=tend, dt=dt, ton=tend / 3.0, toff=2.0 * tend / 3.0,
        enforce_q_limits=enforce_q_limits,
    )
    no_fault = integrate_system(psys, config)
    fault_system = load_psse(str(raw_path))
    add_dyr(fault_system, str(dyr_path))
    fault_system.createYbusComplex()
    fault_system.add_busfault(fault_system.ext2int[int(fault_bus)], 1.0)
    fault_result = integrate_system(fault_system, config)

    fault_theta = np.zeros(fault_system.num_pars)
    for device in fault_system.devices:
        device.initialize_theta(fault_theta)
    event_jacobians = {}
    fault = fault_system.fault_events[0]
    for name, event_time, active in (
        ("fault_on", config.ton, True),
        ("post_clear", config.toff, False),
    ):
        if active:
            fault.apply()
        else:
            fault.remove()
        index = int(np.flatnonzero(np.isclose(fault_result["tvec"], event_time))[0])
        event_jacobians[name] = _sample_jacobian(
            fault_system, fault_result["history"][:, index], fault_theta
        )
    fault.remove()

    numerical_system = load_psse(str(raw_path))
    add_dyr(numerical_system, str(dyr_path))
    numerical_system.createYbusComplex()
    numerical_system.add_busfault(numerical_system.ext2int[int(fault_bus)], 1.0)
    numerical_config = config.model_copy(
        update={"jacobian_mode": "finite_difference"}
    )
    numerical = integrate_system(numerical_system, numerical_config)
    numerical_comparison = {
        "maximum_absolute_error": float(
            np.max(np.abs(numerical["history"] - fault_result["history"]))
        ),
        "final_time": float(numerical["tvec"][-1]),
        "finite": bool(np.all(np.isfinite(numerical["history"]))),
        "color_count": len(
            getattr(numerical_system, "_finite_difference_jacobian_coloring", ((), ()))[1]
        ),
    }

    def trajectory_summary(result):
        history = result["history"]
        return {
            "final_time": float(result["tvec"][-1]),
            "finite": bool(np.all(np.isfinite(history))),
            "maximum_change": float(np.max(np.abs(history - history[:, [0]]))),
            "limit_event_count": len(result["dynamic_limit_diagnostics"]["events"]),
        }

    result = {
        "coverage": coverage.summary_dict(),
        "attachments": attachments,
        "initialization": {
            "state_size": int(state.size),
            "finite": bool(np.all(np.isfinite(state))),
            "residual_infinity_norm": float(np.linalg.norm(residual, np.inf)),
            "residual_by_device_type": _residual_by_type(psys, residual),
        },
        "jacobian": jacobian_result,
        "event_jacobians": event_jacobians,
        "no_fault": trajectory_summary(no_fault),
        "fault": None if fault_result is None else trajectory_summary(fault_result),
        "numerical_jacobian_trajectory": numerical_comparison,
    }
    if not result["initialization"]["finite"] or not np.isfinite(
        result["initialization"]["residual_infinity_norm"]
    ):
        raise AssertionError("Initialization contains non-finite values.")
    if result["initialization"]["residual_infinity_norm"] > residual_tolerance:
        raise AssertionError("Initial residual exceeds acceptance tolerance.")
    if not result["jacobian"]["finite"] or not np.isfinite(
        result["jacobian"]["maximum_absolute_error"]
    ):
        raise AssertionError("Jacobian validation contains non-finite values.")
    for name, event_jacobian in event_jacobians.items():
        if not event_jacobian["finite"] or not np.isfinite(
            event_jacobian["maximum_absolute_error"]
        ):
            raise AssertionError(f"{name} Jacobian contains non-finite values.")
        if event_jacobian["maximum_absolute_error"] > jacobian_tolerance:
            raise AssertionError(f"{name} Jacobian error exceeds acceptance tolerance.")
    if result["jacobian"]["maximum_absolute_error"] > jacobian_tolerance:
        raise AssertionError("Sampled Jacobian error exceeds acceptance tolerance.")
    if result["no_fault"]["maximum_change"] > flat_tolerance:
        raise AssertionError("No-fault trajectory is not flat within tolerance.")
    for name in ("no_fault", "fault"):
        trajectory = result[name]
        if trajectory is None:
            continue
        if not trajectory["finite"]:
            raise AssertionError(f"{name} trajectory contains non-finite values.")
        if abs(trajectory["final_time"] - tend) > 1e-11:
            raise AssertionError(f"{name} trajectory did not reach the requested horizon.")
    if result["fault"]["maximum_change"] <= flat_tolerance:
        raise AssertionError("Fault did not produce a nontrivial trajectory.")
    if not numerical_comparison["finite"]:
        raise AssertionError("Numerical-Jacobian trajectory contains non-finite values.")
    if numerical_comparison["maximum_absolute_error"] > trajectory_tolerance:
        raise AssertionError(
            "Numerical and analytical Jacobian trajectories differ beyond tolerance."
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("ACTIVSg200", "ACTIVSg500", "ACTIVSg2000"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--fault-bus", type=int)
    parser.add_argument("--tend", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--disable-q-limits", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_local_acceptance(
        args.data_dir / f"{args.case}.raw",
        args.data_dir / f"{args.case}.dyr",
        enforce_q_limits=not args.disable_q_limits,
        fault_bus=args.fault_bus,
        tend=args.tend,
        dt=args.dt,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
