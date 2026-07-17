#!/usr/bin/env python
r"""
Power Grid Scenario Generation with Perturbation and Simulation.

This module generates perturbed operating scenarios for power grid transient
stability simulations. It applies stochastic perturbations to load and
generation setpoints, runs dynamic simulations for various fault conditions,
and saves the results for subsequent analysis.

Overview
--------
The scenario generation pipeline consists of:

1. **Perturbation**: Apply multiplicative noise to base load/generation values
2. **Clamping**: Enforce generator limits on active and reactive power
3. **Rebalancing**: Adjust total generation to match total load
4. **Simulation**: Run transient stability analysis for each fault scenario
5. **Storage**: Save time-series results and metadata for analysis

Mathematical Framework
----------------------

Multiplicative Perturbation Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Given a base power value $P_{base}$, the perturbed value is computed as:

    $P_{scaled} = P_{base} \cdot (1 + \epsilon)$

where $\epsilon$ is a zero-mean random variable drawn from the specified
distribution.

Noise Distributions
~~~~~~~~~~~~~~~~~~~
Two noise distributions are supported:

**Normal Distribution**:
    $\epsilon \sim \mathcal{N}(0, \sigma^2)$

    where $\sigma$ is the ``var`` parameter (interpreted as standard deviation).

**Uniform Distribution**:
    $\epsilon \sim \mathcal{U}(-a, a)$

    where $a = \sqrt{3 \cdot \sigma^2}$ is chosen such that
    $\text{Var}(\epsilon) = \sigma^2$.

    This follows from $\text{Var}(\mathcal{U}(-a,a)) = \frac{a^2}{3}$.

Power Factor Preservation
~~~~~~~~~~~~~~~~~~~~~~~~~
When ``preserve_power_factor=True``, reactive power is adjusted to maintain
the original power factor. For a bus with base values $(P_{base}, Q_{base})$:

    $\text{pf} = \frac{Q_{base}}{P_{base}}$ (power factor ratio)

    $Q_{scaled} = \text{pf} \cdot P_{scaled}$

This ensures that $\frac{Q_{scaled}}{P_{scaled}} = \frac{Q_{base}}{P_{base}}$.

Special cases:
- If $P_{base} = 0$ and $Q_{base} \neq 0$ (purely reactive load), the same
  multiplicative noise is applied: $Q_{scaled} = Q_{base} \cdot (1 + \epsilon)$
- If both $P_{base} = 0$ and $Q_{base} = 0$, no perturbation is applied.

Generator Clamping
~~~~~~~~~~~~~~~~~~
When ``clamp_gens=True``, generator outputs are constrained to their limits:

    $P_g^{clamped} = \text{clip}(P_g, P_g^{min}, P_g^{max})$

    $Q_g^{clamped} = \text{clip}(Q_g, Q_g^{min}, Q_g^{max})$

where $\text{clip}(x, a, b) = \max(a, \min(x, b))$.

Active Power Rebalancing
~~~~~~~~~~~~~~~~~~~~~~~~
When ``balance_generation=True``, total generation is adjusted to match total
load while respecting generator limits. Given:

- Current generation: $P_g = [P_{g,1}, ..., P_{g,n}]$
- Generator limits: $P_g^{min}, P_g^{max}$
- Target total: $P_{target} = \sum_i P_{L,i}$ (total load)
- Mismatch: $\Delta = P_{target} - \sum_i P_{g,i}$

**If $\Delta > 0$ (need to increase generation)**:

    For each generator with headroom $h_i = P_{g,i}^{max} - P_{g,i} > 0$:

    $P_{g,i}^{new} = P_{g,i} + h_i \cdot \min\left(1, \frac{\Delta}{\sum_j h_j}\right)$

**If $\Delta < 0$ (need to decrease generation)**:

    For each generator with downward margin $d_i = P_{g,i} - P_{g,i}^{min} > 0$:

    $P_{g,i}^{new} = P_{g,i} - d_i \cdot \min\left(1, \frac{|\Delta|}{\sum_j d_j}\right)$

This distributes the adjustment proportionally to available headroom/margin.

Features
--------
- Multiplicative perturbations with configurable noise distributions
- Independent control over load and generator perturbations
- Power factor preservation for realistic P/Q patterns
- Generator limit enforcement (clamping)
- Generation-load balance maintenance
- Parallel execution with joblib
- Checkpointing for long-running simulations
- Automatic recovery from failed scenarios

Environment Variables
---------------------
The module sets thread count environment variables to prevent oversubscription
when running parallel simulations:

- OMP_NUM_THREADS=1
- MKL_NUM_THREADS=1
- OPENBLAS_NUM_THREADS=1
- NUMEXPR_NUM_THREADS=1

Usage
-----
Command-line execution with a configuration file::

    $ python generate_scenarios.py config/config_IEEE-9.json

Generate default configuration files for all models::

    $ python generate_scenarios.py --generate-configs

Run with default IEEE-9 configuration::

    $ python generate_scenarios.py

Continue from existing simulation, adding more samples::

    $ python generate_scenarios.py config.json --continue --additional-samples 10

This will load existing scenario_metadata.json and simulation_log.json,
then generate 10 additional samples per fault location, starting from
sample_idx = max_existing + 1. Results are merged with existing data.

Programmatic usage::

    from gs import (
        sample_scenarios,
        generate_metadata,
        run_simulation_driver_batched
    )

    # Define scenario space
    scenarios = sample_scenarios(
        n_samples=100,
        fault_locations=[0, 1, 2, 3],
        fault_impedances=[0.00001, 0.0001]
    )
    metadata = generate_metadata(scenarios)

    # Run simulations
    log = run_simulation_driver_batched(
        raw="model.raw",
        dyr="model.dyr",
        scenarios_metadata=metadata,
        noise_type="normal",
        noise_var=0.1,
        balance_generation=True
    )

Output Files
------------
- simulation_data/scenario_*.npz : Per-scenario simulation results
- simulation_log.json : Simulation outcomes and metadata
- scenario_metadata.json : Scenario parameter definitions
- simulation_checkpoint.json : Checkpoint for recovery (removed on completion)

Dependencies
------------
- numpy : Numerical operations
- joblib : Parallel execution
- uqgrid : Power system simulation library

See Also
--------
- TSI_analysis.py : Analyzes simulation results to compute stability indices
- recovery_tool.py : Recovery utilities for failed simulations
- monitor.py : Real-time simulation progress monitoring

"""

import os

# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------
# Disable multi-threading in numerical libraries to prevent thread contention
# when running multiple simulation processes in parallel.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import itertools
import numpy as np
import uuid
import json
import gc
import copy
import time
import traceback
from collections import Counter
from datetime import timedelta
from joblib import Parallel, delayed

from uqgrid.simulation.dynamics import integrate_system
from uqgrid.simulation.config import IntegrationConfig
from uqgrid.simulation.pflow import PowerFlowValidationError, runpf
from uqgrid.io.parse import load_psse, add_dyr
from uqgrid.core.psydef import Bus


# =============================================================================
# Noise Generation Utilities
# =============================================================================

def _draw_noise(shape, *, noise_type="normal", var=0.1, rng=None):
    r"""
    Draw zero-mean multiplicative noise for power perturbations.

    Generates noise values $\epsilon$ such that perturbed quantities can be
    computed as:

        $P_{scaled} = P_{base} \cdot (1 + \epsilon)$

    Parameters
    ----------
    shape : tuple
        Shape of the output noise array.
    noise_type : {'normal', 'uniform', 'none'}
        Distribution type:

        - 'normal': $\epsilon \sim \mathcal{N}(0, \sigma^2)$ where
          $\sigma$ = ``var``
        - 'uniform': $\epsilon \sim \mathcal{U}(-a, a)$ where
          $a = \sqrt{3 \cdot var}$ to achieve $\text{Var}(\epsilon) = var$
        - 'none': Returns zeros (no perturbation)

    var : float, default=0.1
        Distribution parameter. For 'normal', this is the standard deviation
        $\sigma$. For 'uniform', the half-width is computed as
        $a = \sqrt{3 \cdot var}$ so that the variance equals ``var``.
    rng : numpy.random.Generator, optional
        Random number generator. If None, creates a new default generator.

    Returns
    -------
    numpy.ndarray
        Noise array $\epsilon$ with the specified shape.

    Notes
    -----
    The uniform distribution half-width formula derives from:

        $\text{Var}(\mathcal{U}(-a, a)) = \frac{(a - (-a))^2}{12} = \frac{a^2}{3}$

    Setting this equal to ``var`` and solving: $a = \sqrt{3 \cdot var}$

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> eps = _draw_noise((100,), noise_type='normal', var=0.1, rng=rng)
    >>> print(f"Mean: {eps.mean():.3f}, Std: {eps.std():.3f}")
    """
    rng = np.random.default_rng() if rng is None else rng

    if noise_type == "normal":
        # Normal: var parameter is standard deviation
        return rng.normal(0.0, var, size=shape)

    elif noise_type == "uniform":
        # Uniform: compute half-width to achieve desired variance
        # Var(U[-a,a]) = a^2/3 = var  =>  a = sqrt(3*var)
        half = np.sqrt(3 * var)
        return rng.uniform(-half, half, size=shape)

    elif noise_type == "none":
        return np.zeros(shape, dtype=float)

    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'")


def _relative_noise(scaled, base, eps=1e-8):
    r"""
    Compute the relative noise from scaled and base values.

    Given $P_{scaled} = P_{base} \cdot (1 + \epsilon)$, this function
    recovers $\epsilon$ in a numerically safe way:

        $\epsilon = \frac{P_{scaled}}{P_{base}} - 1$

    Parameters
    ----------
    scaled : array-like
        Scaled (perturbed) values.
    base : array-like
        Original base values.
    eps : float, default=1e-8
        Threshold for considering base values as zero. Values with
        $|P_{base}| < \epsilon$ are assigned zero noise.

    Returns
    -------
    numpy.ndarray
        Relative noise array $\epsilon$.

    Notes
    -----
    This function handles the case where $P_{base} = 0$ by returning
    $\epsilon = 0$ for those elements, avoiding division by zero.
    """
    scaled = np.asarray(scaled, dtype=float)
    base = np.asarray(base, dtype=float)

    noise = np.zeros_like(base)
    mask = np.abs(base) > eps
    noise[mask] = scaled[mask] / base[mask] - 1.0

    return noise


def _json_safe(value):
    """Convert numpy/scalar values into JSON-serializable Python values."""
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(val) for val in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _default_operating_point_config():
    """Neutral defaults; operating-point screening is opt-in."""
    return {
        "enabled": False,
        "run_power_flow": True,
        "rebalance_non_slack": True,
        "redistribute_slack_mismatch": True,
        "rebalance_policy": "headroom",
        "loss_compensation": False,
        "loss_compensation_tolerance_pu": 1e-4,
        "loss_compensation_policy": "headroom",
        "q_limit_mitigation": False,
        "q_limit_mitigation_tolerance_pu": 1e-6,
        "q_limit_mitigation_max_passes": 10,
        "q_limit_mitigation_top_n": 10,
        "max_iterations": 5,
        "max_attempts_per_scenario": 50,
        "pf_residual_tol": 1e-8,
        "slack_p_tolerance_pu": 1e-4,
        "max_slack_p_deviation_fraction_of_load": 0.02,
        "voltage_min": 0.90,
        "voltage_max": 1.10,
        "gen_limit_tolerance": 1e-6,
        "branch_loading_max": 1.0,
        "diagnostics_file": "scenario_diagnostics.jsonl",
        "diagnostics_summary_file": "scenario_diagnostics_summary.json",
    }


def _resolve_operating_point_config(config=None):
    """Merge user operating-point settings with neutral defaults."""
    resolved = _default_operating_point_config()
    if config:
        resolved.update(config)
    supported_policies = {
        "headroom",
        "capacity",
        "current_dispatch",
        "base_dispatch",
        "equal",
    }
    for policy_key in ("rebalance_policy", "loss_compensation_policy"):
        resolved[policy_key] = str(resolved[policy_key]).lower()
        if resolved[policy_key] not in supported_policies:
            supported = ", ".join(sorted(supported_policies))
            raise ValueError(
                f"Unsupported {policy_key}={resolved[policy_key]!r}; "
                f"choose one of: {supported}"
            )
    return resolved


def _stress_load(base_p_load, base_q_load, *, load_scale=1.0, load_mean_shift=0.0):
    """Apply deterministic load stress before stochastic perturbations."""
    load_factor = float(load_scale) * (1.0 + float(load_mean_shift))
    return base_p_load * load_factor, base_q_load * load_factor, load_factor


def _scenario_seed_sequence(global_seed, sample_idx, attempt_idx):
    """Keep attempt 0 compatible with the previous sample-level RNG seed."""
    if attempt_idx == 0:
        return np.random.SeedSequence([global_seed, sample_idx])
    return np.random.SeedSequence([global_seed, sample_idx, attempt_idx])


def _generator_bus_type_array(psys):
    """Return each generator's bus type."""
    return np.array([psys.buses[gen.bus].type for gen in psys.gens], dtype=int)


def _slack_generator_mask(psys):
    """Return True for generators connected to slack buses."""
    return _generator_bus_type_array(psys) == Bus.SLACK


def _compute_generator_q_dispatch(base_p_gen, base_q_gen, p_gen, *, keep_power_factor=True):
    """Compute generator Q schedule from the requested active-power dispatch."""
    if not keep_power_factor:
        return np.array(base_q_gen, copy=True, dtype=float)

    q_gen = np.array(base_q_gen, copy=True, dtype=float)
    mask_pg_nonzero = np.abs(base_p_gen) > 1e-8
    if np.any(mask_pg_nonzero):
        ratio = np.zeros_like(base_p_gen, dtype=float)
        ratio[mask_pg_nonzero] = base_q_gen[mask_pg_nonzero] / base_p_gen[mask_pg_nonzero]
        q_gen[mask_pg_nonzero] = ratio[mask_pg_nonzero] * p_gen[mask_pg_nonzero]

    return q_gen


def _limit_violation(values, lower, upper):
    """Return the maximum absolute violation outside lower/upper bounds."""
    if values is None or lower is None or upper is None:
        return 0.0

    values = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    below = np.maximum(lower - values, 0.0)
    above = np.maximum(values - upper, 0.0)
    if values.size == 0:
        return 0.0
    return float(np.max(np.maximum(below, above)))


def _bus_type_label(bus_type):
    """Return a readable bus type label for diagnostics."""
    if bus_type == Bus.PQ:
        return "PQ"
    if bus_type == Bus.PV:
        return "PV"
    if bus_type == Bus.SLACK:
        return "SLACK"
    return str(bus_type)


def _generator_q_limit_diagnostics(psys, q_values, qg_lb, qg_ub, *, top_n=10):
    """Return detailed per-generator reactive-power limit diagnostics."""
    diagnostics = {
        "gen_q_violation_count": 0,
        "gen_q_violation_total_abs": 0.0,
        "gen_q_violation_argmax": None,
        "gen_q_violation_top": [],
    }

    if q_values is None or qg_lb is None or qg_ub is None:
        return diagnostics

    q_values = np.asarray(q_values, dtype=float)
    qg_lb = np.asarray(qg_lb, dtype=float)
    qg_ub = np.asarray(qg_ub, dtype=float)
    if q_values.size == 0:
        return diagnostics

    lower_violation = np.maximum(qg_lb - q_values, 0.0)
    upper_violation = np.maximum(q_values - qg_ub, 0.0)
    violation = np.maximum(lower_violation, upper_violation)
    violator_indices = np.where(violation > 0.0)[0]
    if violator_indices.size == 0:
        return diagnostics

    sorted_indices = sorted(
        violator_indices,
        key=lambda gen_index: violation[gen_index],
        reverse=True,
    )
    top = []
    for gen_index in sorted_indices[:top_n]:
        gen = psys.gens[int(gen_index)]
        bus_index = int(gen.bus)
        bus = psys.buses[bus_index]
        bus_type = getattr(bus, "type", None)
        side = (
            "lower"
            if lower_violation[gen_index] > upper_violation[gen_index]
            else "upper"
        )
        top.append({
            "gen_index": int(gen_index),
            "gen_id": str(getattr(gen, "idx", "")),
            "bus_index": bus_index,
            "bus_id": getattr(bus, "id", None),
            "bus_type": _bus_type_label(bus_type),
            "qg": float(q_values[gen_index]),
            "qmin": float(qg_lb[gen_index]),
            "qmax": float(qg_ub[gen_index]),
            "violation": float(violation[gen_index]),
            "side": side,
            "is_slack": bool(bus_type == Bus.SLACK),
        })

    diagnostics.update({
        "gen_q_violation_count": int(violator_indices.size),
        "gen_q_violation_total_abs": float(np.sum(violation[violator_indices])),
        "gen_q_violation_argmax": int(sorted_indices[0]),
        "gen_q_violation_top": top,
    })
    return diagnostics


def _capture_bus_types(psys):
    """Capture bus types so temporary PV/PQ changes can be restored."""
    return [bus.type for bus in psys.buses]


def _restore_bus_types(psys, bus_types):
    """Restore previously captured bus types."""
    for bus, bus_type in zip(psys.buses, bus_types):
        bus.type = bus_type


def _q_limit_mitigation_defaults(op_cfg):
    """Return default diagnostics for PV-to-PQ Q-limit mitigation."""
    return {
        "q_limit_mitigation_enabled": bool(op_cfg["q_limit_mitigation"]),
        "q_limit_mitigation_applied": False,
        "q_limit_mitigation_passes": 0,
        "q_limit_mitigation_switched_buses": [],
        "q_limit_mitigation_switched_generators": [],
        "q_limit_mitigation_events": [],
        "q_limit_mitigation_limit_reached": False,
    }


def _apply_q_limit_mitigation(
        psys, q_gen, q_values, qg_lb, qg_ub, op_cfg, *,
        q_fixed_mask=None, q_fixed_values=None, pass_idx=0):
    """Clamp violating non-slack PV generator Q and switch those buses to PQ."""
    result = {
        "applied": False,
        "events": [],
        "switched_buses": [],
        "switched_generators": [],
    }

    if (
        not op_cfg["q_limit_mitigation"]
        or q_values is None
        or qg_lb is None
        or qg_ub is None
    ):
        return q_gen, result

    q_gen = np.asarray(q_gen, dtype=float).copy()
    tolerance = float(op_cfg["q_limit_mitigation_tolerance_pu"])
    pass_bus_types = _capture_bus_types(psys)
    q_diagnostics = _generator_q_limit_diagnostics(
        psys,
        q_values,
        qg_lb,
        qg_ub,
        top_n=len(psys.gens),
    )

    for offender in q_diagnostics["gen_q_violation_top"]:
        if offender["violation"] <= tolerance:
            continue

        gen_index = int(offender["gen_index"])
        bus_index = int(offender["bus_index"])
        if pass_bus_types[bus_index] != Bus.PV:
            continue

        side = offender["side"]
        if side == "upper":
            q_limit = float(qg_ub[gen_index])
        else:
            q_limit = float(qg_lb[gen_index])

        q_gen[gen_index] = q_limit
        psys.gens[gen_index].qsch = q_limit
        if q_fixed_mask is not None:
            q_fixed_mask[gen_index] = True
        if q_fixed_values is not None:
            q_fixed_values[gen_index] = q_limit

        psys.buses[bus_index].type = Bus.PQ
        result["applied"] = True
        result["switched_buses"].append(bus_index)
        result["switched_generators"].append(gen_index)
        result["events"].append({
            "pass": int(pass_idx),
            "gen_index": gen_index,
            "gen_id": offender["gen_id"],
            "bus_index": bus_index,
            "bus_id": offender["bus_id"],
            "side": side,
            "qg_solved": float(offender["qg"]),
            "q_limit": q_limit,
            "violation": float(offender["violation"]),
            "from_bus_type": offender["bus_type"],
            "to_bus_type": "PQ",
        })

    result["switched_buses"] = sorted(set(result["switched_buses"]))
    result["switched_generators"] = sorted(set(result["switched_generators"]))
    return q_gen, result


# =============================================================================
# Power Rebalancing
# =============================================================================

def _active_power_margin_sums(p_gen, pg_lb, pg_ub, participation_mask=None):
    """Return total active-power headroom and footroom for selected generators."""
    if pg_lb is None or pg_ub is None:
        return None, None

    p_gen = np.asarray(p_gen, dtype=float)
    pg_lb = np.asarray(pg_lb, dtype=float)
    pg_ub = np.asarray(pg_ub, dtype=float)
    if participation_mask is None:
        participation_mask = np.ones_like(p_gen, dtype=bool)
    else:
        participation_mask = np.asarray(participation_mask, dtype=bool)

    headroom = np.maximum(pg_ub - p_gen, 0.0)
    footroom = np.maximum(p_gen - pg_lb, 0.0)
    return (
        float(np.sum(headroom[participation_mask])),
        float(np.sum(footroom[participation_mask])),
    )


def _active_power_policy_weights(
        policy, p_gen, base_p_gen, pg_lb, pg_ub, margin, eligible_mask):
    """Compute automatic participation weights for a bounded active-power move."""
    if policy == "headroom":
        weights = margin.copy()
    elif policy == "capacity":
        weights = np.maximum(pg_ub - pg_lb, 0.0)
    elif policy == "current_dispatch":
        weights = np.maximum(np.abs(p_gen), 0.0)
    elif policy == "base_dispatch":
        if base_p_gen is None:
            weights = np.maximum(np.abs(p_gen), 0.0)
        else:
            weights = np.maximum(np.abs(np.asarray(base_p_gen, dtype=float)), 0.0)
    elif policy == "equal":
        weights = np.ones_like(p_gen, dtype=float)
    else:
        raise ValueError(f"Unsupported participation policy: {policy!r}")

    weights = np.where(eligible_mask, weights, 0.0)
    if np.sum(weights) <= 1e-12:
        weights = np.where(eligible_mask, 1.0, 0.0)
    return weights


def _allocate_active_power_mismatch(
        p_gen, pg_lb, pg_ub, mismatch, participation_mask=None, *,
        policy="headroom", base_p_gen=None, tolerance=1e-9):
    """Move active-power mismatch with bounded automatic participation weights."""
    p_gen = np.asarray(p_gen, dtype=float)
    p_new = p_gen.copy()
    mismatch = float(mismatch)

    if participation_mask is None:
        participation_mask = np.ones_like(p_new, dtype=bool)
    else:
        participation_mask = np.asarray(participation_mask, dtype=bool)

    headroom_start, footroom_start = _active_power_margin_sums(
        p_new, pg_lb, pg_ub, participation_mask
    )
    touched = np.zeros_like(p_new, dtype=bool)
    applied = 0.0

    if (
        p_new.size == 0
        or not np.any(participation_mask)
        or abs(mismatch) <= tolerance
    ):
        headroom_end, footroom_end = _active_power_margin_sums(
            p_new, pg_lb, pg_ub, participation_mask
        )
        return {
            "p_gen": p_new,
            "applied_mismatch": 0.0,
            "unresolved_mismatch": mismatch,
            "participants": 0,
            "direction": "none",
            "headroom_start": headroom_start,
            "headroom_remaining": headroom_end,
            "footroom_start": footroom_start,
            "footroom_remaining": footroom_end,
        }

    direction_sign = 1.0 if mismatch > 0.0 else -1.0
    direction_name = "increase" if direction_sign > 0.0 else "decrease"

    if pg_lb is None or pg_ub is None:
        selected = np.where(participation_mask)[0]
        selected_total = float(np.sum(p_new[selected]))
        if not np.isclose(selected_total, 0.0):
            p_new[selected] *= (selected_total + mismatch) / selected_total
        else:
            p_new[selected] += mismatch / len(selected)
        touched[selected] = True
        return {
            "p_gen": p_new,
            "applied_mismatch": mismatch,
            "unresolved_mismatch": 0.0,
            "participants": int(len(selected)),
            "direction": direction_name,
            "headroom_start": headroom_start,
            "headroom_remaining": None,
            "footroom_start": footroom_start,
            "footroom_remaining": None,
        }

    pg_lb = np.asarray(pg_lb, dtype=float)
    pg_ub = np.asarray(pg_ub, dtype=float)
    policy = str(policy).lower()

    for _iteration in range(p_new.size):
        unresolved = mismatch - applied
        if abs(unresolved) <= tolerance:
            break

        if direction_sign > 0.0:
            margin = np.maximum(pg_ub - p_new, 0.0)
        else:
            margin = np.maximum(p_new - pg_lb, 0.0)

        eligible_mask = participation_mask & (margin > tolerance)
        if not np.any(eligible_mask):
            break

        weights = _active_power_policy_weights(
            policy, p_new, base_p_gen, pg_lb, pg_ub, margin, eligible_mask
        )
        weight_total = float(np.sum(weights))
        if weight_total <= tolerance:
            break

        requested_abs = abs(unresolved)
        requested = requested_abs * weights / weight_total
        applied_abs = np.minimum(requested, margin)
        applied_step_abs = float(np.sum(applied_abs[eligible_mask]))
        if applied_step_abs <= tolerance:
            break

        p_new[eligible_mask] += direction_sign * applied_abs[eligible_mask]
        touched |= applied_abs > tolerance
        applied += direction_sign * applied_step_abs

    unresolved = mismatch - applied
    headroom_end, footroom_end = _active_power_margin_sums(
        p_new, pg_lb, pg_ub, participation_mask
    )
    return {
        "p_gen": p_new,
        "applied_mismatch": float(applied),
        "unresolved_mismatch": float(unresolved),
        "participants": int(np.sum(touched)),
        "direction": direction_name,
        "headroom_start": headroom_start,
        "headroom_remaining": headroom_end,
        "footroom_start": footroom_start,
        "footroom_remaining": footroom_end,
    }


def _redistribute_active_power_mismatch(
        p_gen, pg_lb, pg_ub, mismatch, participation_mask=None, *,
        policy="headroom", base_p_gen=None, return_diagnostics=False):
    """Move active-power mismatch onto participating generators within limits."""
    allocation = _allocate_active_power_mismatch(
        p_gen,
        pg_lb,
        pg_ub,
        mismatch,
        participation_mask=participation_mask,
        policy=policy,
        base_p_gen=base_p_gen,
    )
    if return_diagnostics:
        return allocation
    return allocation["p_gen"]


def _rebalance_active_power(
        p_gen, pg_lb, pg_ub, target_total, participation_mask=None, *,
        policy="headroom", base_p_gen=None, return_diagnostics=False):
    r"""
    Rebalance generator active power to match a target total.

    Redistributes generation to achieve $\sum_i P_{g,i} = P_{target}$ while
    respecting generator limits. The adjustment is distributed proportionally
    to each generator's available headroom (for increases) or margin (for
    decreases).

    Parameters
    ----------
    p_gen : array-like
        Current generator active power setpoints $P_g$.
    pg_lb : array-like or None
        Lower bounds $P_g^{min}$ for each generator. If None, falls back
        to simple scaling.
    pg_ub : array-like or None
        Upper bounds $P_g^{max}$ for each generator. If None, falls back
        to simple scaling.
    target_total : float or None
        Target total generation. If None, returns unchanged values.

    Returns
    -------
    numpy.ndarray
        Rebalanced generator setpoints $P_g^{new}$.

    Notes
    -----
    **Algorithm for increasing generation** ($\Delta > 0$):

    1. Compute headroom: $h_i = P_{g,i}^{max} - P_{g,i}$
    2. Compute participation factor: $\alpha = \min(1, \Delta / \sum_j h_j)$
    3. Update: $P_{g,i}^{new} = P_{g,i} + \alpha \cdot h_i$

    **Algorithm for decreasing generation** ($\Delta < 0$):

    1. Compute margin: $d_i = P_{g,i} - P_{g,i}^{min}$
    2. Compute participation factor: $\alpha = \min(1, |\Delta| / \sum_j d_j)$
    3. Update: $P_{g,i}^{new} = P_{g,i} - \alpha \cdot d_i$

    If bounds are not provided, simple uniform scaling is applied:

        $P_g^{new} = P_g \cdot \frac{P_{target}}{\sum_i P_{g,i}}$
    """
    p_gen = np.asarray(p_gen, dtype=float)

    if target_total is None:
        if return_diagnostics:
            return _allocate_active_power_mismatch(
                p_gen,
                pg_lb,
                pg_ub,
                0.0,
                participation_mask=participation_mask,
                policy=policy,
                base_p_gen=base_p_gen,
            )
        return p_gen

    current_total = np.sum(p_gen)
    mismatch = target_total - current_total

    # No rebalancing needed if already at target
    if np.isclose(mismatch, 0.0):
        if return_diagnostics:
            return _allocate_active_power_mismatch(
                p_gen,
                pg_lb,
                pg_ub,
                0.0,
                participation_mask=participation_mask,
                policy=policy,
                base_p_gen=base_p_gen,
            )
        return p_gen

    allocation = _redistribute_active_power_mismatch(
        p_gen,
        pg_lb,
        pg_ub,
        mismatch,
        participation_mask=participation_mask,
        policy=policy,
        base_p_gen=base_p_gen,
        return_diagnostics=True,
    )
    if return_diagnostics:
        return allocation
    return allocation["p_gen"]


# =============================================================================
# Perturbation Generation
# =============================================================================

def generate_perturbations(base_p, base_q,
                           *,
                           noise_type="normal", var=0.1,
                           rng=None, return_noise=False,
                           preserve_power_factor=True):
    r"""
    Apply multiplicative perturbations to active and reactive power.

    Generates perturbed power values using the multiplicative noise model:

        $P_{scaled} = P_{base} \cdot (1 + \epsilon_P)$

    For reactive power, the behavior depends on ``preserve_power_factor``.

    Parameters
    ----------
    base_p : array-like
        Base active power values $P_{base}$.
    base_q : array-like
        Base reactive power values $Q_{base}$.
    noise_type : {'normal', 'uniform', 'none'}, default='normal'
        Noise distribution type. See :func:`_draw_noise` for details.
    var : float, default=0.1
        Noise variance parameter.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.
    return_noise : bool, default=False
        If True, also return the noise arrays $(\epsilon_P, \epsilon_Q)$.
    preserve_power_factor : bool, default=True
        If True, adjust reactive power to maintain the original power factor
        ratio $Q/P$. If False, perturb P and Q independently.

    Returns
    -------
    p_scaled : numpy.ndarray
        Perturbed active power values.
    q_scaled : numpy.ndarray
        Perturbed reactive power values.
    p_noise : numpy.ndarray (only if return_noise=True)
        Active power noise $\epsilon_P$.
    q_noise : numpy.ndarray (only if return_noise=True)
        Reactive power noise $\epsilon_Q$.

    Notes
    -----
    **Power Factor Preservation** (``preserve_power_factor=True``):

    For buses where $|P_{base}| > 0$:

        $\text{ratio} = \frac{Q_{base}}{P_{base}}$

        $Q_{scaled} = \text{ratio} \cdot P_{scaled}$

    This ensures $\frac{Q_{scaled}}{P_{scaled}} = \frac{Q_{base}}{P_{base}}$.

    For purely reactive buses ($P_{base} = 0$, $Q_{base} \neq 0$):

        $Q_{scaled} = Q_{base} \cdot (1 + \epsilon_P)$

    **Independent Perturbation** (``preserve_power_factor=False``):

    Both P and Q receive independent noise:

        $P_{scaled} = P_{base} \cdot (1 + \epsilon_P)$
        $Q_{scaled} = Q_{base} \cdot (1 + \epsilon_Q)$

    Examples
    --------
    >>> base_p = np.array([100.0, 50.0, 0.0])
    >>> base_q = np.array([30.0, 15.0, 10.0])
    >>> p_new, q_new = generate_perturbations(base_p, base_q, var=0.1)
    """
    rng = np.random.default_rng() if rng is None else rng

    # Generate active power noise and apply perturbation
    p_noise = _draw_noise(base_p.shape, noise_type=noise_type, var=var, rng=rng)
    p_scaled = base_p * (1.0 + p_noise)

    if preserve_power_factor:
        # Initialize Q from base values
        q_scaled = np.array(base_q, copy=True, dtype=float)

        # For buses with non-zero P: maintain Q/P ratio
        mask_p_nonzero = np.abs(base_p) > 1e-8
        if np.any(mask_p_nonzero):
            ratio = np.zeros_like(base_p, dtype=float)
            ratio[mask_p_nonzero] = base_q[mask_p_nonzero] / base_p[mask_p_nonzero]
            q_scaled[mask_p_nonzero] = ratio[mask_p_nonzero] * p_scaled[mask_p_nonzero]

        # For purely reactive buses (P=0, Q!=0): apply same noise factor
        mask_p_zero_q_nonzero = (~mask_p_nonzero) & (np.abs(base_q) > 1e-8)
        if np.any(mask_p_zero_q_nonzero):
            q_scaled[mask_p_zero_q_nonzero] = base_q[mask_p_zero_q_nonzero] * (
                1.0 + p_noise[mask_p_zero_q_nonzero]
            )

        # Compute effective Q noise for logging
        q_noise = _relative_noise(q_scaled, base_q)

    else:
        # Independent perturbation: draw separate noise for Q
        q_noise = _draw_noise(base_q.shape, noise_type=noise_type, var=var, rng=rng)
        q_scaled = base_q * (1.0 + q_noise)

    if return_noise:
        return p_scaled, q_scaled, p_noise, q_noise
    return p_scaled, q_scaled


# =============================================================================
# Operating Point Diagnostics
# =============================================================================

def _compute_power_flow_residual(psys, pf_solution):
    """Compute the solved PF residual norm using the same bus constraints as runpf."""
    bus_type = np.array([bus.type for bus in psys.buses], dtype=int)
    p_inj = np.zeros(psys.nbuses, dtype=float)
    q_inj = np.zeros(psys.nbuses, dtype=float)

    for gen in psys.gens:
        p_inj[gen.bus] += gen.psch
        q_inj[gen.bus] += gen.qsch

    for load in psys.loads:
        p_inj[load.bus] -= load.pload
        q_inj[load.bus] += load.qload

    residuals = []
    solved = pf_solution.s_inj_vector
    for bus_idx, btype in enumerate(bus_type):
        if btype in (Bus.PQ, Bus.PV):
            residuals.append(solved[2 * bus_idx] - p_inj[bus_idx])
        if btype == Bus.PQ:
            residuals.append(solved[2 * bus_idx + 1] - q_inj[bus_idx])

    if not residuals:
        return 0.0
    return float(np.linalg.norm(np.asarray(residuals, dtype=float)))


def _compute_branch_loading_diagnostics(psys, pf_solution):
    """Compute branch loading against rateA when branch ratings are available."""
    if getattr(psys, "nbranches", 0) == 0:
        return {
            "branch_loading_available": False,
            "branch_loading_max": None,
            "branch_overloaded_count": 0,
            "branch_loading_argmax": None,
        }

    voltages = pf_solution.v_magnitudes * np.exp(1j * pf_solution.v_angles)
    loadings = []
    loading_indices = []

    for idx, branch in enumerate(psys.branches):
        rate = float(getattr(branch, "rateA", 0.0))
        if rate <= 0.0:
            continue

        tap = float(branch.tap)
        shift = float(branch.shift)
        if tap > 0.0:
            tpsh = tap * np.exp(1j * np.pi / 180.0 * shift)
        else:
            tap = 1.0
            tpsh = 1.0

        impedance = branch.r + 1j * branch.x
        if abs(impedance) <= 1e-12:
            continue
        y_series = 1.0 / impedance
        y_shunt = 1j * 0.5 * branch.sh
        v_from = voltages[branch.fr]
        v_to = voltages[branch.to]

        i_from = ((y_series + y_shunt) / (tap * tap)) * v_from - (
            y_series / np.conj(tpsh)
        ) * v_to
        i_to = -(y_series / tpsh) * v_from + (y_series + y_shunt) * v_to

        s_from_mva = abs(v_from * np.conj(i_from)) * psys.basemva
        s_to_mva = abs(v_to * np.conj(i_to)) * psys.basemva
        loadings.append(max(s_from_mva, s_to_mva) / rate)
        loading_indices.append(idx)

    if not loadings:
        return {
            "branch_loading_available": False,
            "branch_loading_max": None,
            "branch_overloaded_count": 0,
            "branch_loading_argmax": None,
        }

    loadings = np.asarray(loadings, dtype=float)
    argmax_local = int(np.argmax(loadings))
    return {
        "branch_loading_available": True,
        "branch_loading_max": float(loadings[argmax_local]),
        "branch_overloaded_count": int(np.sum(loadings > 1.0)),
        "branch_loading_argmax": int(loading_indices[argmax_local]),
    }


def _diagnose_power_flow(
        psys, p_gen_target, q_gen_target, pg_lb, pg_ub, qg_lb, qg_ub,
        op_cfg):
    """Run PF and return solution plus screening diagnostics."""
    diagnostics = {
        "pf_converged": False,
        "pf_residual": None,
        "slack_p_target": None,
        "slack_p_solved": None,
        "slack_p_deviation": None,
        "slack_q_target": None,
        "slack_q_solved": None,
        "slack_q_deviation": None,
        "slack_p_limit_violation": None,
        "voltage_min": None,
        "voltage_max": None,
        "gen_p_violation_max": None,
        "gen_q_violation_max": None,
        "gen_q_violation_count": 0,
        "gen_q_violation_total_abs": 0.0,
        "gen_q_violation_argmax": None,
        "gen_q_violation_top": [],
        "branch_loading_available": False,
        "branch_loading_max": None,
        "branch_overloaded_count": 0,
        "branch_loading_argmax": None,
    }

    try:
        pf_solution = runpf(psys, verbose=False)
    except Exception as exc:
        diagnostics["reject_reason"] = "pf_non_converged"
        diagnostics["pf_error"] = str(exc)
        return None, diagnostics

    slack_mask = _slack_generator_mask(psys)
    diagnostics.update({
        "pf_converged": True,
        "pf_residual": _compute_power_flow_residual(psys, pf_solution),
        "slack_p_target": float(np.sum(p_gen_target[slack_mask])),
        "slack_p_solved": float(np.sum(pf_solution.gen_psch[slack_mask])),
        "slack_q_target": float(np.sum(q_gen_target[slack_mask])),
        "slack_q_solved": float(np.sum(pf_solution.gen_qsch[slack_mask])),
        "voltage_min": float(np.min(pf_solution.v_magnitudes)),
        "voltage_max": float(np.max(pf_solution.v_magnitudes)),
        "gen_p_violation_max": _limit_violation(pf_solution.gen_psch, pg_lb, pg_ub),
        "gen_q_violation_max": _limit_violation(pf_solution.gen_qsch, qg_lb, qg_ub),
        "slack_p_limit_violation": _limit_violation(
            pf_solution.gen_psch[slack_mask],
            pg_lb[slack_mask] if pg_lb is not None else None,
            pg_ub[slack_mask] if pg_ub is not None else None,
        ),
    })
    diagnostics["slack_p_deviation"] = (
        diagnostics["slack_p_solved"] - diagnostics["slack_p_target"]
    )
    diagnostics["slack_q_deviation"] = (
        diagnostics["slack_q_solved"] - diagnostics["slack_q_target"]
    )
    diagnostics.update(
        _generator_q_limit_diagnostics(
            psys,
            pf_solution.gen_qsch,
            qg_lb,
            qg_ub,
            top_n=int(op_cfg["q_limit_mitigation_top_n"]),
        )
    )
    diagnostics.update(_compute_branch_loading_diagnostics(psys, pf_solution))

    return pf_solution, diagnostics


def _screen_power_flow_diagnostics(diagnostics, total_load_p, op_cfg):
    """Return (accepted, reason) for operating-point diagnostics."""
    if not diagnostics.get("pf_converged", False):
        return False, diagnostics.get("reject_reason", "pf_non_converged")

    if diagnostics["pf_residual"] is None or diagnostics["pf_residual"] > op_cfg["pf_residual_tol"]:
        return False, "pf_residual"

    slack_tol = max(
        op_cfg["slack_p_tolerance_pu"],
        abs(total_load_p) * op_cfg["max_slack_p_deviation_fraction_of_load"],
    )
    if abs(diagnostics["slack_p_deviation"]) > slack_tol:
        return False, "slack_p_deviation"

    if diagnostics["voltage_min"] < op_cfg["voltage_min"]:
        return False, "voltage_low"
    if diagnostics["voltage_max"] > op_cfg["voltage_max"]:
        return False, "voltage_high"

    limit_tol = op_cfg["gen_limit_tolerance"]
    if diagnostics["gen_p_violation_max"] > limit_tol:
        return False, "gen_p_limit"
    if diagnostics["gen_q_violation_max"] > limit_tol:
        return False, "gen_q_limit"

    if (
        diagnostics["branch_loading_available"]
        and diagnostics["branch_loading_max"] is not None
        and diagnostics["branch_loading_max"] > op_cfg["branch_loading_max"]
    ):
        return False, "branch_overload"

    return True, None


def _prepare_operating_point(
        psys, p_load, q_load, p_gen, q_gen, base_p_gen, base_q_gen,
        pg_lb, pg_ub, qg_lb, qg_ub, keep_power_factor, clamp_gens,
        balance_generation, op_cfg):
    """PF-aware active-power rebalance and scenario screening."""
    diagnostics = {
        "accepted": False,
        "rebalance_iterations": 0,
        "rebalance_policy": op_cfg["rebalance_policy"],
        "initial_rebalance_applied_pu": 0.0,
        "initial_rebalance_unresolved_pu": 0.0,
        "loss_compensation_enabled": bool(op_cfg["loss_compensation"]),
        "loss_compensation_policy": op_cfg["loss_compensation_policy"],
        "loss_compensation_requested_pu": 0.0,
        "loss_compensation_applied_pu": 0.0,
        "loss_compensation_unresolved_pu": 0.0,
        "loss_compensation_unresolved_abs_pu": 0.0,
        "loss_compensation_participants": 0,
        "loss_compensation_iterations": 0,
        "loss_compensation_effective_tolerance_pu": None,
        "non_slack_headroom_remaining": None,
        "non_slack_footroom_remaining": None,
        "reject_reason": None,
    }
    diagnostics.update(_q_limit_mitigation_defaults(op_cfg))

    if not op_cfg["enabled"]:
        return p_gen, q_gen, None, {**diagnostics, "accepted": True}

    p_gen = np.asarray(p_gen, dtype=float).copy()
    q_gen = np.asarray(q_gen, dtype=float).copy()
    total_load_p = float(np.sum(p_load))
    non_slack_mask = ~_slack_generator_mask(psys)
    original_bus_types = _capture_bus_types(psys)
    q_fixed_mask = np.zeros_like(q_gen, dtype=bool)
    q_fixed_values = np.array(q_gen, copy=True, dtype=float)

    headroom, footroom = _active_power_margin_sums(
        p_gen, pg_lb, pg_ub, non_slack_mask
    )
    diagnostics["non_slack_headroom_remaining"] = headroom
    diagnostics["non_slack_footroom_remaining"] = footroom

    if balance_generation and op_cfg["rebalance_non_slack"]:
        rebalance = _rebalance_active_power(
            p_gen,
            pg_lb,
            pg_ub,
            total_load_p,
            participation_mask=non_slack_mask,
            policy=op_cfg["rebalance_policy"],
            base_p_gen=base_p_gen,
            return_diagnostics=True,
        )
        p_gen = rebalance["p_gen"]
        diagnostics["initial_rebalance_applied_pu"] = rebalance["applied_mismatch"]
        diagnostics["initial_rebalance_unresolved_pu"] = rebalance["unresolved_mismatch"]
        diagnostics["initial_rebalance_participants"] = rebalance["participants"]
        diagnostics["non_slack_headroom_remaining"] = rebalance["headroom_remaining"]
        diagnostics["non_slack_footroom_remaining"] = rebalance["footroom_remaining"]

    pf_solution = None
    max_iterations = int(op_cfg["max_iterations"])
    for iteration in range(max_iterations):
        q_gen = _compute_generator_q_dispatch(
            base_p_gen, base_q_gen, p_gen, keep_power_factor=keep_power_factor
        )
        if np.any(q_fixed_mask):
            q_gen[q_fixed_mask] = q_fixed_values[q_fixed_mask]
        if clamp_gens and qg_lb is not None and qg_ub is not None and q_gen.size:
            q_gen = np.clip(q_gen, qg_lb, qg_ub)
            if np.any(q_fixed_mask):
                q_fixed_values[q_fixed_mask] = q_gen[q_fixed_mask]

        psys.set_load_pq(p_load, q_load)
        psys.set_gen_pq(p_gen, q_gen)
        psys.createYbusComplex()

        diagnostics["rebalance_iterations"] = iteration + 1

        if not op_cfg["run_power_flow"]:
            diagnostics.update({
                "accepted": True,
                "reject_reason": None,
                "pf_converged": None,
                "pf_residual": None,
            })
            _restore_bus_types(psys, original_bus_types)
            return p_gen, q_gen, None, diagnostics

        pf_solution, pf_diag = _diagnose_power_flow(
            psys, p_gen, q_gen, pg_lb, pg_ub, qg_lb, qg_ub, op_cfg
        )
        diagnostics.update(pf_diag)

        if (
            op_cfg["q_limit_mitigation"]
            and pf_solution is not None
            and diagnostics.get("pf_converged", False)
        ):
            if diagnostics["q_limit_mitigation_passes"] >= int(
                op_cfg["q_limit_mitigation_max_passes"]
            ):
                diagnostics["q_limit_mitigation_limit_reached"] = (
                    diagnostics.get("gen_q_violation_max", 0.0)
                    > op_cfg["q_limit_mitigation_tolerance_pu"]
                )
            else:
                q_next, q_mitigation = _apply_q_limit_mitigation(
                    psys,
                    q_gen,
                    pf_solution.gen_qsch,
                    qg_lb,
                    qg_ub,
                    op_cfg,
                    q_fixed_mask=q_fixed_mask,
                    q_fixed_values=q_fixed_values,
                    pass_idx=diagnostics["q_limit_mitigation_passes"] + 1,
                )
                if q_mitigation["applied"]:
                    q_gen = q_next
                    diagnostics["q_limit_mitigation_applied"] = True
                    diagnostics["q_limit_mitigation_passes"] += 1
                    diagnostics["q_limit_mitigation_events"].extend(
                        q_mitigation["events"]
                    )
                    diagnostics["q_limit_mitigation_switched_buses"] = sorted(
                        set(diagnostics["q_limit_mitigation_switched_buses"])
                        | set(q_mitigation["switched_buses"])
                    )
                    diagnostics["q_limit_mitigation_switched_generators"] = sorted(
                        set(diagnostics["q_limit_mitigation_switched_generators"])
                        | set(q_mitigation["switched_generators"])
                    )
                    if iteration + 1 >= max_iterations:
                        diagnostics["reject_reason"] = (
                            "q_limit_mitigation_max_iterations"
                        )
                        break
                    continue

        headroom, footroom = _active_power_margin_sums(
            p_gen, pg_lb, pg_ub, non_slack_mask
        )
        diagnostics["non_slack_headroom_remaining"] = headroom
        diagnostics["non_slack_footroom_remaining"] = footroom

        pf_residual = diagnostics.get("pf_residual")
        pf_ok_for_loss = (
            pf_solution is not None
            and diagnostics.get("pf_converged", False)
            and pf_residual is not None
            and pf_residual <= op_cfg["pf_residual_tol"]
        )
        slack_delta = diagnostics.get("slack_p_deviation")
        loss_tolerance = max(
            min(
                op_cfg["loss_compensation_tolerance_pu"],
                op_cfg["gen_limit_tolerance"],
            ),
            1e-12,
        )
        diagnostics["loss_compensation_effective_tolerance_pu"] = loss_tolerance
        if (
            op_cfg["loss_compensation"]
            and pf_ok_for_loss
            and slack_delta is not None
            and abs(slack_delta) > loss_tolerance
        ):
            loss_rebalance = _redistribute_active_power_mismatch(
                p_gen,
                pg_lb,
                pg_ub,
                slack_delta,
                participation_mask=non_slack_mask,
                policy=op_cfg["loss_compensation_policy"],
                base_p_gen=base_p_gen,
                return_diagnostics=True,
            )
            diagnostics["loss_compensation_requested_pu"] += float(slack_delta)
            diagnostics["loss_compensation_applied_pu"] += float(
                loss_rebalance["applied_mismatch"]
            )
            diagnostics["loss_compensation_unresolved_pu"] = float(
                loss_rebalance["unresolved_mismatch"]
            )
            diagnostics["loss_compensation_unresolved_abs_pu"] = abs(
                diagnostics["loss_compensation_unresolved_pu"]
            )
            diagnostics["loss_compensation_participants"] = int(
                loss_rebalance["participants"]
            )
            diagnostics["loss_compensation_iterations"] += 1
            diagnostics["non_slack_headroom_remaining"] = loss_rebalance[
                "headroom_remaining"
            ]
            diagnostics["non_slack_footroom_remaining"] = loss_rebalance[
                "footroom_remaining"
            ]

            p_next = loss_rebalance["p_gen"]
            if abs(loss_rebalance["applied_mismatch"]) > 1e-12:
                if iteration + 1 >= max_iterations:
                    p_gen = p_next
                    diagnostics["reject_reason"] = "loss_compensation_max_iterations"
                    break
                p_gen = p_next
                continue

        accepted, reason = _screen_power_flow_diagnostics(diagnostics, total_load_p, op_cfg)
        if accepted:
            diagnostics.update({"accepted": True, "reject_reason": None})
            _restore_bus_types(psys, original_bus_types)
            return pf_solution.gen_psch.copy(), pf_solution.gen_qsch.copy(), pf_solution, diagnostics

        diagnostics["reject_reason"] = reason
        if (
            reason != "slack_p_deviation"
            or not op_cfg["redistribute_slack_mismatch"]
            or pf_solution is None
            or iteration + 1 >= max_iterations
        ):
            break

        slack_rebalance = _redistribute_active_power_mismatch(
            p_gen,
            pg_lb,
            pg_ub,
            diagnostics["slack_p_deviation"],
            participation_mask=non_slack_mask,
            policy=op_cfg["loss_compensation_policy"],
            base_p_gen=base_p_gen,
            return_diagnostics=True,
        )
        p_next = slack_rebalance["p_gen"]
        if abs(slack_rebalance["applied_mismatch"]) <= 1e-12:
            break
        p_gen = p_next

    diagnostics["accepted"] = False
    _restore_bus_types(psys, original_bus_types)
    return p_gen, q_gen, pf_solution, diagnostics


# =============================================================================
# Scenario Sampling and Metadata
# =============================================================================

def sample_scenarios(n_samples, fault_locations, fault_impedances):
    """
    Generate all combinations of samples, fault locations, and impedances.

    Creates the Cartesian product of scenario parameters to define the
    complete simulation campaign.

    Parameters
    ----------
    n_samples : int
        Number of perturbation samples per fault configuration.
    fault_locations : list of int
        Bus indices where faults will be applied.
    fault_impedances : list of float
        Fault impedance values in per-unit.

    Returns
    -------
    list of tuple
        List of (sample_idx, fault_location, fault_impedance) tuples
        representing all scenario combinations.

    Examples
    --------
    >>> scenarios = sample_scenarios(2, [0, 1], [0.001])
    >>> len(scenarios)
    4
    >>> scenarios
    [(0, 0, 0.001), (0, 1, 0.001), (1, 0, 0.001), (1, 1, 0.001)]
    """
    return list(itertools.product(range(n_samples), fault_locations, fault_impedances))


def generate_metadata(scenarios):
    """
    Generate unique identifiers and metadata for each scenario.

    Creates a metadata dictionary mapping UUID scenario IDs to their
    parameters and saves it to 'scenario_metadata.json'.

    Parameters
    ----------
    scenarios : list of tuple
        List of (sample_idx, fault_location, fault_impedance) tuples
        from :func:`sample_scenarios`.

    Returns
    -------
    dict
        Dictionary mapping scenario UUIDs to parameter dictionaries
        containing 'sample_idx', 'fault_location', and 'fault_impedance'.

    Side Effects
    ------------
    Creates 'scenario_metadata.json' in the current directory.
    """
    metadata = {}

    for sample_idx, floc, fz in scenarios:
        sid = str(uuid.uuid4())
        metadata[sid] = {
            "sample_idx": sample_idx,
            "fault_location": floc,
            "fault_impedance": fz,
        }

    with open("scenario_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata


# =============================================================================
# Simulation Worker
# =============================================================================

def _load_power_system(raw_file, dyr_file):
    """Load a fresh power-system model with dynamic data."""
    psys = load_psse(raw_file)
    add_dyr(psys, dyr_file)
    return psys


def _get_generator_bounds(psys):
    """Return generator active/reactive bounds when available."""
    try:
        pg_lb, pg_ub = psys.get_pgen_bounds()
    except AttributeError:
        pg_lb = pg_ub = None

    try:
        qg_lb, qg_ub = psys.get_qgen_bounds()
    except AttributeError:
        qg_lb = qg_ub = None

    return pg_lb, pg_ub, qg_lb, qg_ub


def _integration_config_from_dict(integration_config=None):
    """Build an IntegrationConfig from a user dictionary."""
    int_cfg = integration_config or {}
    return IntegrationConfig(
        tend=int_cfg.get("tend", 10.0),
        dt=int_cfg.get("dt", 1 / 120.0),
        power_injection=int_cfg.get("power_injection", False),
        ton=int_cfg.get("ton", 0.25),
        toff=int_cfg.get("toff", 0.4),
        verbose=int_cfg.get("verbose", False),
        petsc=int_cfg.get("petsc", True),
        enforce_q_limits=int_cfg.get("enforce_q_limits", True),
        q_limit_tolerance=int_cfg.get("q_limit_tolerance", 1e-8),
        max_q_limit_iterations=int_cfg.get("max_q_limit_iterations"),
        power_flow_validation=int_cfg.get("power_flow_validation", {}),
    )


def _simulation_file_path(scenario_id):
    """Return the standard output path for a scenario file."""
    return f"simulation_data/scenario_{scenario_id}.npz"


def _prepare_operating_point_candidate(
        raw_file, dyr_file, scenario, scenario_id,
        *, noise_type="normal", noise_var=0.1,
        global_seed=0,
        balance_generation=False,
        perturb_loads=True,
        perturb_gens=True,
        load_noise_type=None,
        gen_noise_type=None,
        load_noise_var=None,
        gen_noise_var=None,
        keep_power_factor=True,
        clamp_gens=True,
        load_scale=1.0,
        load_mean_shift=0.0,
        generation_dispatch_init="perturbed",
        operating_point_config=None):
    """Sample and PF-screen one candidate operating point."""
    op_cfg = _resolve_operating_point_config(operating_point_config)
    max_attempts = int(op_cfg["max_attempts_per_scenario"]) if op_cfg["enabled"] else 1
    load_noise_type = load_noise_type or noise_type
    gen_noise_type = gen_noise_type or noise_type
    load_noise_var = load_noise_var if load_noise_var is not None else noise_var
    gen_noise_var = gen_noise_var if gen_noise_var is not None else noise_var

    diagnostics = None
    psys = None
    accepted = False
    pL_scaled = qL_scaled = pL_noise = qL_noise = None
    pG_scaled = qG_scaled = pG_noise = qG_noise = None
    base_p_load = base_q_load = base_p_gen = base_q_gen = None
    pf_v_magnitudes = pf_v_angles = None
    attempt_diagnostics = []

    try:
        for attempt_idx in range(max_attempts):
            if psys is not None:
                del psys
                gc.collect()

            psys = _load_power_system(raw_file, dyr_file)
            base_p_load, base_q_load = psys.get_load_pq()
            base_p_gen, base_q_gen = psys.get_gen_pq()
            pg_lb, pg_ub, qg_lb, qg_ub = _get_generator_bounds(psys)

            ss = _scenario_seed_sequence(
                global_seed,
                scenario["sample_idx"],
                attempt_idx,
            )
            rng_load, rng_gen = [np.random.default_rng(s) for s in ss.spawn(2)]

            stressed_p_load, stressed_q_load, load_factor = _stress_load(
                base_p_load,
                base_q_load,
                load_scale=load_scale,
                load_mean_shift=load_mean_shift,
            )

            if perturb_loads and base_p_load.size:
                pL_scaled, qL_scaled, _, _ = generate_perturbations(
                    stressed_p_load,
                    stressed_q_load,
                    noise_type=load_noise_type,
                    var=load_noise_var,
                    rng=rng_load,
                    return_noise=True,
                    preserve_power_factor=keep_power_factor,
                )
            else:
                pL_scaled = np.array(stressed_p_load, copy=True)
                qL_scaled = np.array(stressed_q_load, copy=True)

            pL_noise = _relative_noise(pL_scaled, base_p_load)
            qL_noise = _relative_noise(qL_scaled, base_q_load)

            if generation_dispatch_init not in {"base", "perturbed"}:
                raise ValueError(
                    "generation_dispatch_init must be either 'base' or 'perturbed'"
                )

            if generation_dispatch_init == "perturbed" and perturb_gens and base_p_gen.size:
                pG_noise_raw = _draw_noise(
                    base_p_gen.shape,
                    noise_type=gen_noise_type,
                    var=gen_noise_var,
                    rng=rng_gen,
                )
                pG_scaled = base_p_gen * (1.0 + pG_noise_raw)
            else:
                pG_scaled = np.array(base_p_gen, copy=True)

            if clamp_gens and pg_lb is not None and pg_ub is not None and pG_scaled.size:
                pG_scaled = np.clip(pG_scaled, pg_lb, pg_ub)

            if balance_generation and not op_cfg["enabled"]:
                sum_pL = float(np.sum(pL_scaled))
                pG_scaled = _rebalance_active_power(pG_scaled, pg_lb, pg_ub, sum_pL)

            qG_scaled = _compute_generator_q_dispatch(
                base_p_gen, base_q_gen, pG_scaled, keep_power_factor=keep_power_factor
            )

            if clamp_gens and qg_lb is not None and qg_ub is not None and qG_scaled.size:
                qG_scaled = np.clip(qG_scaled, qg_lb, qg_ub)

            pf_solution = None
            diagnostics = {
                "record_type": "operating_point_attempt",
                "scenario_id": scenario_id,
                "operating_point_id": scenario.get("operating_point_id", scenario_id),
                "sample_idx": scenario["sample_idx"],
                "fault_location": scenario.get("fault_location"),
                "fault_impedance": scenario.get("fault_impedance"),
                "accepted_operating_point_index": scenario.get(
                    "accepted_operating_point_index"
                ),
                "attempt_idx": attempt_idx,
                "attempts": attempt_idx + 1,
                "accepted": True,
                "reject_reason": None,
                "load_scale": float(load_scale),
                "load_mean_shift": float(load_mean_shift),
                "load_factor": float(load_factor),
                "generation_dispatch_init": generation_dispatch_init,
                "total_p_load": float(np.sum(pL_scaled)),
                "total_q_load": float(np.sum(qL_scaled)),
                "total_p_gen_scheduled": float(np.sum(pG_scaled)),
                "operating_point_enabled": bool(op_cfg["enabled"]),
            }

            if op_cfg["enabled"]:
                pG_scaled, qG_scaled, pf_solution, op_diagnostics = _prepare_operating_point(
                    psys,
                    pL_scaled,
                    qL_scaled,
                    pG_scaled,
                    qG_scaled,
                    base_p_gen,
                    base_q_gen,
                    pg_lb,
                    pg_ub,
                    qg_lb,
                    qg_ub,
                    keep_power_factor,
                    clamp_gens,
                    balance_generation,
                    op_cfg,
                )
                diagnostics.update(op_diagnostics)
                diagnostics["record_type"] = "operating_point_attempt"
                diagnostics["scenario_id"] = scenario_id
                diagnostics["operating_point_id"] = scenario.get(
                    "operating_point_id", scenario_id
                )
                diagnostics["sample_idx"] = scenario["sample_idx"]
                diagnostics["fault_location"] = scenario.get("fault_location")
                diagnostics["fault_impedance"] = scenario.get("fault_impedance")
                diagnostics["accepted_operating_point_index"] = scenario.get(
                    "accepted_operating_point_index"
                )
                diagnostics["attempt_idx"] = attempt_idx
                diagnostics["attempts"] = attempt_idx + 1
                diagnostics["total_p_gen_scheduled"] = float(np.sum(pG_scaled))

                if not diagnostics["accepted"]:
                    attempt_diagnostics.append(_json_safe(copy.deepcopy(diagnostics)))
                    continue

                if pf_solution is not None:
                    pf_v_magnitudes = np.array(pf_solution.v_magnitudes, copy=True)
                    pf_v_angles = np.array(pf_solution.v_angles, copy=True)

            attempt_diagnostics.append(_json_safe(copy.deepcopy(diagnostics)))
            accepted = True
            break

        if not accepted:
            reject_reason = diagnostics.get("reject_reason", "operating_point_rejected")
            diagnostics.update({
                "accepted": False,
                "attempts": max_attempts,
                "reject_reason": reject_reason,
            })
            print(
                f"Rejected scenario {scenario_id} after {max_attempts} attempts: "
                f"{reject_reason}"
            )
            return {
                "file": None,
                "diverged": True,
                "rejected": True,
                "error": (
                    f"Could not rebalance scenario after {max_attempts} attempts; "
                    f"last reason: {reject_reason}"
                ),
                "diagnostics": _json_safe(diagnostics),
                "diagnostics_attempts": _json_safe(attempt_diagnostics),
            }

        pG_noise = _relative_noise(pG_scaled, base_p_gen)
        qG_noise = _relative_noise(qG_scaled, base_q_gen)

        operating_point = {
            "p_load_scaled": np.array(pL_scaled, copy=True),
            "q_load_scaled": np.array(qL_scaled, copy=True),
            "p_load_noise": np.array(pL_noise, copy=True),
            "q_load_noise": np.array(qL_noise, copy=True),
            "p_gen_scaled": np.array(pG_scaled, copy=True),
            "q_gen_scaled": np.array(qG_scaled, copy=True),
            "p_gen_noise": np.array(pG_noise, copy=True),
            "q_gen_noise": np.array(qG_noise, copy=True),
            "load_scale": float(load_scale),
            "load_mean_shift": float(load_mean_shift),
            "pf_v_magnitudes": pf_v_magnitudes,
            "pf_v_angles": pf_v_angles,
            "diagnostics": _json_safe(diagnostics),
            "operating_point_id": scenario.get("operating_point_id", scenario_id),
            "sample_idx": scenario["sample_idx"],
            "accepted_operating_point_index": scenario.get(
                "accepted_operating_point_index"
            ),
        }

        return {
            "file": None,
            "diverged": False,
            "rejected": False,
            "diagnostics": _json_safe(diagnostics),
            "diagnostics_attempts": _json_safe(attempt_diagnostics),
            "operating_point": operating_point,
        }

    finally:
        if psys is not None:
            del psys
        gc.collect()


def _run_fault_with_operating_point_worker(
        raw_file, dyr_file, scenario, scenario_id, operating_point,
        integration_config=None):
    """Run one dynamic fault simulation from a prepared operating point."""
    psys = None
    sim = None
    diagnostics = copy.deepcopy(operating_point.get("diagnostics", {}))
    diagnostics.update({
        "record_type": "fault_scenario",
        "scenario_id": scenario_id,
        "operating_point_id": scenario.get(
            "operating_point_id",
            operating_point.get("operating_point_id"),
        ),
        "sample_idx": scenario["sample_idx"],
        "fault_location": scenario["fault_location"],
        "fault_impedance": scenario["fault_impedance"],
        "accepted_operating_point_index": scenario.get(
            "accepted_operating_point_index",
            operating_point.get("accepted_operating_point_index"),
        ),
        "accepted": True,
        "reject_reason": None,
    })

    try:
        psys = _load_power_system(raw_file, dyr_file)
        p_load = np.asarray(operating_point["p_load_scaled"], dtype=float)
        q_load = np.asarray(operating_point["q_load_scaled"], dtype=float)
        p_gen = np.asarray(operating_point["p_gen_scaled"], dtype=float)
        q_gen = np.asarray(operating_point["q_gen_scaled"], dtype=float)

        psys.set_load_pq(p_load, q_load)
        psys.set_gen_pq(p_gen, q_gen)

        v_magnitudes = operating_point.get("pf_v_magnitudes")
        v_angles = operating_point.get("pf_v_angles")
        if v_magnitudes is not None and v_angles is not None:
            for bus_idx, bus in enumerate(psys.buses):
                bus.set_vinit(v_magnitudes[bus_idx], v_angles[bus_idx])

        psys.add_busfault(scenario["fault_location"], scenario["fault_impedance"])
        psys.createYbusComplex()

        try:
            sim = integrate_system(psys, _integration_config_from_dict(integration_config))
            diverged = False
            diagnostics["power_flow_validation"] = sim.get(
                "power_flow_diagnostics"
            )
        except Exception as e:
            print(f"Simulation failed for scenario {scenario_id}: {str(e)}")
            sim = {"history": None, "tvec": None}
            diverged = True
            diagnostics["simulation_error"] = str(e)
            if isinstance(e, PowerFlowValidationError):
                diagnostics["reject_reason"] = "power_flow_validation_failed"
                diagnostics["power_flow_validation"] = e.diagnostics

        diagnostics["simulation_diverged"] = diverged
        diagnostics["file"] = None if diverged else _simulation_file_path(scenario_id)

        os.makedirs("simulation_data", exist_ok=True)
        fn = _simulation_file_path(scenario_id)
        np.savez_compressed(
            fn,
            history=sim["history"],
            tvec=sim["tvec"],
            p_load_scaled=p_load,
            q_load_scaled=q_load,
            p_load_noise=np.asarray(operating_point["p_load_noise"], dtype=float),
            q_load_noise=np.asarray(operating_point["q_load_noise"], dtype=float),
            load_scale=float(operating_point["load_scale"]),
            load_mean_shift=float(operating_point["load_mean_shift"]),
            p_gen_scaled=p_gen,
            q_gen_scaled=q_gen,
            p_gen_noise=np.asarray(operating_point["p_gen_noise"], dtype=float),
            q_gen_noise=np.asarray(operating_point["q_gen_noise"], dtype=float),
            operating_point_id=str(diagnostics.get("operating_point_id")),
            accepted_operating_point_index=diagnostics.get(
                "accepted_operating_point_index"
            ),
        )

        diagnostics["file"] = fn
        return {
            "file": fn,
            "diverged": diverged,
            "rejected": False,
            "diagnostics": _json_safe(diagnostics),
        }

    except Exception as e:
        print(f"Worker error for scenario {scenario_id}: {str(e)}")
        traceback.print_exc()
        diagnostics.update({
            "accepted": False,
            "reject_reason": "worker_error",
            "simulation_error": str(e),
            "simulation_diverged": True,
            "file": None,
        })
        return {
            "file": None,
            "diverged": True,
            "rejected": False,
            "error": str(e),
            "diagnostics": _json_safe(diagnostics),
        }
    finally:
        if sim is not None:
            del sim
        if psys is not None:
            del psys
        gc.collect()


def run_single_scenario_worker(
        raw_file, dyr_file, scenario, scenario_id,
        noise_type="normal", noise_var=0.1,
        global_seed=0,
        balance_generation=False,
        perturb_loads=True,
        perturb_gens=True,
        load_noise_type=None,
        gen_noise_type=None,
        load_noise_var=None,
        gen_noise_var=None,
        keep_power_factor=True,
        clamp_gens=True,
        load_scale=1.0,
        load_mean_shift=0.0,
        generation_dispatch_init="perturbed",
        operating_point_config=None,
        integration_config=None):
    r"""
    Worker function for single-scenario simulation.

    Generates a perturbed operating point and runs the transient stability
    simulation for one fault scenario. This function is designed to be
    called in parallel worker processes.

    Parameters
    ----------
    raw_file : str
        Path to PSS/E RAW file containing network data.
    dyr_file : str
        Path to PSS/E DYR file containing dynamic model data.
    scenario : dict
        Scenario parameters with keys 'sample_idx', 'fault_location',
        and 'fault_impedance'.
    scenario_id : str
        Unique identifier for this scenario.
    noise_type : str, default='normal'
        Default noise distribution type for both loads and generators.
    noise_var : float, default=0.1
        Default noise variance for both loads and generators.
    global_seed : int, default=0
        Base seed for random number generation. Combined with sample_idx
        to create reproducible but independent random streams.
    balance_generation : bool, default=False
        If True, rebalance total generation to match total load after
        applying perturbations.
    perturb_loads : bool, default=True
        If True, apply perturbations to load P and Q values.
    perturb_gens : bool, default=True
        If True, apply perturbations to generator P values.
    load_noise_type : str, optional
        Override noise type for loads. If None, uses ``noise_type``.
    gen_noise_type : str, optional
        Override noise type for generators. If None, uses ``noise_type``.
    load_noise_var : float, optional
        Override noise variance for loads. If None, uses ``noise_var``.
    gen_noise_var : float, optional
        Override noise variance for generators. If None, uses ``noise_var``.
    keep_power_factor : bool, default=True
        If True, adjust Q to maintain original Q/P ratio when perturbing.
    clamp_gens : bool, default=True
        If True, clamp generator P and Q to their operational limits.
    integration_config : dict, optional
        Integration parameters for the dynamic simulation. If None, uses
        default values. Supported keys:

        - 'tend': Simulation end time in seconds (default: 10.0)
        - 'dt': Time step in seconds (default: 1/120.0)
        - 'power_injection': Use power injection model (default: False)
        - 'ton': Fault onset time in seconds (default: 0.25)
        - 'toff': Fault clearing time in seconds (default: 0.4)
        - 'verbose': Enable verbose output (default: False)
        - 'petsc': Use PETSc solver (default: True)

    Returns
    -------
    dict
        Result dictionary containing:

        - 'file': str or None, path to saved results
        - 'diverged': bool, True if simulation failed to converge
        - 'error': str (only present if an exception occurred)

    Notes
    -----
    **Perturbation Pipeline**:

    1. Load base values: $(P_L, Q_L, P_G, Q_G)$ from power system model
    2. Retrieve generator limits: $(P_G^{min}, P_G^{max}, Q_G^{min}, Q_G^{max})$
    3. Initialize independent RNG streams for loads and generators
    4. Perturb loads: $P_L' = P_L(1+\epsilon_L)$, preserve power factor for $Q_L'$
    5. Perturb generators: $P_G' = P_G(1+\epsilon_G)$
    6. Clamp: $P_G' = \text{clip}(P_G', P_G^{min}, P_G^{max})$
    7. Rebalance: adjust $P_G'$ so $\sum P_G' = \sum P_L'$
    8. Compute $Q_G'$ from power factor (if enabled)
    9. Clamp: $Q_G' = \text{clip}(Q_G', Q_G^{min}, Q_G^{max})$
    10. Run dynamic simulation with fault

    **Output File Contents** (simulation_data/scenario_*.npz):

    - history: State variable time series (n_states, n_timesteps)
    - tvec: Time vector
    - p_load_scaled, q_load_scaled: Perturbed load values
    - p_gen_scaled, q_gen_scaled: Perturbed generator values
    - p_load_noise, q_load_noise: Load perturbation factors
    - p_gen_noise, q_gen_noise: Generator perturbation factors

    **Random Number Generation**:

    Uses numpy's SeedSequence to create independent RNG streams for loads
    and generators from a combined (global_seed, sample_idx) seed. This
    ensures reproducibility while avoiding correlation between load and
    generator perturbations.
    """
    try:
        op_cfg = _resolve_operating_point_config(operating_point_config)
        max_attempts = int(op_cfg["max_attempts_per_scenario"]) if op_cfg["enabled"] else 1
        load_noise_type = load_noise_type or noise_type
        gen_noise_type = gen_noise_type or noise_type
        load_noise_var = load_noise_var if load_noise_var is not None else noise_var
        gen_noise_var = gen_noise_var if gen_noise_var is not None else noise_var

        diagnostics = None
        psys = None
        pL_scaled = qL_scaled = pL_noise = qL_noise = None
        pG_scaled = qG_scaled = pG_noise = qG_noise = None
        base_p_load = base_q_load = base_p_gen = base_q_gen = None
        accepted = False
        attempt_diagnostics = []

        for attempt_idx in range(max_attempts):
            if psys is not None:
                del psys
                gc.collect()

            # Load fresh model in worker process to avoid MPI communicator issues
            psys = load_psse(raw_file)
            add_dyr(psys, dyr_file)

            # Get base load and generation setpoints
            base_p_load, base_q_load = psys.get_load_pq()
            base_p_gen, base_q_gen = psys.get_gen_pq()

            # Retrieve generator limits (may not exist in older model versions)
            try:
                pg_lb, pg_ub = psys.get_pgen_bounds()
            except AttributeError:
                pg_lb = pg_ub = None

            try:
                qg_lb, qg_ub = psys.get_qgen_bounds()
            except AttributeError:
                qg_lb = qg_ub = None

            # Create independent RNG streams for loads and generators
            ss = _scenario_seed_sequence(global_seed, scenario["sample_idx"], attempt_idx)
            rng_load, rng_gen = [np.random.default_rng(s) for s in ss.spawn(2)]

            # -----------------------------------------------------------------
            # Step 1: Apply deterministic load stress and stochastic perturbation
            # -----------------------------------------------------------------
            stressed_p_load, stressed_q_load, load_factor = _stress_load(
                base_p_load,
                base_q_load,
                load_scale=load_scale,
                load_mean_shift=load_mean_shift,
            )

            if perturb_loads and base_p_load.size:
                pL_scaled, qL_scaled, _, _ = generate_perturbations(
                    stressed_p_load, stressed_q_load,
                    noise_type=load_noise_type, var=load_noise_var, rng=rng_load,
                    return_noise=True, preserve_power_factor=keep_power_factor
                )
            else:
                pL_scaled = np.array(stressed_p_load, copy=True)
                qL_scaled = np.array(stressed_q_load, copy=True)

            pL_noise = _relative_noise(pL_scaled, base_p_load)
            qL_noise = _relative_noise(qL_scaled, base_q_load)

            # -----------------------------------------------------------------
            # Step 2: Initialize generator active power
            # -----------------------------------------------------------------
            if generation_dispatch_init not in {"base", "perturbed"}:
                raise ValueError(
                    "generation_dispatch_init must be either 'base' or 'perturbed'"
                )

            if generation_dispatch_init == "perturbed" and perturb_gens and base_p_gen.size:
                pG_noise_raw = _draw_noise(
                    base_p_gen.shape,
                    noise_type=gen_noise_type,
                    var=gen_noise_var,
                    rng=rng_gen,
                )
                pG_scaled = base_p_gen * (1.0 + pG_noise_raw)
            else:
                pG_scaled = np.array(base_p_gen, copy=True)

            # -----------------------------------------------------------------
            # Step 3: Clamp generator active power to limits
            # -----------------------------------------------------------------
            if clamp_gens and pg_lb is not None and pg_ub is not None and pG_scaled.size:
                pG_scaled = np.clip(pG_scaled, pg_lb, pg_ub)

            # -----------------------------------------------------------------
            # Step 4: Legacy or PF-aware active-power rebalancing
            # -----------------------------------------------------------------
            if balance_generation and not op_cfg["enabled"]:
                sum_pL = float(np.sum(pL_scaled))
                pG_scaled = _rebalance_active_power(pG_scaled, pg_lb, pg_ub, sum_pL)

            qG_scaled = _compute_generator_q_dispatch(
                base_p_gen, base_q_gen, pG_scaled, keep_power_factor=keep_power_factor
            )

            if clamp_gens and qg_lb is not None and qg_ub is not None and qG_scaled.size:
                qG_scaled = np.clip(qG_scaled, qg_lb, qg_ub)

            pf_solution = None
            diagnostics = {
                "scenario_id": scenario_id,
                "sample_idx": scenario["sample_idx"],
                "fault_location": scenario["fault_location"],
                "fault_impedance": scenario["fault_impedance"],
                "attempt_idx": attempt_idx,
                "attempts": attempt_idx + 1,
                "accepted": True,
                "reject_reason": None,
                "load_scale": float(load_scale),
                "load_mean_shift": float(load_mean_shift),
                "load_factor": float(load_factor),
                "generation_dispatch_init": generation_dispatch_init,
                "total_p_load": float(np.sum(pL_scaled)),
                "total_q_load": float(np.sum(qL_scaled)),
                "total_p_gen_scheduled": float(np.sum(pG_scaled)),
                "operating_point_enabled": bool(op_cfg["enabled"]),
            }

            if op_cfg["enabled"]:
                pG_scaled, qG_scaled, pf_solution, op_diagnostics = _prepare_operating_point(
                    psys,
                    pL_scaled,
                    qL_scaled,
                    pG_scaled,
                    qG_scaled,
                    base_p_gen,
                    base_q_gen,
                    pg_lb,
                    pg_ub,
                    qg_lb,
                    qg_ub,
                    keep_power_factor,
                    clamp_gens,
                    balance_generation,
                    op_cfg,
                )
                diagnostics.update(op_diagnostics)
                diagnostics["attempt_idx"] = attempt_idx
                diagnostics["attempts"] = attempt_idx + 1
                diagnostics["total_p_gen_scheduled"] = float(np.sum(pG_scaled))

                if not diagnostics["accepted"]:
                    attempt_diagnostics.append(_json_safe(copy.deepcopy(diagnostics)))
                    continue

                if pf_solution is not None:
                    for bus_idx, bus in enumerate(psys.buses):
                        bus.set_vinit(
                            pf_solution.v_magnitudes[bus_idx],
                            pf_solution.v_angles[bus_idx],
                        )

            attempt_diagnostics.append(_json_safe(copy.deepcopy(diagnostics)))
            accepted = True
            break

        if not accepted:
            reject_reason = diagnostics.get("reject_reason", "operating_point_rejected")
            diagnostics.update({
                "accepted": False,
                "attempts": max_attempts,
                "reject_reason": reject_reason,
            })
            print(
                f"Rejected scenario {scenario_id} after {max_attempts} attempts: "
                f"{reject_reason}"
            )
            if psys is not None:
                del psys
            gc.collect()
            return {
                "file": None,
                "diverged": True,
                "rejected": True,
                "error": (
                    f"Could not rebalance scenario after {max_attempts} attempts; "
                    f"last reason: {reject_reason}"
                ),
                "diagnostics": _json_safe(diagnostics),
                "diagnostics_attempts": _json_safe(attempt_diagnostics),
            }

        # Compute effective perturbation factors for logging
        pG_noise = _relative_noise(pG_scaled, base_p_gen)
        qG_noise = _relative_noise(qG_scaled, base_q_gen)

        # -----------------------------------------------------------------
        # Step 7: Run dynamic simulation
        # -----------------------------------------------------------------
        psys.set_load_pq(pL_scaled, qL_scaled)
        psys.set_gen_pq(pG_scaled, qG_scaled)

        # Apply fault at specified location
        psys.add_busfault(scenario["fault_location"], scenario["fault_impedance"])
        psys.createYbusComplex()

        # Configure integration parameters
        int_cfg = integration_config or {}
        cfg = IntegrationConfig(
            tend=int_cfg.get('tend', 10.0),           # Simulation end time [s]
            dt=int_cfg.get('dt', 1/120.0),            # Time step [s]
            power_injection=int_cfg.get('power_injection', False),
            ton=int_cfg.get('ton', 0.25),             # Fault onset time [s]
            toff=int_cfg.get('toff', 0.4),            # Fault clearing time [s]
            verbose=int_cfg.get('verbose', False),
            petsc=int_cfg.get('petsc', True),
            enforce_q_limits=int_cfg.get('enforce_q_limits', True),
            q_limit_tolerance=int_cfg.get('q_limit_tolerance', 1e-8),
            max_q_limit_iterations=int_cfg.get('max_q_limit_iterations'),
            power_flow_validation=int_cfg.get('power_flow_validation', {}),
        )

        try:
            sim = integrate_system(psys, cfg)
            diverged = False
            diagnostics["power_flow_validation"] = sim.get(
                "power_flow_diagnostics"
            )
        except Exception as e:
            print(f"Simulation failed for scenario {scenario_id}: {str(e)}")
            sim = {"history": None, "tvec": None}
            diverged = True
            diagnostics["simulation_error"] = str(e)
            if isinstance(e, PowerFlowValidationError):
                diagnostics["reject_reason"] = "power_flow_validation_failed"
                diagnostics["power_flow_validation"] = e.diagnostics

        diagnostics["simulation_diverged"] = diverged
        diagnostics["file"] = None if diverged else f"simulation_data/scenario_{scenario_id}.npz"

        # -----------------------------------------------------------------
        # Step 8: Save results
        # -----------------------------------------------------------------
        os.makedirs("simulation_data", exist_ok=True)
        fn = f"simulation_data/scenario_{scenario_id}.npz"

        np.savez_compressed(
            fn,
            # Time series data
            history=sim["history"],
            tvec=sim["tvec"],
            # Load values and perturbations
            p_load_scaled=pL_scaled, q_load_scaled=qL_scaled,
            p_load_noise=pL_noise, q_load_noise=qL_noise,
            load_scale=float(load_scale),
            load_mean_shift=float(load_mean_shift),
            # Generator values and perturbations
            p_gen_scaled=pG_scaled, q_gen_scaled=qG_scaled,
            p_gen_noise=pG_noise, q_gen_noise=qG_noise,
        )

        diagnostics["file"] = fn

        # Clean up resources (important for PETSc/MPI)
        del sim
        del psys
        gc.collect()

        return {
            "file": fn,
            "diverged": diverged,
            "rejected": False,
            "diagnostics": _json_safe(diagnostics),
            "diagnostics_attempts": _json_safe(attempt_diagnostics),
        }

    except Exception as e:
        print(f"Worker error for scenario {scenario_id}: {str(e)}")
        traceback.print_exc()
        return {
            "file": None,
            "diverged": True,
            "error": str(e),
            "diagnostics_attempts": _json_safe(
                locals().get("attempt_diagnostics", [])
            ),
        }


def _write_diagnostics_records(path, records):
    """Append diagnostics records as JSONL."""
    if not path or not records:
        return
    with open(path, "a") as f:
        for record in records:
            f.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


def _load_diagnostics_records(path):
    """Load existing JSONL diagnostics records when continuing a run."""
    if not path or not os.path.exists(path):
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _diagnostic_records_from_output(output):
    """Return all diagnostics records carried by a worker output."""
    if not isinstance(output, dict):
        return []
    records = output.get("diagnostics_attempts")
    if records:
        return records
    record = output.get("diagnostics")
    return [record] if record else []


def _summarize_diagnostics(records):
    """Create a compact diagnostics summary from simulation-log records."""
    records = [record for record in records if record]
    accepted = [record for record in records if record.get("accepted")]
    rejected = [record for record in records if not record.get("accepted")]
    reasons = Counter(record.get("reject_reason") or "accepted" for record in records)

    def _finite_values(key):
        vals = []
        for record in records:
            val = record.get(key)
            if val is not None and np.isfinite(val):
                vals.append(float(val))
        return vals

    summary = {
        "total_records": len(records),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": (len(accepted) / len(records)) if records else None,
        "reject_reasons": dict(reasons),
    }

    for key in [
        "attempts",
        "pf_residual",
        "voltage_min",
        "voltage_max",
        "slack_p_deviation",
        "slack_q_deviation",
        "slack_p_limit_violation",
        "gen_p_violation_max",
        "gen_q_violation_max",
        "gen_q_violation_count",
        "gen_q_violation_total_abs",
        "branch_loading_max",
        "initial_rebalance_unresolved_pu",
        "loss_compensation_requested_pu",
        "loss_compensation_applied_pu",
        "loss_compensation_unresolved_pu",
        "loss_compensation_unresolved_abs_pu",
        "loss_compensation_iterations",
        "loss_compensation_effective_tolerance_pu",
        "q_limit_mitigation_passes",
        "non_slack_headroom_remaining",
        "non_slack_footroom_remaining",
        "target_accepted_scenarios",
        "total_candidate_attempts",
        "max_total_attempts",
        "faults_required",
        "faults_attempted",
        "faults_successful",
        "fault_files_saved",
    ]:
        vals = _finite_values(key)
        if vals:
            summary[key] = {
                "min": float(np.min(vals)),
                "mean": float(np.mean(vals)),
                "max": float(np.max(vals)),
            }

    offender_rollup = {}
    for record in records:
        for offender in record.get("gen_q_violation_top", []) or []:
            key = (str(offender.get("bus_id")), str(offender.get("gen_id")))
            entry = offender_rollup.setdefault(
                key,
                {
                    "bus_id": offender.get("bus_id"),
                    "gen_id": offender.get("gen_id"),
                    "bus_index": offender.get("bus_index"),
                    "gen_index": offender.get("gen_index"),
                    "bus_type": offender.get("bus_type"),
                    "is_slack": offender.get("is_slack"),
                    "count": 0,
                    "total_abs_violation": 0.0,
                    "max_violation": 0.0,
                },
            )
            violation = float(offender.get("violation") or 0.0)
            entry["count"] += 1
            entry["total_abs_violation"] += violation
            entry["max_violation"] = max(entry["max_violation"], violation)

    if offender_rollup:
        top_offenders = sorted(
            offender_rollup.values(),
            key=lambda entry: (
                entry["max_violation"],
                entry["total_abs_violation"],
                entry["count"],
            ),
            reverse=True,
        )[:10]
        for offender in top_offenders:
            offender["mean_abs_violation"] = (
                offender["total_abs_violation"] / offender["count"]
            )
        summary["gen_q_violation_top_offenders"] = top_offenders

    return summary


def _write_diagnostics_summary(path, simulation_log):
    """Write summary JSON for diagnostics already stored in the simulation log."""
    if not path:
        return
    records = [
        entry.get("diagnostics")
        for entry in simulation_log.values()
        if isinstance(entry, dict) and entry.get("diagnostics")
    ]
    with open(path, "w") as f:
        json.dump(_json_safe(_summarize_diagnostics(records)), f, indent=4)


def _write_diagnostics_summary_records(path, records):
    """Write summary JSON from an explicit diagnostics record list."""
    if not path:
        return
    with open(path, "w") as f:
        json.dump(_json_safe(_summarize_diagnostics(records)), f, indent=4)


def _print_batch_diagnostics(batch_out):
    """Print compact acceptance/rejection and stress diagnostics for a batch."""
    records = [
        out.get("diagnostics")
        for out in batch_out
        if isinstance(out, dict) and out.get("diagnostics")
    ]
    if not records:
        return

    summary = _summarize_diagnostics(records)
    accepted = summary["accepted"]
    total = summary["total_records"]
    rejected = summary["rejected"]
    acceptance_rate = summary["acceptance_rate"] or 0.0
    reasons = Counter(
        record.get("reject_reason") or "accepted"
        for record in records
        if not record.get("accepted")
    )
    top_reasons = ", ".join(
        f"{reason}={count}" for reason, count in reasons.most_common(3)
    ) or "none"

    def _max_text(key, fmt="{:.3e}"):
        stats = summary.get(key)
        if not stats:
            return "n/a"
        return fmt.format(stats["max"])

    message = (
        "Batch diagnostics | "
        f"accepted {accepted}/{total} ({acceptance_rate:.1%}) | "
        f"rejected {rejected} | top rejects: {top_reasons} | "
        f"avg attempts: {summary.get('attempts', {}).get('mean', 1.0):.2f} | "
        f"max PF residual: {_max_text('pf_residual')} | "
        f"voltage min/max: {_max_text('voltage_min', '{:.4f}')}/"
        f"{_max_text('voltage_max', '{:.4f}')} | "
        f"max branch loading: {_max_text('branch_loading_max', '{:.3f}')}"
    )
    unresolved_loss = summary.get("loss_compensation_unresolved_abs_pu")
    if unresolved_loss and unresolved_loss["max"] > 1e-9:
        message += (
            " | max unresolved loss comp: "
            f"{unresolved_loss['max']:.3e}"
        )
    q_violation = summary.get("gen_q_violation_max")
    if q_violation and q_violation["max"] > 0.0:
        message += f" | max Q viol: {q_violation['max']:.3e}"
        top_offenders = summary.get("gen_q_violation_top_offenders") or []
        if top_offenders:
            worst = top_offenders[0]
            message += (
                f" at gen {worst.get('gen_id')} "
                f"bus {worst.get('bus_id')}"
            )
    print(message)


def _build_target_fault_metadata(
        sample_idx, fault_locations, fault_impedances, *,
        operating_point_id, accepted_operating_point_index):
    """Create fault-level metadata for one accepted operating point."""
    metadata = {}
    for fault_location, fault_impedance in itertools.product(
            fault_locations, fault_impedances):
        scenario_id = str(uuid.uuid4())
        metadata[scenario_id] = {
            "sample_idx": sample_idx,
            "fault_location": fault_location,
            "fault_impedance": fault_impedance,
            "operating_point_id": operating_point_id,
            "accepted_operating_point_index": accepted_operating_point_index,
        }
    return metadata


def _write_metadata_file(metadata, metadata_file="scenario_metadata.json"):
    """Write scenario metadata to disk."""
    with open(metadata_file, "w") as f:
        json.dump(_json_safe(metadata), f, indent=4)


def _successful_fault_output(out):
    """Return True when a fault simulation produced a usable trajectory file."""
    return (
        isinstance(out, dict)
        and not out.get("rejected", False)
        and not out.get("diverged", False)
        and bool(out.get("file"))
        and os.path.exists(out["file"])
    )


def _successful_log_fault(entry):
    """Return True when a simulation-log row represents a usable fault file."""
    return (
        isinstance(entry, dict)
        and not entry.get("rejected", False)
        and not entry.get("diverged", False)
        and bool(entry.get("file"))
        and os.path.exists(entry["file"])
    )


def _existing_complete_operating_point_count(
        simulation_log, fault_locations, fault_impedances):
    """Count complete accepted target-mode operating-point groups in a log."""
    expected_faults = {
        (fault_location, fault_impedance)
        for fault_location, fault_impedance in itertools.product(
            fault_locations,
            fault_impedances,
        )
    }
    if not expected_faults:
        return 0, 0

    grouped = {}
    max_existing_index = -1
    for entry in simulation_log.values():
        if not _successful_log_fault(entry):
            continue
        operating_point_id = entry.get("operating_point_id")
        if operating_point_id is None:
            continue
        fault_key = (entry.get("fault_location"), entry.get("fault_impedance"))
        if fault_key not in expected_faults:
            continue
        grouped.setdefault(operating_point_id, set()).add(fault_key)
        accepted_index = entry.get("accepted_operating_point_index")
        if accepted_index is not None:
            max_existing_index = max(max_existing_index, int(accepted_index))

    complete_count = sum(
        1 for completed_faults in grouped.values()
        if expected_faults.issubset(completed_faults)
    )
    next_accepted_index = max(max_existing_index + 1, complete_count)
    return complete_count, next_accepted_index


def _remove_fault_outputs(batch_out):
    """Remove partial files from an incomplete target-mode operating point."""
    for out in batch_out:
        if not isinstance(out, dict):
            continue
        path = out.get("file")
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _print_target_mode_progress(
        *, accepted_operating_points, target_accepted_scenarios,
        total_candidate_attempts, max_total_attempts,
        fault_files_saved, last_record, start_time):
    """Print compact progress for target accepted operating-point mode."""
    elapsed = time.time() - start_time
    acceptance_rate = (
        accepted_operating_points / total_candidate_attempts
        if total_candidate_attempts else 0.0
    )
    if accepted_operating_points:
        attempts_per_accept = total_candidate_attempts / accepted_operating_points
        remaining_accepted = target_accepted_scenarios - accepted_operating_points
        estimated_remaining_attempts = remaining_accepted * attempts_per_accept
        estimated_total = (
            elapsed
            * (total_candidate_attempts + estimated_remaining_attempts)
            / max(total_candidate_attempts, 1)
        )
        eta = max(0.0, estimated_total - elapsed)
    else:
        eta = 0.0

    reason = last_record.get("reject_reason") or "accepted"
    print(
        "Target mode | "
        f"accepted OPs {accepted_operating_points}/{target_accepted_scenarios} | "
        f"candidate attempts {total_candidate_attempts}/{max_total_attempts} | "
        f"OP acceptance/attempt {acceptance_rate:.1%} | "
        f"fault files saved {fault_files_saved} | "
        f"last: {reason} | "
        f"elapsed {timedelta(seconds=int(elapsed))} | "
        f"ETA {timedelta(seconds=int(eta))}"
    )


def _run_target_accepted_driver(
        raw, dyr, *,
        target_accepted_scenarios,
        max_total_attempts,
        sample_idx_start,
        fault_locations,
        fault_impedances,
        noise_type="normal",
        noise_var=0.1,
        balance_generation=True,
        n_jobs=-1,
        perturb_loads=True,
        perturb_gens=True,
        load_noise_type=None,
        gen_noise_type=None,
        load_noise_var=None,
        gen_noise_var=None,
        keep_power_factor=True,
        clamp_gens=True,
        load_scale=1.0,
        load_mean_shift=0.0,
        generation_dispatch_init="perturbed",
        operating_point_config=None,
        existing_log=None,
        existing_metadata=None,
        integration_config=None):
    """Run until the requested number of complete operating-point groups exists."""
    op_cfg = _resolve_operating_point_config(operating_point_config)
    target_accepted_scenarios = int(target_accepted_scenarios)
    max_total_attempts = int(max_total_attempts)
    if target_accepted_scenarios <= 0:
        raise ValueError("target_accepted_scenarios must be positive")
    if max_total_attempts <= 0:
        raise ValueError("max_total_attempts must be positive")

    diagnostics_file = op_cfg.get("diagnostics_file")
    diagnostics_summary_file = op_cfg.get("diagnostics_summary_file")
    if diagnostics_file and not existing_log:
        open(diagnostics_file, "w").close()
    if diagnostics_summary_file and not existing_log:
        _write_diagnostics_summary_records(diagnostics_summary_file, [])
    diagnostics_records = (
        _load_diagnostics_records(diagnostics_file)
        if diagnostics_file and existing_log
        else []
    )

    simulation_log = dict(existing_log) if existing_log else {}
    scenario_metadata = dict(existing_metadata) if existing_metadata else {}
    _write_metadata_file(scenario_metadata)

    base_psys = _load_power_system(raw, dyr)
    base_psys.export_state_metadata()
    del base_psys

    group_records = list(diagnostics_records)
    accepted_operating_points, next_accepted_index = (
        _existing_complete_operating_point_count(
            simulation_log,
            fault_locations,
            fault_impedances,
        )
        if existing_log
        else (0, 0)
    )
    total_candidate_attempts = 0
    fault_files_saved = sum(
        1 for entry in simulation_log.values()
        if isinstance(entry, dict) and entry.get("file")
    )
    sample_idx = int(sample_idx_start)
    start_time = time.time()
    fault_count = len(fault_locations) * len(fault_impedances)

    while (
        accepted_operating_points < target_accepted_scenarios
        and total_candidate_attempts < max_total_attempts
    ):
        remaining_attempts = max_total_attempts - total_candidate_attempts
        candidate_op_cfg = dict(op_cfg)
        candidate_op_cfg["max_attempts_per_scenario"] = min(
            int(op_cfg["max_attempts_per_scenario"]),
            remaining_attempts,
        )

        operating_point_id = str(uuid.uuid4())
        candidate_scenario = {
            "sample_idx": sample_idx,
            "operating_point_id": operating_point_id,
            "accepted_operating_point_index": next_accepted_index,
        }
        prep = _prepare_operating_point_candidate(
            raw,
            dyr,
            candidate_scenario,
            operating_point_id,
            noise_type=noise_type,
            noise_var=noise_var,
            global_seed=1234,
            balance_generation=balance_generation,
            perturb_loads=perturb_loads,
            perturb_gens=perturb_gens,
            load_noise_type=load_noise_type,
            gen_noise_type=gen_noise_type,
            load_noise_var=load_noise_var,
            gen_noise_var=gen_noise_var,
            keep_power_factor=keep_power_factor,
            clamp_gens=clamp_gens,
            load_scale=load_scale,
            load_mean_shift=load_mean_shift,
            generation_dispatch_init=generation_dispatch_init,
            operating_point_config=candidate_op_cfg,
        )

        prep_diag = prep.get("diagnostics") or {}
        candidate_attempts = int(prep_diag.get("attempts") or 1)
        total_candidate_attempts += candidate_attempts
        group_record = copy.deepcopy(prep_diag)
        group_record.update({
            "record_type": "operating_point_group",
            "operating_point_id": operating_point_id,
            "sample_idx": sample_idx,
            "accepted_operating_point_index": next_accepted_index,
            "target_accepted_scenarios": target_accepted_scenarios,
            "total_candidate_attempts": total_candidate_attempts,
            "max_total_attempts": max_total_attempts,
            "faults_required": fault_count,
            "faults_attempted": 0,
            "faults_successful": 0,
            "fault_files_saved": fault_files_saved,
        })

        if prep.get("rejected") or not prep.get("operating_point"):
            group_record.update({
                "accepted": False,
                "reject_reason": prep_diag.get(
                    "reject_reason", "operating_point_rejected"
                ),
            })
            prep_attempt_records = prep.get("diagnostics_attempts") or [prep_diag]
            records_to_write = [
                *prep_attempt_records,
                _json_safe(group_record),
            ]
            group_records.extend(_json_safe(records_to_write))
            _write_diagnostics_records(diagnostics_file, records_to_write)
            _write_diagnostics_summary_records(diagnostics_summary_file, group_records)
            _print_target_mode_progress(
                accepted_operating_points=accepted_operating_points,
                target_accepted_scenarios=target_accepted_scenarios,
                total_candidate_attempts=total_candidate_attempts,
                max_total_attempts=max_total_attempts,
                fault_files_saved=fault_files_saved,
                last_record=group_record,
                start_time=start_time,
            )
            sample_idx += 1
            continue

        fault_metadata = _build_target_fault_metadata(
            sample_idx,
            fault_locations,
            fault_impedances,
            operating_point_id=operating_point_id,
            accepted_operating_point_index=next_accepted_index,
        )
        fault_ids = list(fault_metadata.keys())
        fault_args = [
            (raw, dyr, fault_metadata[sid], sid, prep["operating_point"], integration_config)
            for sid in fault_ids
        ]

        try:
            batch_out = Parallel(n_jobs=n_jobs, backend="loky", timeout=600)(
                delayed(_run_fault_with_operating_point_worker)(*args)
                for args in fault_args
            )
        except Exception as e:
            print(
                f"Fault group failed for operating point {operating_point_id}: {e}"
            )
            traceback.print_exc()
            batch_out = [
                {
                    "file": None,
                    "diverged": True,
                    "rejected": False,
                    "error": f"Fault group failure: {str(e)}",
                    "diagnostics": {
                        "record_type": "fault_scenario",
                        "accepted": False,
                        "reject_reason": "fault_group_error",
                        "simulation_error": str(e),
                    },
                }
                for _ in fault_ids
            ]

        faults_successful = sum(_successful_fault_output(out) for out in batch_out)
        group_accepted = faults_successful == fault_count
        group_record.update({
            "faults_attempted": fault_count,
            "faults_successful": faults_successful,
            "fault_files_saved": fault_files_saved + faults_successful,
            "accepted": group_accepted,
            "reject_reason": None if group_accepted else "fault_simulation_failed",
        })

        if group_accepted:
            for sid, out in zip(fault_ids, batch_out):
                simulation_log[sid] = _json_safe({**fault_metadata[sid], **out})
            scenario_metadata.update(fault_metadata)
            accepted_operating_points += 1
            next_accepted_index += 1
            fault_files_saved += faults_successful
            _write_metadata_file(scenario_metadata)
            with open("simulation_log.json", "w") as f:
                json.dump(_json_safe(simulation_log), f, indent=4)
        else:
            _remove_fault_outputs(batch_out)
            failure_reasons = Counter(
                (out.get("diagnostics") or {}).get("reject_reason")
                or out.get("error")
                or "simulation_diverged"
                for out in batch_out
                if not _successful_fault_output(out)
            )
            group_record["fault_failure_reasons"] = dict(failure_reasons)

        prep_attempt_records = prep.get("diagnostics_attempts") or [prep_diag]
        records_to_write = [
            *prep_attempt_records,
            _json_safe(group_record),
        ]
        group_records.extend(_json_safe(records_to_write))
        _write_diagnostics_records(diagnostics_file, records_to_write)
        _write_diagnostics_summary_records(diagnostics_summary_file, group_records)
        _print_target_mode_progress(
            accepted_operating_points=accepted_operating_points,
            target_accepted_scenarios=target_accepted_scenarios,
            total_candidate_attempts=total_candidate_attempts,
            max_total_attempts=max_total_attempts,
            fault_files_saved=fault_files_saved,
            last_record=group_record,
            start_time=start_time,
        )

        sample_idx += 1
        gc.collect()

    if accepted_operating_points < target_accepted_scenarios:
        print(
            "Target mode stopped before reaching requested accepted operating "
            f"points: accepted {accepted_operating_points}/"
            f"{target_accepted_scenarios} after {total_candidate_attempts} "
            f"candidate attempts."
        )
    else:
        print(
            "Target mode complete: accepted "
            f"{accepted_operating_points}/{target_accepted_scenarios} "
            f"operating points and saved {fault_files_saved} fault simulations."
        )

    return simulation_log


# =============================================================================
# Batch Simulation Driver
# =============================================================================

def run_simulation_driver_batched(
        raw, dyr, scenarios_metadata,
        *, noise_type="normal", noise_var=0.1,
        balance_generation=True,
        n_jobs=-1, batch_size=10,
        checkpoint_interval=100,
        perturb_loads=True,
        perturb_gens=True,
        load_noise_type=None,
        gen_noise_type=None,
        load_noise_var=None,
        gen_noise_var=None,
        keep_power_factor=True,
        clamp_gens=True,
        load_scale=1.0,
        load_mean_shift=0.0,
        generation_dispatch_init="perturbed",
        operating_point_config=None,
        existing_log=None,
        integration_config=None,
        target_accepted_scenarios=None,
        max_total_attempts=None,
        sample_idx_start=0,
        fault_locations=None,
        fault_impedances=None,
        existing_metadata=None):
    """
    Batched simulation driver with checkpointing and error handling.

    Orchestrates parallel execution of multiple scenarios with automatic
    checkpointing, progress tracking, and recovery from failures.

    Parameters
    ----------
    raw : str
        Path to PSS/E RAW file.
    dyr : str
        Path to PSS/E DYR file.
    scenarios_metadata : dict
        Dictionary mapping scenario IDs to parameter dictionaries,
        as returned by :func:`generate_metadata`.
    noise_type : str, default='normal'
        Default noise distribution type.
    noise_var : float, default=0.1
        Default noise variance.
    balance_generation : bool, default=True
        If True, rebalance generation to match load.
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores.
    batch_size : int, default=10
        Number of scenarios per batch.
    checkpoint_interval : int, default=100
        Save checkpoint every N batches.
    perturb_loads : bool, default=True
        Enable load perturbations.
    perturb_gens : bool, default=True
        Enable generator perturbations.
    load_noise_type : str, optional
        Override noise type for loads.
    gen_noise_type : str, optional
        Override noise type for generators.
    load_noise_var : float, optional
        Override noise variance for loads.
    gen_noise_var : float, optional
        Override noise variance for generators.
    keep_power_factor : bool, default=True
        Maintain power factor when perturbing.
    clamp_gens : bool, default=True
        Enforce generator limits.
    existing_log : dict, optional
        Existing simulation log to merge results into. Used for continuation
        mode (--continue) to append new scenarios to previous runs. If None,
        starts with an empty log.
    integration_config : dict, optional
        Integration parameters for the dynamic simulation. Passed to each
        worker. See :func:`run_single_scenario_worker` for supported keys.

    Returns
    -------
    dict
        Simulation log mapping scenario IDs to result dictionaries
        containing scenario parameters and simulation outcomes.

    Notes
    -----
    **Checkpointing**:

    Progress is saved periodically to enable recovery from interruptions:

    - simulation_log.json: Updated after each batch
    - simulation_checkpoint.json: Full checkpoint at specified intervals

    On restart, the driver automatically resumes from the last checkpoint.

    **Error Handling**:

    - Individual scenario failures are recorded but don't stop execution
    - Batch-level failures mark all scenarios in the batch as failed
    - The 'loky' backend provides timeout protection (600s per batch)

    **Memory Management**:

    - Garbage collection runs between batches
    - Power system models are loaded fresh for each batch
    - State metadata is exported once per batch

    See Also
    --------
    run_single_scenario_worker : Worker function for individual scenarios
    """
    if target_accepted_scenarios is not None:
        op_cfg_for_target = _resolve_operating_point_config(operating_point_config)
        if max_total_attempts is None:
            max_total_attempts = (
                int(target_accepted_scenarios)
                * int(op_cfg_for_target["max_attempts_per_scenario"])
            )
        if fault_locations is None or fault_impedances is None:
            raise ValueError(
                "fault_locations and fault_impedances are required in target mode"
            )
        return _run_target_accepted_driver(
            raw,
            dyr,
            target_accepted_scenarios=target_accepted_scenarios,
            max_total_attempts=max_total_attempts,
            sample_idx_start=sample_idx_start,
            fault_locations=fault_locations,
            fault_impedances=fault_impedances,
            noise_type=noise_type,
            noise_var=noise_var,
            balance_generation=balance_generation,
            n_jobs=n_jobs,
            perturb_loads=perturb_loads,
            perturb_gens=perturb_gens,
            load_noise_type=load_noise_type,
            gen_noise_type=gen_noise_type,
            load_noise_var=load_noise_var,
            gen_noise_var=gen_noise_var,
            keep_power_factor=keep_power_factor,
            clamp_gens=clamp_gens,
            load_scale=load_scale,
            load_mean_shift=load_mean_shift,
            generation_dispatch_init=generation_dispatch_init,
            operating_point_config=operating_point_config,
            existing_log=existing_log,
            existing_metadata=existing_metadata,
            integration_config=integration_config,
        )

    scenario_ids = list(scenarios_metadata.keys())
    op_cfg = _resolve_operating_point_config(operating_point_config)
    diagnostics_enabled = (
        op_cfg["enabled"]
        or not np.isclose(load_scale, 1.0)
        or not np.isclose(load_mean_shift, 0.0)
    )
    diagnostics_file = op_cfg.get("diagnostics_file") if diagnostics_enabled else None
    diagnostics_summary_file = (
        op_cfg.get("diagnostics_summary_file") if diagnostics_enabled else None
    )
    diagnostics_records = []
    
    # Initialize log, optionally starting from existing data (for --continue mode)
    simulation_log = dict(existing_log) if existing_log else {}

    # Load checkpoint if it exists (for resuming interrupted runs)
    checkpoint_file = "simulation_checkpoint.json"
    start_batch = 0

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
                simulation_log = checkpoint.get("simulation_log", {})
                start_batch = checkpoint.get("last_batch", 0) + 1
                print(f"Resuming from batch {start_batch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    if diagnostics_file and start_batch == 0 and not existing_log:
        open(diagnostics_file, "w").close()
    if diagnostics_summary_file and start_batch == 0 and not existing_log:
        with open(diagnostics_summary_file, "w") as f:
            json.dump(_summarize_diagnostics([]), f, indent=4)
    if diagnostics_file and (existing_log or start_batch > 0):
        diagnostics_records = _load_diagnostics_records(diagnostics_file)

    t0 = time.time()
    total_batches = int(np.ceil(len(scenario_ids) / batch_size))

    # Process batches
    for batch_idx, batch_start in enumerate(
            range(start_batch * batch_size, len(scenario_ids), batch_size),
            start=start_batch):

        batch_ids = scenario_ids[batch_start: batch_start + batch_size]

        # Progress tracking with ETA estimation
        elapsed = time.time() - t0
        if batch_idx > start_batch:
            est_total = elapsed * (total_batches - start_batch) / (batch_idx - start_batch)
            remaining = max(0.0, est_total - elapsed)
        else:
            est_total = remaining = 0

        progress = (batch_idx + 1) / total_batches if total_batches else 1.0
        bar_width = 24
        filled = int(round(bar_width * progress))
        progress_bar = "#" * filled + "-" * (bar_width - filled)

        print(
            f"Processing batch {batch_idx + 1} / {total_batches} | "
            f"[{progress_bar}] {progress:.1%} | "
            f"elapsed {timedelta(seconds=int(elapsed))} | "
            f"est total {timedelta(seconds=int(est_total))} | "
            f"ETA {timedelta(seconds=int(remaining))}"
        )

        # Export state metadata once per batch (outside parallel region)
        base_psys = load_psse(raw)
        add_dyr(base_psys, dyr)
        base_psys.export_state_metadata()
        del base_psys

        # Prepare worker arguments
        batch_args = [
            (raw, dyr, scenarios_metadata[sid], sid,
             noise_type, noise_var, 1234, balance_generation,
             perturb_loads, perturb_gens,
             load_noise_type, gen_noise_type,
             load_noise_var, gen_noise_var,
             keep_power_factor, clamp_gens,
             load_scale, load_mean_shift,
             generation_dispatch_init, op_cfg,
             integration_config)
            for sid in batch_ids
        ]

        try:
            # Execute batch in parallel with timeout protection
            # TODO: Ensure timeout doesn't prematurely terminate long scenarios
            batch_out = Parallel(n_jobs=n_jobs, backend='loky', timeout=600)(
                delayed(run_single_scenario_worker)(*args) for args in batch_args
            )

            # Update simulation log with results
            for sid, out in zip(batch_ids, batch_out):
                simulation_log[sid] = _json_safe({**scenarios_metadata[sid], **out})

            batch_diagnostics = []
            for out in batch_out:
                batch_diagnostics.extend(_diagnostic_records_from_output(out))
            _write_diagnostics_records(diagnostics_file, batch_diagnostics)
            diagnostics_records.extend(_json_safe(batch_diagnostics))
            _write_diagnostics_summary_records(
                diagnostics_summary_file,
                diagnostics_records,
            )
            _print_batch_diagnostics(batch_out)

        except Exception as e:
            print(f"Batch {batch_idx + 1} failed with error: {e}")
            traceback.print_exc()

            # Mark all scenarios in failed batch
            for sid in batch_ids:
                if sid not in simulation_log:
                    simulation_log[sid] = {
                        **scenarios_metadata[sid],
                        "file": None,
                        "diverged": True,
                        "error": f"Batch failure: {str(e)}"
                    }

        # Save progress after each batch
        with open("simulation_log.json", "w") as f:
            json.dump(simulation_log, f, indent=4)

        # Periodic checkpointing
        if (batch_idx + 1) % checkpoint_interval == 0:
            with open(checkpoint_file, "w") as f:
                json.dump({
                    "last_batch": batch_idx,
                    "simulation_log": simulation_log
                }, f, indent=4)
            print(f"Checkpoint saved at batch {batch_idx + 1}")

        # Force garbage collection between batches
        gc.collect()

    # Clean up checkpoint file on successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return simulation_log


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: str) -> dict:
    """
    Load simulation configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Configuration dictionary with the following structure:

        - model: Model configuration (raw, dyr, n_bus paths)
        - scenarios: Scenario sampling parameters
        - execution: Parallel execution parameters
        - perturbation: Noise and perturbation settings
        - integration: Dynamic simulation parameters (tend, dt, ton, toff, etc.)

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    json.JSONDecodeError
        If the configuration file is not valid JSON.

    Examples
    --------
    >>> config = load_config("config_IEEE9.json")
    >>> print(config["model"]["n_bus"])
    9
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Loaded configuration from: {config_path}")
    return config


def get_default_config(model_name: str) -> dict:
    """
    Get the default configuration for a specified power grid model.

    Parameters
    ----------
    model_name : str
        Name of the power grid model. Supported values:
        'IEEE-9', 'IEEE-39', 'ACTIVSg200', 'ACTIVSg500'

    Returns
    -------
    dict
        Default configuration dictionary for the specified model.

    Raises
    ------
    ValueError
        If the model name is not recognized.

    Notes
    -----
    The returned configuration includes:

    - Model paths (raw, dyr files) and bus count
    - Default scenario sampling parameters
    - Default execution parameters (n_jobs, batch_size)
    - Default perturbation settings (noise type, variance, flags)
    - Default integration settings (tend, dt, ton, toff, petsc, etc.)
    """
    # Model-specific paths and bus counts
    model_configs = {
        "IEEE-9": {
            "raw": "../data/ieee9_v33.raw",
            "dyr": "../data/ieee9bus_gov.dyr",
            "n_bus": 9
        },
        "IEEE-39": {
            "raw": "data/IEEE39_v33.raw",
            "dyr": "data/IEEE39_gov.dyr",
            "n_bus": 39
        },
        "ACTIVSg200": {
            "raw": "data/ACTIVSg200.raw",
            "dyr": "data/ACTIVSg200.dyr",
            "n_bus": 200
        },
        "ACTIVSg500": {
            "raw": "data/ACTIVSg500.raw",
            "dyr": "data/ACTIVSg500.dyr",
            "n_bus": 500
        }
    }

    if model_name not in model_configs:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported models: {list(model_configs.keys())}"
        )

    model_info = model_configs[model_name]

    config = {
        # Model configuration
        "model": {
            "name": model_name,
            "raw": model_info["raw"],
            "dyr": model_info["dyr"],
            "n_bus": model_info["n_bus"]
        },

        # Scenario sampling configuration
        "scenarios": {
            "samples_per_fault_location": 5,
            "fault_impedances": [0.00001],
            "fault_locations": "all",  # "all" means list(range(n_bus)), or specify list
            "target_accepted_scenarios": None,
            "max_total_attempts": None
        },

        # Execution parameters
        "execution": {
            "n_jobs": 5,
            "batch_size": 10,
            "checkpoint_interval": 5
        },

        # Perturbation configuration
        "perturbation": {
            # Noise settings for loads
            "load_noise_type": "normal",
            "load_noise_var": 0.25,

            # Noise settings for generators
            "gen_noise_type": "normal",
            "gen_noise_var": 0.25,

            # Control flags
            "balance_generation": True,
            "perturb_loads": True,
            "perturb_gens": True,
            "keep_power_factor": True,
            "clamp_gens": True,
            "load_scale": 1.0,
            "load_mean_shift": 0.0,
            "generation_dispatch_init": "perturbed"
        },

        # PF-aware operating-point screening (opt-in; disabled preserves legacy behavior)
        "operating_point": {
            "enabled": False,
            "run_power_flow": True,
            "rebalance_non_slack": True,
            "redistribute_slack_mismatch": True,
            "rebalance_policy": "headroom",
            "loss_compensation": False,
            "loss_compensation_tolerance_pu": 1e-4,
            "loss_compensation_policy": "headroom",
            "q_limit_mitigation": False,
            "q_limit_mitigation_tolerance_pu": 1e-6,
            "q_limit_mitigation_max_passes": 10,
            "q_limit_mitigation_top_n": 10,
            "max_iterations": 5,
            "max_attempts_per_scenario": 50,
            "pf_residual_tol": 1e-8,
            "slack_p_tolerance_pu": 1e-4,
            "max_slack_p_deviation_fraction_of_load": 0.02,
            "voltage_min": 0.90,
            "voltage_max": 1.10,
            "gen_limit_tolerance": 1e-6,
            "branch_loading_max": 1.0,
            "diagnostics_file": "scenario_diagnostics.jsonl",
            "diagnostics_summary_file": "scenario_diagnostics_summary.json"
        },

        # Integration/simulation configuration
        "integration": {
            "tend": 10.0,              # Simulation end time [s]
            "dt": 0.008333333333333333, # Time step [s] (1/120)
            "power_injection": False,
            "ton": 0.25,               # Fault onset time [s]
            "toff": 0.4,               # Fault clearing time [s]
            "verbose": False,
            "petsc": True,
            "enforce_q_limits": True,
            "q_limit_tolerance": 1e-8,
            "max_q_limit_iterations": None,
            "power_flow_validation": {
                "enabled": False,
                "residual_tolerance": 1e-8,
                "generator_limit_tolerance": 1e-6,
                "voltage_min": 0.9,
                "voltage_max": 1.1,
                "branch_loading_max": 1.0,
                "branch_limit_tolerance": 1e-5,
                "active_set_voltage_tolerance": 1e-6
            }
        }
    }

    return config


def save_config(config: dict, config_path: str) -> None:
    """
    Save a configuration dictionary to a JSON file.

    Parameters
    ----------
    config : dict
        Configuration dictionary to save.
    config_path : str
        Output path for the JSON file.
    """
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to: {config_path}")


def generate_default_configs(output_dir: str = ".") -> None:
    """
    Generate default configuration files for all supported models.

    Creates JSON configuration files for IEEE-9, IEEE-39, ACTIVSg200,
    and ACTIVSg500 models in the specified output directory.

    Parameters
    ----------
    output_dir : str, default='.'
        Directory where configuration files will be saved.

    Side Effects
    ------------
    Creates the following files:
        - config_IEEE-9.json
        - config_IEEE-39.json
        - config_ACTIVSg200.json
        - config_ACTIVSg500.json
    """
    os.makedirs(output_dir, exist_ok=True)

    models = ["IEEE-9", "IEEE-39", "ACTIVSg200", "ACTIVSg500"]

    for model_name in models:
        config = get_default_config(model_name)
        config_path = os.path.join(output_dir, f"config_{model_name}.json")
        save_config(config, config_path)

    print(f"\nGenerated {len(models)} configuration files in: {output_dir}")


# =============================================================================
# Continuation Support
# =============================================================================

def load_existing_state(metadata_file="scenario_metadata.json", 
                        log_file="simulation_log.json"):
    """
    Load existing simulation state for continuation mode.

    Reads existing metadata and simulation log files to enable adding
    new samples to a previous simulation run.

    Parameters
    ----------
    metadata_file : str, default='scenario_metadata.json'
        Path to existing scenario metadata file.
    log_file : str, default='simulation_log.json'
        Path to existing simulation log file.

    Returns
    -------
    tuple of (dict, dict, int)
        - existing_metadata : dict
            Previously generated scenario metadata (empty if file not found).
        - existing_log : dict
            Previous simulation results (empty if file not found).
        - max_sample_idx : int
            Highest sample_idx in existing data (-1 if no existing data).

    Notes
    -----
    This function is used by the --continue CLI option to determine where
    to start numbering new samples and to merge results with existing data.
    """
    existing_metadata = {}
    existing_log = {}
    max_sample_idx = -1

    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            existing_metadata = json.load(f)
        # Find the highest sample_idx to continue from
        for scenario in existing_metadata.values():
            idx = scenario.get('sample_idx', -1)
            max_sample_idx = max(max_sample_idx, idx)
        print(f"Loaded existing metadata: {len(existing_metadata):,} scenarios, "
              f"max sample_idx={max_sample_idx}")

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            existing_log = json.load(f)
        print(f"Loaded existing log: {len(existing_log):,} entries")

    return existing_metadata, existing_log, max_sample_idx


def generate_metadata_continued(scenarios, existing_metadata=None,
                                metadata_file="scenario_metadata.json"):
    """
    Generate metadata for new scenarios and merge with existing metadata.

    Creates unique identifiers for new scenarios and optionally merges
    them with existing metadata from a previous run.

    Parameters
    ----------
    scenarios : list of tuple
        List of (sample_idx, fault_location, fault_impedance) tuples
        from :func:`sample_scenarios`.
    existing_metadata : dict, optional
        Existing metadata to merge with. If None, starts fresh.
    metadata_file : str, default='scenario_metadata.json'
        Output path for the merged metadata file.

    Returns
    -------
    dict
        Dictionary mapping scenario UUIDs to parameter dictionaries,
        including both existing and new scenarios.

    Side Effects
    ------------
    Creates or overwrites the metadata file with merged metadata.
    """
    # Start with existing metadata if provided
    metadata = dict(existing_metadata) if existing_metadata else {}
    new_count = 0

    for sample_idx, floc, fz in scenarios:
        sid = str(uuid.uuid4())
        metadata[sid] = {
            "sample_idx": sample_idx,
            "fault_location": floc,
            "fault_impedance": fz,
        }
        new_count += 1

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Generated {new_count:,} new scenarios (total: {len(metadata):,})")
    return metadata


# =============================================================================
# Main Entry Point
# =============================================================================

def main(config_path: str = None):
    """
    Main entry point for scenario generation.

    Loads configuration from a JSON file and runs a complete simulation
    campaign for the specified power grid model.

    Parameters
    ----------
    config_path : str, optional
        Path to the JSON configuration file. If not provided, looks for
        a config file specified via command-line argument, or uses the
        default IEEE-9 configuration.

    Configuration File Structure
    ----------------------------
    The JSON configuration file should have the following structure::

        {
            "model": {
                "name": "IEEE-9",
                "raw": "path/to/model.raw",
                "dyr": "path/to/model.dyr",
                "n_bus": 9
            },
            "scenarios": {
                "samples_per_fault_location": 5,
                "fault_impedances": [0.00001],
                "fault_locations": "all",  // or [0, 1, 2, ...]
                "target_accepted_scenarios": null,
                "max_total_attempts": null
            },
            "execution": {
                "n_jobs": 5,
                "batch_size": 10,
                "checkpoint_interval": 5
            },
            "perturbation": {
                "load_noise_type": "normal",
                "load_noise_var": 0.25,
                "gen_noise_type": "normal",
                "gen_noise_var": 0.25,
                "balance_generation": true,
                "perturb_loads": true,
                "perturb_gens": true,
                "keep_power_factor": true,
                "clamp_gens": true,
                "load_scale": 1.0,
                "load_mean_shift": 0.0,
                "generation_dispatch_init": "perturbed"
            },
            "operating_point": {
                "enabled": false,
                "run_power_flow": true,
                "rebalance_non_slack": true,
                "redistribute_slack_mismatch": true,
                "rebalance_policy": "headroom",
                "loss_compensation": false,
                "loss_compensation_tolerance_pu": 1e-4,
                "loss_compensation_policy": "headroom",
                "q_limit_mitigation": false,
                "q_limit_mitigation_tolerance_pu": 1e-6,
                "q_limit_mitigation_max_passes": 10,
                "q_limit_mitigation_top_n": 10,
                "max_iterations": 5,
                "max_attempts_per_scenario": 50,
                "pf_residual_tol": 1e-8,
                "slack_p_tolerance_pu": 1e-4,
                "max_slack_p_deviation_fraction_of_load": 0.02,
                "voltage_min": 0.9,
                "voltage_max": 1.1,
                "gen_limit_tolerance": 1e-6,
                "branch_loading_max": 1.0,
                "diagnostics_file": "scenario_diagnostics.jsonl",
                "diagnostics_summary_file": "scenario_diagnostics_summary.json"
            },
            "integration": {
                "tend": 10.0,
                "dt": 0.008333333333333333,
                "power_injection": false,
                "ton": 0.25,
                "toff": 0.4,
                "verbose": false,
                "petsc": true,
                "enforce_q_limits": true,
                "q_limit_tolerance": 1e-8,
                "max_q_limit_iterations": null,
                "power_flow_validation": {
                    "enabled": false,
                    "residual_tolerance": 1e-8,
                    "generator_limit_tolerance": 1e-6,
                    "voltage_min": 0.9,
                    "voltage_max": 1.1,
                    "branch_loading_max": 1.0,
                    "branch_limit_tolerance": 1e-5,
                    "active_set_voltage_tolerance": 1e-6
                }
            }
        }

    Examples
    --------
    Run with a specific configuration file::

        $ python generate_scenarios.py config_IEEE-9.json

    Generate default configuration files::

        $ python generate_scenarios.py --generate-configs

    Run with default IEEE-9 configuration::

        $ python generate_scenarios.py
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Power Grid Scenario Generation with Perturbation and Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_scenarios.py config_IEEE-9.json    Run with config file
  python generate_scenarios.py --generate-configs    Generate default configs
  python generate_scenarios.py                       Run with default IEEE-9 config
  python generate_scenarios.py config.json --continue --additional-samples 10
                                                     Add 10 more samples to existing run
        """
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--generate-configs",
        action="store_true",
        help="Generate default configuration files for all models and exit"
    )
    parser.add_argument(
        "--config-dir",
        default=".",
        help="Output directory for generated config files (default: current directory)"
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Continue from existing simulation, adding more samples"
    )
    parser.add_argument(
        "--additional-samples",
        type=int,
        default=None,
        help="Number of additional samples per fault location (required with --continue)"
    )

    args = parser.parse_args()

    # Handle config generation mode
    if args.generate_configs:
        generate_default_configs(args.config_dir)
        return

    # Validate continuation mode arguments
    if args.continue_run and args.additional_samples is None:
        parser.error("--continue requires --additional-samples")

    # Load configuration
    if args.config is not None:
        config = load_config(args.config)
    elif config_path is not None:
        config = load_config(config_path)
    else:
        print("No configuration file specified. Using default IEEE-9 configuration.")
        config = get_default_config("IEEE-9")

    # -------------------------------------------------------------------------
    # Extract configuration values
    # -------------------------------------------------------------------------

    # Model configuration
    model_cfg = config["model"]
    raw = model_cfg["raw"]
    dyr = model_cfg["dyr"]
    n_bus = model_cfg["n_bus"]

    # Scenario configuration
    scenario_cfg = config["scenarios"]
    samples_per_fault = scenario_cfg["samples_per_fault_location"]
    fault_impedances = scenario_cfg["fault_impedances"]
    target_accepted_scenarios = scenario_cfg.get("target_accepted_scenarios")
    max_total_attempts = scenario_cfg.get("max_total_attempts")

    # Handle fault_locations: "all" means all buses, otherwise use the list
    fault_locations_cfg = scenario_cfg["fault_locations"]
    if fault_locations_cfg == "all":
        fault_locations = list(range(n_bus))
    else:
        fault_locations = fault_locations_cfg

    # Execution configuration
    exec_cfg = config["execution"]
    n_jobs = exec_cfg["n_jobs"]
    batch_size = exec_cfg["batch_size"]
    checkpoint_interval = exec_cfg["checkpoint_interval"]

    # Perturbation configuration
    pert_cfg = config["perturbation"]
    load_noise_type = pert_cfg.get("load_noise_type", config.get("noise_type", "normal"))
    load_noise_var = pert_cfg.get("load_noise_var", config.get("noise_var", 0.1))
    gen_noise_type = pert_cfg.get("gen_noise_type", load_noise_type)
    gen_noise_var = pert_cfg.get("gen_noise_var", load_noise_var)
    balance_generation = pert_cfg.get("balance_generation", True)
    perturb_loads = pert_cfg.get("perturb_loads", True)
    perturb_gens = pert_cfg.get("perturb_gens", True)
    keep_power_factor = pert_cfg.get("keep_power_factor", True)
    clamp_gens = pert_cfg.get("clamp_gens", True)
    load_scale = pert_cfg.get("load_scale", 1.0)
    load_mean_shift = pert_cfg.get("load_mean_shift", 0.0)
    generation_dispatch_init = pert_cfg.get("generation_dispatch_init", "perturbed")

    # Operating-point screening is opt-in for backward compatibility.
    operating_point_config = _resolve_operating_point_config(config.get("operating_point", {}))
    target_mode = target_accepted_scenarios is not None
    if target_mode:
        target_accepted_scenarios = int(target_accepted_scenarios)
        if max_total_attempts is None:
            max_total_attempts = (
                target_accepted_scenarios
                * int(operating_point_config["max_attempts_per_scenario"])
            )
        else:
            max_total_attempts = int(max_total_attempts)

    # Integration configuration
    integration_cfg = config.get("integration", {})
    validation_cfg = integration_cfg.get("power_flow_validation", {}) or {}
    integration_config = {
        "tend": integration_cfg.get("tend", 10.0),
        "dt": integration_cfg.get("dt", 1/120.0),
        "power_injection": integration_cfg.get("power_injection", False),
        "ton": integration_cfg.get("ton", 0.25),
        "toff": integration_cfg.get("toff", 0.4),
        "verbose": integration_cfg.get("verbose", False),
        "petsc": integration_cfg.get("petsc", True),
        "enforce_q_limits": integration_cfg.get("enforce_q_limits", True),
        "q_limit_tolerance": integration_cfg.get("q_limit_tolerance", 1e-8),
        "max_q_limit_iterations": integration_cfg.get("max_q_limit_iterations"),
        "power_flow_validation": {
            "enabled": validation_cfg.get("enabled", False),
            "residual_tolerance": validation_cfg.get(
                "residual_tolerance", 1e-8
            ),
            "generator_limit_tolerance": validation_cfg.get(
                "generator_limit_tolerance", 1e-6
            ),
            "voltage_min": validation_cfg.get("voltage_min"),
            "voltage_max": validation_cfg.get("voltage_max"),
            "branch_loading_max": validation_cfg.get("branch_loading_max"),
            "branch_limit_tolerance": validation_cfg.get(
                "branch_limit_tolerance", 1e-5
            ),
            "active_set_voltage_tolerance": validation_cfg.get(
                "active_set_voltage_tolerance", 1e-6
            ),
        },
    }

    # -------------------------------------------------------------------------
    # Print configuration summary
    # -------------------------------------------------------------------------
    if target_mode:
        total_scenarios = (
            target_accepted_scenarios * len(fault_locations) * len(fault_impedances)
        )
    else:
        total_scenarios = samples_per_fault * len(fault_locations) * len(fault_impedances)

    print("\n" + "=" * 60)
    print("SIMULATION CONFIGURATION")
    print("=" * 60)
    print(f"Model: {model_cfg.get('name', 'Unknown')}")
    print(f"  - RAW file: {raw}")
    print(f"  - DYR file: {dyr}")
    print(f"  - Buses: {n_bus}")
    print()
    
    # Handle continuation mode
    existing_metadata = {}
    existing_log = {}
    sample_idx_offset = 0
    
    if args.continue_run:
        print("=" * 60)
        print("CONTINUATION MODE")
        print("=" * 60)
        existing_metadata, existing_log, max_sample_idx = load_existing_state()
        sample_idx_offset = max_sample_idx + 1
        samples_per_fault = args.additional_samples
        if target_mode:
            total_scenarios = (
                target_accepted_scenarios
                * len(fault_locations)
                * len(fault_impedances)
            )
        else:
            total_scenarios = samples_per_fault * len(fault_locations) * len(fault_impedances)
        print(f"  - Starting sample_idx from: {sample_idx_offset}")
        print(f"  - Additional samples per fault: {samples_per_fault}")
        print(f"  - New scenarios to generate: {total_scenarios:,}")
        print(f"  - Total scenarios after: {len(existing_metadata) + total_scenarios:,}")
        print()
    
    print(f"Scenarios: {total_scenarios:,} {'new' if args.continue_run else 'total'}")
    print(f"  - Samples per fault location: {samples_per_fault}")
    print(f"  - Fault locations: {len(fault_locations)}")
    print(f"  - Fault impedances: {fault_impedances}")
    if target_mode:
        print("  - Target mode: enabled")
        print(f"  - Target accepted operating points: {target_accepted_scenarios}")
        print(f"  - Max total candidate attempts: {max_total_attempts}")
    print()
    print(f"Execution:")
    print(f"  - Parallel jobs: {n_jobs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Checkpoint interval: {checkpoint_interval} batches")
    print()
    print(f"Perturbation:")
    print(f"  - Load noise: {load_noise_type} (var={load_noise_var})")
    print(f"  - Gen noise: {gen_noise_type} (var={gen_noise_var})")
    print(f"  - Balance generation: {balance_generation}")
    print(f"  - Perturb loads: {perturb_loads}")
    print(f"  - Perturb generators: {perturb_gens}")
    print(f"  - Keep power factor: {keep_power_factor}")
    print(f"  - Clamp generators: {clamp_gens}")
    print(f"  - Load scale: {load_scale}")
    print(f"  - Load mean shift: {load_mean_shift}")
    print(f"  - Generation dispatch init: {generation_dispatch_init}")
    print()
    print(f"Operating point:")
    print(f"  - Enabled: {operating_point_config['enabled']}")
    print(f"  - Run power flow: {operating_point_config['run_power_flow']}")
    print(f"  - Rebalance non-slack: {operating_point_config['rebalance_non_slack']}")
    print(f"  - Redistribute slack mismatch: {operating_point_config['redistribute_slack_mismatch']}")
    print(f"  - Rebalance policy: {operating_point_config['rebalance_policy']}")
    print(f"  - Loss compensation: {operating_point_config['loss_compensation']}")
    print(f"  - Loss compensation policy: {operating_point_config['loss_compensation_policy']}")
    print(f"  - Q-limit mitigation: {operating_point_config['q_limit_mitigation']}")
    print(f"  - Max iterations: {operating_point_config['max_iterations']}")
    print(f"  - Max attempts/scenario: {operating_point_config['max_attempts_per_scenario']}")
    print()
    print(f"Integration:")
    print(f"  - End time (tend): {integration_config['tend']} s")
    print(f"  - Time step (dt): {integration_config['dt']:.6f} s")
    print(f"  - Fault onset (ton): {integration_config['ton']} s")
    print(f"  - Fault clearing (toff): {integration_config['toff']} s")
    print(f"  - Power injection: {integration_config['power_injection']}")
    print(f"  - PETSc solver: {integration_config['petsc']}")
    print(f"  - Enforce PF Q limits: {integration_config['enforce_q_limits']}")
    print(f"  - PF Q-limit tolerance: {integration_config['q_limit_tolerance']}")
    print(
        "  - PF Q-limit max solves: "
        f"{integration_config['max_q_limit_iterations']}"
    )
    validation_summary = integration_config["power_flow_validation"]
    print(f"  - Validate final PF: {validation_summary['enabled']}")
    if validation_summary["enabled"]:
        print(
            "  - Final PF voltage range: "
            f"{validation_summary['voltage_min']} to "
            f"{validation_summary['voltage_max']}"
        )
        print(
            "  - Final PF branch loading max: "
            f"{validation_summary['branch_loading_max']}"
        )
    print(f"  - Verbose: {integration_config['verbose']}")
    print("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # Generate scenario definitions
    # -------------------------------------------------------------------------
    if target_mode:
        metadata = dict(existing_metadata)
        new_metadata = {}
    else:
        # Generate sample indices with offset for continuation mode
        sample_indices = range(sample_idx_offset, sample_idx_offset + samples_per_fault)
        scenarios = list(itertools.product(sample_indices, fault_locations, fault_impedances))

        # Generate metadata (merged with existing if continuing)
        if args.continue_run:
            metadata = generate_metadata_continued(scenarios, existing_metadata)
            # Extract only the NEW scenario metadata for simulation
            new_scenario_ids = set(metadata.keys()) - set(existing_metadata.keys())
            new_metadata = {sid: metadata[sid] for sid in new_scenario_ids}
        else:
            metadata = generate_metadata(scenarios)
            new_metadata = metadata

    # -------------------------------------------------------------------------
    # Execute Simulation Campaign
    # -------------------------------------------------------------------------
    # Use the more general noise_type/noise_var as fallback (set to load values)
    run_simulation_driver_batched(
        raw, dyr, new_metadata,
        noise_type=load_noise_type,
        noise_var=load_noise_var,
        balance_generation=balance_generation,
        n_jobs=n_jobs,
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        perturb_loads=perturb_loads,
        perturb_gens=perturb_gens,
        load_noise_type=load_noise_type,
        gen_noise_type=gen_noise_type,
        load_noise_var=load_noise_var,
        gen_noise_var=gen_noise_var,
        keep_power_factor=keep_power_factor,
        clamp_gens=clamp_gens,
        load_scale=load_scale,
        load_mean_shift=load_mean_shift,
        generation_dispatch_init=generation_dispatch_init,
        operating_point_config=operating_point_config,
        existing_log=existing_log if args.continue_run else None,
        integration_config=integration_config,
        target_accepted_scenarios=target_accepted_scenarios if target_mode else None,
        max_total_attempts=max_total_attempts if target_mode else None,
        sample_idx_start=sample_idx_offset,
        fault_locations=fault_locations,
        fault_impedances=fault_impedances,
        existing_metadata=metadata if target_mode else None
    )


if __name__ == "__main__":
    main()
