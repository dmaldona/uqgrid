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

Author
------
Power Grid Simulation Team
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
from datetime import timedelta
from joblib import Parallel, delayed

from uqgrid.simulation.dynamics import integrate_system
from uqgrid.simulation.config import IntegrationConfig
from uqgrid.io.parse import load_psse, add_dyr


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


# =============================================================================
# Power Rebalancing
# =============================================================================

def _rebalance_active_power(p_gen, pg_lb, pg_ub, target_total):
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
        return p_gen

    current_total = np.sum(p_gen)
    mismatch = target_total - current_total

    # No rebalancing needed if already at target
    if np.isclose(mismatch, 0.0):
        return p_gen

    # Fall back to uniform scaling if no bounds available
    if pg_lb is None or pg_ub is None:
        if current_total != 0.0:
            return p_gen * (target_total / current_total)
        return p_gen

    pg_lb = np.asarray(pg_lb, dtype=float)
    pg_ub = np.asarray(pg_ub, dtype=float)

    if mismatch > 0.0:
        # Need to increase generation - use upward headroom
        headroom = pg_ub - p_gen
        mask = headroom > 1e-9
        total_headroom = np.sum(headroom[mask])

        if total_headroom <= 0.0:
            # Already at upper limits everywhere
            return p_gen

        # Participation factor (capped at 1.0)
        factor = min(1.0, mismatch / total_headroom)
        p_new = p_gen.copy()
        p_new[mask] += headroom[mask] * factor

    else:
        # Need to decrease generation - use downward margin
        down_margin = p_gen - pg_lb
        mask = down_margin > 1e-9
        total_down = np.sum(down_margin[mask])

        if total_down <= 0.0:
            # Already at lower limits everywhere
            return p_gen

        # Participation factor (capped at 1.0)
        factor = min(1.0, -mismatch / total_down)
        p_new = p_gen.copy()
        p_new[mask] -= down_margin[mask] * factor

    return p_new


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
        clamp_gens=True):
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
        ss = np.random.SeedSequence([global_seed, scenario["sample_idx"]])
        rng_load, rng_gen = [np.random.default_rng(s) for s in ss.spawn(2)]

        # Resolve per-type noise parameters (use defaults if not specified)
        load_noise_type = load_noise_type or noise_type
        gen_noise_type = gen_noise_type or noise_type
        load_noise_var = load_noise_var if load_noise_var is not None else noise_var
        gen_noise_var = gen_noise_var if gen_noise_var is not None else noise_var

        # -----------------------------------------------------------------
        # Step 1: Perturb loads
        # -----------------------------------------------------------------
        if perturb_loads and base_p_load.size:
            pL_scaled, qL_scaled, pL_noise, qL_noise = generate_perturbations(
                base_p_load, base_q_load,
                noise_type=load_noise_type, var=load_noise_var, rng=rng_load,
                return_noise=True, preserve_power_factor=keep_power_factor
            )
        else:
            # Keep base values unchanged
            pL_scaled = np.array(base_p_load, copy=True)
            qL_scaled = np.array(base_q_load, copy=True)
            pL_noise = np.zeros_like(base_p_load, dtype=float)
            qL_noise = np.zeros_like(base_q_load, dtype=float)

        # -----------------------------------------------------------------
        # Step 2: Perturb generator active power
        # -----------------------------------------------------------------
        if perturb_gens and base_p_gen.size:
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
        # Step 4: Rebalance generation to match load
        # -----------------------------------------------------------------
        if balance_generation:
            sum_pL = float(np.sum(pL_scaled))
            pG_scaled = _rebalance_active_power(pG_scaled, pg_lb, pg_ub, sum_pL)

        # -----------------------------------------------------------------
        # Step 5: Compute generator reactive power
        # -----------------------------------------------------------------
        if keep_power_factor:
            # Adjust Q to maintain original power factor
            qG_scaled = np.array(base_q_gen, copy=True, dtype=float)

            mask_pg_nonzero = np.abs(base_p_gen) > 1e-8
            if np.any(mask_pg_nonzero):
                ratio = np.zeros_like(base_p_gen, dtype=float)
                ratio[mask_pg_nonzero] = base_q_gen[mask_pg_nonzero] / base_p_gen[mask_pg_nonzero]
                qG_scaled[mask_pg_nonzero] = ratio[mask_pg_nonzero] * pG_scaled[mask_pg_nonzero]

            # For purely reactive units (P=0, Q!=0): keep original Q
            mask_pg_zero_q_nonzero = (~mask_pg_nonzero) & (np.abs(base_q_gen) > 1e-8)
            if np.any(mask_pg_zero_q_nonzero):
                qG_scaled[mask_pg_zero_q_nonzero] = base_q_gen[mask_pg_zero_q_nonzero]
        else:
            # Keep original Q schedule unchanged
            qG_scaled = np.array(base_q_gen, copy=True, dtype=float)

        # -----------------------------------------------------------------
        # Step 6: Clamp generator reactive power to limits
        # -----------------------------------------------------------------
        if clamp_gens and qg_lb is not None and qg_ub is not None and qG_scaled.size:
            qG_scaled = np.clip(qG_scaled, qg_lb, qg_ub)

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
        cfg = IntegrationConfig(
            tend=10.0,           # Simulation end time [s]
            dt=1/120.0,          # Time step [s]
            power_injection=False,
            ton=0.25,            # Fault onset time [s]
            toff=0.4,            # Fault clearing time [s]
            verbose=False,
            petsc=True
        )

        try:
            sim = integrate_system(psys, cfg)
            diverged = False
        except Exception as e:
            print(f"Simulation failed for scenario {scenario_id}: {str(e)}")
            sim = {"history": None, "tvec": None}
            diverged = True

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
            # Generator values and perturbations
            p_gen_scaled=pG_scaled, q_gen_scaled=qG_scaled,
            p_gen_noise=pG_noise, q_gen_noise=qG_noise,
        )

        # Clean up resources (important for PETSc/MPI)
        del sim
        del psys
        gc.collect()

        return {"file": fn, "diverged": diverged}

    except Exception as e:
        print(f"Worker error for scenario {scenario_id}: {str(e)}")
        traceback.print_exc()
        return {"file": None, "diverged": True, "error": str(e)}


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
        existing_log=None):
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
    scenario_ids = list(scenarios_metadata.keys())
    
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

        print(
            f"Processing batch {batch_idx + 1} / {total_batches} | "
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
             keep_power_factor, clamp_gens)
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
                simulation_log[sid] = {**scenarios_metadata[sid], **out}

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
            "fault_locations": "all"  # "all" means list(range(n_bus)), or specify list
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
            "clamp_gens": True
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
                "fault_locations": "all"  // or [0, 1, 2, ...]
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
                "clamp_gens": true
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
    load_noise_type = pert_cfg["load_noise_type"]
    load_noise_var = pert_cfg["load_noise_var"]
    gen_noise_type = pert_cfg["gen_noise_type"]
    gen_noise_var = pert_cfg["gen_noise_var"]
    balance_generation = pert_cfg["balance_generation"]
    perturb_loads = pert_cfg["perturb_loads"]
    perturb_gens = pert_cfg["perturb_gens"]
    keep_power_factor = pert_cfg["keep_power_factor"]
    clamp_gens = pert_cfg["clamp_gens"]

    # -------------------------------------------------------------------------
    # Print configuration summary
    # -------------------------------------------------------------------------
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
    print("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # Generate scenario definitions
    # -------------------------------------------------------------------------
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
        existing_log=existing_log if args.continue_run else None
    )


if __name__ == "__main__":
    main()
