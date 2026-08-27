# Dynamics Simulation

UQGrid models power-system dynamics as differential-algebraic equations (DAEs).
This guide covers the core APIs for preparing a system, configuring disturbances,
executing time integration, and interpreting the results.

## Load a system

```python
from uqgrid.io.parse import load_psse, add_dyr

psys = load_psse(raw_filename="data/IEEE39_v33.raw")
add_dyr(psys, "data/IEEE39.dyr")
```

`load_psse` parses the steady-state network description while `add_dyr` attaches
dynamic device models (generators, exciters, governors, loads, and monitors).

## Configure disturbances

Add shunt faults or other events before integration. Fault timing is controlled
externally via the integration settings.

```python
psys.add_busfault(bus=1, rfault=0.01)
```

The first fault registered is toggled automatically using `IntegrationConfig.ton`
(fault-on time) and `IntegrationConfig.toff` (fault clearing time). The automatic
scheduler currently supports one fault event.

## Initialize with a power flow

```python
from uqgrid.simulation.pflow import runpf

psys.createYbusComplex()
v_init, s_inj = runpf(psys, verbose=False)
```

The solved voltages seed the dynamic state vector and ensure algebraic
constraints are satisfied at the start of the simulation.

## Build an integration configuration

`IntegrationConfig` encapsulates solver and timing options. Validation is handled
by [Pydantic](https://docs.pydantic.dev/).

```python
from uqgrid.simulation.config import IntegrationConfig

config = IntegrationConfig(
    tend=10.0,
    dt=1.0 / 120.0,
    method="cn",
    ton=0.25,
    toff=0.40,
    steps=-1,
    verbose=False,
    comp_sens=False,
    fsolve=False,
    petsc=True,
    enforce_q_limits=True,
    q_limit_tolerance=1e-8,
    max_q_limit_iterations=None,
    enforce_dynamic_limits=True,
    dynamic_limit_tolerance=1e-8,
    dynamic_limit_release_tolerance=1e-10,
    max_dynamic_limit_iterations=20,
    power_flow_validation={
        "enabled": False,
        "residual_tolerance": 1e-8,
        "generator_limit_tolerance": 1e-6,
        "voltage_min": None,
        "voltage_max": None,
        "branch_loading_max": None,
        "branch_limit_tolerance": 1e-5,
        "active_set_voltage_tolerance": 1e-6,
    },
    check_jacobian=False,
    jacobian_check_tol=1e-6,
    jacobian_check_top_k=10,
    jacobian_check_csv=None,
)
```

Key fields:

- **tend**: Final simulation time (seconds).
- **dt**: Integration step size (seconds).
- **steps**: Number of nominal advances. A positive value overrides `tend` and
  produces `steps + 1` base samples, including `t=0`.
- **method**: `"beuler"`, `"cn"`, `"herk2"`, or `"herk4"`.
- **ton/toff**: Fault activation and removal times.
- **comp_sens**: Enable adjoint-based sensitivities (requires PETSc and
  `method="cn"`).
- **petsc**: Switch to PETSc-backed integrators for improved robustness and
  adjoint capabilities.
- **enforce_q_limits**: Enforce non-slack PV generator reactive-power limits
  during the power flow used for dynamic initialization.
- **q_limit_tolerance**: Per-unit tolerance for activating a generator
  reactive-power limit.
- **max_q_limit_iterations**: Optional cap on active-set power-flow solves.
- **power_flow_validation**: Optional final operating-point checks performed
  after PF convergence and before dynamic device initialization.
- **enforce_dynamic_limits**: Validate enabled hard dynamic-state limits at
  initialization and enforce them in supported integration methods.
- **dynamic_limit_tolerance**: State-bound tolerance used by hard limits.
- **dynamic_limit_release_tolerance**: Complementarity tolerance reserved for
  directional release from an active bound.
- **max_dynamic_limit_iterations**: Maximum active-set iterations reserved for
  implicit hard-limit solves.
- **check_jacobian**: Run a finite-difference Jacobian check (non-PETSc only).
- **jacobian_check_tol**: Absolute tolerance for reporting FD mismatches.
- **jacobian_check_top_k**: Number of mismatches to report.
- **jacobian_check_csv**: Optional CSV file path for mismatch report.

### Integration methods

| Configuration | Backend |
|---|---|
| `method="beuler", petsc=False` | Native backward Euler |
| `method="beuler", petsc=True` | PETSc `TSBEULER`, or ordinary SNES when hard limits are active |
| `method="cn", petsc=True` | PETSc `TSCN`, or ordinary SNES when hard limits are active |
| `method="herk2", petsc=False` | HERK2 |
| `method="herk4", petsc=False` | HERK4 |
| `arkimex=True, petsc=True, enforce_dynamic_limits=False` | PETSc ARKIMEX |

Without enabled hard-limit states, `cn` is explicitly mapped to PETSc `TSCN`.
This is not the same as selecting the midpoint form of `TSTHETA`; PETSc
distinguishes endpoint Crank-Nicolson from its theta-method midpoint
formulation for DAEs. See the
[PETSc TSTHETA documentation](https://petsc.org/release/manualpages/TS/TSTHETA/).

`cn` requires PETSc, HERK requires the native backend, and ARKIMEX cannot be
combined with `cn` or HERK. Hard dynamic limits do not support ARKIMEX,
sensitivities, or the legacy fsolve path; set `enforce_dynamic_limits=False`
explicitly to use those paths. The library default remains `method="beuler"`.
Configurations that previously omitted `method` therefore select backward
Euler. PETSc command-line options may tune solver internals, but options that
override the configured TS type, time step, horizon, exact-final-time policy,
or adaptivity are rejected.

### Initial reactive-power limits

Q-limit enforcement is enabled by default and projects generator reactive
dispatch onto its RAW/MATPOWER bounds before dynamic initialization. Set
`enforce_q_limits=False` only when an unconstrained legacy operating point is
required. If a non-slack PV bus cannot hold its voltage setpoint within
aggregate generator Q capability, the power flow switches that bus to PQ and
solves again. The same initialization is used for PETSc, backward Euler,
HERK2, and HERK4.

This constrains the operating point only. Exciter field-voltage limits use the
separate dynamic-limit configuration below.

### Hard dynamic-state limits

Hard dynamic limits default to enabled. Valid finite SEXS limits parsed from a
DYR file are enabled when `EMIN < EMAX`; malformed or non-increasing limits
fail DYR loading with the bus and generator identity. Manually constructed
SEXS models retain their explicit `enable_limits` setting. UQGrid validates
enabled Efd states after `IntegrationCtx` state and parameter overrides. Values
outside EMIN/EMAX beyond `dynamic_limit_tolerance` raise `DynamicLimitError`
before Jacobian allocation, PETSc setup, fault scheduling, or time stepping.
Initial values are never silently clamped.

Parsed HYGOV desired-gate limits and IEEEG1 valve-position limits use the same
shared hard-limit machinery. By default, `add_dyr(...,
limit_initialization_policy="adjust")` widens only a source bound violated by
the initialized gate or valve and records both source and effective values in
the dynamic-limit diagnostics. Set `limit_initialization_policy="strict"` to
reject such an operating point instead. HYGOV retains its model-local gate-rate
clip, while IEEEG1 retains its model-local valve-rate clip. IEEEG1 power and
power-rate parameters are converted from generator MBASE to system SBASE;
its HP and LP shaft coefficients are preserved when their branch sum is at
most one and normalized independently when a branch sum exceeds one.

ESAC1A applies its internal `VRMIN`/`VRMAX` clamp between the regulator state
and field-voltage block. Initialization is strict: the required steady-state
signal must satisfy both `VAMIN`/`VAMAX` and effective `VRMIN`/`VRMAX`; UQGrid
does not widen those limits. ESST4B uses model-local directional anti-windup
for both PI loops. An integrator is blocked only while its limited PI output is
at the corresponding bound and its raw derivative points farther outward;
inward motion releases immediately. These moving PI-output conditions are not
fixed state bounds and therefore do not use the shared active-set layer.

IEEEST computes its washout output as `T5 * washout_derivative`. Consequently,
`T5=0` produces exactly zero stabilizer output and output Jacobian even though
the internal washout state can continue to evolve. An IEEEST record may appear
after its generator but before its exciter in a DYR file; attachment is deferred
until that exciter is available. A missing dynamic generator is still skipped
with a warning, while a dynamic generator that never receives an exciter is an
error.

Every successful result contains a JSON-safe `dynamic_limit_diagnostics`
summary, including disabled and zero-state cases. Native HERK2 and HERK4 enforce
enabled bounded-state limits at every RK stage and weighted endpoint. A
tentative stage is projected before its algebraic solve, only outward
derivatives are blocked, and the final weighted state is projected before its
endpoint algebraic solve. Inward derivatives release immediately, while
upstream controller states continue to evolve when a bounded state is pinned.

HERK limiter activation is evaluated only at RK stages and endpoints. It does
not localize the exact crossing, backtrack, or add timestamps. Accuracy is
locally first order around a limit transition even though HERK2/HERK4 retain
their nominal order on smooth intervals. Diagnostics record actual clamps and
compact `activate`/`release` transitions, not every repeatedly blocked stage.

Native and PETSc backward Euler use a fixed-active-set nonlinear solve at every
endpoint. Free states retain the ordinary BE equation. An active bounded-state
row is replaced by `state_next - bound = 0` with an identity Jacobian row,
while all algebraic equations remain in the coupled solve. After convergence,
the discarded free BE residual determines whether an active state remains
pinned or releases inward. The solve repeats until the active set is
complementarity-consistent. Cycling, nonlinear-solver failure, or exceeding
`max_dynamic_limit_iterations` raises a structured `DynamicLimitError`.
The same configured iteration cap applies when voltage-scaled bounds move at a
fault application or clearing. Algebraic nonconvergence and projection
exhaustion at those topology transitions use the common runtime limiter
diagnostics, including method, backend, event time, fault stage, and previously
accepted limiter events.

When PETSc BE has at least one enabled limited state, UQGrid advances the shared
time grid one interval at a time with ordinary PETSc SNES rather than TS. The
default SNES type is `newtonls`; KSP, preconditioner, monitor, tolerance, and
line-search options may still be supplied through `petsc_args`. PETSc
variational-inequality types `vinewtonrsls` and `vinewtonssls` are rejected:
UQGrid owns the active set and never calls SNESVI or configures variable bounds.
PETSc BE with limits disabled or no enabled states continues to use TSBEULER.

Backward Euler transitions are represented at the interval endpoint; they do
not localize the crossing, backtrack, or add timestamps. Stored endpoints are
bound-feasible and algebraically consistent, but activation or release may be
delayed by one step and contributes an expected O(dt) switching-time error.
PETSc CN uses the same endpoint active-set contract when enabled limited states
exist, but assembles the trapezoidal equations explicitly and solves them with
ordinary SNES instead of delegating to TSCN. At each accepted interval start,
an outward raw derivative is zeroed only for an inherited active bound; inward
derivatives pass through immediately. That effective starting derivative stays
fixed while UQGrid repeats endpoint SNES solves until the active set is
complementarity-consistent. Free differential rows use the average of the fixed
starting derivative and the raw endpoint derivative. Active rows remain exact
bound equations, and algebraic rows remain endpoint equations.

This limited CN path uses neither TS limiter state nor SNESVI. It retains
second-order accuracy on smooth intervals, but endpoint-only activation and
release are locally first order and may shift by one step. Crossings are not
localized and add no timestamps. PETSc CN with limits disabled or no enabled
limited states continues to use TSCN. Set `enforce_dynamic_limits=False` for
legacy unconstrained behavior.

All supported integrators return limiter events with the same required fields:

```text
device_type, device_id, bus, state_index,
side, action, time, stage_or_endpoint,
raw_derivative, state_before, state_after, bound,
active_set_iterations
```

Supported actions are `project`, `block_outward_derivative`, `activate`, and
`release`. Runtime event times are finite. HERK events identify `stage_N` or
`endpoint`; implicit BE/CN events identify `endpoint`. `raw_derivative` is null
when it is not applicable to a projection or implicit complementarity event,
and `active_set_iterations` is null for explicit methods. Existing descriptor
bounds, enabled status, and implicit `free_residual` fields remain available.
The schema constants are exported as `DYNAMIC_LIMIT_EVENT_FIELDS` and
`DYNAMIC_LIMIT_EVENT_ACTIONS`.

Cross-integrator validation requires stored and HERK stage bounded-state values
to remain inside their configured limits, algebraic endpoint residuals to
remain below solver tolerance, and implicit free or active integration rows to
satisfy their method equations and complementarity conditions. A raw
differential residual during a disturbance is the physical state derivative
and is not expected to be zero. No-fault initialized trajectories remain flat.
Native and PETSc BE use the same equations and agree to nonlinear-solver
tolerance. All methods converge toward the same trajectory as the step is
reduced, while stage- or endpoint-only limiter transitions can differ by up to
one step because UQGrid does not localize crossings.

### Final operating-point validation

Validation is disabled by default. When enabled, UQGrid checks the final Newton
residual, finite voltages, generator P/Q limits, optional voltage and branch
loading bounds, Q-limit active-set consistency, and one slack bus per electrical
island. A failed check raises `PowerFlowValidationError` before
`initialize_system()` or PETSc setup.

Successful backward Euler, PETSc, HERK2, and HERK4 runs include the JSON-safe
diagnostics under `results["power_flow_diagnostics"]`.

### Jacobian diagnostics (optional)

When running without PETSc, you can enable a finite-difference Jacobian check:

```python
config = IntegrationConfig(
    check_jacobian=True,
    jacobian_check_tol=1e-6,
    jacobian_check_top_k=10,
    jacobian_check_csv="jacobian_mismatches.csv",
)
```

The solver prints the top mismatches (row/column labels included) and can
optionally write a CSV report.

## Run the integrator

```python
from uqgrid.simulation.dynamics import integrate_system

results = integrate_system(psys, config)
```

The solver returns a dictionary with time stamps (`tvec`), state trajectory
(`history`), `dynamic_limit_diagnostics`, and optional adjoint outputs when
sensitivities are enabled. The first column of `history` is the initialized
state at `t=0`.

## Interpret the trajectory

Generator speed deviations, bus voltages, and other state variables can be
plotted directly from the result arrays. With `steps=N`, the regular grid runs
from `0` through `N*dt` and contains `N+1` samples. With `steps=-1`, integration
ends exactly at `tend`; a shortened final interval is added when needed.

Exact `ton`, `toff`, and `tend` values are inserted when they are off the regular
grid, so fault events can add samples beyond the base count. Timestamps are
strictly increasing and are stored once. The fault is active on `[ton, toff)`.
At each transition, UQGrid holds differential states fixed, solves the
algebraic equations for the new topology, and stores that post-switch state.
`ton=0` is rejected when a fault is registered because both the initialized and
post-switch states cannot occupy the single `t=0` sample.

```python
import matplotlib.pyplot as plt

speed_idx = psys.genspeed_idx_set()
plt.plot(results["tvec"], results["history"][speed_idx, :].T)
plt.xlabel("Time [s]")
plt.ylabel("Speed deviation [p.u.]")
plt.title("Generator speed response after Bus 1 fault")
plt.show()
```

This normalized time contract changes the axes and potentially the TSI values of
trajectories produced by older releases. Do not append corrected trajectories to
an existing dense-history dataset created with the previous convention.

## Sensitivities and post-processing

Setting `comp_sens=True`, `petsc=True`, and `method="cn"` activates the
adjoint solve and adds:

- `adjoint_cost`: Scalar performance index over the simulated interval.
- `adjoint_gradient_trajectory`: Contribution from the trajectory (`μᵢ`).
- `adjoint_gradient_initial`: Contribution from the initial condition (`λᵢ ∂y₀/∂p`).
- `adjoint_gradient_complete`: Sum of trajectory and initial-condition terms.

Faulted adjoints replay the saved trajectory by time segment and apply the
fixed-differential algebraic projection jump at each topology transition.
Backward Euler remains supported for forward PETSc simulation, but its integral
adjoint is rejected because that combination does not satisfy the repository's
finite-difference validation.

Example: rank loads by their influence on the monitored quantity.

```python
import numpy as np

grad = results["adjoint_gradient_complete"]
per_load = grad.reshape(psys.nloads, 2)
magnitudes = np.linalg.norm(per_load, axis=1)
for load, value in zip(psys.loads, magnitudes):
    print(f"{load.name:>12}: {value: .3e}")
```

## Batch studies

Use `bin/generate_scenarios.py` for Monte Carlo sweeps that sample load
perturbations, execute dynamics simulations in parallel, and store summary
statistics. Customize the script or import its helpers when building larger
workflows.

!!! tip "Data management"
    Large sweeps can create multi-gigabyte traces. Use the script's output
    options (e.g., `--outdir`) to segment runs by study.
