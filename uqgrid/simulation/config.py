from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np


class PowerFlowValidationConfig(BaseModel):
    enabled: bool = False
    residual_tolerance: float = Field(1e-8, ge=0.0)
    generator_limit_tolerance: float = Field(1e-6, ge=0.0)
    voltage_min: Optional[float] = None
    voltage_max: Optional[float] = None
    branch_loading_max: Optional[float] = Field(None, gt=0.0)
    branch_limit_tolerance: float = Field(1e-5, ge=0.0)
    active_set_voltage_tolerance: float = Field(1e-6, ge=0.0)

    @model_validator(mode="after")
    def validate_voltage_range(self):
        if (
            self.voltage_min is not None
            and self.voltage_max is not None
            and self.voltage_min > self.voltage_max
        ):
            raise ValueError("power_flow_validation.voltage_min must not exceed voltage_max")
        return self


class IntegrationConfig(BaseModel):
    power_injection: bool = False
    tend: float = Field(10.0, ge=0.0, description="Integration end time in seconds.")
    dt: float = Field(1.0 / 120.0, gt=0.0, description="Nominal time step in seconds.")
    steps: int = Field(
        -1,
        description=(
            "Number of nominal integration advances. If positive, overrides "
            "tend and produces steps + 1 base samples including t=0."
        ),
    )
    verbose: bool = Field(False, description="Enable verbose output.")
    comp_sens: bool = Field(False, description="Compute first and second-order sensitivities.")
    fsolve: bool = Field(False, description="Use fsolve for solving nonlinear equations.")
    newton_tol: float = Field(1e-10, gt=0.0, description="Newton residual norm tolerance.")
    newton_max_iter: int = Field(500, gt=0, description="Maximum Newton iterations per integration step.")
    ton: float = Field(0.25, ge=0.0, description="Fault activation time.")
    toff: float = Field(0.4, ge=0.0, description="Fault deactivation time.")
    petsc: bool = Field(False, description="Enable PETSc integration.")
    petsc_args: List[str] = Field(
        default_factory=list,
        description="PETSc-specific command-line options to pass to petsc4py.init.",
    )
    enforce_q_limits: bool = Field(
        True,
        description="Enforce non-slack PV generator Q limits during initial power flow.",
    )
    q_limit_tolerance: float = Field(
        1e-8,
        ge=0.0,
        description="Per-unit tolerance for initial power-flow Q-limit activation.",
    )
    max_q_limit_iterations: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum active-set power-flow solves; defaults to the PV-bus count plus one.",
    )
    power_flow_validation: PowerFlowValidationConfig = Field(
        default_factory=PowerFlowValidationConfig,
        description="Optional final operating-point checks before dynamic initialization.",
    )
    enforce_dynamic_limits: bool = Field(
        True,
        description="Enable hard dynamic-state limits for models that expose them.",
    )
    dynamic_limit_tolerance: float = Field(
        1e-8,
        ge=0.0,
        description="State-bound tolerance for hard dynamic limits.",
    )
    dynamic_limit_release_tolerance: float = Field(
        1e-10,
        ge=0.0,
        description="Complementarity tolerance for releasing hard dynamic limits.",
    )
    max_dynamic_limit_iterations: int = Field(
        20,
        gt=0,
        description="Maximum active-set iterations for hard dynamic limits.",
    )
    solve_powerflow_dynamics: bool = Field(True, description="Solve power flow before dynamics.")
    arkimex: bool = Field(False, description="Use ARKIMEX integrator.")
    check_jacobian: bool = Field(False, description="Run FD Jacobian check (non-PETSc only).")
    jacobian_mode: str = Field(
        "analytical", description="Residual Jacobian assembly: analytical | finite_difference."
    )
    finite_difference_epsilon: float = Field(1e-7, gt=0.0, allow_inf_nan=False)
    jacobian_check_tol: float = Field(1e-6, description="Absolute tolerance for Jacobian FD checks.")
    jacobian_check_top_k: int = Field(10, description="Number of Jacobian mismatches to report.")
    jacobian_check_csv: Optional[str] = Field(None, description="Optional CSV path for Jacobian mismatch report.")
    arkimex_slow_differential: Optional[List[int]] = Field(
        default=None,
        description=(
            "Optional list with the global indexes of differential equations that"
            " must be treated as slow in the ARKIMEX fast/slow split."
        ),
    )
    arkimex_fast_differential: Optional[List[int]] = Field(
        default=None,
        description=(
            "Optional list with the global indexes of differential equations that"
            " must be treated as fast in the ARKIMEX fast/slow split."
        ),
    )
    method: str = Field(
        "beuler",
        description="Integrator method: beuler | cn | herk2 | herk4.",
    )
    herk_alg_tol: float = Field(1e-10, gt=0.0, description="HERK stage algebraic tolerance.")
    herk_alg_max_iter: int = Field(50, gt=0, description="HERK stage Newton max iterations.")

    @field_validator('steps', mode='after')
    def steps_non_negative(cls, v, info: Any):
        if v == 0 or v < -1:
            raise ValueError("`steps` must be -1 or a positive integer.")
        return v

    @field_validator("method")
    def supported_method(cls, value):
        if value not in {"beuler", "cn", "herk2", "herk4"}:
            raise ValueError(
                "`method` must be one of: beuler, cn, herk2, herk4."
            )
        return value

    @field_validator("jacobian_mode")
    def supported_jacobian_mode(cls, value):
        if value not in {"analytical", "finite_difference"}:
            raise ValueError("`jacobian_mode` must be 'analytical' or 'finite_difference'.")
        return value

    @model_validator(mode="after")
    def validate_integration_contract(self):
        slow = self.arkimex_slow_differential
        fast = self.arkimex_fast_differential
        if self.enforce_dynamic_limits:
            conflicts = [
                name
                for name, enabled in (
                    ("arkimex", self.arkimex),
                    ("comp_sens", self.comp_sens),
                    ("fsolve", self.fsolve),
                )
                if enabled
            ]
            if conflicts:
                formatted = ", ".join(f"`{name}=True`" for name in conflicts)
                raise ValueError(
                    "Hard dynamic limits are incompatible with "
                    f"{formatted}. Set `enforce_dynamic_limits=False` to use "
                    "these legacy integration paths."
                )
        if slow is not None and fast is not None:
            raise ValueError(
                "Specify only one of `arkimex_slow_differential` or `arkimex_fast_differential`."
            )
        if self.toff < self.ton:
            raise ValueError("`toff` must be greater than or equal to `ton`.")
        if self.method == "cn" and not self.petsc:
            raise ValueError("`method='cn'` requires `petsc=True`.")
        if self.method in {"herk2", "herk4"} and self.petsc:
            raise ValueError(f"`method='{self.method}'` requires `petsc=False`.")
        if self.jacobian_mode == "finite_difference" and (
            self.petsc or self.method != "beuler"
        ):
            raise ValueError(
                "`jacobian_mode='finite_difference'` requires native backward Euler."
            )
        if self.jacobian_mode == "finite_difference" and self.check_jacobian:
            raise ValueError(
                "`check_jacobian=True` requires `jacobian_mode='analytical'`."
            )
        if self.comp_sens and (not self.petsc or self.method != "cn"):
            raise ValueError(
                "`comp_sens=True` requires `petsc=True` and `method='cn'`."
            )
        if self.arkimex:
            if not self.petsc:
                raise ValueError("`arkimex=True` requires `petsc=True`.")
            if self.method != "beuler":
                raise ValueError(
                    "`arkimex=True` cannot be combined with `cn`, `herk2`, or `herk4`."
                )

        reserved_petsc_options = {
            "ts_type",
            "ts_theta_theta",
            "ts_theta_endpoint",
            "ts_dt",
            "ts_max_time",
            "ts_max_steps",
            "ts_exact_final_time",
            "ts_time_span",
            "ts_adapt_type",
        }
        conflicts = sorted({
            arg.lstrip("-").split("=", 1)[0]
            for arg in self.petsc_args
            if arg.startswith("-")
            and arg.lstrip("-").split("=", 1)[0] in reserved_petsc_options
        })
        if conflicts:
            raise ValueError(
                "PETSc options cannot override IntegrationConfig method or time grid: "
                + ", ".join(f"-{name}" for name in conflicts)
            )
        return self
    
class IntegrationCtx:
    def __init__(self):
        self.z0_user = None
        self.theta_user = None

    def set_initial_conditions(self, z0):
        if not isinstance(z0, np.ndarray):
            raise ValueError("Initial conditions `z0` must be a numpy array.")
        self.z0_user = z0

    def set_theta(self, theta):
        if not isinstance(theta, np.ndarray):
            raise ValueError("Theta must be a numpy array.")
        self.theta_user = theta
