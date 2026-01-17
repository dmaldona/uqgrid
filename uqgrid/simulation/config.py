from typing import Any, List, Optional, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
import numpy as np
from uqgrid.simulation.sparse_solvers import DEFAULT_SPARSE_SOLVER

class IntegrationConfig(BaseModel):
    power_injection: bool = False
    tend: float = Field(10.0, description="Integration end time in seconds.")
    dt: float = Field(1.0 / 120.0, description="Time step in seconds.")
    steps: int = Field(-1, description="Number of integration steps. If >0, overrides tend.")
    verbose: bool = Field(False, description="Enable verbose output.")
    comp_sens: bool = Field(False, description="Compute first and second-order sensitivities.")
    fsolve: bool = Field(False, description="Use fsolve for solving nonlinear equations.")
    ton: float = Field(0.25, description="Fault activation time.")
    toff: float = Field(0.4, description="Fault deactivation time.")
    petsc: bool = Field(False, description="Enable PETSc integration.")
    sparse_solver: Literal["scipy", "klu"] = Field(
        DEFAULT_SPARSE_SOLVER,
        description="Sparse solver for power flow and non-PETSc dynamics."
    )
    solve_powerflow_dynamics: bool = Field(True, description="Solve power flow before dynamics.")
    arkimex: bool = Field(False, description="Use ARKIMEX integrator.")
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

    @field_validator('tend', 'dt', 'ton', 'toff', mode='after')
    def positive_values(cls, v, info: Any):
        if v < 0:
            raise ValueError(f"`{info.field_name}` must be non-negative.")
        return v

    @field_validator('steps', mode='after')
    def steps_non_negative(cls, v, info: Any):
        if v < -1:
            raise ValueError("`steps` must be -1 or a non-negative integer.")
        return v

    @model_validator(mode="after")
    def validate_partition_spec(cls, values):
        slow = values.arkimex_slow_differential
        fast = values.arkimex_fast_differential
        if slow is not None and fast is not None:
            raise ValueError(
                "Specify only one of `arkimex_slow_differential` or `arkimex_fast_differential`."
            )
        return values
    
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
