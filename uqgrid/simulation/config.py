from pydantic import BaseModel, Field, field_validator

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

    @field_validator('tend', 'dt', 'ton', 'toff', mode='after')
    def positive_values(cls, v, info):
        if v < 0:
            raise ValueError(f"`{info.field_name}` must be non-negative.")
        return v

    @field_validator('steps', mode='after')
    def steps_non_negative(cls, v, info):
        if v < -1:
            raise ValueError("`steps` must be -1 or a non-negative integer.")
        return v