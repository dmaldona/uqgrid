"""uqgrid.osl — Oscillation Source Location case generation.

Generates synthetic PMU datasets with labeled forced-oscillation sources
and colored-noise loads, in the style of the 2021 IEEE-NASPI Oscillation
Source Location Contest cases.

Reference
---------
Maslennikov, S. and Wang, B. (2022). *Creation of Simulated Test Cases
for the Oscillation Source Location Contest.* NREL/CP-6A40-81394.
2022 IEEE PES General Meeting.
https://www.nrel.gov/docs/fy22osti/81394.pdf

Quick start
-----------
    from uqgrid.osl import build_osl_case, ForcedOscillation

    case = build_osl_case(
        raw="data/ACTIVSg200.raw",
        dyr="data/ACTIVSg200.dyr",
        forced_oscillations=[
            ForcedOscillation(target=("gov", 7), freq_hz=0.82,
                              amplitude=0.02, t_start=5.0, t_end=30.0),
        ],
        colored_noise=True,
        tend=30.0, dt=1/240,
        pmu=dict(rate_hz=30, observed_buses="50%", missing_rate=0.001, seed=42),
        label="ACTIVSg200_case01",
    )
    case.export("cases/ACTIVSg200_case01")
"""

from .injectors import (
    SignalInjector,
    ForcedOscillation,
    ColoredNoise,
)
from .pmu import PMUEmulator
from .scenario import OSLCase, build_osl_case

__all__ = [
    "SignalInjector",
    "ForcedOscillation",
    "ColoredNoise",
    "PMUEmulator",
    "OSLCase",
    "build_osl_case",
]
