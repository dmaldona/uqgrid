"""Correctness tests for uqgrid.osl on the 2-bus PSSE fixture.

Asserts:
1. Empty signal_injectors produces bit-identical results to the baseline
   integration path (zero-overhead regression guard).
2. ForcedOscillation resolves the right theta index for a load target and
   mutates theta in place at every integration step (within its active
   interval).
3. build_osl_case wires the integrator + PMU emulator together, produces
   PMU output with the right shape, and round-trips through .export().
"""

from pathlib import Path

import json

import numpy as np

from uqgrid.io.parse import load_psse, add_dyr
from uqgrid.simulation.config import IntegrationConfig
from uqgrid.simulation.dynamics import integrate_system

from uqgrid.osl import (
    build_osl_case,
    ForcedOscillation,
    PMUEmulator,
)
from uqgrid.osl.injectors import LOAD_PL_OFFSET


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "2bus_33.raw"
DYR = REPO_ROOT / "data" / "2bus_TGOV1.dyr"

SIM_TEND = 2.0
SIM_DT = 1.0 / 240.0


def _build_psys():
    psys = load_psse(str(RAW))
    add_dyr(psys, str(DYR))
    psys.createYbusComplex()
    return psys


def test_zero_overhead_when_no_injectors():
    """Empty signal_injectors must produce bit-identical results."""
    psys_a = _build_psys()
    cfg = IntegrationConfig(tend=SIM_TEND, dt=SIM_DT, method="beuler")
    res_a = integrate_system(psys_a, cfg)

    psys_b = _build_psys()
    assert psys_b.signal_injectors == []
    res_b = integrate_system(psys_b, cfg)

    np.testing.assert_array_equal(res_a["history"], res_b["history"])
    np.testing.assert_array_equal(res_a["tvec"], res_b["tvec"])


def test_forced_oscillation_resolves_and_mutates_theta():
    """Injector must resolve the right theta index and mutate it on update."""
    psys = _build_psys()
    load = psys.loads[0]
    expected_idx = load.par_ptr + LOAD_PL_OFFSET
    psse_bus = next(k for k, v in psys.ext2int.items() if v == load.bus)

    fo = ForcedOscillation(
        target=("load_p", psse_bus),
        freq_hz=1.0,
        amplitude=0.1,
        t_start=0.5,
        t_end=1.5,
        waveform="sine",
    )

    theta = np.zeros(psys.num_pars)
    theta[expected_idx] = load.pload  # mimic initialize_theta
    theta[load.par_ptr + 4] = 1.0     # v0
    theta0 = theta.copy()

    fo.update(0.0, theta, psys)
    assert fo._par_idx == expected_idx
    # before t_start: theta restored to baseline
    np.testing.assert_allclose(theta[expected_idx], theta0[expected_idx])

    fo.update(0.75, theta, psys)
    # inside active window with phase 0, sin(2π·1·0.75) = -1 → -amplitude
    assert theta[expected_idx] != theta0[expected_idx]
    np.testing.assert_allclose(
        theta[expected_idx] - theta0[expected_idx],
        0.1 * np.sin(2.0 * np.pi * 0.75),
        atol=1e-12,
    )

    fo.update(2.0, theta, psys)
    # after t_end: restored to baseline
    np.testing.assert_allclose(theta[expected_idx], theta0[expected_idx])


def test_build_osl_case_shape_and_export(tmp_path):
    case = build_osl_case(
        raw=RAW,
        dyr=DYR,
        forced_oscillations=[
            ForcedOscillation(
                target=("load_p", 2),
                freq_hz=1.0,
                amplitude=0.05,
                t_start=0.5,
                t_end=SIM_TEND,
            ),
        ],
        tend=SIM_TEND,
        dt=SIM_DT,
        pmu=PMUEmulator(rate_hz=30.0, seed=1),
        label="2bus_test_fo",
    )

    pmu = case.pmu
    assert pmu["V_mag"].shape[0] == 2
    assert pmu["V_mag"].shape == pmu["V_ang"].shape
    assert pmu["V_mag"].shape[1] == pmu["t"].shape[0]
    assert pmu["t"][-1] <= SIM_TEND + SIM_DT * 2
    assert pmu["pmu_class"].shape == (2,)
    assert set(pmu["pmu_class"].tolist()).issubset({"P", "M"})

    npz_path, json_path = case.export(tmp_path / "case")
    assert npz_path.exists() and json_path.exists()
    loaded = np.load(npz_path)
    assert "V_mag" in loaded.files

    meta = json.loads(json_path.read_text())
    assert meta["sources"][0]["kind"] == "load_p"
    assert meta["sources"][0]["freq_hz"] == 1.0
    assert meta["pmu"]["rate_hz"] == 30.0
