from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from uqgrid.core.psydef import Psystem

from .indexing import build_index_cache
from .params import extract_lambda


@dataclass(frozen=True)
class Dobson5Setup:
    """Container for the Dobson five-bus closest-SNB benchmark."""

    psys: Psystem
    x_init: np.ndarray
    w_init: np.ndarray
    lambda_init: np.ndarray
    lambda0: np.ndarray
    lambda_reference: np.ndarray
    expected_delta: np.ndarray
    k_init: float


def build_dobson5_case() -> Dobson5Setup:
    """Return the Dobson & Lu five-bus system and benchmark data."""

    psys = Psystem(basemva=1.0)

    # Buses: id tags follow the paper's numbering (1-based).
    psys.add_bus(1, 3)  # Slack
    psys.add_bus(2, 1)  # PQ
    psys.add_bus(3, 2)  # PV
    psys.add_bus(4, 1)  # PQ
    psys.add_bus(5, 1)  # PQ

    # Voltage set-points / initial guesses (per-unit magnitude, radians).
    psys.buses[0].set_vinit(1.04, 0.0)
    psys.buses[1].set_vinit(1.0, 0.0)
    psys.buses[2].set_vinit(1.02, 0.0)
    psys.buses[3].set_vinit(1.0, 0.0)
    psys.buses[4].set_vinit(1.0, 0.0)

    # Branch data (converted to zero-based indexing for psys).
    lines = [
        (0, 1, 1.40056, -5.60224),
        (0, 4, 1.84118, -7.48352),
        (1, 2, 1.84118, -7.48352),
        (2, 3, 0.70028, -2.80112),
        (2, 4, 1.12985, -4.47675),
        (3, 4, 0.93372, -3.43483),
    ]
    for fr, to, g, b in lines:
        admittance = g + 1j * b
        impedance = 1.0 / admittance
        psys.add_branch(fr, to, r=float(np.real(impedance)), x=float(np.imag(impedance)))

    # Generators: slack at bus 1, PV at bus 3 (P = +1.1 per Dobson sign convention).
    psys.add_gen(bus=0, idx_name="G1", psch=0.0, qsch=0.0)
    psys.add_gen(bus=2, idx_name="G3", psch=-1.1, qsch=0.0)

    # PQ loads correspond to lambda_0 in the paper (positive loads in this project).
    psys.add_load(bus=1, tag="LD2", pload=1.1500, qload=0.6000)
    psys.add_load(bus=3, tag="LD4", pload=0.7000, qload=0.3000)
    psys.add_load(bus=4, tag="LD5", pload=0.7000, qload=0.4000)

    psys.assemble()
    psys.createYbusComplex()

    cache = build_index_cache(psys)
    lambda0 = extract_lambda(psys, cache)

    # Reorder the published initial guesses into the solver's state ordering:
    # x = [V_PQ, angles_{PQ,PV}]. Published data lists angles first.
    voltage_guess = {1: 0.95, 3: 0.95, 4: 0.95}
    angle_guess = {1: -0.10, 2: -0.05, 3: -0.17, 4: -0.10}

    voltage_block = np.array([voltage_guess[bus] for bus in cache.pq_buses], dtype=float)
    angle_block = np.zeros(cache.n_pq + cache.n_pv, dtype=float)
    for bus_idx in range(psys.nbuses):
        slot = cache.pqv_indices[bus_idx]
        if slot >= 0:
            angle_block[slot] = angle_guess[bus_idx]
    x_init = np.concatenate([voltage_block, angle_block])

    # Left-null initial guess (published order: angles, then voltages).
    w_voltage_guess = {1: 0.0055, 3: -0.0326, 4: 0.0074}
    w_angle_guess = {1: 0.35, 2: 0.48, 3: 0.71, 4: 0.35}

    w_voltage_block = np.array([w_voltage_guess[bus] for bus in cache.pq_buses], dtype=float)
    w_angle_block = np.zeros(cache.n_pq + cache.n_pv, dtype=float)
    for bus_idx in range(psys.nbuses):
        slot = cache.pqv_indices[bus_idx]
        if slot >= 0:
            w_angle_block[slot] = w_angle_guess[bus_idx]
    w_init = np.concatenate([w_voltage_block, w_angle_block])

    lambda_init = np.array([1.7919, 2.0080, 1.3522, 0.6101, 0.2404, 0.4136])
    lambda_star = np.array([1.1584, 1.1659, 0.7491, 0.6008, 1.0495, 0.5320])
    expected_delta = lambda_star - lambda0

    return Dobson5Setup(
        psys=psys,
        x_init=x_init,
        w_init=w_init,
        lambda_init=lambda_init,
        lambda0=lambda0,
        lambda_reference=lambda_star,
        expected_delta=expected_delta,
        k_init=1.0,
    )
