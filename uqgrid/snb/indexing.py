from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np

from uqgrid.core.psydef import Psystem


@dataclass(frozen=True)
class PFIndexCache:
    """Cached structural metadata for a power-flow system."""

    bus_type: np.ndarray
    pq_buses: np.ndarray
    pv_buses: np.ndarray
    slack_buses: np.ndarray
    pq_indices: np.ndarray
    pqv_indices: np.ndarray
    n_pq: int
    n_pv: int
    n_slack: int

    @property
    def n_pqv(self) -> int:
        return int(self.n_pq + self.n_pv)

    @property
    def n_unknowns(self) -> int:
        return int(2 * self.n_pq + self.n_pv)


def _classify_buses(psys: Psystem) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bus_type = np.array([bus.type for bus in psys.buses], dtype=int)
    pq = np.where(bus_type == 1)[0]
    pv = np.where(bus_type == 2)[0]
    slack = np.where(bus_type == 3)[0]
    return pq, pv, slack


def _build_unknown_indices(psys: Psystem, bus_type: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pq_mask = (bus_type == 1).astype(int)
    pq_indices = np.where(pq_mask == 1, np.cumsum(pq_mask) - 1, -1)

    pqv_mask = np.logical_or(bus_type == 1, bus_type == 2).astype(int)
    pqv_indices = np.where(pqv_mask == 1, np.cumsum(pqv_mask) - 1, -1)

    return pq_indices, pqv_indices


def build_index_cache(psys: Psystem) -> PFIndexCache:
    """Compute and cache bus-type masks and unknown ordering for `psys`."""

    if psys.nbuses == 0:
        raise ValueError("Power system has no buses; cannot build index cache.")

    bus_type = np.array([bus.type for bus in psys.buses], dtype=int)
    pq_buses, pv_buses, slack_buses = _classify_buses(psys)
    pq_indices, pqv_indices = _build_unknown_indices(psys, bus_type)

    return PFIndexCache(
        bus_type=bus_type,
        pq_buses=pq_buses,
        pv_buses=pv_buses,
        slack_buses=slack_buses,
        pq_indices=pq_indices,
        pqv_indices=pqv_indices,
        n_pq=int(pq_buses.size),
        n_pv=int(pv_buses.size),
        n_slack=int(slack_buses.size),
    )
