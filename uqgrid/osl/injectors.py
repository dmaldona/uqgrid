"""Signal injectors for forced oscillations and colored-noise loads.

Each injector exposes ``update(t, theta, psys)`` and is appended to
``psys.signal_injectors``. The dynamics integrator calls ``update`` once
per step before solving, mutating ``theta`` in place. Empty injector
list means zero overhead.

Targeted theta offsets are model-specific and documented inline:

* TGOV1 ``pref``  at ``gov.par_ptr + 8`` (uqgrid/models/tgov1_imp.py:141)
* SEXS  ``vref``  at ``exc.par_ptr + 7`` (uqgrid/models/sexs_imp.py:120)
* Load  ``pl,ql`` at ``load.par_ptr + 0/+1`` (uqgrid/models/load_imp.py:15-16)

Reference
---------
Maslennikov, S. and Wang, B. (2022). *Creation of Simulated Test Cases
for the Oscillation Source Location Contest.* NREL/CP-6A40-81394.
2022 IEEE PES General Meeting.
https://www.nrel.gov/docs/fy22osti/81394.pdf

The forced-oscillation injectors implement §III-C (sinusoidal and
rectangular forcing in TGOV1 governors, SEXS exciters, and loads). The
colored-noise injector implements §III-D (sum of low- and high-frequency
random components on every load).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


TGOV1_PREF_OFFSET = 8
SEXS_VREF_OFFSET = 7
LOAD_PL_OFFSET = 0
LOAD_QL_OFFSET = 1


def _resolve_bus(psys, bus) -> int:
    """Accept an internal bus index or a PSSE bus number; return internal index."""
    if hasattr(psys, "ext2int") and bus in psys.ext2int:
        return psys.ext2int[bus]
    if 0 <= int(bus) < psys.nbuses:
        return int(bus)
    raise KeyError(f"Bus {bus!r} not found (neither internal index nor PSSE id).")


def _find_governor_param(psys, bus_internal: int, tag=None) -> int:
    matches = [g for g in psys.gov if g.bus == bus_internal and (tag is None or str(g.id_tag) == str(tag))]
    if not matches:
        raise ValueError(f"No governor found at internal bus {bus_internal} (tag={tag}).")
    if len(matches) > 1:
        raise ValueError(f"Multiple governors at bus {bus_internal}; specify tag.")
    return matches[0].par_ptr + TGOV1_PREF_OFFSET


def _find_exciter_param(psys, bus_internal: int, tag=None) -> int:
    matches = [e for e in psys.exc if e.bus == bus_internal and (tag is None or str(e.id_tag) == str(tag))]
    if not matches:
        raise ValueError(f"No exciter found at internal bus {bus_internal} (tag={tag}).")
    if len(matches) > 1:
        raise ValueError(f"Multiple exciters at bus {bus_internal}; specify tag.")
    return matches[0].par_ptr + SEXS_VREF_OFFSET


def _find_load_params(psys, bus_internal: int, tag=None) -> Tuple[int, int]:
    matches = [l for l in psys.loads if l.bus == bus_internal and (tag is None or str(l.id_tag) == str(tag))]
    if not matches:
        raise ValueError(f"No load found at internal bus {bus_internal} (tag={tag}).")
    if len(matches) > 1:
        raise ValueError(f"Multiple loads at bus {bus_internal}; specify tag.")
    load = matches[0]
    return load.par_ptr + LOAD_PL_OFFSET, load.par_ptr + LOAD_QL_OFFSET


class SignalInjector(ABC):
    """Base class. Subclasses mutate theta in place every integration step."""

    @abstractmethod
    def update(self, t: float, theta: np.ndarray, psys) -> None:
        ...


@dataclass
class ForcedOscillation(SignalInjector):
    """Periodic forced oscillation on a single device parameter.

    Parameters
    ----------
    target : tuple
        One of:
          ``("gov", bus)``       -> TGOV1 pref
          ``("exc", bus)``       -> SEXS vref
          ``("load_p", bus)``    -> load active power
          ``("load_q", bus)``    -> load reactive power
        ``bus`` may be a PSSE bus number (preferred) or internal index.
        Optionally append a tag: ``("gov", bus, tag)``.
    freq_hz : float
        Fundamental forcing frequency.
    amplitude : float
        Peak deviation, in the *same per-unit system* as the targeted
        parameter (e.g. p.u. on system MVA for TGOV1 pref/load).
    t_start, t_end : float
        Active interval in seconds (relative to t=0 of integration).
    waveform : {"sine", "rectangular"}
        See paper §III-C, Fig. 2.
    harmonics : list[(rel_freq, rel_amp)], optional
        Additional harmonic components added on top of the fundamental,
        each ``(multiplier_of_freq, fraction_of_amplitude)``.
    phase : float
        Phase offset in radians (sine only).
    """

    target: Tuple
    freq_hz: float
    amplitude: float
    t_start: float = 0.0
    t_end: float = np.inf
    waveform: str = "sine"
    harmonics: Optional[Sequence[Tuple[float, float]]] = None
    phase: float = 0.0

    _par_idx: int = field(default=-1, init=False)
    _theta0: float = field(default=0.0, init=False)
    _resolved: bool = field(default=False, init=False)
    # for load targets: paired (pl_idx, ql_idx, y_real_idx, y_imag_idx, v0)
    _load_pair: Optional[tuple] = field(default=None, init=False, repr=False)

    def _resolve(self, psys) -> None:
        kind = self.target[0]
        bus_user = self.target[1]
        tag = self.target[2] if len(self.target) > 2 else None
        bus_int = _resolve_bus(psys, bus_user)

        if kind == "gov":
            self._par_idx = _find_governor_param(psys, bus_int, tag)
        elif kind == "exc":
            self._par_idx = _find_exciter_param(psys, bus_int, tag)
        elif kind in ("load_p", "load_q"):
            p_idx, q_idx = _find_load_params(psys, bus_int, tag)
            self._par_idx = p_idx if kind == "load_p" else q_idx
            # cache the load's full theta block so we can keep ZIP Y entries
            # consistent with the time-varying P/Q (load_imp.py reads
            # theta[pp+5,+6] as the constant-admittance part).
            self._load_pair = (p_idx, q_idx, p_idx + 5, p_idx + 6)
        else:
            raise ValueError(f"Unknown ForcedOscillation target kind {kind!r}.")
        self._resolved = True

    def _wave(self, t: float) -> float:
        omega = 2.0 * np.pi * self.freq_hz
        if self.waveform == "sine":
            val = np.sin(omega * t + self.phase)
        elif self.waveform == "rectangular":
            val = 1.0 if np.sin(omega * t + self.phase) >= 0.0 else -1.0
        else:
            raise ValueError(f"Unknown waveform {self.waveform!r}.")
        if self.harmonics:
            for k, rel_amp in self.harmonics:
                val += rel_amp * np.sin(k * omega * t + self.phase)
        return val

    def update(self, t: float, theta: np.ndarray, psys) -> None:
        if not self._resolved:
            self._resolve(psys)
            self._theta0 = float(theta[self._par_idx])
        if self.t_start <= t <= self.t_end:
            theta[self._par_idx] = self._theta0 + self.amplitude * self._wave(t)
        else:
            theta[self._par_idx] = self._theta0
        # Keep ZIP-load constant-admittance entries consistent with the new P/Q.
        if self._load_pair is not None:
            p_idx, q_idx, yr_idx, yi_idx = self._load_pair
            pl = theta[p_idx]
            ql = theta[q_idx]
            # v0 was captured at init in load_imp; re-derive from existing y0:
            # y0 = (pl0 + j*ql0) / v0**2  =>  v0**2 = (pl0+jql0)/y0
            # but simplest: just refresh y = (pl + j ql) / v0**2 using the
            # load's stored v0 (par_ptr+4).
            v0 = theta[p_idx + 4]
            if v0 > 0.0:
                inv_v02 = 1.0 / (v0 * v0)
                theta[yr_idx] = pl * inv_v02
                theta[yi_idx] = ql * inv_v02


@dataclass
class ColoredNoise(SignalInjector):
    """Additive load-power noise modelled as low-frequency + high-frequency
    components, applied to every load's pl and ql (paper §III-D, Fig. 3).

    The low-frequency component changes value at random intervals in
    ``tau_lf_range`` seconds. The high-frequency component changes every
    integration step. Both are zero-mean uniform variates scaled by
    ``sigma_*`` (interpreted as a fraction of the load's nominal pl/ql).
    """

    sigma_lf: float = 0.002
    sigma_hf: float = 0.001
    tau_lf_range: Tuple[float, float] = (0.5, 5.0)
    seed: Optional[int] = None
    apply_to_q: bool = True

    _rng: np.random.Generator = field(default=None, init=False, repr=False)
    _load_p_idx: np.ndarray = field(default=None, init=False, repr=False)
    _load_q_idx: np.ndarray = field(default=None, init=False, repr=False)
    _load_v0_idx: np.ndarray = field(default=None, init=False, repr=False)
    _load_yr_idx: np.ndarray = field(default=None, init=False, repr=False)
    _load_yi_idx: np.ndarray = field(default=None, init=False, repr=False)
    _theta0_p: np.ndarray = field(default=None, init=False, repr=False)
    _theta0_q: np.ndarray = field(default=None, init=False, repr=False)
    _lf_value_p: np.ndarray = field(default=None, init=False, repr=False)
    _lf_value_q: np.ndarray = field(default=None, init=False, repr=False)
    _lf_next_change: np.ndarray = field(default=None, init=False, repr=False)
    _resolved: bool = field(default=False, init=False)

    def _resolve(self, psys, t: float) -> None:
        self._rng = np.random.default_rng(self.seed)
        nloads = len(psys.loads)
        self._load_p_idx = np.array([l.par_ptr + LOAD_PL_OFFSET for l in psys.loads], dtype=np.int64)
        self._load_q_idx = np.array([l.par_ptr + LOAD_QL_OFFSET for l in psys.loads], dtype=np.int64)
        self._load_v0_idx = np.array([l.par_ptr + 4 for l in psys.loads], dtype=np.int64)
        self._load_yr_idx = np.array([l.par_ptr + 5 for l in psys.loads], dtype=np.int64)
        self._load_yi_idx = np.array([l.par_ptr + 6 for l in psys.loads], dtype=np.int64)
        self._lf_value_p = np.zeros(nloads)
        self._lf_value_q = np.zeros(nloads)
        self._lf_next_change = np.full(nloads, t)
        self._resolved = True

    def _maybe_refresh_lf(self, t: float, scale_p: np.ndarray, scale_q: np.ndarray) -> None:
        due = t >= self._lf_next_change
        if not np.any(due):
            return
        n_due = int(np.sum(due))
        self._lf_value_p[due] = self._rng.uniform(-self.sigma_lf, self.sigma_lf, size=n_due) * scale_p[due]
        if self.apply_to_q:
            self._lf_value_q[due] = self._rng.uniform(-self.sigma_lf, self.sigma_lf, size=n_due) * scale_q[due]
        dt_low, dt_high = self.tau_lf_range
        self._lf_next_change[due] = t + self._rng.uniform(dt_low, dt_high, size=n_due)

    def update(self, t: float, theta: np.ndarray, psys) -> None:
        if not self._resolved:
            self._resolve(psys, t)
            self._theta0_p = theta[self._load_p_idx].copy()
            self._theta0_q = theta[self._load_q_idx].copy()

        scale_p = np.abs(self._theta0_p) + 1e-12
        scale_q = np.abs(self._theta0_q) + 1e-12

        self._maybe_refresh_lf(t, scale_p, scale_q)

        hf_p = self._rng.uniform(-self.sigma_hf, self.sigma_hf, size=self._theta0_p.size) * scale_p
        theta[self._load_p_idx] = self._theta0_p + self._lf_value_p + hf_p
        if self.apply_to_q:
            hf_q = self._rng.uniform(-self.sigma_hf, self.sigma_hf, size=self._theta0_q.size) * scale_q
            theta[self._load_q_idx] = self._theta0_q + self._lf_value_q + hf_q

        v0 = theta[self._load_v0_idx]
        valid = v0 > 0.0
        if np.any(valid):
            inv_v02 = 1.0 / (v0[valid] * v0[valid])
            theta[self._load_yr_idx[valid]] = theta[self._load_p_idx[valid]] * inv_v02
            theta[self._load_yi_idx[valid]] = theta[self._load_q_idx[valid]] * inv_v02
