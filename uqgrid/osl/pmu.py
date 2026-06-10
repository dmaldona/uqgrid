"""PMU-style post-processing of a uqgrid integration result.

Takes the dense ``history`` returned by ``integrate_system`` and turns it
into a PMU-shaped measurement bundle: down-sampled at the configured rate
(default 30 Hz), masked to a configurable observed-bus set, and with
randomly-injected missing-sample packets.

The P/M-class filter is approximated by a single first-order Butterworth
lowpass. The actual EPRI PMU Emulator referenced in the paper is more
sophisticated; this is an admitted stand-in and the cutoff is recorded in
the case metadata.

Reference
---------
Maslennikov, S. and Wang, B. (2022). NREL/CP-6A40-81394, §III-A, III-E,
III-F. https://www.nrel.gov/docs/fy22osti/81394.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

try:
    from scipy.signal import butter, filtfilt
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass
class PMUEmulator:
    """Configurable PMU post-processor.

    Parameters
    ----------
    rate_hz : float
        Output sampling rate (paper uses 30 Hz).
    p_class_fraction : float
        Fraction of PMUs assigned P-class (rest are M-class). The paper
        used 30% M, 70% P; we expose the P-fraction directly. Currently
        only used as metadata + per-bus filter cutoff selection.
    p_class_cutoff_hz, m_class_cutoff_hz : float
        Lowpass cutoffs used as a stand-in for the actual EPRI filter.
    observed_buses : Sequence[int] | float | str | None
        Which buses are observed. ``None`` → all. A float in (0,1] or a
        string like ``"50%"`` → that fraction picked at random with the
        provided seed.
    missing_rate : float
        Average fraction of samples replaced by NaN (per channel),
        applied as small packets to mimic real PMU data drops.
    packet_size_range : (int, int)
        Min/max packet length of consecutive missing samples.
    seed : int | None
        RNG seed for observability + missing-sample draws.
    """

    rate_hz: float = 30.0
    p_class_fraction: float = 0.70
    p_class_cutoff_hz: float = 7.0
    m_class_cutoff_hz: float = 4.0
    observed_buses: Union[Sequence[int], float, str, None] = None
    missing_rate: float = 0.0
    packet_size_range: tuple = (1, 8)
    seed: Optional[int] = None

    def _resolve_observed(self, psys, rng) -> np.ndarray:
        if self.observed_buses is None:
            return np.arange(psys.nbuses, dtype=np.int64)
        if isinstance(self.observed_buses, str) and self.observed_buses.endswith("%"):
            frac = float(self.observed_buses[:-1]) / 100.0
        elif isinstance(self.observed_buses, float):
            frac = float(self.observed_buses)
        else:
            return np.asarray([_to_internal(psys, b) for b in self.observed_buses], dtype=np.int64)
        k = max(1, int(round(frac * psys.nbuses)))
        return np.sort(rng.choice(psys.nbuses, size=k, replace=False)).astype(np.int64)

    def _lowpass(self, x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
        if not _HAS_SCIPY or cutoff <= 0.0 or cutoff >= fs / 2.0:
            return x
        b, a = butter(N=1, Wn=cutoff / (fs / 2.0), btype="low")
        return filtfilt(b, a, x, axis=-1)

    def process(self, history: np.ndarray, tvec: np.ndarray, psys) -> dict:
        """Build the PMU bundle.

        Parameters
        ----------
        history : ndarray, shape (sys_size, nsteps)
            Dense state history from ``integrate_system``.
        tvec : ndarray, shape (nsteps,)
            Simulation time vector.
        psys : Psystem

        Returns
        -------
        dict with keys: t, observed_buses_internal, observed_buses_psse,
        V_mag, V_ang, I_mag, I_ang, branches, missing_mask_V, missing_mask_I,
        pmu_class, filter_cutoff_hz_per_bus, rate_hz.
        """
        rng = np.random.default_rng(self.seed)

        alg_size = psys.num_dof_alg
        dif_size = psys.num_dof_dif
        nbus = psys.nbuses
        v_block_start = dif_size + alg_size

        nsteps = history.shape[1]
        sim_dt = float(tvec[1] - tvec[0]) if nsteps > 1 else 1.0
        sim_rate = 1.0 / sim_dt
        decim = max(1, int(round(sim_rate / self.rate_hz)))

        v_all = history[v_block_start:v_block_start + 2 * nbus, :]

        # Real/imag vs polar form depends on psys.power_injection. Detect:
        if getattr(psys, "power_injection", False):
            vm_full = v_all[0::2, :]
            va_full = v_all[1::2, :]
        else:
            vr_full = v_all[0::2, :]
            vi_full = v_all[1::2, :]
            vm_full = np.sqrt(vr_full**2 + vi_full**2)
            va_full = np.arctan2(vi_full, vr_full)

        observed = self._resolve_observed(psys, rng)
        n_obs = observed.size

        is_p_class = rng.random(n_obs) < self.p_class_fraction
        cutoffs = np.where(is_p_class, self.p_class_cutoff_hz, self.m_class_cutoff_hz)

        vm_obs = np.empty((n_obs, nsteps))
        va_obs = np.empty((n_obs, nsteps))
        for k, (bus, cut) in enumerate(zip(observed, cutoffs)):
            vm_obs[k, :] = self._lowpass(vm_full[bus, :], sim_rate, float(cut))
            va_obs[k, :] = self._lowpass(np.unwrap(va_full[bus, :]), sim_rate, float(cut))

        # Down-sample
        sel = np.arange(0, nsteps, decim)
        V_mag = vm_obs[:, sel]
        V_ang = va_obs[:, sel]
        t_out = tvec[sel]

        # Branch currents
        I_mag, I_ang, branches_obs = self._branch_currents(psys, v_all, observed, sel, sim_rate, cutoffs)

        missing_V = self._draw_missing_mask(V_mag.shape, rng)
        missing_I = self._draw_missing_mask(I_mag.shape, rng) if I_mag.size else np.zeros_like(I_mag, dtype=bool)

        V_mag = np.where(missing_V, np.nan, V_mag)
        V_ang = np.where(missing_V, np.nan, V_ang)
        if I_mag.size:
            I_mag = np.where(missing_I, np.nan, I_mag)
            I_ang = np.where(missing_I, np.nan, I_ang)

        int2ext = {v: k for k, v in getattr(psys, "ext2int", {}).items()}
        observed_psse = [int2ext.get(int(b), int(b)) for b in observed]

        return {
            "t": t_out,
            "observed_buses_internal": observed,
            "observed_buses_psse": np.asarray(observed_psse),
            "V_mag": V_mag,
            "V_ang": V_ang,
            "I_mag": I_mag,
            "I_ang": I_ang,
            "branches": branches_obs,
            "missing_mask_V": missing_V,
            "missing_mask_I": missing_I,
            "pmu_class": np.where(is_p_class, "P", "M"),
            "filter_cutoff_hz_per_bus": cutoffs,
            "rate_hz": float(self.rate_hz),
        }

    def _branch_currents(self, psys, v_all, observed, sel, sim_rate, cutoffs):
        observed_set = set(int(b) for b in observed)
        chosen = [(k, br) for k, br in enumerate(psys.branches)
                  if br.fr in observed_set or br.to in observed_set]
        if not chosen:
            return (np.zeros((0, sel.size)), np.zeros((0, sel.size)),
                    np.zeros((0, 2), dtype=np.int64))

        nbus = psys.nbuses
        if getattr(psys, "power_injection", False):
            vm = v_all[0::2, :]
            va = v_all[1::2, :]
            vr = vm * np.cos(va)
            vi = vm * np.sin(va)
        else:
            vr = v_all[0::2, :]
            vi = v_all[1::2, :]

        Vc = vr + 1j * vi  # shape (nbus, nsteps)

        nsteps = Vc.shape[1]
        I_mag_full = np.empty((len(chosen), nsteps))
        I_ang_full = np.empty((len(chosen), nsteps))
        br_pairs = np.empty((len(chosen), 2), dtype=np.int64)

        bus_to_obs_k = {int(b): k for k, b in enumerate(observed)}

        for row, (br_idx, br) in enumerate(chosen):
            z = complex(br.r, br.x)
            if abs(z) < 1e-12:
                I_series = np.zeros(nsteps, dtype=np.complex128)
            else:
                I_series = (Vc[br.fr, :] - Vc[br.to, :]) / z
                if getattr(br, "sh", 0.0):
                    I_series = I_series + 1j * br.sh / 2.0 * Vc[br.fr, :]
            I_mag_full[row, :] = np.abs(I_series)
            I_ang_full[row, :] = np.unwrap(np.angle(I_series))
            br_pairs[row, :] = (br.fr, br.to)

            obs_bus = br.fr if br.fr in bus_to_obs_k else br.to
            cut = float(cutoffs[bus_to_obs_k[obs_bus]])
            I_mag_full[row, :] = self._lowpass(I_mag_full[row, :], sim_rate, cut)
            I_ang_full[row, :] = self._lowpass(I_ang_full[row, :], sim_rate, cut)

        return I_mag_full[:, sel], I_ang_full[:, sel], br_pairs

    def _draw_missing_mask(self, shape, rng) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        if self.missing_rate <= 0.0 or mask.size == 0:
            return mask
        n_channels, n_samples = shape
        target_missing = self.missing_rate * n_samples
        pmin, pmax = self.packet_size_range
        for ch in range(n_channels):
            placed = 0
            while placed < target_missing:
                pkt = int(rng.integers(pmin, pmax + 1))
                start = int(rng.integers(0, max(1, n_samples - pkt)))
                mask[ch, start:start + pkt] = True
                placed += pkt
        return mask


def _to_internal(psys, bus):
    if hasattr(psys, "ext2int") and bus in psys.ext2int:
        return psys.ext2int[bus]
    return int(bus)
