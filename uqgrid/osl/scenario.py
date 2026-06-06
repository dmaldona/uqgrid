"""High-level builder for Oscillation Source Location (OSL) test cases.

End-to-end pipeline:
    load_psse + add_dyr  ->  attach injectors  ->  integrate_system
                          ->  PMUEmulator.process  ->  OSLCase

The output OSLCase can be exported to an ``.npz`` (PMU signals) plus a
``.json`` (metadata: source description, frequency, PMU class assignment,
observed buses, seeds) — one row of Table III of the reference paper.

Reference
---------
Maslennikov, S. and Wang, B. (2022). *Creation of Simulated Test Cases
for the Oscillation Source Location Contest.* NREL/CP-6A40-81394.
2022 IEEE PES General Meeting.
https://www.nrel.gov/docs/fy22osti/81394.pdf

Out of scope in this first cut (see plan): PSS auto-tuning, HVDC sources,
DEF/OSLp scoring, EPRI-accurate P/M filter, PETSc integration path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from uqgrid.core.psydef import Psystem
from uqgrid.io.parse import load_psse, add_dyr
from uqgrid.simulation.dynamics import integrate_system
from uqgrid.simulation.config import IntegrationConfig

from .injectors import ForcedOscillation, ColoredNoise, SignalInjector
from .pmu import PMUEmulator


@dataclass
class OSLCase:
    """One generated OSL test case: PMU signals + ground-truth metadata."""

    pmu: Dict[str, Any]
    metadata: Dict[str, Any]
    raw_history: Optional[np.ndarray] = None
    raw_tvec: Optional[np.ndarray] = None

    def export(self, stem: Union[str, Path], include_raw: bool = False) -> Tuple[Path, Path]:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        npz_path = stem.with_suffix(".npz")
        json_path = stem.with_suffix(".json")

        npz_payload = {k: np.asarray(v) for k, v in self.pmu.items()
                       if isinstance(v, (np.ndarray, list, tuple))}
        # scalars/strings -> metadata only
        if include_raw and self.raw_history is not None:
            npz_payload["raw_history"] = self.raw_history
            npz_payload["raw_tvec"] = self.raw_tvec

        np.savez(npz_path, **npz_payload)
        json_path.write_text(json.dumps(self.metadata, indent=2, default=_json_default))
        return npz_path, json_path


def build_osl_case(
    raw: Union[str, Path],
    dyr: Union[str, Path],
    *,
    forced_oscillations: Optional[List[ForcedOscillation]] = None,
    colored_noise: Union[bool, ColoredNoise] = False,
    extra_injectors: Optional[List[SignalInjector]] = None,
    tend: float = 90.0,
    dt: float = 1.0 / 240.0,
    method: str = "beuler",
    pmu: Optional[Union[PMUEmulator, Dict[str, Any]]] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    label: Optional[str] = None,
    keep_raw: bool = False,
) -> OSLCase:
    """Build one OSL case end-to-end.

    Parameters
    ----------
    raw, dyr : path
        PSSE RAW/DYR pair for the test system (e.g. ACTIVSg200).
    forced_oscillations : list[ForcedOscillation]
        Sources to inject. Use the convenience ``ForcedOscillation(...)``
        constructor. Empty/None → no FO injected (ambient case).
    colored_noise : bool | ColoredNoise
        ``True`` → attach a default ColoredNoise instance. A
        ``ColoredNoise`` instance is used directly. ``False`` → none.
    extra_injectors : list[SignalInjector]
        Any additional custom injectors to attach.
    tend, dt, method : simulation controls passed to IntegrationConfig.
    pmu : PMUEmulator | dict | None
        Post-processor. A dict is treated as PMUEmulator(**dict). None
        uses defaults (30 Hz, all buses observed, no missing samples).
    extra_config : dict
        Extra IntegrationConfig overrides (e.g. ``{"verbose": True}``).
        Cannot include the keys this function sets (``tend, dt, method``).
    label : str
        Human label stored in metadata.
    keep_raw : bool
        If True, the dense simulation history is attached to the OSLCase
        (and written by ``.export(..., include_raw=True)``).
    """

    if extra_config and any(k in extra_config for k in ("tend", "dt", "method")):
        raise ValueError("extra_config must not redefine tend/dt/method.")

    raw = Path(raw)
    dyr = Path(dyr)
    psys = load_psse(str(raw))
    add_dyr(psys, str(dyr))
    psys.createYbusComplex()

    injectors: List[SignalInjector] = []
    if forced_oscillations:
        injectors.extend(forced_oscillations)
    if colored_noise:
        injectors.append(colored_noise if isinstance(colored_noise, ColoredNoise) else ColoredNoise())
    if extra_injectors:
        injectors.extend(extra_injectors)

    for inj in injectors:
        psys.add_signal_injector(inj)

    cfg_kwargs = dict(tend=tend, dt=dt, method=method)
    if extra_config:
        cfg_kwargs.update(extra_config)
    if cfg_kwargs.get("petsc", False) and injectors:
        raise NotImplementedError(
            "PETSc integration path does not yet honour signal_injectors. "
            "TODO: hook DAE_petsc.evalFunction (uqgrid/simulation/dynamics.py:1153)."
        )
    config = IntegrationConfig(**cfg_kwargs)

    results = integrate_system(psys, config)
    history = results["history"]
    tvec = results["tvec"]

    if pmu is None:
        emulator = PMUEmulator()
    elif isinstance(pmu, PMUEmulator):
        emulator = pmu
    elif isinstance(pmu, dict):
        emulator = PMUEmulator(**pmu)
    else:
        raise TypeError(f"pmu must be PMUEmulator | dict | None, got {type(pmu)!r}")

    pmu_bundle = emulator.process(history, tvec, psys)

    metadata = _build_metadata(
        label=label,
        raw=raw,
        dyr=dyr,
        psys=psys,
        forced_oscillations=forced_oscillations or [],
        colored_noise=colored_noise,
        tend=tend,
        dt=dt,
        method=method,
        emulator=emulator,
        pmu_bundle=pmu_bundle,
    )

    return OSLCase(
        pmu=pmu_bundle,
        metadata=metadata,
        raw_history=history if keep_raw else None,
        raw_tvec=tvec if keep_raw else None,
    )


def _build_metadata(*, label, raw, dyr, psys, forced_oscillations, colored_noise,
                    tend, dt, method, emulator, pmu_bundle) -> Dict[str, Any]:
    sources = []
    for fo in forced_oscillations:
        sources.append({
            "kind": fo.target[0],
            "bus_user": fo.target[1],
            "tag": fo.target[2] if len(fo.target) > 2 else None,
            "freq_hz": float(fo.freq_hz),
            "amplitude": float(fo.amplitude),
            "t_start": float(fo.t_start),
            "t_end": float(fo.t_end) if np.isfinite(fo.t_end) else None,
            "waveform": fo.waveform,
            "harmonics": list(fo.harmonics) if fo.harmonics else None,
            "phase": float(fo.phase),
        })

    noise_meta: Optional[Dict[str, Any]]
    if isinstance(colored_noise, ColoredNoise):
        noise_meta = {
            "sigma_lf": colored_noise.sigma_lf,
            "sigma_hf": colored_noise.sigma_hf,
            "tau_lf_range": list(colored_noise.tau_lf_range),
            "apply_to_q": colored_noise.apply_to_q,
            "seed": colored_noise.seed,
        }
    elif colored_noise:
        noise_meta = {"defaults": True}
    else:
        noise_meta = None

    return {
        "label": label,
        "reference": "NREL/CP-6A40-81394 (Maslennikov & Wang, 2022)",
        "system": {
            "raw": str(raw),
            "dyr": str(dyr),
            "nbuses": psys.nbuses,
            "ngens": psys.ngens,
            "nloads": psys.nloads,
        },
        "simulation": {
            "tend": tend,
            "dt": dt,
            "method": method,
        },
        "sources": sources,
        "colored_noise": noise_meta,
        "pmu": {
            "rate_hz": emulator.rate_hz,
            "p_class_fraction": emulator.p_class_fraction,
            "p_class_cutoff_hz": emulator.p_class_cutoff_hz,
            "m_class_cutoff_hz": emulator.m_class_cutoff_hz,
            "missing_rate": emulator.missing_rate,
            "seed": emulator.seed,
            "observed_buses_psse": [int(b) for b in pmu_bundle["observed_buses_psse"].tolist()],
            "filter_note": (
                "First-order Butterworth lowpass as a stand-in for the EPRI "
                "P/M-class PMU emulator referenced in the paper."
            ),
        },
    }


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Not JSON serialisable: {type(obj)!r}")
