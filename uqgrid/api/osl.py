"""Stable API for OSL-style dataset generation.

This module is the package-level equivalent of ``scripts/osl/generate_dataset.py``:
it keeps config merging, scenario expansion, and artifact writing importable for
callers that should not know about scripts, CLI flags, or repository layout.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from uqgrid.osl import ColoredNoise, ForcedOscillation, PMUEmulator, build_osl_case


DEFAULT_OSL_DATASET_CONFIG: Dict[str, Any] = {
    "raw": None,
    "dyr": None,
    "outdir": "outputs/osl_dataset",
    "tend": 8.0,
    "dt": 1.0 / 240.0,
    "fo_start": 2.0,
    "fo_buses": [49],
    "freqs": [0.6, 0.8, 1.0],
    "amplitudes": [0.10, 0.20],
    "seed_start": 1000,
    "limit": None,
    "observed_buses": "all",
    "pmu_rate_hz": 30.0,
    "p_class_fraction": 0.70,
    "missing_rate": 0.0,
    "colored_noise": True,
    "noise_sigma_lf": 0.002,
    "noise_sigma_hf": 0.001,
    "noise_tau_lf_range": [0.5, 5.0],
    "overwrite": False,
}

_CONFIG_KEYS = set(DEFAULT_OSL_DATASET_CONFIG)


@dataclass
class OSLDatasetConfig:
    """Configuration for generating an OSL-style PMU dataset."""

    raw: Optional[Union[str, Path]] = DEFAULT_OSL_DATASET_CONFIG["raw"]
    dyr: Optional[Union[str, Path]] = DEFAULT_OSL_DATASET_CONFIG["dyr"]
    outdir: Union[str, Path] = DEFAULT_OSL_DATASET_CONFIG["outdir"]
    tend: float = DEFAULT_OSL_DATASET_CONFIG["tend"]
    dt: float = DEFAULT_OSL_DATASET_CONFIG["dt"]
    fo_start: float = DEFAULT_OSL_DATASET_CONFIG["fo_start"]
    fo_buses: List[int] = field(
        default_factory=lambda: list(DEFAULT_OSL_DATASET_CONFIG["fo_buses"])
    )
    freqs: List[float] = field(default_factory=lambda: list(DEFAULT_OSL_DATASET_CONFIG["freqs"]))
    amplitudes: List[float] = field(
        default_factory=lambda: list(DEFAULT_OSL_DATASET_CONFIG["amplitudes"])
    )
    seed_start: int = DEFAULT_OSL_DATASET_CONFIG["seed_start"]
    limit: Optional[int] = DEFAULT_OSL_DATASET_CONFIG["limit"]
    observed_buses: Any = DEFAULT_OSL_DATASET_CONFIG["observed_buses"]
    pmu_rate_hz: float = DEFAULT_OSL_DATASET_CONFIG["pmu_rate_hz"]
    p_class_fraction: float = DEFAULT_OSL_DATASET_CONFIG["p_class_fraction"]
    missing_rate: float = DEFAULT_OSL_DATASET_CONFIG["missing_rate"]
    colored_noise: bool = DEFAULT_OSL_DATASET_CONFIG["colored_noise"]
    noise_sigma_lf: float = DEFAULT_OSL_DATASET_CONFIG["noise_sigma_lf"]
    noise_sigma_hf: float = DEFAULT_OSL_DATASET_CONFIG["noise_sigma_hf"]
    noise_tau_lf_range: List[float] = field(
        default_factory=lambda: list(DEFAULT_OSL_DATASET_CONFIG["noise_tau_lf_range"])
    )
    overwrite: bool = DEFAULT_OSL_DATASET_CONFIG["overwrite"]
    config_path: Optional[Union[str, Path]] = None

    def __post_init__(self) -> None:
        missing_paths = [
            name for name in ("raw", "dyr")
            if getattr(self, name) is None or str(getattr(self, name)) == ""
        ]
        if missing_paths:
            joined = " and ".join(missing_paths)
            raise ValueError(
                f"{joined} must be provided either in a config file or as explicit overrides."
            )

        if self.fo_buses is None:
            self.fo_buses = list(DEFAULT_OSL_DATASET_CONFIG["fo_buses"])
        else:
            self.fo_buses = [int(bus) for bus in self.fo_buses]

        if self.freqs is None:
            self.freqs = list(DEFAULT_OSL_DATASET_CONFIG["freqs"])
        else:
            self.freqs = [float(freq) for freq in self.freqs]

        if self.amplitudes is None:
            self.amplitudes = list(DEFAULT_OSL_DATASET_CONFIG["amplitudes"])
        else:
            self.amplitudes = [float(amplitude) for amplitude in self.amplitudes]

        if self.noise_tau_lf_range is None:
            self.noise_tau_lf_range = list(DEFAULT_OSL_DATASET_CONFIG["noise_tau_lf_range"])
        else:
            self.noise_tau_lf_range = [float(value) for value in self.noise_tau_lf_range]

        if len(self.noise_tau_lf_range) != 2:
            raise ValueError("noise_tau_lf_range must contain exactly two values.")
        if self.limit is not None and self.limit < 0:
            raise ValueError("limit must be non-negative or None.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OSLDatasetConfig":
        config_values = dict(values)
        if "config" in config_values:
            if "config_path" in config_values:
                raise ValueError("Specify only one of config or config_path.")
            config_values["config_path"] = config_values.pop("config")

        valid_keys = _CONFIG_KEYS | {"config_path"}
        unknown = sorted(set(config_values) - valid_keys)
        if unknown:
            raise ValueError(f"Unknown OSL dataset config keys: {', '.join(unknown)}")

        return cls(**config_values)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["raw"] = str(self.raw)
        data["dyr"] = str(self.dyr)
        data["outdir"] = str(self.outdir)
        if self.config_path is not None:
            data["config_path"] = str(self.config_path)
        return data


@dataclass
class OSLDatasetResult:
    """Summary of a generated OSL dataset."""

    outdir: Path
    cases_dir: Path
    manifest_path: Path
    case_count: int
    rows: List[Dict[str, Any]]

    def to_dict(self, include_rows: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "outdir": str(self.outdir),
            "cases_dir": str(self.cases_dir),
            "manifest_path": str(self.manifest_path),
            "case_count": self.case_count,
        }
        if include_rows:
            data["rows"] = self.rows
        return data


@dataclass
class OSLDatasetInspection:
    """Read-only summary of an existing OSL dataset manifest."""

    outdir: Path
    manifest_path: Path
    case_count: int
    rows: List[Dict[str, Any]]
    observed_bus_counts: List[int]
    pmu_sample_counts: List[int]

    def to_dict(self, include_rows: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "outdir": str(self.outdir),
            "manifest_path": str(self.manifest_path),
            "case_count": self.case_count,
            "observed_bus_counts": self.observed_bus_counts,
            "pmu_sample_counts": self.pmu_sample_counts,
        }
        if include_rows:
            data["rows"] = self.rows
        return data


def load_osl_dataset_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate an OSL dataset JSON config file."""

    config_path = Path(path)
    with config_path.open() as f:
        config = json.load(f)

    unknown = sorted(set(config) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"{config_path} contains unknown keys: {', '.join(unknown)}")
    return config


def merge_osl_dataset_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> OSLDatasetConfig:
    """Merge built-in defaults, an optional JSON config, and explicit overrides."""

    values = dict(DEFAULT_OSL_DATASET_CONFIG)
    if config_path:
        values.update(load_osl_dataset_config(config_path))

    if overrides:
        override_values = {key: value for key, value in overrides.items() if value is not None}
        if "config" in override_values:
            if config_path:
                raise ValueError("Specify only one config path, either config_path or config.")
            config_path = override_values.pop("config")
            values.update(load_osl_dataset_config(config_path))
        if "config_path" in override_values:
            if config_path:
                raise ValueError("Specify only one config path.")
            config_path = override_values.pop("config_path")
            values.update(load_osl_dataset_config(config_path))
        unknown = sorted(set(override_values) - _CONFIG_KEYS)
        if unknown:
            raise ValueError(f"Unknown OSL dataset config overrides: {', '.join(unknown)}")
        values.update(override_values)

    values["config_path"] = str(config_path) if config_path else None
    return OSLDatasetConfig.from_mapping(values)


def parse_observed_buses(value: Any) -> Any:
    """Normalize an observed-bus selector for ``PMUEmulator``."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(bus) for bus in value]
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return [int(value)]
    if not isinstance(value, str):
        raise TypeError("observed_buses must be None, a list, a fraction, or a string.")

    if value.lower() == "all":
        return None
    if value.endswith("%"):
        return value
    return [int(part) for part in value.split(",") if part.strip()]


def scenario_label(kind: str, bus: int, freq_hz: float, amplitude: float, seed: int) -> str:
    freq = f"{freq_hz:g}".replace(".", "p")
    amp = f"{amplitude:g}".replace(".", "p")
    return f"{kind}{bus}_f{freq}_a{amp}_s{seed}"


def iter_osl_scenarios(config: OSLDatasetConfig) -> List[Dict[str, Any]]:
    """Expand an OSL dataset config into case scenario dictionaries."""

    scenarios: List[Dict[str, Any]] = []
    for idx, (bus, freq_hz, amplitude) in enumerate(
        itertools.product(config.fo_buses, config.freqs, config.amplitudes)
    ):
        seed = config.seed_start + idx
        scenarios.append({
            "case_id": f"case_{idx:04d}",
            "label": scenario_label("gov", bus, freq_hz, amplitude, seed),
            "target": ("gov", bus),
            "freq_hz": freq_hz,
            "amplitude": amplitude,
            "seed": seed,
        })
    if config.limit is not None:
        scenarios = scenarios[:config.limit]
    return scenarios


def generate_osl_dataset(
    config: OSLDatasetConfig,
    *,
    progress: Optional[Callable[[int, int, Mapping[str, Any]], None]] = None,
) -> OSLDatasetResult:
    """Generate an OSL-style dataset and return a manifest summary."""

    outdir = Path(config.outdir)
    cases_dir = outdir / "cases"
    manifest_path = outdir / "manifest.jsonl"

    if manifest_path.exists() and not config.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass overwrite=True to replace it.")

    cases_dir.mkdir(parents=True, exist_ok=True)
    scenarios = iter_osl_scenarios(config)
    observed_buses = parse_observed_buses(config.observed_buses)

    rows: List[Dict[str, Any]] = []
    with manifest_path.open("w") as manifest:
        for i, scenario in enumerate(scenarios, start=1):
            if progress:
                progress(i, len(scenarios), scenario)

            noise: Union[bool, ColoredNoise] = False
            if config.colored_noise:
                noise = ColoredNoise(
                    sigma_lf=config.noise_sigma_lf,
                    sigma_hf=config.noise_sigma_hf,
                    tau_lf_range=tuple(config.noise_tau_lf_range),
                    seed=scenario["seed"],
                )

            case = build_osl_case(
                raw=config.raw,
                dyr=config.dyr,
                forced_oscillations=[
                    ForcedOscillation(
                        target=scenario["target"],
                        freq_hz=scenario["freq_hz"],
                        amplitude=scenario["amplitude"],
                        t_start=config.fo_start,
                        t_end=config.tend,
                    )
                ],
                colored_noise=noise,
                tend=config.tend,
                dt=config.dt,
                pmu=PMUEmulator(
                    rate_hz=config.pmu_rate_hz,
                    p_class_fraction=config.p_class_fraction,
                    observed_buses=observed_buses,
                    missing_rate=config.missing_rate,
                    seed=scenario["seed"],
                ),
                label=scenario["label"],
            )
            stem = cases_dir / scenario["case_id"]
            npz_path, json_path = case.export(stem)
            row = {
                **scenario,
                "target": list(scenario["target"]),
                "npz": str(npz_path.relative_to(outdir)),
                "json": str(json_path.relative_to(outdir)),
                "n_observed_buses": int(case.pmu["observed_buses_internal"].shape[0]),
                "n_pmu_samples": int(case.pmu["t"].shape[0]),
                "config": str(config.config_path) if config.config_path else None,
            }
            manifest.write(json.dumps(row) + "\n")
            rows.append(row)

    return OSLDatasetResult(
        outdir=outdir,
        cases_dir=cases_dir,
        manifest_path=manifest_path,
        case_count=len(rows),
        rows=rows,
    )


def inspect_osl_dataset(outdir: Union[str, Path]) -> OSLDatasetInspection:
    """Inspect an existing OSL dataset manifest without loading case arrays."""

    outdir = Path(outdir)
    manifest_path = outdir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} does not exist.")

    rows: List[Dict[str, Any]] = []
    with manifest_path.open() as manifest:
        for line in manifest:
            if line.strip():
                rows.append(json.loads(line))

    observed_bus_counts = sorted({
        int(row["n_observed_buses"]) for row in rows if "n_observed_buses" in row
    })
    pmu_sample_counts = sorted({
        int(row["n_pmu_samples"]) for row in rows if "n_pmu_samples" in row
    })

    return OSLDatasetInspection(
        outdir=outdir,
        manifest_path=manifest_path,
        case_count=len(rows),
        rows=rows,
        observed_bus_counts=observed_bus_counts,
        pmu_sample_counts=pmu_sample_counts,
    )
