#!/usr/bin/env python
"""Generate a small OSL-style PMU dataset.

Light batch driver: it defines a compact scenario grid,
runs ``uqgrid.osl.build_osl_case`` for each scenario, and writes paired
``.npz``/``.json`` case files plus a ``manifest.jsonl`` index.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any

# Keep numerical libraries from oversubscribing if the script is launched in a
# larger batch job. This script itself runs serially.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO_ROOT / "data" / "ACTIVSg200.raw"
DEFAULT_DYR = REPO_ROOT / "data" / "ACTIVSg200.dyr"

DEFAULT_CONFIG = {
    "raw": str(DEFAULT_RAW),
    "dyr": str(DEFAULT_DYR),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="Optional JSON config path.")
    parser.add_argument("--config", dest="config_option", help="Optional JSON config path.")
    parser.add_argument("--raw", default=argparse.SUPPRESS)
    parser.add_argument("--dyr", default=argparse.SUPPRESS)
    parser.add_argument("--outdir", default=argparse.SUPPRESS)
    parser.add_argument("--tend", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--dt", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--fo-start", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--fo-buses", type=int, nargs="+", default=argparse.SUPPRESS)
    parser.add_argument("--freqs", type=float, nargs="+", default=argparse.SUPPRESS)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=argparse.SUPPRESS)
    parser.add_argument("--seed-start", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--limit",
        type=int,
        default=argparse.SUPPRESS,
        help="Run only the first N scenarios.",
    )
    parser.add_argument(
        "--observed-buses",
        default=argparse.SUPPRESS,
        help='PMU bus set: "all", a percent like "50%%", or comma-separated PSSE buses.',
    )
    parser.add_argument("--pmu-rate-hz", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--p-class-fraction", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--missing-rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--colored-noise",
        dest="colored_noise",
        action="store_true",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-noise",
        dest="colored_noise",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--noise-sigma-lf", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--noise-sigma-hf", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--noise-tau-lf-range", type=float, nargs=2, default=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true", default=argparse.SUPPRESS)

    parsed = parser.parse_args()
    cli_values = vars(parsed).copy()
    config_path = cli_values.pop("config", None)
    config_option = cli_values.pop("config_option", None)
    if config_path and config_option:
        parser.error("pass only one config path, either positional or --config")
    config_path = config_path or config_option

    config = DEFAULT_CONFIG.copy()
    if config_path:
        config.update(load_config(config_path))
    config.update(cli_values)
    config["config"] = config_path
    return argparse.Namespace(**config)


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open() as f:
        config = json.load(f)
    unknown = sorted(set(config) - set(DEFAULT_CONFIG))
    if unknown:
        raise SystemExit(f"{config_path} contains unknown keys: {', '.join(unknown)}")
    return config


def parse_observed_buses(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [int(bus) for bus in value]
    if value.lower() == "all":
        return None
    if value.endswith("%"):
        return value
    return [int(part) for part in value.split(",") if part.strip()]


def scenario_label(kind: str, bus: int, freq_hz: float, amplitude: float, seed: int) -> str:
    freq = f"{freq_hz:g}".replace(".", "p")
    amp = f"{amplitude:g}".replace(".", "p")
    return f"{kind}{bus}_f{freq}_a{amp}_s{seed}"


def iter_scenarios(args: argparse.Namespace) -> list[dict[str, Any]]:
    scenarios = []
    for idx, (bus, freq_hz, amplitude) in enumerate(
        itertools.product(args.fo_buses, args.freqs, args.amplitudes)
    ):
        seed = args.seed_start + idx
        scenarios.append({
            "case_id": f"case_{idx:04d}",
            "label": scenario_label("gov", bus, freq_hz, amplitude, seed),
            "target": ("gov", bus),
            "freq_hz": freq_hz,
            "amplitude": amplitude,
            "seed": seed,
        })
    if args.limit is not None:
        scenarios = scenarios[:args.limit]
    return scenarios


def main() -> None:
    args = parse_args()
    from uqgrid.osl import ColoredNoise, ForcedOscillation, PMUEmulator, build_osl_case

    outdir = Path(args.outdir)
    cases_dir = outdir / "cases"
    manifest_path = outdir / "manifest.jsonl"

    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"{manifest_path} exists; pass --overwrite to replace it.")

    cases_dir.mkdir(parents=True, exist_ok=True)
    scenarios = iter_scenarios(args)
    observed_buses = parse_observed_buses(args.observed_buses)

    with manifest_path.open("w") as manifest:
        for i, scenario in enumerate(scenarios, start=1):
            print(f"[{i}/{len(scenarios)}] {scenario['label']}")
            noise = False
            if args.colored_noise:
                noise = ColoredNoise(
                    sigma_lf=args.noise_sigma_lf,
                    sigma_hf=args.noise_sigma_hf,
                    tau_lf_range=tuple(args.noise_tau_lf_range),
                    seed=scenario["seed"],
                )
            case = build_osl_case(
                raw=args.raw,
                dyr=args.dyr,
                forced_oscillations=[
                    ForcedOscillation(
                        target=scenario["target"],
                        freq_hz=scenario["freq_hz"],
                        amplitude=scenario["amplitude"],
                        t_start=args.fo_start,
                        t_end=args.tend,
                    )
                ],
                colored_noise=noise,
                tend=args.tend,
                dt=args.dt,
                pmu=PMUEmulator(
                    rate_hz=args.pmu_rate_hz,
                    p_class_fraction=args.p_class_fraction,
                    observed_buses=observed_buses,
                    missing_rate=args.missing_rate,
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
                "config": args.config,
            }
            manifest.write(json.dumps(row) + "\n")

    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
