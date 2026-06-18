#!/usr/bin/env python
"""Generate a small OSL-style PMU dataset.

Light batch driver: it defines a compact scenario grid,
runs ``uqgrid.osl.build_osl_case`` for each scenario, and writes paired
``.npz``/``.json`` case files plus a ``manifest.jsonl`` index.
"""

from __future__ import annotations

import argparse
import os

# Keep numerical libraries from oversubscribing if the script is launched in a
# larger batch job. This script itself runs serially.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from uqgrid.api.osl import generate_osl_dataset, merge_osl_dataset_config


def parse_args():
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

    return merge_osl_dataset_config(config_path=config_path, overrides=cli_values)


def main() -> None:
    try:
        config = parse_args()

        def show_progress(i, total, scenario):
            print(f"[{i}/{total}] {scenario['label']}")

        result = generate_osl_dataset(config, progress=show_progress)
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"wrote {result.manifest_path}")


if __name__ == "__main__":
    main()
