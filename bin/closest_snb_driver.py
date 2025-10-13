#!/usr/bin/env python3
"""Command-line driver for closest-SNB computations."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from uqgrid.snb import build_index_cache, build_param_selector, closest_snb_fsolve
from uqgrid.snb.viewer import print_snb_result

try:
	from tests.fixtures_snb import build_dobson5_fixture
except ImportError as exc:  # pragma: no cover - diagnostic for installed package
	raise SystemExit(
		"Dobson 5-bus fixture is only available in the source tree."
	) from exc


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Closest SNB solver driver")
	parser.add_argument(
		"--case",
		default="dobson5",
		choices=["dobson5"],
		help="Benchmark case to run (default: dobson5).",
	)
	parser.add_argument(
		"--alpha",
		type=float,
		default=1e-3,
		help="Step size used when seeding λ from w (ignored if λ_init supplied).",
	)
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	if args.case != "dobson5":
		raise SystemExit(f"Unsupported case '{args.case}'")

	setup = build_dobson5_fixture()
	cache = build_index_cache(setup.psys)
	selector = build_param_selector(cache)

	result = closest_snb_fsolve(
		setup.psys,
		alpha=args.alpha,
		c_vector=np.ones(cache.n_unknowns),
		x_init=setup.x_init,
		w_init=setup.w_init,
		lambda_init=setup.lambda_init,
		k_init=setup.k_init,
	)

	print_snb_result(result, selector)


if __name__ == "__main__":
	main()
