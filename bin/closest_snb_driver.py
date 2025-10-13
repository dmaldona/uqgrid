#!/usr/bin/env python3
"""Command-line driver for closest-SNB computations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from uqgrid.snb import (
	build_index_cache,
	build_param_selector,
	closest_snb_fsolve,
	extract_lambda,
)
from uqgrid.snb.ldt import as_linear_op_Sigma_inv
from uqgrid.snb.mc import ellipse_collapse_mask, sample_gaussian
from uqgrid.snb.viewer import print_snb_result

try:
	from tests.fixtures_snb import build_dobson5_fixture
except ImportError as exc:  # pragma: no cover - diagnostic for installed package
	raise SystemExit(
		"Dobson 5-bus fixture is only available in the source tree."
	) from exc


def _build_two_bus_psys():
	from uqgrid.core.psydef import Psystem

	psys = Psystem(basemva=1.0)
	psys.add_bus(1, 3)
	psys.add_bus(2, 1)

	for bus in psys.buses:
		bus.set_vinit(1.0, 0.0)

	psys.add_branch(0, 1, r=0.0, x=0.25)
	psys.add_gen(bus=0, idx_name="G1", psch=0.0, qsch=0.0)
	psys.add_load(bus=1, tag="LD1", pload=0.5, qload=0.3)

	psys.assemble()
	psys.createYbusComplex()
	return psys


def _load_mu(mu_arg: Optional[str], lambda0: np.ndarray) -> np.ndarray:
	if mu_arg is None or mu_arg == "lambda0":
		return lambda0.copy()

	path = Path(mu_arg)
	if not path.exists():
		raise SystemExit(f"Mu path '{mu_arg}' not found.")
	if path.suffix == ".npy":
		return np.load(path)
	if path.suffix == ".json":
		with path.open("r", encoding="utf8") as fh:
			data = json.load(fh)
		return np.asarray(data, dtype=float)

	raise SystemExit("Unsupported mu format. Use .npy or .json.")


def _load_sigma(sigma_arg: Optional[str], m: int) -> Optional[np.ndarray]:
	if sigma_arg is None:
		return None
	if sigma_arg == "identity":
		return np.ones(m, dtype=float)
	if sigma_arg.startswith("diag:"):
		payload = sigma_arg.split(":", 1)[1]
		values = np.array([float(x) for x in payload.split(",") if x], dtype=float)
		if values.size == 1:
			values = np.full(m, values[0], dtype=float)
		if values.size != m:
			raise SystemExit(f"Diagonal sigma must have {m} entries (got {values.size}).")
		return values

	path = Path(sigma_arg)
	if not path.exists():
		raise SystemExit(f"Sigma path '{sigma_arg}' not found.")
	if path.suffix == ".npy":
		return np.load(path)
	if path.suffix == ".json":
		with path.open("r", encoding="utf8") as fh:
			data = json.load(fh)
		return np.asarray(data, dtype=float)

	raise SystemExit("Unsupported sigma format. Use identity, diag:..., or .npy/.json.")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Closest SNB solver driver")
	parser.add_argument(
		"--case",
		default="dobson5",
		choices=["dobson5", "two_bus"],
		help="Benchmark case to run (default: dobson5).",
	)
	parser.add_argument(
		"--alpha",
		type=float,
		default=1e-3,
		help="Step size used when seeding λ from w (ignored if λ_init supplied).",
	)
	parser.add_argument(
		"--mu",
		type=str,
		default="lambda0",
		help="Mean load vector: 'lambda0' or path to .npy/.json",
	)
	parser.add_argument(
		"--sigma",
		type=str,
		default=None,
		help="Covariance: 'identity', 'diag:x,y', or path to .npy/.json",
	)
	parser.add_argument(
		"--scale",
		type=float,
		default=1.0,
		help="Scalar multiplier applied to Sigma (default: 1.0)",
	)
	parser.add_argument(
		"--mc",
		type=int,
		default=None,
		help="If set, run Monte Carlo with N samples (two-bus case only)",
	)
	return parser.parse_args()


def main() -> None:
	args = _parse_args()

	if args.case == "dobson5":
		fixture = build_dobson5_fixture()
		psys = fixture.psys
		x_init = fixture.x_init
		w_init = fixture.w_init
		lambda_init = fixture.lambda_init
		k_init = fixture.k_init
		mc_available = False
	else:
		psys = _build_two_bus_psys()
		x_init = w_init = lambda_init = k_init = None
		mc_available = True

	cache = build_index_cache(psys)
	selector = build_param_selector(cache)
	lambda0 = extract_lambda(psys, cache)
	m = lambda0.size

	mu_vec = _load_mu(args.mu, lambda0)
	if mu_vec.shape[0] != m:
		raise SystemExit(f"mu dimension {mu_vec.shape[0]} does not match lambda ({m}).")

	Sigma_base = _load_sigma(args.sigma, m)
	Sigma = None
	Sigma_inv_input = None
	if Sigma_base is not None:
		Sigma = Sigma_base * args.scale
		if Sigma.ndim == 1:
			Sigma_inv_input = 1.0 / Sigma
		else:
			Sigma_inv_input = np.linalg.inv(Sigma)

	Sigma_inv_arg = None
	if Sigma_inv_input is not None:
		Sigma_inv_arg = as_linear_op_Sigma_inv(Sigma_inv_input, m)

	result = closest_snb_fsolve(
		psys,
		alpha=args.alpha,
		c_vector=np.ones(cache.n_unknowns),
		x_init=x_init,
		w_init=w_init,
		lambda_init=lambda_init,
		k_init=k_init,
		mu=mu_vec,
		Sigma_inv=Sigma_inv_arg,
	)

	print_snb_result(result, selector)

	if args.mc:
		if not mc_available:
			print("Monte Carlo estimation is only available for the two_bus case.", file=sys.stderr)
		elif Sigma is None:
			print("Monte Carlo requires --sigma to be provided.", file=sys.stderr)
		elif Sigma.ndim != 1 or mu_vec.size != 2:
			print("Monte Carlo helper currently supports diagonal Sigma with two parameters.", file=sys.stderr)
		else:
			samples = sample_gaussian(mu_vec, Sigma, args.mc)
			P = samples[:, 0]
			Q = samples[:, 1]
			mask = ellipse_collapse_mask(P, Q)
			phat = mask.mean()
			p_first = result.p_ldt_first if result.p_ldt_first is not None else 0.0
			rel_err = abs(phat - p_first) / max(p_first, 1e-12)
			print(
				f"MC: N={args.mc}, p̂={phat:.3e}, rel.err vs P1st={rel_err:.2%}",
				file=sys.stdout,
			)


if __name__ == "__main__":
	main()
