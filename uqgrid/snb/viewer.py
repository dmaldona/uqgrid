from __future__ import annotations

import sys
from typing import IO, Optional

import numpy as np
from scipy.sparse import csr_matrix

from .solver import ClosestSNBResult

_HEADER = "Closest SNB — solved with 4-eq KKT (Dobson)"
_LAMBDA_LABELS = ("P2", "P4", "P5", "Q2", "Q4", "Q5")


def _top_components(vec: np.ndarray, count: int = 3) -> list[tuple[int, float]]:
	if vec.size == 0:
		return []
	indices = np.argsort(np.abs(vec))[-count:][::-1]
	return [(int(i), float(vec[i])) for i in indices]


def print_snb_result(
	result: ClosestSNBResult,
	f_lambda: csr_matrix,
	*,
	file: Optional[IO[str]] = None,
) -> None:
	"""Pretty-print summary information for a closest-SNB solve."""

	stream: IO[str] = sys.stdout if file is None else file

	normal = np.asarray(f_lambda.transpose().dot(result.w_star)).ravel()
	delta = result.lambda_star - result.lambda0

	distance = float(result.distance)
	angle_deg = float(np.degrees(result.angle))
	normal_norm = float(np.linalg.norm(normal))

	kkt = result.kkt_residuals
	pf_res = kkt.get("pf", float("nan"))
	left_res = kkt.get("left_null", float("nan"))
	stat_res = kkt.get("stationarity", float("nan"))
	norm_res = kkt.get("normalization", float("nan"))

	stream.write(f"{_HEADER}\n")
	stream.write(f"  Distance ||Δλ||₂       : {distance: .6e}\n")
	stream.write(f"  Normal angle (deg)     : {angle_deg: .6e}\n")
	stream.write(f"  k*                     : {result.k_star: .6e}\n")
	stream.write(f"  ||f_λᵀ w*||₂           : {normal_norm: .6e}\n")
	stream.write(f"  σ_min(f_x)             : {result.sigma_min: .6e}\n")
	stream.write("  KKT residuals (∞-norm):\n")
	stream.write(f"    PF              : {pf_res: .6e}\n")
	stream.write(f"    Left-null       : {left_res: .6e}\n")
	stream.write(f"    Stationarity    : {stat_res: .6e}\n")
	stream.write(f"    Normalization   : {norm_res: .6e}\n")

	stream.write("  Lambda components (canonical order P2,P4,P5,Q2,Q4,Q5):\n")
	for label, lam0, lam_star, dlam in zip(_LAMBDA_LABELS, result.lambda0, result.lambda_star, delta):
		stream.write(
			f"    {label:>3}: λ0={lam0: .6f}  λ*={lam_star: .6f}  Δ={dlam: .6f}\n"
		)

	w_top = _top_components(result.w_star)
	normal_top = _top_components(normal)
	stream.write("  Top components:\n")
	stream.write("    w*    : " + ", ".join(f"[{i}]={val: .6e}" for i, val in w_top) + "\n")
	stream.write("    normal: " + ", ".join(f"[{i}]={val: .6e}" for i, val in normal_top) + "\n")

	metadata = result.metadata
	stream.write("  Solver diagnostics:\n")
	stream.write(f"    nfev   : {metadata.get('nfev', 'n/a')}\n")
	stream.write(f"    ier    : {metadata.get('ier', 'n/a')}\n")
	stream.write(f"    status : {metadata.get('message', '')}\n")
	stream.flush()
