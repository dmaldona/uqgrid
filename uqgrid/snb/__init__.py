"""Tools for saddle–node bifurcation (SNB) analysis atop the power-flow module."""

from .indexing import PFIndexCache, build_index_cache
from .params import (build_fixed_injections, extract_lambda, pack_params,
                     scatter_lambda, unpack_params)
from .pf import PFFunctions, build_pf_operators, solution_to_state_vector
from .selectors import build_param_selector
from .solver import (ClosestSNBResult, SolverDiagnostics, closest_snb_fsolve)
from .viewer import print_snb_result

__all__ = [
    "build_index_cache",
    "PFIndexCache",
    "build_pf_operators",
    "PFFunctions",
    "solution_to_state_vector",
    "build_param_selector",
    "pack_params",
    "unpack_params",
    "extract_lambda",
    "build_fixed_injections",
    "scatter_lambda",
    "closest_snb_fsolve",
    "ClosestSNBResult",
    "SolverDiagnostics",
    "print_snb_result",
]
