# uqgrid/__init__.py
"""
UQGrid: Uncertainty quantification for the electrical grid.

This package provides tools for uncertainty quantification-type computations
in power grid systems, including DAE integration and sensitivity analysis.
"""

__version__ = "0.1.0"

# Core imports - main user-facing components
from .simulation.dynamics import integrate_system, initialize_system
from .core.psydef import Psystem
from .io.parse import load_psse, add_dyr, load_matpower, load_gic
from .simulation.pflow import runpf
from .simulation.config import IntegrationConfig, IntegrationCtx
from .simulation.sparse_solvers import klu_available

# Check if PETSc is available
try:
    import petsc4py
    _PETSC_AVAILABLE = True
except ImportError:
    _PETSC_AVAILABLE = False

def get_info():
    """Return basic information about the UQGrid installation."""
    info = {
        "version": __version__,
        "petsc_available": _PETSC_AVAILABLE,
        "klu_available": klu_available(),
    }
    return info
