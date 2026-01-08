import os

import numpy as np
import pytest

from uqgrid.io.parse import load_matpower, load_matpower_raw
from uqgrid.simulation.pflow import runpf


pytest.importorskip("pypower")
from pypower.ext2int import ext2int
from pypower.makeYbus import makeYbus
from pypower.ppoption import ppoption
from pypower.runpf import runpf as pp_runpf


CASES = ["case9.m", "case14.m", "case30.m", "case118.m"]

VMAG_ATOL = 1e-5
VANG_ATOL = 1e-4  # radians
YBUS_ATOL = 1e-6


@pytest.fixture
def data_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )


def _ppc_from_matpower(case_path):
    base_mva, bus, gen, branch = load_matpower_raw(case_path)
    gen = gen.copy()
    return {
        "baseMVA": base_mva,
        "bus": bus,
        "gen": gen,
        "branch": branch,
        "version": "2",
    }


@pytest.mark.parametrize("case_file", CASES)
def test_pypower_matches_uqgrid_powerflow(case_file, data_dir):
    case_path = os.path.join(data_dir, case_file)

    # UQGrid power flow
    psys = load_matpower(case_path)
    psys.createYbusComplex()
    res = runpf(psys, verbose=False)
    v = res.v_vector
    vmag_uq = v[0::2]
    vang_uq = v[1::2]

    # PYPOWER power flow
    ppc = _ppc_from_matpower(case_path)
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
    results, success = pp_runpf(ppc, ppopt)
    assert success

    vmag_pp = results["bus"][:, 7]
    vang_pp = np.deg2rad(results["bus"][:, 8])

    np.testing.assert_allclose(vmag_uq, vmag_pp, atol=VMAG_ATOL)
    np.testing.assert_allclose(vang_uq, vang_pp, atol=VANG_ATOL)


@pytest.mark.parametrize("case_file", CASES)
def test_pypower_ybus_matches_uqgrid(case_file, data_dir):
    case_path = os.path.join(data_dir, case_file)

    psys = load_matpower(case_path)
    psys.createYbusComplex()

    ppc = _ppc_from_matpower(case_path)
    ppc_int = ext2int(ppc)
    ybus_pp, _, _ = makeYbus(ppc_int["baseMVA"], ppc_int["bus"], ppc_int["branch"])

    ybus_uq = psys.ybus_spa.toarray()
    ybus_pp_dense = ybus_pp.toarray()

    diff = np.max(np.abs(ybus_pp_dense - ybus_uq))
    assert diff <= YBUS_ATOL
