import os

import pytest
from uqgrid.io.parse import load_psse, add_dyr
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.dynamics import initialize_system, preallocate_jacobian
from uqgrid.simulation.residual import residual_function
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.models.sexs_imp import sexs_resdiff, sexs_jac
from uqgrid.models.tgov1_imp import tgov1_resdiff, tgov1_jac
from uqgrid.models.esdc1a_imp import esdc1a_resdiff, esdc1a_jac
from uqgrid.models.ieesgo_imp import ieesgo_resdiff, ieesgo_jac


@pytest.fixture
def data_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )


def _run_case(raw_path, dyr_path):
    psys = load_psse(raw_filename=raw_path)
    add_dyr(psys, dyr_path)
    psys.createYbusComplex()
    pf_solution = runpf(psys, verbose=False)
    sysvec, theta = initialize_system(psys, pf_solution)
    F = sysvec * 0.0
    residual_function(F, sysvec, theta, psys)
    residual_function(F, sysvec, theta, psys)
    J = preallocate_jacobian(psys)
    residual_jacobian(J, sysvec, theta, psys)
    residual_jacobian(J, sysvec, theta, psys)


def test_controller_numba_kernels_compile(data_dir, tmp_path):
    esdc1a_dyr = tmp_path / "2bus_ESDC1A.dyr"
    esdc1a_dyr.write_text(
        """
1 'GENROU'  1            6.1          0.05         1.0          0.15
                3.38         0.0          1.575        1.512        0.291
                0.39         0.1733       0.0787       0.0       0.0      /
1 'ESDC1A'  1            0.02         20.0         1.0          0.7
                0.7          10.0        -10.0         7.0          0.5
                0.7          0.7          0.0          1.0          0.006
                1.2          0.9                                             /
""".strip()
    )

    _run_case(os.path.join(data_dir, "2bus_33.raw"), os.path.join(data_dir, "2bus_SEXS.dyr"))
    _run_case(os.path.join(data_dir, "2bus_33.raw"), os.path.join(data_dir, "2bus_TGOV1.dyr"))
    _run_case(os.path.join(data_dir, "2bus_CIM5.raw"), os.path.join(data_dir, "2bus_IEESGO.dyr"))
    _run_case(os.path.join(data_dir, "2bus_33.raw"), str(esdc1a_dyr))

    kernels = [
        sexs_resdiff,
        sexs_jac,
        tgov1_resdiff,
        tgov1_jac,
        ieesgo_resdiff,
        ieesgo_jac,
        esdc1a_resdiff,
        esdc1a_jac,
    ]
    for kernel in kernels:
        assert kernel.signatures
