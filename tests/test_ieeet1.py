import os

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.models import ExcIEEET1
from uqgrid.simulation.dynamics import initialize_system
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function


def _model(tr=0.06):
    model = ExcIEEET1(
        "1", tr, 25.0, 0.2, 1.0, -1.0, 0.0, 0.646, 0.103, 1.0,
        0.0, 2.9479, 0.0822, 3.9306, 0.3766,
    )
    model.set_bus(0)
    model.vref = 1.1
    return model


def _evaluate(model, z, v):
    model.set_pointers(0, 0, 0, 0)
    theta = np.zeros(model.par_dim)
    model.initialize_theta(theta)
    residual_idxs = np.array([0, 0, 0], dtype=np.int64)
    jac_idxs = np.array([0, 0, model.dif_dim], dtype=np.int64)
    coords = model.preallocate_jacobian(jac_idxs, None, False)
    rows = []
    cols = []
    for row, row_cols in coords:
        rows.extend([row] * len(row_cols))
        cols.extend(row_cols)
    J = csr_matrix(
        (np.zeros(len(rows)), (rows, cols)),
        shape=(model.dif_dim + 2, model.dif_dim + 2),
    )
    F = np.zeros(model.dif_dim)
    model.residual_diff(F, z, v, theta, residual_idxs, False)
    model.residual_jac(J, z, v, theta, jac_idxs, False)
    return F, J.toarray()[:model.dif_dim], theta


@pytest.mark.parametrize("tr", [0.0, 0.06])
@pytest.mark.parametrize("efd", [1.0, 4.2])
def test_ieeet1_analytical_jacobian_matches_finite_difference(tr, efd):
    model = _model(tr)
    z = np.array(([0.97] if tr else []) + [0.2, efd, 0.8])
    v = np.array([0.98, 0.12])
    _, analytical, _ = _evaluate(model, z, v)

    numerical = np.zeros_like(analytical)
    eps = 1e-6
    for col in range(model.dif_dim + 2):
        z_plus, z_minus = z.copy(), z.copy()
        v_plus, v_minus = v.copy(), v.copy()
        if col < model.dif_dim:
            z_plus[col] += eps
            z_minus[col] -= eps
        else:
            v_plus[col - model.dif_dim] += eps
            v_minus[col - model.dif_dim] -= eps
        f_plus = _evaluate(model, z_plus, v_plus)[0]
        f_minus = _evaluate(model, z_minus, v_minus)[0]
        numerical[:, col] = (f_plus - f_minus) / (2.0 * eps)

    np.testing.assert_allclose(analytical, numerical, rtol=2e-8, atol=2e-8)


def test_ieeet1_initialization_is_an_equilibrium():
    model = _model(0.06)
    model.e_fd0 = 1.2
    model.set_pointers(0, 0, 0, 0)
    z = np.zeros(model.dif_dim)
    model.initialize(1.03, 0.0, 0.0, 0.0, z, np.empty(0), None)
    residual, _, theta = _evaluate(model, z, np.array([1.03, 0.0]))

    np.testing.assert_allclose(residual, 0.0, atol=1e-13)
    assert theta[9] == 0.0


def test_ieeet1_zero_vrmax_disables_only_upper_limit():
    model = ExcIEEET1(
        "1", 0.0, 25.0, 0.2, 0.0, -1.0, 0.0, 0.5, 0.1, 1.0,
        1.0, 2.0, 0.1, 3.0, 0.2,
    )
    assert model.effective_vrmax == 999.0
    assert model.Vrmin == -1.0
    assert model.Switch == 1.0


def test_ieeet1_initialization_adjusts_required_upper_limit():
    model = _model()
    model.e_fd0 = 4.2
    model.set_pointers(0, 0, 0, 0)
    z = np.zeros(model.dif_dim)

    model.initialize(1.0, 0.0, 0.0, 0.0, z, np.empty(0), None)

    assert model.effective_vrmax > model.Vrmax


def test_ieeet1_switch_is_retained_but_does_not_change_equations():
    first = _model()
    second = _model()
    second.Switch = 1.0
    z = np.array([0.97, 0.2, 1.0, 0.8])
    v = np.array([0.98, 0.12])

    np.testing.assert_array_equal(_evaluate(first, z, v)[0], _evaluate(second, z, v)[0])


def test_ieeet1_parser_and_system_initialization(tmp_path):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    dyr = tmp_path / "ieeet1.dyr"
    dyr.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        "1 'IEEET1' 1 0 25 0.2 1 -1 0 0.646 0.103 1 0 2.9479 0.0822 3.9306 0.3766 /\n"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))
    psys.createYbusComplex()
    solution = runpf(psys, verbose=False)
    z, theta = initialize_system(psys, solution)
    residual = np.zeros_like(z)
    residual_function(residual, z, theta, psys)

    assert len(psys.exc) == 1
    assert isinstance(psys.exc[0], ExcIEEET1)
    assert psys.exc[0].dif_dim == 3
    np.testing.assert_allclose(residual, 0.0, atol=1e-7)
