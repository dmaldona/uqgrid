import os

import numpy as np
import pytest

from uqgrid.core.base_models import Governor, Stabilizer
from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.simulation.dynamics import initialize_system, preallocate_jacobian
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.residual import residual_function
from uqgrid.utils.tools import csr_set_row


@pytest.fixture
def data_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )


class StubStabilizer(Stabilizer):
    def __init__(self, id_tag):
        super().__init__(id_tag, initdim=0, ddim=0, adim=1, pdim=0, state_list=["v_pss"])

    def initialize(self, vm, va, p, q, x, y, psys):
        y[self.alg_ptr] = 0.0
        self.initialized = True

    def initialize_theta(self, theta):
        return None

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        F[idxs[1]] = -z[idxs[1]]

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys, power_injection):
        return [[idxs[1], [idxs[1]]]]

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        col = np.array([idxs[1]], dtype=np.int64)
        val = np.array([-1.0])
        csr_set_row(J.data, J.indptr, J.indices, 1, idxs[1], col, val)


class StubTwoOutputGovernor(Governor):
    secondary_output_offset = 1

    def __init__(self, id_tag):
        super().__init__(id_tag, initdim=0, ddim=0, adim=2, pdim=0, state_list=["p_hp", "p_lp"])

    def initialize(self, vm, va, p, q, x, y, psys):
        y[self.alg_ptr] = self.p_m0
        y[self.alg_ptr + 1] = self.p_m0_secondary
        self.initialized = True

    def initialize_theta(self, theta):
        return None

    def residual_diff(self, F, z, v, theta, idxs, power_injection):
        F[idxs[1]] = self.p_m0 - z[idxs[1]]
        F[idxs[1] + 1] = self.p_m0_secondary - z[idxs[1] + 1]

    def residual_pinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def residual_cinj(self, F, z, v, theta, idxs, alpha=False):
        return None

    def preallocate_jacobian(self, idxs, psys, power_injection):
        return [[idxs[1], [idxs[1]]], [idxs[1] + 1, [idxs[1] + 1]]]

    def residual_jac(self, J, z, v, theta, idxs, power_injection):
        for row in (idxs[1], idxs[1] + 1):
            col = np.array([row], dtype=np.int64)
            val = np.array([-1.0])
            csr_set_row(J.data, J.indptr, J.indices, 1, row, col, val)


def _initialized_sexs_case(data_dir):
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, "2bus_SEXS.dyr"))
    pss = StubStabilizer("1")
    psys.add_pss(psys.gendyn[0], pss)
    psys.createYbusComplex()
    solution = runpf(psys, verbose=False)
    z, theta = initialize_system(psys, solution)
    return psys, pss, z, theta


def _initialized_esdc1a_case(data_dir, tmp_path):
    dyr = tmp_path / "2bus_ESDC1A.dyr"
    dyr.write_text(
        """1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /
1 'ESDC1A' 1 0.02 20.0 1.0 0.7 0.7 10.0 -10.0 7.0 0.5 0.7 0.7 0.0 1.0 0.006 1.2 0.9 /
"""
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))
    pss = StubStabilizer("1")
    psys.add_pss(psys.gendyn[0], pss)
    psys.createYbusComplex()
    solution = runpf(psys, verbose=False)
    z, theta = initialize_system(psys, solution)
    return psys, pss, z, theta


def test_stabilizer_attachment_uses_precomputed_indexes(data_dir):
    psys, pss, _, _ = _initialized_sexs_case(data_dir)
    gen = psys.gendyn[0]
    exc = psys.exc[0]

    assert psys.pss == [pss]
    assert gen.stabilizer is pss
    assert pss.generator is gen
    assert pss.exciter is exc
    assert pss.gen_index == gen.device_index
    assert pss.w_idx == gen.dif_ptr + 4
    assert exc.pss_input_idx == psys.num_dof_dif + pss.alg_ptr


def test_stabilizer_output_changes_sexs_residual_and_jacobian(data_dir):
    psys, pss, z, theta = _initialized_sexs_case(data_dir)
    exc = psys.exc[0]
    pss_col = psys.num_dof_dif + pss.alg_ptr
    exc_row = exc.dif_ptr + 1

    baseline = np.zeros_like(z)
    residual_function(baseline, z, theta, psys)
    perturbed = z.copy()
    perturbation = 1e-4
    perturbed[pss_col] += perturbation
    changed = np.zeros_like(z)
    residual_function(changed, perturbed, theta, psys)

    assert changed[exc_row] - baseline[exc_row] == pytest.approx(
        exc.K * exc.TA_TB * perturbation / exc.TE
    )

    jacobian = preallocate_jacobian(psys)
    residual_jacobian(jacobian, z, theta, psys)
    assert jacobian[exc_row, pss_col] == pytest.approx(exc.K * exc.TA_TB / exc.TE)


def test_sexs_without_stabilizer_keeps_existing_behavior(data_dir):
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, "2bus_SEXS.dyr"))
    psys.createYbusComplex()
    solution = runpf(psys, verbose=False)
    z, theta = initialize_system(psys, solution)

    assert psys.exc[0].pss_input_idx == -1
    residual = np.zeros_like(z)
    residual_function(residual, z, theta, psys)
    assert np.linalg.norm(residual, np.inf) < 1e-8


def test_stabilizer_output_changes_esdc1a_residual_and_jacobian(data_dir, tmp_path):
    psys, pss, z, theta = _initialized_esdc1a_case(data_dir, tmp_path)
    exc = psys.exc[0]
    pss_col = psys.num_dof_dif + pss.alg_ptr
    regulator_row = exc.dif_ptr

    baseline = np.zeros_like(z)
    residual_function(baseline, z, theta, psys)
    perturbed = z.copy()
    perturbation = 1e-4
    perturbed[pss_col] += perturbation
    changed = np.zeros_like(z)
    residual_function(changed, perturbed, theta, psys)

    assert changed[regulator_row] - baseline[regulator_row] == pytest.approx(
        exc.Ka * perturbation / exc.Ta
    )

    jacobian = preallocate_jacobian(psys)
    residual_jacobian(jacobian, z, theta, psys)
    assert jacobian[regulator_row, pss_col] == pytest.approx(exc.Ka / exc.Ta)


def test_secondary_governor_output_routes_to_exact_generator(data_dir):
    psys = load_psse(os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus.dyr"))
    primary = psys.gendyn[0]
    secondary = psys.gendyn[1]
    governor = StubTwoOutputGovernor(primary.id_tag)
    psys.add_gov(primary, governor, secondary_gen=secondary)
    psys.createYbusComplex()
    solution = runpf(psys, verbose=False)
    z, theta = initialize_system(psys, solution)

    dif = psys.num_dof_dif
    assert primary.governor is governor
    assert secondary.governor is governor
    assert governor.primary_generator is primary
    assert governor.secondary_generator is secondary
    assert psys.gen_pm_ctrl_col[primary.device_index] == dif + governor.alg_ptr
    assert psys.gen_pm_ctrl_col[secondary.device_index] == dif + governor.alg_ptr + 1
    assert governor.p_m0 == pytest.approx(primary.p_m)
    assert governor.p_m0_secondary == pytest.approx(secondary.p_m)

    residual = np.zeros_like(z)
    residual_function(residual, z, theta, psys)
    assert np.linalg.norm(residual, np.inf) < 1e-8

    jacobian = preallocate_jacobian(psys)
    residual_jacobian(jacobian, z, theta, psys)
    primary_row = psys.gen_pm_out_idx[primary.device_index]
    secondary_row = psys.gen_pm_out_idx[secondary.device_index]
    assert jacobian[primary_row, dif + governor.alg_ptr] == pytest.approx(-1.0)
    assert jacobian[secondary_row, dif + governor.alg_ptr + 1] == pytest.approx(-1.0)


def test_controller_attachment_rejects_duplicate_generator_control(data_dir):
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, "2bus_SEXS.dyr"))

    with pytest.raises(ValueError, match="already has a stabilizer"):
        psys.add_pss(psys.gendyn[0], StubStabilizer("1"))
        psys.add_pss(psys.gendyn[0], StubStabilizer("1"))


def test_stabilizer_requires_exciter(data_dir):
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, os.path.join(data_dir, "GENROU.dyr"))

    with pytest.raises(ValueError, match="requires an attached exciter"):
        psys.add_pss(psys.gendyn[0], StubStabilizer("1"))


def test_secondary_governor_target_must_be_distinct(data_dir):
    psys = load_psse(os.path.join(data_dir, "ieee9_v33.raw"))
    add_dyr(psys, os.path.join(data_dir, "ieee9bus.dyr"))
    gen = psys.gendyn[0]

    with pytest.raises(ValueError, match="must differ"):
        psys.add_gov(gen, StubTwoOutputGovernor(gen.id_tag), secondary_gen=gen)
