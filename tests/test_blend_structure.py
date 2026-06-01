"""
Structural tests for controller masking blend architecture.

These tests verify that:
1. System initialization creates correct index arrays
2. Generator blend rows have correct Jacobian structure
3. Controllers write only their own rows
"""

import os
import pytest
import numpy as np
from uqgrid.io.parse import load_psse, add_dyr
from uqgrid.models import GenGENROU
from uqgrid.simulation.pflow import runpf
from uqgrid.simulation.jacobian import residual_jacobian
from uqgrid.simulation.residual import residual_function
from uqgrid.simulation.dynamics import preallocate_jacobian, initialize_system
from scipy.sparse import csr_matrix


@pytest.fixture
def data_dir():
    """Fixture to provide the absolute path to the data directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'
    )


def _initialized_jacobian_case(raw_path, dyr_path=None):
    psys = load_psse(raw_filename=raw_path)
    if dyr_path is not None:
        add_dyr(psys, dyr_path)
    psys.createYbusComplex()
    pf_solution = runpf(psys, verbose=False)
    sysvec, theta = initialize_system(psys, pf_solution)
    J = preallocate_jacobian(psys)
    residual_jacobian(J, sysvec, theta, psys)
    return psys, J


def _assert_genrou_preallocation_rows_sorted(psys):
    dif_size = psys.num_dof_dif
    alg_size = psys.num_dof_alg
    for gen in psys.gendyn:
        idxs = np.array([
            gen.dif_ptr,
            dif_size + gen.alg_ptr,
            dif_size + alg_size,
        ], dtype=np.int32)
        for row, cols in gen.preallocate_jacobian(idxs, psys, psys.power_injection):
            assert cols == sorted(cols), f"GENROU row {row} columns are not sorted: {cols}"


def _assert_csr_rows_sorted(J):
    for row in range(J.shape[0]):
        cols = J.indices[J.indptr[row]:J.indptr[row + 1]]
        assert np.all(cols[:-1] < cols[1:]), f"CSR row {row} columns are not sorted: {cols.tolist()}"


def test_genrou_jacobian_rows_are_sorted_for_controller_cases(data_dir):
    cases = [
        ("2bus_33.raw", "GENROU.dyr"),
        ("2bus_33.raw", "2bus_SEXS.dyr"),
        ("2bus_33.raw", "2bus_TGOV1.dyr"),
    ]

    for raw_name, dyr_name in cases:
        psys, J = _initialized_jacobian_case(
            os.path.join(data_dir, raw_name),
            os.path.join(data_dir, dyr_name),
        )
        _assert_genrou_preallocation_rows_sorted(psys)
        _assert_csr_rows_sorted(J)


def test_genrou_constructor_defaults_to_no_saturation():
    gen = GenGENROU(
        0, 1.575, 1.512, 0.291, 0.39, 0.1733,
        0.0787, 3.38, 0.0, 6.1, 1.0, 0.05, 0.15
    )

    assert gen.S1 == 0.0
    assert gen.S2 == 0.0


def test_controller_setpoints_are_read_from_theta(data_dir, tmp_path):
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

    cases = [
        ("2bus_33.raw", os.path.join(data_dir, "2bus_SEXS.dyr"), "exc", "vref"),
        ("2bus_33.raw", os.path.join(data_dir, "2bus_TGOV1.dyr"), "gov", "pref"),
        ("2bus_CIM5.raw", os.path.join(data_dir, "2bus_IEESGO.dyr"), "gov", "pref"),
        ("2bus_33.raw", str(esdc1a_dyr), "exc", "vref"),
    ]

    for raw_name, dyr_path, collection, attr in cases:
        psys = load_psse(raw_filename=os.path.join(data_dir, raw_name))
        add_dyr(psys, dyr_path)
        psys.createYbusComplex()
        pf_solution = runpf(psys, verbose=False)
        sysvec, theta = initialize_system(psys, pf_solution)

        devices = getattr(psys, collection)
        ctrl = devices[0]
        par_ptr = ctrl.par_ptr
        setpoint_offset = ctrl.par_dim - 1
        assert theta[par_ptr + setpoint_offset] == pytest.approx(getattr(ctrl, attr))

        F_before = np.zeros_like(sysvec)
        residual_function(F_before, sysvec, theta, psys)

        setattr(ctrl, attr, getattr(ctrl, attr) + 10.0)
        F_after = np.zeros_like(sysvec)
        residual_function(F_after, sysvec, theta, psys)

        assert np.allclose(F_after, F_before)


def test_init_mappings_no_controllers(data_dir):
    """Test initialization mapping for system without controllers."""
    # Load system without controllers
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    psys.add_gen_dynamics(
        psys.gens[0],
        GenGENROU(
            0, 1.575, 1.512, 0.291, 0.39, 0.1733,
            0.0787, 3.38, 0.0, 6.1, 1.0, 0.05, 0.15, 0.0, 0.0
        )
    )
    psys.initialize()
    
    dif = psys.num_dof_dif
    for i, gen in enumerate(psys.gendyn):
        ap = dif + gen.alg_ptr
        
        # Check index arrays
        assert psys.gen_pm_ref_idx[i] == gen.dif_ptr + 6, "p_m0 index wrong"
        assert psys.gen_efd_ref_idx[i] == gen.dif_ptr + 7, "e_fd0 index wrong"
        assert psys.gen_pm_out_idx[i] == ap + 4, "p_m_out index wrong"
        assert psys.gen_efd_out_idx[i] == ap + 5, "e_fd_out index wrong"
        
        # Check generator fields
        assert gen.pm_idx == ap + 4, "gen.pm_idx wrong"
        assert gen.efd_idx == ap + 5, "gen.efd_idx wrong"
        
        # Check no controllers
        assert not gen.has_governor, "Should not have governor"
        assert not gen.has_exciter, "Should not have exciter"
        assert psys.gen_pm_ctrl_col[i] == -1, "Should have no governor column"
        assert psys.gen_efd_ctrl_col[i] == -1, "Should have no exciter column"
        assert psys.gov_mask[i] == 0.0, "Governor mask should be 0"
        assert psys.exc_mask[i] == 0.0, "Exciter mask should be 0"


def test_init_mappings_with_governor(data_dir):
    """Test initialization mapping for system with governor."""
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_CIM5.raw"))
    add_dyr(psys, os.path.join(data_dir, "2bus_IEESGO.dyr"))
    psys.initialize()
    
    dif = psys.num_dof_dif
    for i, gen in enumerate(psys.gendyn):
        if gen.has_governor:
            gov = gen.governor
            ap = dif + gen.alg_ptr
            
            # Check governor mapping
            assert psys.gen_pm_ctrl_col[i] == dif + gov.alg_ptr, "Governor column index wrong"
            assert gov.w_idx == gen.dif_ptr + 4, "Governor w_idx wrong"
            assert psys.gov_mask[i] == 1.0, "Governor mask should be 1"
            assert gov.gen_index == i, "Governor gen_index wrong"


def test_blend_structure_no_controllers(data_dir):
    """Test Jacobian structure of blend rows for generators without controllers."""
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    psys.add_gen_dynamics(
        psys.gens[0],
        GenGENROU(
            0, 1.575, 1.512, 0.291, 0.39, 0.1733,
            0.0787, 3.38, 0.0, 6.1, 1.0, 0.05, 0.15, 0.0, 0.0
        )
    )
    psys.createYbusComplex()
    pf_solution = runpf(psys)
    sysvec, theta = initialize_system(psys, pf_solution)

    # Build Jacobian
    dif_size = psys.num_dof_dif
    alg_size = psys.num_dof_alg
    J = preallocate_jacobian(psys)
    residual_jacobian(J, sysvec, theta, psys)
    
    # Check blend row structure
    dif = psys.num_dof_dif
    for i, gen in enumerate(psys.gendyn):
        if psys.gov_mask[i] == 0.0 and psys.exc_mask[i] == 0.0:
            ap = dif + gen.alg_ptr
            
            # Row ap+4 (p_m_out): should have columns [dp+6, ap+4] sorted
            ap4_cols = J.indices[J.indptr[ap+4]:J.indptr[ap+5]].tolist()
            expected_ap4 = sorted([ap+4, gen.dif_ptr+6])
            assert ap4_cols == expected_ap4, f"Row ap+4 columns wrong: {ap4_cols} != {expected_ap4}"
            
            # Row ap+5 (e_fd_out): should have columns [dp+7, ap+5] sorted
            ap5_cols = J.indices[J.indptr[ap+5]:J.indptr[ap+6]].tolist()
            expected_ap5 = sorted([ap+5, gen.dif_ptr+7])
            assert ap5_cols == expected_ap5, f"Row ap+5 columns wrong: {ap5_cols} != {expected_ap5}"


def test_blend_values_at_t0(data_dir):
    """Test that blend outputs equal references at t=0 (pre-fault)."""
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_33.raw"))
    psys.add_gen_dynamics(
        psys.gens[0],
        GenGENROU(
            0, 1.575, 1.512, 0.291, 0.39, 0.1733,
            0.0787, 3.38, 0.0, 6.1, 1.0, 0.05, 0.15, 0.0, 0.0
        )
    )
    psys.createYbusComplex()
    pf_solution = runpf(psys)
    sysvec, _ = initialize_system(psys, pf_solution)

    # Build state vector
    dif_size = psys.num_dof_dif
    alg_size = psys.num_dof_alg
    z = sysvec
    
    # Check blend outputs equal references
    for i, gen in enumerate(psys.gendyn):
        pm_out = z[psys.gen_pm_out_idx[i]]
        efd_out = z[psys.gen_efd_out_idx[i]]
        pm0 = z[psys.gen_pm_ref_idx[i]]
        efd0 = z[psys.gen_efd_ref_idx[i]]
        
        assert abs(pm_out - pm0) < 1e-8, f"p_m_out != p_m0: {pm_out} != {pm0}"
        assert abs(efd_out - efd0) < 1e-8, f"e_fd_out != e_fd0: {efd_out} != {efd0}"


def test_blend_structure_with_controllers(data_dir):
    """Test Jacobian structure of blend rows when controllers are present."""
    psys = load_psse(raw_filename=os.path.join(data_dir, "2bus_CIM5.raw"))
    add_dyr(psys, os.path.join(data_dir, "2bus_IEESGO.dyr"))
    psys.createYbusComplex()
    pf_solution = runpf(psys)
    sysvec, theta = initialize_system(psys, pf_solution)

    dif_size = psys.num_dof_dif
    alg_size = psys.num_dof_alg
    total_size = dif_size + alg_size
    J = preallocate_jacobian(psys)
    residual_jacobian(J, sysvec, theta, psys)

    dif = psys.num_dof_dif
    for i, gen in enumerate(psys.gendyn):
        if psys.gov_mask[i] == 1.0 or psys.exc_mask[i] == 1.0:
            ap = dif + gen.alg_ptr

            if psys.gov_mask[i] == 1.0:
                expected = sorted([ap + 4, gen.dif_ptr + 6, psys.gen_pm_ctrl_col[i]])
                ap4_cols = J.indices[J.indptr[ap + 4]:J.indptr[ap + 5]].tolist()
                assert ap4_cols == expected, f"Governor blend row mismatch: {ap4_cols} != {expected}"

            if psys.exc_mask[i] == 1.0:
                expected = sorted([ap + 5, gen.dif_ptr + 7, psys.gen_efd_ctrl_col[i]])
                ap5_cols = J.indices[J.indptr[ap + 5]:J.indptr[ap + 6]].tolist()
                assert ap5_cols == expected, f"Exciter blend row mismatch: {ap5_cols} != {expected}"
