# tests/test_gen_line_properties.py
# Test generator and line properties extraction

import os
import pytest
import numpy as np
from uqgrid.io.parse import load_psse, load_matpower


@pytest.fixture
def data_dir():
    """Fixture to provide the absolute path to the data directory."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'
    )


class TestGeneratorProperties:
    """Tests for generator properties getters and setters."""

    def test_get_gen_pq_psse(self, data_dir):
        """Test get_gen_pq returns correct values from PSSE file."""
        psys = load_psse(os.path.join(data_dir, 'ieee9_v33.raw'))
        p_gen, q_gen = psys.get_gen_pq()
        
        assert len(p_gen) == psys.ngens
        assert len(q_gen) == psys.ngens
        assert all(isinstance(p, (int, float, np.floating)) for p in p_gen)
        assert all(isinstance(q, (int, float, np.floating)) for q in q_gen)

    def test_get_pgen_bounds_psse(self, data_dir):
        """Test get_pgen_bounds returns correct values from PSSE file."""
        psys = load_psse(os.path.join(data_dir, 'ieee9_v33.raw'))
        pg_lb, pg_ub = psys.get_pgen_bounds()
        
        assert len(pg_lb) == psys.ngens
        assert len(pg_ub) == psys.ngens
        # Upper bound should be >= lower bound
        assert all(pg_ub[i] >= pg_lb[i] for i in range(psys.ngens))

    def test_get_qgen_bounds_psse(self, data_dir):
        """Test get_qgen_bounds returns correct values from PSSE file."""
        psys = load_psse(os.path.join(data_dir, 'ieee9_v33.raw'))
        qg_lb, qg_ub = psys.get_qgen_bounds()
        
        assert len(qg_lb) == psys.ngens
        assert len(qg_ub) == psys.ngens
        # Upper bound should be >= lower bound
        assert all(qg_ub[i] >= qg_lb[i] for i in range(psys.ngens))

    def test_get_gen_properties_keys(self, data_dir):
        """Test get_gen_properties returns dictionary with correct keys."""
        psys = load_psse(os.path.join(data_dir, 'ieee9_v33.raw'))
        gen_props = psys.get_gen_properties()
        
        expected_keys = ['p_gen', 'q_gen', 'pg_lb', 'pg_ub', 'qg_lb', 'qg_ub', 'mbase', 'bus']
        assert set(gen_props.keys()) == set(expected_keys)

    def test_set_gen_pq(self, data_dir):
        """Test set_gen_pq correctly updates generator power."""
        psys = load_psse(os.path.join(data_dir, 'ieee9_v33.raw'))
        
        # Get original values
        p_gen_orig, q_gen_orig = psys.get_gen_pq()
        
        # Set new values
        new_p = p_gen_orig * 1.1
        new_q = q_gen_orig * 0.9
        psys.set_gen_pq(new_p, new_q)
        
        # Get updated values
        p_gen_new, q_gen_new = psys.get_gen_pq()
        
        np.testing.assert_array_almost_equal(p_gen_new, new_p)
        np.testing.assert_array_almost_equal(q_gen_new, new_q)

    def test_set_pgen_bounds(self, data_dir):
        """Test set_pgen_bounds correctly updates generator bounds."""
        psys = load_psse(os.path.join(data_dir, 'ieee9_v33.raw'))
        
        # Get original values
        pg_lb_orig, pg_ub_orig = psys.get_pgen_bounds()
        
        # Set new values
        new_lb = pg_lb_orig * 0.9
        new_ub = pg_ub_orig * 1.1
        psys.set_pgen_bounds(new_lb, new_ub)
        
        # Get updated values
        pg_lb_new, pg_ub_new = psys.get_pgen_bounds()
        
        np.testing.assert_array_almost_equal(pg_lb_new, new_lb)
        np.testing.assert_array_almost_equal(pg_ub_new, new_ub)

    def test_set_qgen_bounds(self, data_dir):
        """Test set_qgen_bounds correctly updates generator bounds."""
        psys = load_psse(os.path.join(data_dir, 'ieee9_v33.raw'))
        
        # Get original values
        qg_lb_orig, qg_ub_orig = psys.get_qgen_bounds()
        
        # Set new values
        new_lb = qg_lb_orig * 0.9
        new_ub = qg_ub_orig * 1.1
        psys.set_qgen_bounds(new_lb, new_ub)
        
        # Get updated values
        qg_lb_new, qg_ub_new = psys.get_qgen_bounds()
        
        np.testing.assert_array_almost_equal(qg_lb_new, new_lb)
        np.testing.assert_array_almost_equal(qg_ub_new, new_ub)

    def test_get_gen_pq_matpower(self, data_dir):
        """Test get_gen_pq returns correct values from MATPOWER file."""
        psys = load_matpower(os.path.join(data_dir, 'case14.mat'))
        p_gen, q_gen = psys.get_gen_pq()
        
        assert len(p_gen) == psys.ngens
        assert len(q_gen) == psys.ngens

    def test_get_pgen_bounds_matpower(self, data_dir):
        """Test get_pgen_bounds returns correct values from MATPOWER file."""
        psys = load_matpower(os.path.join(data_dir, 'case14.mat'))
        pg_lb, pg_ub = psys.get_pgen_bounds()
        
        assert len(pg_lb) == psys.ngens
        assert len(pg_ub) == psys.ngens


class TestBranchProperties:
    """Tests for branch/line properties getters and setters."""

    def test_get_branch_properties_keys(self, data_dir):
        """Test get_branch_properties returns dictionary with correct keys."""
        psys = load_psse(os.path.join(data_dir, 'ieee9_v33.raw'))
        branch_props = psys.get_branch_properties()
        
        expected_keys = ['fr', 'to', 'r', 'x', 'sh', 'tap', 'shift', 'rateA', 'rateB', 'rateC']
        assert set(branch_props.keys()) == set(expected_keys)

    def test_get_branch_ratings_psse(self, data_dir):
        """Test get_branch_ratings returns correct values from PSSE file."""
        psys = load_psse(os.path.join(data_dir, 'ieee9_v33.raw'))
        rateA, rateB, rateC = psys.get_branch_ratings()
        
        assert len(rateA) == psys.nbranches
        assert len(rateB) == psys.nbranches
        assert len(rateC) == psys.nbranches
        assert all(isinstance(r, (int, float, np.floating)) for r in rateA)

    def test_set_branch_ratings(self, data_dir):
        """Test set_branch_ratings correctly updates branch ratings."""
        psys = load_psse(os.path.join(data_dir, 'ieee9_v33.raw'))
        
        # Get original values
        rateA_orig, rateB_orig, rateC_orig = psys.get_branch_ratings()
        
        # Set new values
        new_rateA = rateA_orig * 1.1
        new_rateB = rateB_orig * 1.1
        new_rateC = rateC_orig * 1.1
        psys.set_branch_ratings(new_rateA, new_rateB, new_rateC)
        
        # Get updated values
        rateA_new, rateB_new, rateC_new = psys.get_branch_ratings()
        
        np.testing.assert_array_almost_equal(rateA_new, new_rateA)
        np.testing.assert_array_almost_equal(rateB_new, new_rateB)
        np.testing.assert_array_almost_equal(rateC_new, new_rateC)

    def test_branch_properties_matpower(self, data_dir):
        """Test branch properties are correctly loaded from MATPOWER file."""
        psys = load_matpower(os.path.join(data_dir, 'case14.mat'))
        branch_props = psys.get_branch_properties()
        
        assert len(branch_props['fr']) == psys.nbranches
        assert len(branch_props['to']) == psys.nbranches
        assert len(branch_props['rateA']) == psys.nbranches


class TestSetterValidation:
    """Tests for setter validation."""

    def test_set_gen_pq_length_mismatch(self, data_dir):
        """Test set_gen_pq raises assertion error on length mismatch."""
        psys = load_psse(os.path.join(data_dir, 'ieee9_v33.raw'))
        
        # Try to set with wrong length
        with pytest.raises(AssertionError):
            psys.set_gen_pq(np.zeros(1), np.zeros(1))

    def test_set_pgen_bounds_length_mismatch(self, data_dir):
        """Test set_pgen_bounds raises assertion error on length mismatch."""
        psys = load_psse(os.path.join(data_dir, 'ieee9_v33.raw'))
        
        # Try to set with wrong length
        with pytest.raises(AssertionError):
            psys.set_pgen_bounds(np.zeros(1), np.zeros(1))

    def test_set_branch_ratings_length_mismatch(self, data_dir):
        """Test set_branch_ratings raises assertion error on length mismatch."""
        psys = load_psse(os.path.join(data_dir, 'ieee9_v33.raw'))
        
        # Try to set with wrong length
        with pytest.raises(AssertionError):
            psys.set_branch_ratings(np.zeros(1), np.zeros(1), np.zeros(1))
