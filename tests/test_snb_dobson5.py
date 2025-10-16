import numpy as np
import pytest

from tests.fixtures_snb import build_dobson5_fixture

from uqgrid.snb import (
	build_fixed_injections,
	build_index_cache,
	build_param_selector,
	closest_snb_fsolve,
	scatter_lambda,
)
from uqgrid.snb.pf import solution_to_state_vector
from uqgrid.simulation.pflow import jac_wrapper, resfun_wrapper, runpf


@pytest.fixture(scope="module")
def dobson_fixture():
	return build_dobson5_fixture()


@pytest.fixture(scope="module")
def dobson_cache_and_selector(dobson_fixture):
	cache = build_index_cache(dobson_fixture.psys)
	selector = build_param_selector(cache)
	return cache, selector


@pytest.fixture(scope="module")
def dobson_pf_context(dobson_fixture, dobson_cache_and_selector):
	cache, _ = dobson_cache_and_selector
	pf_solution = runpf(dobson_fixture.psys, verbose=False)
	x_pf = solution_to_state_vector(dobson_fixture.psys, pf_solution, cache)
	p_fixed, q_fixed = build_fixed_injections(dobson_fixture.psys, cache)
	return {
		"x_pf": x_pf,
		"p_fixed": p_fixed,
		"q_fixed": q_fixed,
		"ybus": dobson_fixture.psys.ybus_mat,
		"graph": dobson_fixture.psys.graph_mat,
	}


@pytest.fixture(scope="module")
def dobson_result(dobson_fixture, dobson_cache_and_selector):
	cache, _ = dobson_cache_and_selector
	result = closest_snb_fsolve(
		dobson_fixture.psys,
		c_vector=np.ones(cache.n_unknowns),
		x_init=dobson_fixture.x_init,
		w_init=dobson_fixture.w_init,
		lambda_init=dobson_fixture.lambda_init,
		k_init=dobson_fixture.k_init,
	)
	return result


def test_lambda0_matches_load_order(dobson_fixture):
	cache = build_index_cache(dobson_fixture.psys)
	lambda0 = dobson_fixture.lambda0
	assert lambda0.shape[0] == 2 * cache.n_pq

	# Positive loads at PQ buses should align with lambda ordering: [P2,P4,P5,Q2,Q4,Q5].
	for offset, bus_idx in enumerate(cache.pq_buses):
		load = next(ld for ld in dobson_fixture.psys.loads if ld.bus == bus_idx)
		assert np.isclose(lambda0[offset], load.pload)
		assert np.isclose(lambda0[cache.n_pq + offset], load.qload)


def test_selector_respects_canonical_order(dobson_fixture, dobson_cache_and_selector, dobson_pf_context):
	cache, selector = dobson_cache_and_selector
	ctx = dobson_pf_context

	lambda_base = dobson_fixture.lambda0
	p_load, q_load = scatter_lambda(lambda_base, dobson_fixture.psys, cache)
	pinj = ctx["p_fixed"] + p_load
	qinj = ctx["q_fixed"] + q_load

	base_F = resfun_wrapper(
		ctx["x_pf"],
		dobson_fixture.psys.buses_v_magnitudes.copy() if hasattr(dobson_fixture.psys, "buses_v_magnitudes") else np.array([bus.v0m for bus in dobson_fixture.psys.buses]),
		dobson_fixture.psys.buses_v_angles.copy() if hasattr(dobson_fixture.psys, "buses_v_angles") else np.array([bus.v0a for bus in dobson_fixture.psys.buses]),
		pinj.copy(),
		qinj.copy(),
		ctx["ybus"],
		cache.bus_type,
		cache.pq_indices,
		cache.pqv_indices,
		ctx["graph"],
	)

	h = 1e-6
	# Check first active column (P2) and first reactive column (Q2).
	for idx in (0, cache.n_pq):
		lam_pert = lambda_base.copy()
		lam_pert[idx] += h
		p_load_pert, q_load_pert = scatter_lambda(lam_pert, dobson_fixture.psys, cache)
		pinj_pert = ctx["p_fixed"] + p_load_pert
		qinj_pert = ctx["q_fixed"] + q_load_pert

		F_pert = resfun_wrapper(
			ctx["x_pf"],
			dobson_fixture.psys.buses_v_magnitudes.copy() if hasattr(dobson_fixture.psys, "buses_v_magnitudes") else np.array([bus.v0m for bus in dobson_fixture.psys.buses]),
			dobson_fixture.psys.buses_v_angles.copy() if hasattr(dobson_fixture.psys, "buses_v_angles") else np.array([bus.v0a for bus in dobson_fixture.psys.buses]),
			pinj_pert.copy(),
			qinj_pert.copy(),
			ctx["ybus"],
			cache.bus_type,
			cache.pq_indices,
			cache.pqv_indices,
			ctx["graph"],
		)

		fd_column = (F_pert - base_F) / h
		selector_column = selector[:, idx].toarray().ravel()
		assert np.allclose(fd_column, selector_column, atol=5e-6)


def test_dobson5_regression(dobson_fixture, dobson_cache_and_selector, dobson_pf_context, dobson_result):
	cache, selector = dobson_cache_and_selector
	ctx = dobson_pf_context
	result = dobson_result

	assert np.allclose(dobson_fixture.lambda0, result.lambda0, atol=1e-9)

	delta = result.lambda_star - result.lambda0
	assert np.linalg.norm(delta - dobson_fixture.expected_delta, ord=np.inf) < 1e-3

	normal = np.asarray(selector.transpose().dot(result.w_star)).ravel()
	angle = np.arccos(
		np.clip((delta @ normal) / (np.linalg.norm(delta) * np.linalg.norm(normal)), -1.0, 1.0)
	)
	assert angle < 1e-3
	assert result.angle < 1e-3

	p_load_star, q_load_star = scatter_lambda(result.lambda_star, dobson_fixture.psys, cache)
	pinj_star = ctx["p_fixed"] + p_load_star
	qinj_star = ctx["q_fixed"] + q_load_star

	jac_star = jac_wrapper(
		result.x_star,
		dobson_fixture.psys.buses_v_magnitudes.copy() if hasattr(dobson_fixture.psys, "buses_v_magnitudes") else np.array([bus.v0m for bus in dobson_fixture.psys.buses]),
		dobson_fixture.psys.buses_v_angles.copy() if hasattr(dobson_fixture.psys, "buses_v_angles") else np.array([bus.v0a for bus in dobson_fixture.psys.buses]),
		pinj_star.copy(),
		qinj_star.copy(),
		ctx["ybus"],
		cache.bus_type,
		cache.pq_indices,
		cache.pqv_indices,
		ctx["graph"],
	)
	left_null_res = jac_star.transpose().dot(result.w_star)
	assert np.linalg.norm(left_null_res, ord=np.inf) < 1e-7

	stationarity = delta - result.k_star * normal
	assert np.linalg.norm(stationarity, ord=np.inf) < 5e-6

	expected_distance = np.linalg.norm(dobson_fixture.expected_delta)
	assert abs(np.linalg.norm(delta) - expected_distance) < 1e-3
	assert abs(result.distance - expected_distance) < 1e-3

	assert np.linalg.norm(result.lambda_star - dobson_fixture.lambda_reference, ord=np.inf) < 1e-3

	for key, value in result.kkt_residuals.items():
		assert value < 1e-5, f"Residual {key} too large: {value}"
