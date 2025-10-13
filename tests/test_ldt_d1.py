import numpy as np

from tests.fixtures_snb import build_dobson5_fixture

from uqgrid.snb import build_index_cache, closest_snb_fsolve, extract_lambda


def _build_two_bus_system():
    from uqgrid.core.psydef import Psystem

    psys = Psystem(basemva=1.0)
    psys.add_bus(1, 3)
    psys.add_bus(2, 1)

    for bus in psys.buses:
        bus.set_vinit(1.0, 0.0)

    psys.add_branch(0, 1, r=0.0, x=0.25)
    psys.add_gen(bus=0, idx_name="G1", psch=0.0, qsch=0.0)
    psys.add_load(bus=1, tag="LD1", pload=0.5, qload=0.3)

    psys.assemble()
    psys.createYbusComplex()
    return psys


def _reference_two_bus_result():
    psys = _build_two_bus_system()
    return closest_snb_fsolve(psys)


def test_euclidean_reduction_two_bus():
    ref = _reference_two_bus_result()

    psys = _build_two_bus_system()
    cache = build_index_cache(psys)
    lambda0 = extract_lambda(psys, cache)

    diag_inv = np.ones_like(lambda0)
    res = closest_snb_fsolve(psys, mu=lambda0, Sigma_inv=diag_inv)

    assert np.allclose(res.lambda_star, ref.lambda_star, atol=1e-9)
    assert np.allclose(res.w_star, ref.w_star, atol=1e-9)
    assert np.isclose(res.distance, ref.distance, atol=1e-8)
    assert np.isclose(res.angle, ref.angle, atol=1e-7)
    for key in ref.kkt_residuals:
        assert np.isclose(res.kkt_residuals[key], ref.kkt_residuals[key], atol=1e-9)

    assert res.beta is not None
    assert np.isclose(res.beta, np.linalg.norm(res.lambda_star - lambda0), atol=1e-10)


def test_euclidean_reduction_dobson():
    fixture_ref = build_dobson5_fixture()
    ref = closest_snb_fsolve(fixture_ref.psys)

    fixture = build_dobson5_fixture()
    lambda0 = fixture.lambda0
    diag_inv = np.ones_like(lambda0)

    res = closest_snb_fsolve(fixture.psys, mu=lambda0, Sigma_inv=diag_inv)

    assert np.allclose(res.lambda_star, ref.lambda_star, atol=1e-9)
    assert np.allclose(res.w_star, ref.w_star, atol=1e-9)
    assert np.isclose(res.distance, ref.distance, atol=1e-8)
    assert np.isclose(res.angle, ref.angle, atol=1e-7)
    for key in ref.kkt_residuals:
        assert np.isclose(res.kkt_residuals[key], ref.kkt_residuals[key], atol=1e-9)

    assert res.beta is not None
    assert np.isclose(res.beta, np.linalg.norm(res.lambda_star - lambda0), atol=1e-10)


def test_anisotropy_tilts_two_bus():
    ref = _reference_two_bus_result()

    psys = _build_two_bus_system()
    cache = build_index_cache(psys)
    lambda0 = extract_lambda(psys, cache)
    Sigma_inv_diag = np.array([10.0, 1.0])

    res = closest_snb_fsolve(psys, mu=lambda0, Sigma_inv=Sigma_inv_diag)

    delta_ref = ref.lambda_star - ref.lambda0
    delta_ldt = res.lambda_star - lambda0

    ratio_ref = abs(delta_ref[1]) / max(abs(delta_ref[0]), 1e-12)
    ratio_ldt = abs(delta_ldt[1]) / max(abs(delta_ldt[0]), 1e-12)
    assert ratio_ldt > ratio_ref

    beta_expected = np.sqrt(np.sum(Sigma_inv_diag * (delta_ldt ** 2)))
    assert res.beta is not None and np.isclose(res.beta, beta_expected, atol=1e-10)
    assert res.I_value is not None and np.isclose(res.I_value, 0.5 * beta_expected ** 2, atol=1e-12)