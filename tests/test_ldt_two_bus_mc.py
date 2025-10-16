import numpy as np

from uqgrid.snb import closest_snb_fsolve
from uqgrid.snb.mc import ellipse_collapse_mask, sample_gaussian


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

def test_ldt_mc_two_bus_asymptotics():
    base_res = closest_snb_fsolve(_build_two_bus_system())
    delta = base_res.lambda_star - base_res.lambda0
    mus = [
        base_res.lambda0 + 0.6 * delta,
        base_res.lambda0 + 0.7 * delta,
    ]

    Sigma0 = np.array([0.1, 0.1])
    Sigma0_inv = 1.0 / Sigma0
    scales = [0.5, 0.2, 0.1, 0.05]
    samples_per_scale = {0.5: 20000, 0.2: 50000, 0.1: 120000, 0.05: 150000}
    rng = np.random.default_rng(321)

    for mu in mus:
        rel_errors = []
        for c in scales:
            Sigma = c * Sigma0
            Sigma_inv_diag = (1.0 / c) * Sigma0_inv

            psys = _build_two_bus_system()
            res = closest_snb_fsolve(psys, mu=mu, Sigma_inv=Sigma_inv_diag)

            assert res.beta is not None and res.p_ldt_first is not None

            delta = res.lambda_star - mu
            beta_expected = np.sqrt(np.sum(Sigma_inv_diag * (delta ** 2)))
            assert np.isclose(res.beta, beta_expected, atol=1e-10)

            samples = sample_gaussian(mu, Sigma, samples_per_scale[c], rng=rng)
            mask = ellipse_collapse_mask(samples[:, 0], samples[:, 1])
            phat = mask.mean()
            rel_err = abs(phat - res.p_ldt_first) / max(res.p_ldt_first, 1e-12)
            rel_errors.append(rel_err)

            if c <= 0.1:
                assert rel_err <= 0.2
            if c <= 0.05:
                assert rel_err <= 0.1

        increases = sum(diff > 0 for diff in np.diff(rel_errors))
        assert increases <= 1
        assert rel_errors[-1] <= rel_errors[0]
