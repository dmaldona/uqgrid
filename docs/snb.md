# Saddle-Node Bifurcation (SNB) Module

The SNB module locates the closest saddle-node bifurcation of a steady-state operating point. It solves a four-equation Karush–Kuhn–Tucker (KKT) system that enforces power-flow balance while aligning a perturbation direction with the null-space of the Jacobian.

## Mathematical Formulation

We solve for $(x^\star, \lambda^\star, w^\star, k^\star)$ such that

$$
\begin{aligned}
F(x^\star, \lambda^\star) &= 0,\\
J_x(x^\star, \lambda^\star)^\top w^\star &= 0,\\
\|w^\star\|_2 &= 1,\\
\lambda^\star - \lambda_0 &= k^\star J_\lambda(x^\star, \lambda^\star)^\top w^\star,
\end{aligned}
$$

where $F$ is the power-flow residual, $J_x$ and $J_\lambda$ are its Jacobians with respect to the state variables $x$ and parameters $\lambda$, and $\lambda_0$ denotes the base load vector in canonical order $[P_2,P_4,P_5,Q_2,Q_4,Q_5]$.

The saddle-node distance is

$$
\|\lambda^\star - \lambda_0\|_2,
$$

and the alignment angle

$$
\theta = \cos^{-1}\left(\frac{(\lambda^\star - \lambda_0)^\top n}{\|\lambda^\star - \lambda_0\|_2 \|n\|_2}\right), \quad n = J_\lambda(x^\star, \lambda^\star)^\top w^\star,
$$

quantifies how closely the perturbation direction matches the left nullspace.

## Key Components

- `closest_snb_fsolve`: Newton-style solver that calls SciPy's `fsolve` on the KKT system.
- `build_param_selector`: Constructs the sparse $J_\lambda$ selector matrix in canonical order.
- `scatter_lambda`: Expands the parameter vector back into per-bus active and reactive loads.
- `ClosestSNBResult`: Dataclass container exposing $(x^\star, \lambda^\star, w^\star, k^\star)$ plus diagnostics.

## Worked Examples

### 1. CLI: Dobson 5-Bus Benchmark

```bash
python bin/closest_snb_driver.py --case dobson5
```

This command loads the Dobson & Lu five-bus fixture, computes the closest SNB, and prints a summary table with the distance, alignment angle, and dominant components of the null vector.

### 2. Python API Snippet

```python
from tests.fixtures_snb import build_dobson5_fixture
from uqgrid.snb import build_index_cache, closest_snb_fsolve

fixture = build_dobson5_fixture()
cache = build_index_cache(fixture.psys)
result = closest_snb_fsolve(
    fixture.psys,
    c_vector=None,
    x_init=fixture.x_init,
    w_init=fixture.w_init,
    lambda_init=fixture.lambda_init,
    k_init=fixture.k_init,
)
print(result.distance)
print(result.kkt_residuals)
```

The result object also exposes `lambda_star`, `w_star`, and the unit-normal vector via `result.normal` for further analysis.

## Further Reading

- `tests/test_snb_phaseA.py`: Unit-level tests that validate selector ordering and finite-difference checks.
- `tests/test_snb_dobson5.py`: Regression test anchoring the Dobson benchmark expectations.
