# Generator Controller Blend Architecture

This document summarizes the refactor that decouples generator models from their
controllers while keeping the data layout friendly to vectorization and adjoint
workflows. The key idea is that generators expose *frozen* differential
references alongside algebraic outputs. Controllers only touch their own rows and
communicate with generators through blend equations that live inside the
generator model.

## GENROU state layout

GENROU now allocates two additional differential states and two algebraic
outputs:

- Differential block (8 states):
  `[e_qp, e_dp, phi_1d, phi_2q, w, delta, p_m0, e_fd0]`
- Algebraic block (6 states):
  `[v_q, v_d, i_q, i_d, p_m_out, e_fd_out]`

The last two differential entries (`p_m0`, `e_fd0`) are frozen references that
enter the mass matrix as identity rows. The algebraic outputs (`p_m_out`,
`e_fd_out`) are the only quantities the rest of the system reads.

## System initialization helpers

`Psystem.initialize` now builds global index arrays and controller masks:

- `gen_pm_ref_idx`, `gen_efd_ref_idx`: absolute indices of the frozen
  references.
- `gen_pm_out_idx`, `gen_efd_out_idx`: absolute indices of the algebraic
  outputs.
- `gen_pm_ctrl_col`, `gen_efd_ctrl_col`: column indices written by governors and
  exciters (or `-1` if no controller is attached).
- `gov_mask`, `exc_mask`: `float64` arrays with values `0.0` or `1.0` that allow
  branch-free blends.

During this pass GENROU records its own output indices (`pm_idx`, `efd_idx`) so
its residual and Jacobian never need to look up controller state locations.

Governor routing uses each governor's stored `primary_generator` and optional
`secondary_generator` references. It does not infer the primary from generator
list order or from the first matching back-reference. Initialization validates
that both machines belong to the current system, still reference that governor,
and map to distinct output columns. The speed input always comes from the
configured primary machine, while a dual-output governor retains its declared
secondary-output offset.

## Blend equations

GENROU owns two algebraic equations that mix references with controller output:

```python
p_m_target  = mask_gov * p_m_ctrl  + (1.0 - mask_gov) * p_m0
p_m_out_res = p_m_out - p_m_target

 e_fd_target = mask_exc * e_fd_ctrl + (1.0 - mask_exc) * e_fd0
 e_fd_out_res = e_fd_out - e_fd_target
```

If a controller is absent (`mask == 0.0`) we simply freeze the algebraic output
at the reference value. Because both controller columns and masks are stored in
SoA form, the residual and its Jacobian can be vectorized across generators in
future work.

## Controller responsibilities

Controllers (e.g. IEESGO, ESDC1A) now write only their own rows:

- Governors own an algebraic row that outputs the mechanical power command.
- Exciters own differential rows that compute the field voltage command.

They never mutate generator rows directly. Any generator/controller coupling is
mediated via the blend equations above.

## Assembly changes

The residual and Jacobian assemblers iterate through devices once to let each
model write its equations. They then loop over `psys.gendyn` to call
`residual_blend` and `residual_blend_jac`. This ensures a single writer per row
in the CSR structure while keeping the logic concentrated in one place.

## Testing

`tests/test_blend_structure.py` exercises the new layout:

- Initialization mappings (`gen_*_idx`, controller masks/columns).
- Blend Jacobian sparsity with and without controllers.
- Equality of blend outputs and references at the initial operating point.
- Exact primary/secondary routing when generator order differs from governor
  output order and when multiple governors are present.

`tests/test_pssedyn.py` now derives indices programmatically from the same
arrays, removing the brittle hard-coded offsets it used before.

These checks should accompany any future controller or generator updates to make
sure the contracts remain stable.
