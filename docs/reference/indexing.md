# Indexing

Understanding UQGrid's indexing scheme is essential when extending device
models, wiring custom diagnostics, or coupling external solvers. This page
summarizes how external identifiers are mapped into contiguous internal vectors
and how the global state is organized.

## Bus renumbering

- RAW files supply *external* bus numbers (e.g., 101, 325, …). During parsing,
  each bus is wrapped in a `Bus` object.
- The `Bus` constructor assigns a zero-based sequential identifier (`Bus.i`) via
  an internal counter. These internal IDs are used everywhere else in the code
  base, guaranteeing a dense range `[0, nbuses-1]`.
- Device records (loads, generators, shunts) store the internal bus index so
  that the algebraic network equations and the device residuals align with the
  same voltage slots in the global vectors.

```text
External bus → Internal index
       101 → 0
       102 → 1
       325 → 2
       ...
```

Keep the original number around (`Bus.id`) if you need to report results using
utility naming conventions.

## Global vector layout

After all devices are added, `Psystem` tracks:

- `num_dof_dif`: total differential degrees of freedom (`ndif`).
- `num_dof_alg`: total algebraic degrees of freedom (`nalg`).
- `num_pars`: total parameter entries (`npar`).

During integration the coupled vector `z` is split as:

```text
z = [ x | y | v ]
        ↑     ↑
        ndif  ndif + nalg
```

- `x = z[:ndif]` – dynamic states.
- `y = z[ndif:ndif+nalg]` – device algebraic variables.
- `v = z[ndif+nalg:]` – network voltages (length `2 * nbuses`).

Power-flow injections occupy the tail of the residual vector using the same
ordering, which is why bus indices are doubled when working with rectangular
components.

## Device pointers

Every dynamic device derives from `DeviceModel`. When `Psystem.add_device` is
called, the model receives **pointers** into the global vectors:

| Attribute | Meaning |
| --- | --- |
| `dif_ptr` | Offset of the device's differential block within `x`. |
| `alg_ptr` | Offset of the device's algebraic block within `y`. |
| `par_ptr` | Offset of the device's parameters within the `theta` vector. |
| `ndev`    | Sequential device number (useful for diagnostics). |

These pointers are contiguous and never overlap because UQGrid increments the
running totals (`num_dof_dif`, `num_dof_alg`, `num_pars`) every time a device is
registered. Devices expose their own block sizes via `getdim()`.

### Example

```python
for device in psys.devices:
    print(device.model_type, device.dif_ptr, device.alg_ptr, device.par_ptr)
```

Output might look like:

ernor  6   4   10
```text
generator 0   0   0
governor  6   4   10
exciter   8   5   18
ZIPLoad   8   5   25
```

(Exact numbers depend on the case.) The key point is that each device knows
where its block lives in the concatenated vectors, so you can slice into `z`
and `theta` deterministically.

## Convenience selectors

`Psystem` ships a few helpers that return index sets for common analysis tasks,
for example `psys.genspeed_idx_set()` (generator speed states) or
`psys.genangle_idx_set()` (rotor angles). These helpers respect the same global
layout described above.

When building new diagnostics, construct similar utilities instead of hard
coding offsets. Doing so keeps scripts robust even if the underlying models add
states in the future.

## Spotting inconsistencies

If you ever observe:

- Overlapping pointer ranges.
- Device residuals writing outside their allotted slice.
- Buses whose internal index is out of range when addressing voltage entries.

please flag it immediately—these symptoms indicate a parsing bug. The current
review of the indexing logic did **not** surface issues, but we recommend
instrumenting custom scripts with assertions like:

```python
assert device.dif_ptr + device.dif_dim <= psys.num_dof_dif
```

whenever new device types are introduced.

Diagrams or more elaborate examples are welcome additions—feel free to extend
this section as the project evolves.
