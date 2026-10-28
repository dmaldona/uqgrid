from pathlib import Path
from types import SimpleNamespace

import numpy as np

from uqgrid.io import parse
from uqgrid.io.parse_psse import read_raw
from uqgrid.simulation.pflow import runpf, validate_power_flow_solution


DATA = Path(__file__).resolve().parents[1] / "data"


def test_two_winding_ratings_preserve_mva_and_zero_ratings(monkeypatch):
    case = read_raw(str(DATA / "ieee9_v33.raw"))
    case.baseMVA = 50.0
    ratings = [(10.0, 20.0, 30.0), (0.0, 0.0, 0.0), (40.0, 50.0, 60.0)]
    for transformer, values in zip(case.transformers, ratings):
        transformer.rateA, transformer.rateB, transformer.rateC = values
    monkeypatch.setattr(parse, "read_raw", lambda path: case)

    psys = parse.load_psse("fixture.raw")

    imported = psys.branches[len(case.branches):]
    assert psys.basemva == 50.0
    assert all(b.raw_identity["element_type"] == "line" for b in psys.branches[:len(case.branches)])
    assert [b.raw_identity["circuit_id"] for b in imported] == [t.ckt.strip("' ") for t in case.transformers]
    assert all(b.raw_identity["element_type"] == "transformer" for b in imported)
    np.testing.assert_array_equal(
        [(branch.rateA, branch.rateB, branch.rateC) for branch in imported], ratings
    )


def test_three_winding_ratings_follow_their_star_branches(monkeypatch):
    case = read_raw(str(DATA / "pf_tests" / "three_winding.raw"))
    case.baseMVA = 50.0
    transformer = case.transthree[0]
    ratings = [(11.0, 12.0, 13.0), (21.0, 22.0, 23.0), (0.0, 0.0, 0.0)]
    for winding, values in enumerate(ratings, start=1):
        for name, value in zip(("rateA", "rateB", "rateC"), values):
            setattr(transformer, f"{name}{winding}", value)
    monkeypatch.setattr(parse, "read_raw", lambda path: case)

    psys = parse.load_psse("fixture.raw")

    assert [b.raw_identity["winding"] for b in psys.branches] == [1, 2, 3]
    assert all(b.raw_identity["parent_bus_ids"] == [transformer.ibus, transformer.jbus, transformer.kbus] for b in psys.branches)
    np.testing.assert_array_equal(
        [(branch.rateA, branch.rateB, branch.rateC) for branch in psys.branches],
        ratings,
    )


def test_imported_transformer_overload_is_checked(monkeypatch):
    case = read_raw(str(DATA / "ieee9_v33.raw"))
    case.transformers[0].rateA = 5.0
    monkeypatch.setattr(parse, "read_raw", lambda path: case)
    psys = parse.load_psse("fixture.raw")
    psys.createYbusComplex()

    solution = runpf(psys)
    diagnostics = validate_power_flow_solution(psys, solution, branch_loading_max=1.0)

    assert "branch_overload" in diagnostics["failure_reasons"]
    assert diagnostics["branch"]["rated_branch_count"] == len(case.branches) + 1
    assert diagnostics["branch"]["loading_argmax"] == len(case.branches)


def test_import_preserves_fixed_switched_and_magnetizing_provenance(monkeypatch):
    case = read_raw(str(DATA / "ieee9_v33.raw"))
    case.shunts = [SimpleNamespace(status=1, busn=1, gshunt=2.0, bshunt=10.0)]
    case.switched_shunts = [
        SimpleNamespace(status=1, busn=1, binit=5.0),
        SimpleNamespace(status=0, busn=1, binit=50.0),
    ]
    case.transformers[0].COD1 = 1
    case.transformers[0].MAG1 = 0.001
    case.transformers[0].MAG2 = 0.002
    monkeypatch.setattr(parse, "read_raw", lambda path: case)

    psys = parse.load_psse("fixture.raw")

    assert len(psys.shunts) == 3
    assert [shunt.is_switched for shunt in psys.shunts] == [False, False, True]
    np.testing.assert_allclose(
        [(shunt.gsh, shunt.bsh) for shunt in psys.shunts],
        [(0.001, 0.002), (0.02, 0.10), (0.0, 0.05)],
    )
