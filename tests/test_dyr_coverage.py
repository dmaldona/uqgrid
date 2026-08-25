from pathlib import Path

import pytest

from scripts.validation.dyr_coverage import (
    ACTIVSG_TARGET_NATIVE_MODELS,
    ACTIVSG_TARGET_REDIRECTS,
    DyrCoverageError,
    MachineRecordPolicy,
    analyze_dyr_coverage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_case(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "case.raw"
    raw.write_text(
        """0,   100.00          / PSS/E-33
COMMENT
COMMENT
1,'BUS1',230.0,3,1,1,1,1.0,0.0
0 / END OF BUS DATA, BEGIN LOAD DATA
0 / END OF LOAD DATA, BEGIN FIXED SHUNT DATA
0 / END OF FIXED SHUNT DATA, BEGIN GENERATOR DATA
1,'1',50.0,0.0,100.0,-100.0,1.0,0,100.0,0.0,0.2,0.0,0.1,1.0,1,100.0,100.0,0.0,1,1.0
1,'2',10.0,0.0,100.0,-100.0,1.0,0,100.0,0.0,0.2,0.0,0.1,1.0,0,100.0,100.0,0.0,1,1.0
0 / END OF GENERATOR DATA, BEGIN BRANCH DATA
0 / END OF BRANCH DATA, BEGIN TRANSFORMER DATA
0 / END OF TRANSFORMER DATA, BEGIN AREA DATA
0 / END OF AREA DATA, BEGIN TWO-TERMINAL DC DATA
0 / END OF TWO-TERMINAL DC DATA, BEGIN VSC DC LINE DATA
0 / END OF VSC DC LINE DATA, BEGIN IMPEDANCE CORRECTION DATA
0 / END OF IMPEDANCE CORRECTION DATA, BEGIN MULTI-TERMINAL DC DATA
0 / END OF MULTI-TERMINAL DC DATA, BEGIN MULTI-SECTION LINE DATA
0 / END OF MULTI-SECTION LINE DATA, BEGIN ZONE DATA
0 / END OF ZONE DATA, BEGIN INTER-AREA TRANSFER DATA
0 / END OF INTER-AREA TRANSFER DATA, BEGIN OWNER DATA
0 / END OF OWNER DATA, BEGIN FACTS DEVICE DATA
0 / END OF FACTS DEVICE DATA, BEGIN SWITCHED SHUNT DATA
0 / END OF SWITCHED SHUNT DATA
Q
"""
    )
    dyr = tmp_path / "case.dyr"
    dyr.write_text(
        """1 'GENROU' '1' 1 1 1 1 1 0 1 1 1 1 1 0 0 0 /
1 'GGOV1' '1' 0 0 0.05 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 /
1 'IEEEST' '1' 1 0 1 1 1 1 1 1 1 1 1 1 1 0.1 -0.1 999 -999 /
1 'GENROU' '2' 1 1 1 1 1 0 1 1 1 1 1 0 0 0 /
99 'MYSTERY' '1' 1 /
"""
    )
    return raw, dyr


def test_coverage_classifies_native_redirect_inactive_and_unmatched(tmp_path):
    raw, dyr = _write_case(tmp_path)

    report = analyze_dyr_coverage(
        raw,
        dyr,
        native_models={"GENROU", "IEEEST"},
        redirects={"GGOV1": "TGOV1"},
    )

    assert report.counts == {
        "active": 3,
        "inactive": 1,
        "native": 2,
        "redirected": 1,
        "unsupported": 0,
        "unmatched": 1,
        "duplicate": 0,
    }
    assert report.by_source_model["GGOV1"].redirected == 1
    assert report.records[1].effective_model == "TGOV1"
    assert report.machine_policy == MachineRecordPolicy.SOURCE_DYR
    assert report.active_generators_without_machine == ()


def test_strict_coverage_rejects_unmatched_active_record(tmp_path):
    raw, dyr = _write_case(tmp_path)

    with pytest.raises(DyrCoverageError, match="unmatched=1"):
        analyze_dyr_coverage(
            raw,
            dyr,
            native_models={"GENROU", "IEEEST"},
            redirects={"GGOV1": "TGOV1"},
            strict=True,
        )


def test_strict_coverage_rejects_unsupported_active_record(tmp_path):
    raw, dyr = _write_case(tmp_path)
    dyr.write_text(dyr.read_text().replace("1 'IEEEST'", "1 'MYSTERY'"))

    with pytest.raises(DyrCoverageError, match="unsupported=1"):
        analyze_dyr_coverage(
            raw,
            dyr,
            native_models={"GENROU", "IEEEST"},
            redirects={"GGOV1": "TGOV1"},
            strict=True,
        )


def test_unknown_model_on_active_load_is_unsupported(tmp_path):
    raw, dyr = _write_case(tmp_path)
    raw.write_text(
        raw.read_text().replace(
            "0 / END OF BUS DATA, BEGIN LOAD DATA\n",
            "0 / END OF BUS DATA, BEGIN LOAD DATA\n"
            "1,'L1',1,1,1,10.0,2.0,0.0,0.0,0.0,0.0,1\n",
        )
    )
    dyr.write_text("1 'LOADX' 'L1' 1 /\n")

    report = analyze_dyr_coverage(raw, dyr, native_models=set())

    assert report.counts["unsupported"] == 1
    assert report.counts["unmatched"] == 0


def test_source_policy_reports_machine_less_active_generator(tmp_path):
    raw, dyr = _write_case(tmp_path)
    dyr.write_text(dyr.read_text().replace("1 'GENROU' '1'", "1 'SEXS' '1'"))

    report = analyze_dyr_coverage(
        raw,
        dyr,
        native_models={"GENROU", "SEXS", "IEEEST"},
        redirects={"GGOV1": "TGOV1"},
    )

    assert report.active_generators_without_machine == ((1, "1"),)
    assert report.machine_policy == MachineRecordPolicy.SOURCE_DYR

    summary = report.summary_dict()
    assert summary["source_machine_count"] == 0
    assert summary["synthetic_machine_count"] == 0
    assert summary["effective_machine_count"] == 0

    synthetic = analyze_dyr_coverage(
        raw,
        dyr,
        native_models={"GENROU", "SEXS", "IEEEST"},
        redirects={"GGOV1": "TGOV1"},
        machine_policy=MachineRecordPolicy.SYNTHETIC,
    ).summary_dict()
    assert synthetic["source_machine_count"] == 0
    assert synthetic["synthetic_machine_count"] == 1
    assert synthetic["effective_machine_count"] == 1


def test_duplicate_source_record_is_reported(tmp_path):
    raw, dyr = _write_case(tmp_path)
    first = dyr.read_text().splitlines()[0]
    dyr.write_text(dyr.read_text() + first + "\n")

    report = analyze_dyr_coverage(
        raw,
        dyr,
        native_models={"GENROU", "IEEEST"},
        redirects={"GGOV1": "TGOV1"},
    )

    assert report.counts["duplicate"] == 1
    assert report.counts["active"] == 4
    with pytest.raises(DyrCoverageError, match="duplicate=1"):
        report.require_complete()


def test_conflicting_models_for_same_controller_are_duplicates(tmp_path):
    raw, dyr = _write_case(tmp_path)
    dyr.write_text(
        dyr.read_text()
        + "1 'TGOV1' '1' 0.05 0.1 1.2 0 0.2 10 0 /\n"
    )
    report = analyze_dyr_coverage(
        raw,
        dyr,
        native_models={"GENROU", "IEEEST", "TGOV1"},
        redirects={"GGOV1": "TGOV1"},
    )
    assert report.counts["duplicate"] == 1


def test_duplicate_inactive_record_remains_inactive(tmp_path):
    raw, dyr = _write_case(tmp_path)
    inactive = dyr.read_text().splitlines()[3]
    dyr.write_text(dyr.read_text() + inactive + "\n")

    report = analyze_dyr_coverage(
        raw,
        dyr,
        native_models={"GENROU", "IEEEST"},
        redirects={"GGOV1": "TGOV1"},
    )

    assert report.counts["active"] == 3
    assert report.counts["inactive"] == 2
    assert report.counts["duplicate"] == 1


@pytest.mark.parametrize(
    ("case", "expected_counts", "machine_less"),
    [
        (
            "ACTIVSg200",
            {"active": 114, "inactive": 33, "native": 114, "redirected": 0},
            0,
        ),
        (
            "ACTIVSg500",
            {"active": 170, "inactive": 102, "native": 170, "redirected": 0},
            0,
        ),
        (
            "ACTIVSg2000",
            {"active": 1335, "inactive": 404, "native": 989, "redirected": 346},
            98,
        ),
    ],
)
def test_target_case_policy_inventory(case, expected_counts, machine_less):
    raw = PROJECT_ROOT / "data" / f"{case}.raw"
    dyr = PROJECT_ROOT / "data" / f"{case}.dyr"
    if not raw.exists() or not dyr.exists():
        pytest.skip(f"{case} data files are not installed")

    report = analyze_dyr_coverage(
        raw,
        dyr,
        native_models=ACTIVSG_TARGET_NATIVE_MODELS,
        redirects=ACTIVSG_TARGET_REDIRECTS,
        strict=True,
    )

    for name, expected in expected_counts.items():
        assert report.counts[name] == expected
    assert len(report.active_generators_without_machine) == machine_less


@pytest.mark.parametrize(
    ("case", "native", "unsupported"),
    [("ACTIVSg200", 114, 0), ("ACTIVSg500", 170, 0), ("ACTIVSg2000", 989, 0)],
)
def test_target_case_current_implementation_inventory(case, native, unsupported):
    raw = PROJECT_ROOT / "data" / f"{case}.raw"
    dyr = PROJECT_ROOT / "data" / f"{case}.dyr"
    if not raw.exists() or not dyr.exists():
        pytest.skip(f"{case} data files are not installed")

    report = analyze_dyr_coverage(raw, dyr)

    assert report.counts["native"] == native
    assert report.counts["unsupported"] == unsupported
    assert report.counts["unmatched"] == 0
    assert report.counts["duplicate"] == 0
