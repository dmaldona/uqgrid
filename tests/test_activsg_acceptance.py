from collections import Counter
from pathlib import Path

import pytest

from scripts.validation.activsg_acceptance import (
    _require_target_inventory,
    reconcile_runtime_attachments,
    runtime_attachment_counts,
)
from scripts.validation.dyr_coverage import (
    ACTIVSG_TARGET_NATIVE_MODELS,
    ACTIVSG_TARGET_REDIRECTS,
    analyze_dyr_coverage,
)
from uqgrid.io.parse import add_dyr, load_psse


ROOT = Path(__file__).resolve().parents[1]


EXPECTED = {
    "ACTIVSg200": {
        "GENROU": 38,
        "SEXS": 38,
        "TGOV1": 38,
    },
    "ACTIVSg500": {
        "GAST": 6,
        "GENROU": 56,
        "HYGOV": 35,
        "IEEEST": 2,
        "SEXS": 56,
        "TGOV1": 15,
    },
    "ACTIVSg2000": {
        "ESAC1A": 2,
        "ESAC6A": 2,
        "ESDC1A": 10,
        "ESDC2A": 1,
        "ESST4B": 212,
        "EXAC1": 4,
        "EXAC2": 31,
        "EXPIC1": 52,
        "GENROU": 314,
        "GENSAL": 20,
        "GGOV1": 288,
        "HYGOV": 20,
        "IEEEG1": 26,
        "IEEEST": 333,
        "IEEET1": 16,
        "SCRX": 4,
    },
}


@pytest.mark.parametrize("case", EXPECTED)
def test_target_runtime_attachments_match_strict_coverage(case):
    raw = ROOT / "data" / f"{case}.raw"
    dyr = ROOT / "data" / f"{case}.dyr"
    if not raw.exists() or not dyr.exists():
        pytest.skip(f"{case} data files are not installed")
    report = analyze_dyr_coverage(
        raw,
        dyr,
        native_models=ACTIVSG_TARGET_NATIVE_MODELS,
        redirects=ACTIVSG_TARGET_REDIRECTS,
        strict=True,
    )
    psys = load_psse(str(raw))
    add_dyr(psys, str(dyr))

    summary = reconcile_runtime_attachments(report, psys)
    assert summary["by_source_model"] == EXPECTED[case]
    assert sum(runtime_attachment_counts(psys).values()) == report.counts["active"]

    expected_redirects = Counter(
        f"{record.source_model}->{record.effective_model}"
        for record in report.records
        if record.status == "redirected"
    )
    assert summary["redirects"] == dict(sorted(expected_redirects.items()))


def test_target_inventory_rejects_incomplete_report(tmp_path):
    raw = ROOT / "data" / "ACTIVSg200.raw"
    dyr = tmp_path / "ACTIVSg200.dyr"
    dyr.write_text("")
    report = analyze_dyr_coverage(
        raw,
        dyr,
        native_models=ACTIVSG_TARGET_NATIVE_MODELS,
        redirects=ACTIVSG_TARGET_REDIRECTS,
        strict=True,
    )

    with pytest.raises(AssertionError, match="coverage active"):
        _require_target_inventory("ACTIVSg200", report)
