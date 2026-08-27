import os

import pytest

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.models import ExcSEXS


def test_expic1_redirect_uses_frozen_sexs_defaults(tmp_path, caplog):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    source = tuple(range(1, 25))
    dyr = tmp_path / "expic1.dyr"
    dyr.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        + "1 'EXPIC1' 1 " + " ".join(str(value) for value in source) + " /\n"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))

    add_dyr(psys, str(dyr))

    exciter = psys.exc[0]
    assert isinstance(exciter, ExcSEXS)
    assert exciter.source_model == "EXPIC1"
    assert exciter.source_parameters == tuple(str(value) for value in source)
    assert (
        exciter.TA_TB, exciter.TB, exciter.K, exciter.TE,
        exciter.Emin, exciter.Emax,
    ) == pytest.approx((0.4, 5.0, 20.0, 1.0, -99.0, 99.0))
    assert exciter.enable_limits
    assert psys.dynamic_model_redirects == [
        {
            "source_model": "EXPIC1",
            "effective_model": "SEXS",
            "bus": 1,
            "device_id": "1",
            "source_parameters": exciter.source_parameters,
        }
    ]
    assert "EXPIC1->SEXS: 1" in caplog.text


def _load_with_dyr(tmp_path, records):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    dyr = tmp_path / "redirects.dyr"
    dyr.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        + "".join(records)
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))
    return psys


@pytest.mark.parametrize(
    ("source_model", "parameter_count"),
    [("SCRX", 8), ("ESAC6A", 23)],
)
def test_phase4_redirects_preserve_source_and_use_frozen_sexs_defaults(
    tmp_path, source_model, parameter_count
):
    source = tuple(str(value) for value in range(1, parameter_count + 1))
    psys = _load_with_dyr(
        tmp_path,
        [f"1 '{source_model}' 1 {' '.join(source)} /\n"],
    )

    exciter = psys.exc[0]
    assert isinstance(exciter, ExcSEXS)
    assert exciter.source_model == source_model
    assert exciter.source_parameters == source
    assert (
        exciter.TA_TB, exciter.TB, exciter.K, exciter.TE,
        exciter.Emin, exciter.Emax,
    ) == pytest.approx((0.4, 5.0, 20.0, 1.0, -99.0, 99.0))
    assert exciter.enable_limits
    assert psys.dynamic_model_redirects == [
        {
            "source_model": source_model,
            "effective_model": "SEXS",
            "bus": 1,
            "device_id": "1",
            "source_parameters": source,
        }
    ]


@pytest.mark.parametrize(
    ("source_model", "parameter_count"),
    [("SCRX", 8), ("ESAC6A", 23)],
)
def test_phase4_redirects_require_exact_field_count(
    tmp_path, source_model, parameter_count
):
    source = " ".join(str(value) for value in range(parameter_count - 1))
    with pytest.raises(
        ValueError, match=rf"{source_model}.*requires {parameter_count} parameters"
    ):
        _load_with_dyr(tmp_path, [f"1 '{source_model}' 1 {source} /\n"])


def test_phase4_redirects_skip_inactive_and_unmatched_generators(tmp_path, caplog):
    psys = _load_with_dyr(
        tmp_path,
        [
            "1 'SCRX' 2 1 2 3 4 5 6 7 8 /\n",
            "2 'ESAC6A' 1 " + " ".join(str(value) for value in range(23)) + " /\n",
        ],
    )

    assert psys.exc == []
    assert psys.dynamic_model_redirects == []
    assert "Cannot pair ESAC6A with bus 2 and idx 1. Skipping." in caplog.text
    assert "Applied dynamic-model compatibility redirects" not in caplog.text


def test_phase4_redirects_emit_one_aggregate_warning(tmp_path, caplog):
    psys = _load_with_dyr(
        tmp_path,
        [
            "1 'SCRX' 1 1 2 3 4 5 6 7 8 /\n",
            "1 'ESAC6A' 1 " + " ".join(str(value) for value in range(23)) + " /\n",
        ],
    )

    messages = [
        record.message for record in caplog.records
        if "Applied dynamic-model compatibility redirects" in record.message
    ]
    assert len(messages) == 1
    assert "ESAC6A->SEXS: 1" in messages[0]
    assert "SCRX->SEXS: 1" in messages[0]
    assert len(psys.dynamic_model_redirects) == 2
