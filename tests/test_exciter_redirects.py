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
