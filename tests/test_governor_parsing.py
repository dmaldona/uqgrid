import os

import pytest

from uqgrid.io.parse import add_dyr, load_psse
from uqgrid.models import GovGAST, GovHYGOV, GovIEEEG1, GovTGOV1
from uqgrid.simulation.config import IntegrationConfig
from uqgrid.simulation.dynamics import integrate_system


@pytest.fixture
def data_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )


def _write_dyr(tmp_path, governor_record):
    path = tmp_path / "governor.dyr"
    path.write_text(
        "1 'GENROU' 1 6.1 0.05 1.0 0.15 3.38 0.0 1.575 1.512 0.291 0.39 0.1733 0.0787 0.0 0.0 /\n"
        + governor_record
        + "\n"
    )
    return path


@pytest.mark.parametrize(
    "model_name",
    ("GAST2A", "HYGOVX", "IEEEG1D", "IEEEG1_GE", "EXAC1A"),
)
def test_parser_rejects_unsupported_model_variants(
    data_dir, tmp_path, model_name
):
    dyr = _write_dyr(tmp_path, f"1 '{model_name}' 1 /")
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))

    with pytest.raises(
        ValueError,
        match=rf"Unsupported DYR model '{model_name}'.*bus 1",
    ):
        add_dyr(psys, str(dyr))


@pytest.mark.parametrize("model_token", ("'gast'", '"GaSt"'))
def test_parser_normalizes_exact_model_name(data_dir, tmp_path, model_token):
    dyr = _write_dyr(
        tmp_path,
        f"1 {model_token} 1 0.05 0.4 0.1 3 1 2 1 0 0 /",
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))

    add_dyr(psys, str(dyr))

    assert len(psys.gov) == 1
    assert isinstance(psys.gov[0], GovGAST)


@pytest.mark.parametrize(
    ("model_name", "parameters", "expected_count"),
    (
        ("GAST", "0.05 0.4 0.1 3 1 2 1 0 0", 9),
        ("HYGOV", "0.05 0.4 5.0 0.2 0.5 0.167 1 0 1.2 1.25 0.2 0.08", 12),
        (
            "IEEEG1",
            "0 0 20 0.2 0 0.1 0.3 -0.3 1.2 0 0.4 0.5 0 1.0 0.5 0 0 0 0 0 0 0",
            22,
        ),
    ),
)
@pytest.mark.parametrize("count_delta", (-1, 1))
def test_parser_rejects_incorrect_governor_parameter_count(
    data_dir, tmp_path, model_name, parameters, expected_count, count_delta
):
    values = parameters.split()
    if count_delta < 0:
        values.pop()
    else:
        values.append("999")
    dyr = _write_dyr(
        tmp_path,
        f"1 '{model_name}' 1 {' '.join(values)} /",
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))

    with pytest.raises(
        ValueError,
        match=rf"{model_name}.*requires {expected_count} parameters.*got",
    ):
        add_dyr(psys, str(dyr))


def test_ggov1_redirect_uses_source_r_and_frozen_tgov1_defaults(
    data_dir, tmp_path, caplog
):
    dyr = _write_dyr(
        tmp_path,
        "1 'GGOV1' 1 1 1 0.05 1 0.05 -0.05 5 1 0 1 1 0.15 0.1 1 0 0.1 0 3 1 1 1 0 1 -1 0 0.01 10 0.1 100 0 4 5 99 -99 /",
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))

    add_dyr(psys, str(dyr))

    gov = psys.gov[0]
    assert isinstance(gov, GovTGOV1)
    assert gov.source_model == "GGOV1"
    assert gov.R == pytest.approx(0.05)
    assert (gov.T1, gov.VMAX, gov.VMIN, gov.T2, gov.T3, gov.DT) == pytest.approx(
        (0.1, 1.2, 0.0, 0.2, 10.0, 0.0)
    )
    assert psys.dynamic_model_redirects == [
        {
            "source_model": "GGOV1",
            "effective_model": "TGOV1",
            "bus": 1,
            "device_id": "1",
            "source_parameters": gov.source_parameters,
        }
    ]
    assert "GGOV1->TGOV1: 1" in caplog.text


def test_ggov1_redirect_rejects_unapproved_selector(data_dir, tmp_path):
    dyr = _write_dyr(
        tmp_path,
        "1 'GGOV1' 1 2 1 0.05 1 0.05 -0.05 5 1 0 1 1 0.15 0.1 1 0 0.1 0 3 1 1 1 0 1 -1 0 0.01 10 0.1 100 0 4 5 99 -99 /",
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))

    with pytest.raises(ValueError, match="Rselect=1 and Fswitch=1"):
        add_dyr(psys, str(dyr))


def test_gast_parser_maps_fields_and_power_base(data_dir, tmp_path):
    dyr = _write_dyr(tmp_path, "1 'GAST' 1 0.05 0.4 0.1 3 1 2 1 0 0 /")
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))

    add_dyr(psys, str(dyr))

    gov = psys.gov[0]
    assert isinstance(gov, GovGAST)
    assert (gov.R, gov.T1, gov.T2, gov.T3, gov.AT, gov.KT) == pytest.approx(
        (0.05, 0.4, 0.1, 3.0, 1.0, 2.0)
    )
    assert (gov.VMAX, gov.VMIN, gov.DT) == pytest.approx((1.0, 0.0, 0.0))
    assert gov.enable_limits is True


def test_gast_parser_converts_power_quantities(data_dir, tmp_path):
    dyr = _write_dyr(tmp_path, "1 'GAST' 1 0.05 0.4 0.1 3 1 2 1.2 -0.1 0.3 /")
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    psys.gens[0].mbase = 50.0

    add_dyr(psys, str(dyr))

    gov = psys.gov[0]
    assert gov.R == pytest.approx(0.1)
    assert (gov.AT, gov.VMAX, gov.VMIN, gov.DT) == pytest.approx(
        (0.5, 0.6, -0.05, 0.15)
    )


def test_hygov_parser_maps_fields_and_power_base(data_dir, tmp_path):
    dyr = _write_dyr(
        tmp_path, "1 'HYGOV' 1 0.05 0.4 5.0 0.2 0.5 0.167 1 0 1.2 1.25 0.2 0.08 /"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))

    add_dyr(psys, str(dyr))

    gov = psys.gov[0]
    assert isinstance(gov, GovHYGOV)
    assert (
        gov.R, gov.r, gov.Tr, gov.Tf, gov.Tg, gov.VELM, gov.GMAX,
        gov.GMIN, gov.Tw, gov.At, gov.DT, gov.qNL,
    ) == pytest.approx((0.05, 0.4, 5.0, 0.2, 0.5, 0.167, 1.0, 0.0, 1.2, 1.25, 0.2, 0.08))
    assert gov.enable_limits is True
    assert gov.adjust_initial_limits is True


def test_hygov_parser_converts_power_quantities(data_dir, tmp_path):
    dyr = _write_dyr(
        tmp_path, "1 'HYGOV' 1 0.05 0.4 5.0 0.2 0.5 0.2 1.2 0 1.2 1.25 0.4 0.08 /"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    psys.gens[0].mbase = 50.0

    add_dyr(psys, str(dyr))

    gov = psys.gov[0]
    assert (gov.R, gov.r) == pytest.approx((0.1, 0.8))
    assert (gov.VELM, gov.GMAX, gov.GMIN, gov.DT, gov.qNL) == pytest.approx(
        (0.1, 0.6, 0.0, 0.2, 0.04)
    )


def test_ieeeg1_parser_maps_active_single_machine_mode(data_dir, tmp_path):
    dyr = _write_dyr(
        tmp_path,
        "1 'IEEEG1' 1 0 0 20 0.2 0 0.1 0.3 -0.3 1.2 0 0.4 0.5 0 1.0 0.5 0 0 0 0 0 0 0 /",
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))

    add_dyr(psys, str(dyr))

    gov = psys.gov[0]
    assert isinstance(gov, GovIEEEG1)
    assert gov.has_secondary_output is False
    assert gov.adjust_initial_limits is True
    assert (gov.K, gov.T1, gov.T2, gov.T3, gov.UO, gov.UC) == pytest.approx(
        (20.0, 0.2, 0.0, 0.1, 0.3, -0.3)
    )
    assert sum(gov.normalized_K) == pytest.approx(1.0)


def test_ieeeg1_parser_converts_power_quantities(data_dir, tmp_path):
    dyr = _write_dyr(
        tmp_path,
        "1 'IEEEG1' 1 0 0 20 0.2 0 0.1 0.3 -0.3 1.2 0 0.4 0.5 0 1.0 0.5 0 0 0 0 0 0 0 /",
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    psys.gens[0].mbase = 50.0

    add_dyr(psys, str(dyr))

    gov = psys.gov[0]
    assert (gov.K, gov.PMAX, gov.PMIN) == pytest.approx((10.0, 0.6, 0.0))
    assert (gov.UO, gov.UC) == pytest.approx((0.15, -0.15))


def test_limit_initialization_policy_is_validated(data_dir, tmp_path):
    dyr = _write_dyr(
        tmp_path,
        "1 'HYGOV' 1 0.05 0.4 5.0 0.2 0.5 0.167 1 0 1.2 1.25 0.2 0.08 /",
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))

    with pytest.raises(ValueError, match="must be 'adjust' or 'strict'"):
        add_dyr(psys, str(dyr), limit_initialization_policy="invalid")


@pytest.mark.parametrize("model", ("HYGOV", "IEEEG1"))
def test_strict_limit_initialization_policy_reaches_parsed_governor(
    data_dir, tmp_path, model
):
    record = {
        "HYGOV": "1 'HYGOV' 1 0.05 0.4 5.0 0.2 0.5 0.167 1 0 1.2 1.25 0.2 0.08 /",
        "IEEEG1": "1 'IEEEG1' 1 0 0 20 0.2 0 0.1 0.3 -0.3 1.2 0 0.4 0.5 0 1.0 0.5 0 0 0 0 0 0 0 /",
    }[model]
    dyr = _write_dyr(tmp_path, record)
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))

    add_dyr(psys, str(dyr), limit_initialization_policy="strict")

    assert psys.gov[0].adjust_initial_limits is False


@pytest.mark.parametrize("model", ("HYGOV", "IEEEG1"))
def test_strict_limit_initialization_policy_rejects_violated_source_bound(
    data_dir, tmp_path, model
):
    record = {
        "HYGOV": "1 'HYGOV' 1 0.05 0.4 5.0 0.2 0.5 0.167 0.1 0 1.2 1.25 0.2 0.08 /",
        "IEEEG1": "1 'IEEEG1' 1 0 0 20 0.2 0 0.1 0.3 -0.3 0.1 0 0.4 0.5 0 1.0 0.5 0 0 0 0 0 0 0 /",
    }[model]
    dyr = _write_dyr(tmp_path, record)
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr), limit_initialization_policy="strict")
    psys.createYbusComplex()

    expected = "GMIN/GMAX" if model == "HYGOV" else "PMIN/PMAX"
    with pytest.raises(ValueError, match=expected):
        integrate_system(
            psys,
            IntegrationConfig(steps=1, dt=1.0 / 120.0, ton=1.0, toff=2.0),
        )


@pytest.mark.parametrize("model", ("HYGOV", "IEEEG1"))
def test_adjust_limit_initialization_policy_reports_source_and_effective_bound(
    data_dir, tmp_path, model
):
    record = {
        "HYGOV": "1 'HYGOV' 1 0.05 0.4 5.0 0.2 0.5 0.167 0.1 0 1.2 1.25 0.2 0.08 /",
        "IEEEG1": "1 'IEEEG1' 1 0 0 20 0.2 0 0.1 0.3 -0.3 0.1 0 0.4 0.5 0 1.0 0.5 0 0 0 0 0 0 0 /",
    }[model]
    dyr = _write_dyr(tmp_path, record)
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))
    psys.createYbusComplex()

    result = integrate_system(
        psys,
        IntegrationConfig(steps=1, dt=1.0 / 120.0, ton=1.0, toff=2.0),
    )

    adjustment = result["dynamic_limit_diagnostics"]["parameter_adjustments"][0]
    assert adjustment["bounds_adjusted"] is True
    assert adjustment["initialization_policy"] == "adjust"
    if model == "HYGOV":
        assert adjustment["source_GMAX"] == pytest.approx(0.1)
        assert adjustment["effective_GMAX"] > adjustment["source_GMAX"]
    else:
        assert adjustment["source_PMAX"] == pytest.approx(0.1)
        assert adjustment["effective_PMAX"] > adjustment["source_PMAX"]


def test_tgov1_parser_enables_limits_and_uses_power_base(data_dir, tmp_path):
    dyr = _write_dyr(tmp_path, "1 'TGOV1' 1 0.05 0.1 1.2 0.0 0.2 10.0 0.3 /")
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    psys.gens[0].mbase = 50.0

    add_dyr(psys, str(dyr))

    gov = psys.gov[0]
    assert gov.R == pytest.approx(0.1)
    assert gov.VMAX == pytest.approx(0.6)
    assert gov.VMIN == 0.0
    assert gov.DT == pytest.approx(0.15)
    assert gov.enable_limits is True


def test_parser_bound_adjustment_is_reported_in_integration_result(data_dir, tmp_path):
    dyr = _write_dyr(
        tmp_path, "1 'HYGOV' 1 0.05 0.4 5.0 0.2 0.5 0.167 0.5 0 1.2 1.25 0.2 0.08 /"
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))
    psys.createYbusComplex()

    result = integrate_system(
        psys,
        IntegrationConfig(steps=1, dt=1.0 / 120.0, ton=1.0, toff=2.0),
    )

    adjustments = result["dynamic_limit_diagnostics"]["parameter_adjustments"]
    assert len(adjustments) == 1
    assert adjustments[0]["device_type"] == "GovHYGOV"
    assert adjustments[0]["device_id"] == "1"
    assert adjustments[0]["source_GMAX"] == pytest.approx(0.5)
    assert adjustments[0]["effective_GMAX"] > adjustments[0]["source_GMAX"]


def test_ieeeg1_coefficient_normalization_is_reported_in_integration_result(
    data_dir, tmp_path
):
    dyr = _write_dyr(
        tmp_path,
        "1 'IEEEG1' 1 0 0 20 0.2 0 0.1 0.3 -0.3 2.0 0 0.4 1.0 0 1.0 0.5 0 0 0 0 0 0 0 /",
    )
    psys = load_psse(os.path.join(data_dir, "2bus_33.raw"))
    add_dyr(psys, str(dyr))
    psys.createYbusComplex()

    result = integrate_system(
        psys,
        IntegrationConfig(steps=1, dt=1.0 / 120.0, ton=1.0, toff=2.0),
    )

    adjustment = result["dynamic_limit_diagnostics"]["parameter_adjustments"][0]
    assert adjustment["device_type"] == "GovIEEEG1"
    assert adjustment["coefficients_adjusted"] is True
    assert adjustment["source_hp_coefficient_sum"] == pytest.approx(1.5)
    assert adjustment["effective_hp_coefficient_sum"] == pytest.approx(1.0)
