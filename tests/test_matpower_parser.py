import numpy as np

from uqgrid.io.parse import load_matpower


def test_matpower_m_parser_handles_comments_and_inline_rows(tmp_path):
    case_path = tmp_path / "case_inline.m"
    case_path.write_text(
        "\n".join(
            [
                "function mpc = case_inline",
                "% Comment line",
                "mpc.baseMVA = 100; % inline comment",
                "mpc.bus = [ 1 3 0 0 0 0 1 1 0 230 1 1.1 0.9; 2 1 10 5 0 0 1 1 0 230 1 1.1 0.9; ];",
                "mpc.gen = [",
                "  1 50 10 30 -30 1.0 100 1 80 10; % generator row",
                "];",
                "mpc.branch = [ 1 2 0.01 0.05 0 100 100 100 0 0; ];",
            ]
        )
    )

    psys = load_matpower(str(case_path))

    assert psys.basemva == 100.0
    assert psys.nbuses == 2
    assert psys.ngens == 1
    assert psys.nbranches == 1
    assert psys.nloads == 2


def test_matpower_m_parser_handles_multiline_blocks(tmp_path):
    case_path = tmp_path / "case_multiline.m"
    case_path.write_text(
        "\n".join(
            [
                "function mpc = case_multiline",
                "mpc.baseMVA = 50;",
                "mpc.bus = [",
                "  1 3 0 0 0 0 1 1 0 115 1 1.1 0.9;",
                "  2 1 5 2 0 0 1 1 0 115 1 1.1 0.9;",
                "];",
                "mpc.gen = [",
                "  1 20 5 10 -10 1.0 50 1 25 5;",
                "];",
                "mpc.branch = [",
                "  1 2 0.02 0.04 0 0 0 0 0 0;",
                "];",
            ]
        )
    )

    psys = load_matpower(str(case_path))

    assert psys.basemva == 50.0
    assert psys.nbuses == 2
    assert psys.ngens == 1
    assert psys.nbranches == 1
    np.testing.assert_allclose(psys.buses[0].v0m, 1.0)
