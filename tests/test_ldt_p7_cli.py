import sys
import subprocess


def test_p7_cli_smoke():
    completed = subprocess.run(
        [sys.executable, "-m", "bin.exp_ldt_v2"],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout

    assert "Case: Euclidean" in stdout
    assert "Case: Σ=I" in stdout
    assert "Case: Anisotropy c=1.0" in stdout
    assert "Case: Anisotropy c=0.5" in stdout
    assert "Case: Anisotropy c=0.2" in stdout

    # Check key metrics printed
    assert "||delta||2" in stdout
    assert "sigma_min(J)" in stdout
    assert "beta" in stdout
    assert "prefactor C" in stdout
    assert "S_perp eigenvalues" in stdout
    assert "MC:" in stdout

    assert completed.stderr == ""
