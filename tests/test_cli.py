import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module",
    [
        "scripts.train_supervised",
        "scripts.train_simclr",
        "scripts.train_linear_probe",
    ],
)
def test_training_cli_help(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
