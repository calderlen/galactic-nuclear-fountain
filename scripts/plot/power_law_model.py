"""Read the analytic power-law solution from the existing C++ model runner.

This adapter only handles CSV transport; the equations live in model/physics.cpp.
"""

import csv
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "build" / "galactic-nuclear-fountain-model"


def power_law_profiles(radius_kpc, parameters):
    """Evaluate at the supplied radii, preserving their order and array shape."""
    if not RUNNER.exists():
        raise RuntimeError("Build the core model first with make -j4")
    radius = np.asarray(radius_kpc, dtype=float)
    with TemporaryDirectory(prefix="gnf-power-law-") as temporary:
        directory = Path(temporary)
        source, destination = directory / "input", directory / "output"
        source.mkdir()
        with (source / "parameters.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("name", "value"))
            writer.writerows(parameters.items())
        with (source / "radii.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("R_kpc",))
            writer.writerows((value,) for value in radius.ravel())
        result = subprocess.run(
            [str(RUNNER), "power-law", str(source), str(destination)],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise ValueError(result.stderr.strip())
        rows = np.atleast_1d(np.genfromtxt(destination / "profiles.csv", delimiter=",", names=True))
    return {name: rows[name].reshape(radius.shape) for name in rows.dtype.names}
