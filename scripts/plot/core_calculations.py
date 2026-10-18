"""CSV calls to the compiled research calculations; no equations live here."""

import csv
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "build/galactic-nuclear-fountain-calculations"
OBSERVATION_COLUMNS = ("source_seq", "offRA_arcsec", "offDE_arcsec", "R_kpc", "OH12", "e_OH12")


def run_calculation(command, columns, rows, parameters=None, curve=None):
    if not RUNNER.is_file():
        raise RuntimeError("Build the C++ calculations first with make -j4")
    with TemporaryDirectory(prefix="gnf-calculation-") as temporary:
        source, output = Path(temporary) / "input", Path(temporary) / "output"
        source.mkdir()
        for filename, header, values in (
            ("points.csv", columns, rows),
            ("parameters.csv", ("name", "value"), (parameters or {}).items()),
        ):
            with (source / filename).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(values)
        if curve is not None:
            with (source / "curve.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("R_kpc", "Z"))
                writer.writerows(curve)
        result = subprocess.run([str(RUNNER), command, str(source), str(output)], capture_output=True, text=True)
        if result.returncode:
            raise ValueError(result.stderr.strip())
        tables = {}
        for path in output.glob("*.csv"):
            with path.open(newline="") as stream:
                tables[path.stem] = list(csv.DictReader(stream))
        return tables


def collapse_abundances(rows, scatter=0.05):
    if not rows:
        return []
    values = ((row[key] for key in OBSERVATION_COLUMNS) for row in rows)
    tables = run_calculation("collapse-abundances", OBSERVATION_COLUMNS, values, {"abundance_scatter_dex": scatter})
    result = []
    for row in tables["values"]:
        result.append({key: (value if key == "source_sequences" else
                            int(value) if key in ("source_seq", "n_catalog_rows") else
                            float(value) if value else None) for key, value in row.items()})
    order = {int(row["source_seq"]): index for index, row in enumerate(rows)}
    return sorted(result, key=lambda row: order[row["source_seq"]])


def fit_abundances(rows, sample_count=0, scatter=0.05):
    tables = run_calculation("fit-abundance", ("R_kpc", "OH12", "e_OH12"), rows,
                             {"sample_count": sample_count, "abundance_scatter_dex": scatter})
    fit = {key: float(value) for key, value in tables["fit"][0].items()}
    fit["fit_dof"] = int(fit["fit_dof"])
    profiles = [{key: float(value) for key, value in row.items()} for row in tables["profiles"]]
    return fit, profiles


def compare_abundances(rows, curve):
    tables = run_calculation("compare-abundance", OBSERVATION_COLUMNS,
                             ((row[key] for key in OBSERVATION_COLUMNS) for row in rows), curve=curve)
    return {key: (value if key == "fit_status" else int(value) if key.startswith("n_") else
                  float(value) if value else None) for key, value in tables["fit"][0].items()}
