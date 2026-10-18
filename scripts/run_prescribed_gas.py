#!/usr/bin/env python3
"""Run the separate prescribed-gas / required-angular-momentum model.

Choose --config and --output explicitly. Input-file paths in a config are
relative to the repository root unless absolute. Snapshot the selected inputs;
the C++ prescribed-gas module owns all model calculations.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/plot"))
from surface_rate_units import YEARS_PER_GYR
RUNNER = ROOT / "build/galactic-nuclear-fountain-model"


def read_parameters(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    settings = {row["name"]: row["value"] for row in rows}
    if len(settings) != len(rows):
        raise ValueError("Duplicate configuration parameter")
    return settings


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_model(config, output, overrides=None, skip_build=True):
    """Snapshot, run and validate one explicit supplied-gas case."""
    config, output = Path(config).resolve(), Path(output).resolve()
    settings = {**read_parameters(config), **(overrides or {})}
    if settings.get("closure") != "required_j":
        raise ValueError("Select a prescribed-gas required_j configuration")
    if not skip_build:
        subprocess.run(["make", "-j4", "build/galactic-nuclear-fountain-model"], cwd=ROOT, check=True)
    source = output / "input"
    source.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for key, filename in (("gas_file", "gas_profiles.csv"), ("rotation_file", "rotation_profiles.csv"),
                          ("sfr_file", "sfr_profiles.csv"), ("sfr_parameters_file", "sfr_parameters.csv")):
        if key not in settings:
            continue
        original = Path(settings[key])
        original = original if original.is_absolute() else ROOT / original
        original = original.resolve(strict=True)
        shutil.copyfile(original, source / filename)
        settings[key] = filename
        provenance[key] = {"original_path": str(original), "sha256": digest(original), "snapshot": str(source / filename)}
    with (source / "parameters.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("name", "value"))
        writer.writerows(settings.items())
    result = subprocess.run([str(RUNNER), "prescribed-gas", str(source), str(output)],
                            cwd=ROOT, text=True, capture_output=True)
    (output / "run.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Model failed; see {output / 'run.log'}: {result.stderr.strip()}")
    with (output / "profiles.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with (output/"summary.csv").open(newline="") as stream:
        summary = {row["quantity"]: row["value"] for row in csv.DictReader(stream)}
    p = {key: np.array([float(row[key]) for row in rows]) for key in rows[0] if key != "model"}
    r, gas = p["R_kpc"], p["Sigma_g_Msun_kpc2"]
    land, star, acc = p["Sigmadot_land_Msun_Gyr_kpc2"], p["Sigmadot_star_Msun_Gyr_kpc2"], p["Mdot_acc_Msun_yr"]
    assert all(np.all(np.isfinite(value)) for value in p.values())
    np.testing.assert_allclose(acc[-1], float(settings["Mdot_out_Msun_yr"]), rtol=1e-12)
    np.testing.assert_allclose(p["cumulative_landing_Msun_yr"][-1], float(summary["Mdot_land_Msun_yr"]), rtol=1e-12)
    np.testing.assert_allclose(p["v_R_kpc_yr"], -acc/(2*np.pi*r*gas), rtol=1e-12)
    np.testing.assert_allclose(gas*p["v_R_kpc_yr"]*YEARS_PER_GYR*(p["v_c_kms"]+r*p["dv_c_dR_kms_kpc"]),
        land*(p["j_land_required_kpc_kms"]-p["j_disk_kpc_kms"]), rtol=1e-11, atol=1e-13)
    mass_rhs = 2*np.pi*r*(star-land)/YEARS_PER_GYR
    mass_error = float(np.max(abs(np.gradient(acc,r,edge_order=2)-mass_rhs)) / max(np.max(abs(mass_rhs)), 1e-30))
    if mass_error > 0.005:
        raise ValueError(f"Output-grid mass conservation check failed: {mass_error:g}")
    checks = {"mass_equation_output_grid_relative_error": mass_error,
              "negative_required_j_samples": int(np.count_nonzero(p["j_land_required_kpc_kms"]<0)),
              "outward_flow_samples": int(np.count_nonzero(p["v_R_kpc_yr"]>0)), "samples": len(r)}
    if "Z" in p:
        assert np.all(p["Z"]>0) and np.all(p["Z"]<=1)
        checks["metallicity_solver"] = summary["metallicity_solver"]
        if summary["metallicity_solver"] == "RK4":
            if np.all(acc>0):
                np.testing.assert_allclose(p["Z"][-1], float(settings["Z_outer_boundary"]), rtol=1e-11)
            elif np.all(acc<0):
                np.testing.assert_allclose(p["Z"][0], float(settings["Z_inner_boundary"]), rtol=1e-11)
            metal_rhs=2*np.pi*r*(land*p["Z_land"]+(float(settings["yield_y"])-p["Z"])*star)/YEARS_PER_GYR
            metal_error=float(np.max(abs(np.gradient(-acc*p["Z"],r,edge_order=2)-metal_rhs)) /
                              max(np.max(abs(metal_rhs)),1e-30))
            if metal_error>0.005:
                raise ValueError(f"Output-grid metal conservation check failed: {metal_error:g}")
            checks["metal_equation_output_grid_relative_error"]=metal_error
        else:
            cell_error=float(summary["metallicity_max_cell_balance_error"])
            if cell_error>1e-12:
                raise ValueError(f"Conservative metal balance failed: {cell_error:g}")
            checks["metallicity_max_cell_balance_error"]=cell_error
    files = [config, RUNNER, Path(__file__), *(ROOT / "model" / name for name in (
        "units.h",
        "prescribed_gas.cpp", "prescribed_gas.h", "prescribed_gas_io.cpp", "prescribed_gas_io.h",
        "physics.cpp", "physics.h", "models.cpp", "models.h", "profiles.cpp", "profiles.h", "rk4.cpp", "rk4.h"))]
    manifest = {"command": "prescribed-gas", "config": str(config), "parameters": settings,
                "inputs": provenance, "source_sha256": {str(path): digest(path) for path in files},
                "stellar_sink": summary["stellar_sink"], "return_fraction": float(summary["return_fraction"]),
                "tabulated_interpolation": "linear between supplied samples; no extrapolation",
                "supply_model": summary["supply_model"], "Mdot_land_Msun_yr": float(summary["Mdot_land_Msun_yr"]),
                "chemistry": "Conditional on prescribed incident composition and abundance boundaries" if "Z" in p else "disabled",
                "checks": checks}
    if summary["stellar_sink"] == "net_observed_sfr_spline":
        manifest["sfr_input_return_fraction"] = float(summary["sfr_input_return_fraction"])
        manifest["sfr_net_rescale_factor"] = float(summary["sfr_net_rescale_factor"])
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {output / 'profiles.csv'}")
    print(json.dumps(manifest["checks"]))
    return p, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    run_model(args.config, args.output, skip_build=args.skip_build)


if __name__ == "__main__":
    main()
