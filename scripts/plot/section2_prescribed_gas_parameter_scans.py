#!/usr/bin/env python3
"""Combined Section 2.1.2 flow and chemistry scans with gas and direct landing supply.

The shared C++ prescribed-gas solver computes the flow and required landing
angular momentum, then predicts chemistry from the supplied incident abundance,
yield and inflow boundaries. This adapter selects cases and shares the existing
flow and metallicity renderers. Observations remain comparison overlays.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from color_palette import color_cycle

import section3_mixed_parameter_scans as figures

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_prescribed_gas import run_model

TITLE = r"Analytic solution, prescribed $\Sigma_g$ and $\dot\Sigma_{\rm land}$"
SCAN_FIELDS = {
    "gas_scale": "R_g_kpc",
    "gas_mass": "M_g_Msun",
    "landing_rate": "Mdot_land_Msun_yr",
    "landing_slope": "lambda",
    "outer_accretion": "Mdot_out_Msun_yr",
}


def calculate_cases(args, base, cases):
    profiles, resolved = {}, {}
    for case in cases:
        overrides = {field: case[field] for field in SCAN_FIELDS.values() if case.get(field)}
        expected = {**base, **overrides}
        destination = args.run_directory / "models" / case["case"]
        if args.plot_only:
            manifest = json.loads((destination / "run_manifest.json").read_text())
            if manifest["parameters"] != expected:
                raise ValueError(f"Saved inputs differ for {case['case']}; rerun without --plot-only")
            rows = figures.read_csv(destination / "profiles.csv")
            profile = {key: np.array([float(row[key]) for row in rows]) for key in rows[0] if key != "model"}
        else:
            profile, manifest = run_model(args.config, destination, overrides)
        profiles[case["case"]] = profile
        deficit = figures.required_j_display(profile, "deficit")[0]
        lower, upper = figures.DEFICIT_Y_LIMITS
        resolved[case["case"]] = {
            "parameters": manifest["parameters"],
            "Mdot_land_Msun_yr": manifest["Mdot_land_Msun_yr"],
            "checks": manifest["checks"],
            "deficit_outside_display_samples": int(np.count_nonzero((deficit < lower) | (deficit > upper))),
            "deficit_range": [float(np.min(deficit)), float(np.max(deficit))],
        }

    fiducial = profiles["fiducial"]
    for case in cases:
        profile = profiles[case["case"]]
        for field in ("R_kpc", "v_c_kms", "dv_c_dR_kms_kpc", "j_disk_kpc_kms"):
            np.testing.assert_array_equal(profile[field], fiducial[field])
        if case["group"] not in ("gas_scale", "gas_mass"):
            for field in ("Sigma_g_Msun_kpc2", "Sigmadot_star_Msun_Gyr_kpc2"):
                np.testing.assert_array_equal(profile[field], fiducial[field])
        if case["group"] not in ("landing_rate", "landing_slope"):
            np.testing.assert_array_equal(profile["Sigmadot_land_Msun_Gyr_kpc2"],
                                          fiducial["Sigmadot_land_Msun_Gyr_kpc2"])
        if case["group"] == "outer_accretion":
            delta = float(case["Mdot_out_Msun_yr"]) - float(base["Mdot_out_Msun_yr"])
            np.testing.assert_allclose(profile["Mdot_acc_Msun_yr"] - fiducial["Mdot_acc_Msun_yr"],
                                       delta, rtol=1e-10, atol=1e-12)
        if case["group"] == "gas_mass":
            ratio = float(case["M_g_Msun"]) / float(base["M_g_Msun"])
            np.testing.assert_allclose(profile["Sigma_g_Msun_kpc2"], ratio * fiducial["Sigma_g_Msun_kpc2"], rtol=1e-12)
            np.testing.assert_allclose(profile["Sigmadot_star_Msun_Gyr_kpc2"],
                                       ratio ** float(base["KS_N"]) * fiducial["Sigmadot_star_Msun_Gyr_kpc2"],
                                       rtol=1e-12)
        if case["group"] == "landing_rate":
            ratio = float(case["Mdot_land_Msun_yr"]) / float(base["Mdot_land_Msun_yr"])
            np.testing.assert_allclose(profile["Sigmadot_land_Msun_Gyr_kpc2"],
                                       ratio * fiducial["Sigmadot_land_Msun_Gyr_kpc2"], rtol=1e-12)
    return profiles, resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/idealized/section2_prescribed_gas_scans.csv")
    parser.add_argument("--cases", type=Path, default=ROOT / "config/idealized/section2_prescribed_gas_cases.csv")
    parser.add_argument("--run-directory", type=Path, default=ROOT / "output/section2_prescribed_gas")
    parser.add_argument("--figure-directory", type=Path, default=ROOT / "output/pdf")
    parser.add_argument("--observation-directory", type=Path, default=ROOT / "output/section3")
    parser.add_argument("--galaxy", default="NGC2403", help="Comparison overlays only")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    args.run_directory.mkdir(parents=True, exist_ok=True)
    if not args.skip_build and not args.plot_only:
        figures.run(["make", "-j4", "build/galactic-nuclear-fountain-model"], args.run_directory / "logs/build.log")
    base, cases = figures.parameters(args.config), figures.read_csv(args.cases)
    profiles, resolved = calculate_cases(args, base, cases)
    disk_fit, abundance, fit, observation_paths = figures.observations(args, base)
    files = [args.config, args.cases, Path(__file__), Path(figures.__file__),
             ROOT / "scripts/run_prescribed_gas.py", ROOT / "scripts/plot/observed_disk_comparison.py",
             ROOT / "scripts/plot/axis_labels.py", ROOT / "scripts/plot/surface_rate_units.py",
             *observation_paths]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Section 2.1.2: prescribed exponential gas, KS sink, power-law landing and flat rotation",
        "supply": "Direct Mdot_land_Msun_yr; no reservoir supply parameters",
        "angular_momentum": "Required j_land is inferred; j_land=j_nuc is not imposed",
        "observations": "Saved disk fits, metallicity measurements and abundance gradient are comparisons only",
        "chemistry": {key: float(base[key]) for key in ("Z_land", "yield_y", "Z_inner_boundary", "Z_outer_boundary")},
        "scan_fields": SCAN_FIELDS,
        "profiles_reused": args.plot_only,
        "cases": resolved,
        "source_sha256": {str(path.resolve()): figures.sha256(path) for path in files},
        "figure": {"title": TITLE, "j_display": "1-j_land/j_disk", "j_limits": list(figures.DEFICIT_Y_LIMITS),
                   "layout": "4x2", "size_inches": [7.2, 7.2], "row_height_ratios": [1, 1, 1, 2],
                   "output": "section2_prescribed_gas_combined_scans.pdf"},
    }
    (args.run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.run_directory / "figure_notes.txt").write_text(
        "SECTION 2.1.2 PRESCRIBED-GAS FLOW AND METALLICITY SCANS\n\n"
        "One combined 4x2 flow and metallicity/gradient PDF; one input changes in each case.\n"
        "The bottom metallicity/gradient panels have twice the height of each flow panel.\n"
        "R_g=2,3,4 kpc at fixed annular M_g=9e8 Msun.\n"
        "M_g=4.5e8,9e8,1.8e9 Msun at fixed R_g=3 kpc.\n"
        "Mdot_land=0.125,0.25,0.5 Msun/yr; lambda=1.5,2,2.5.\n"
        "Mdot_acc,out=0.025,0.05,0.1 Msun/yr.\n"
        "The receiving annulus is 1.1 to 15 kpc; flat v_c=200 km/s.\n"
        "The C++ solver owns gas normalization, KS sink, landing, flow and required j.\n"
        "No incident angular-momentum mixture or nuclear launch j is imposed.\n"
        "This is a required-j calculation, not an assertion of compatibility with an unmixed fountain.\n"
        "Observed NGC 2403 gas and net stellar-sink splines are dotted black display overlays.\n"
        "Read across rows: landing and stellar sink; gas and radial velocity; accretion and angular deficit; metallicity and gradient.\n"
        "The angular-deficit axis spans -1 to 2; full values remain in the per-case CSVs.\n"
        "Negative required j and outward flow are retained and counted in each run manifest.\n"
        "Chemistry uses Z_land=Z_sun=0.0139, yield_y=0.01, Z_out=0.3 Z_sun and Z_in=Z_sun.\n"
        "Chemical assumptions are held fixed across the same eleven flow cases.\n"
        "The shared C++ chemistry solver uses conservative metal balance when flow changes sign.\n"
        "The chemical gradient is dlog10(Z)/dR in dex/kpc; its displayed range is -0.2 to 0.\n"
        "NGC 2403 KK04 measurement points and their errors are retained; the dashed line is its fitted gradient.\n"
        "Surface rates use Msun/Gyr/kpc^2; integrated mass rates use Msun/yr.\n"
    )
    figures.smplotlib.set_style(usetex=False, fontsize=9, dpi=144)
    plt.rcParams.update({"axes.prop_cycle": color_cycle(), "axes.grid": False, "legend.frameon": False, "pdf.fonttype": 42,
                         "savefig.bbox": None})
    metallicity_max=max(3, 1.05*max(
        max(float(np.max(profile["Z"]))/figures.SOLAR_Z for profile in profiles.values()),
        max((float(row["Z"])+float(row["e_Z_high"]))/figures.SOLAR_Z for row in abundance)))
    figures.plot_flow_metallicity(args, cases, profiles, disk_fit, abundance, fit,
                                 stem="section2_prescribed_gas_combined_scans", gradient_min=-0.2,
                                 title=TITLE, legend_parameter_columns=True,
                                 metallicity_max=metallicity_max)
    for stem in ("section2_prescribed_gas_flow_scans", "section2_prescribed_gas_metallicity_scans"):
        (args.figure_directory / f"{stem}.pdf").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
