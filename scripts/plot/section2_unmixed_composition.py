#!/usr/bin/env python3
"""Observed NGC 2403 predictions with prescribed unmixed fountain composition.

The existing C++ forward runner owns the dynamics and chemistry. All cases use
mu=0, j_land=j_nuc and Z_land=Z_nuc, with fixed observed disk and nuclear source.
Launch-radius cases use the saved rotation profile at the selected launch point.
The existing total-landing normalization determines the outer accretion flux.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import section3_models as models
import section3_mixing_figures as figures

STEM = "section2_observed_unmixed_composition_combined_scans"
TITLE = "Unmixed fountain, prescribed composition for NGC 2403"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-input", type=Path, default=ROOT / "output/section3/inputs/NGC2403/unmixed")
    parser.add_argument("--cases", type=Path,
                        default=ROOT / "config/galaxies/NGC2403/section2_unmixed_composition_cases.csv")
    parser.add_argument("--run-directory", type=Path, default=ROOT / "output/observed_unmixed_composition")
    parser.add_argument("--observation-directory", type=Path, default=ROOT / "output/section3")
    parser.add_argument("--figure-directory", type=Path, default=ROOT / "output/pdf")
    parser.add_argument("--runner", type=Path, default=ROOT / "build/galactic-nuclear-fountain-model")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    rows = models.read_csv(args.cases)
    base = models.parameters(args.base_input / "parameters.csv")
    if float(base["mu"]) != 0:
        raise ValueError("Select the observed unmixed forward inputs (mu=0)")
    definitions = {row["case"]: ({row["parameter"]: float(row["value"])} if row["parameter"] else {}) for row in rows}
    rotation = [row for row in models.read_csv(args.base_input / "rotation_profiles.csv")
                if row["source"] == figures.SOURCE]
    rotation_radius = figures.values(rotation, "R_kpc")
    rotation_speed = figures.values(rotation, "V_kms")
    for overrides in definitions.values():
        inner_radius = overrides.get("R_in_kpc", float(base["R_in_kpc"]))
        launch_radius = overrides.get("R_nucl_kpc", float(base["R_nucl_kpc"]))
        if not launch_radius < inner_radius < float(base["R_out_kpc"]):
            raise ValueError("Require R_nuc < R_in < R_out for the normalized unmixed cases")
        if "R_in_kpc" in overrides:
            for filename in ("rotation_profiles.csv", "gas_profiles.csv", "profiles.csv"):
                support = figures.values(models.read_csv(args.base_input / filename), "R_kpc")
                if not support.min() <= inner_radius <= support.max():
                    raise ValueError(f"R_in lies outside the saved observed support in {filename}")
        if "R_nucl_kpc" not in overrides:
            continue
        radius = launch_radius
        if not rotation_radius[0] <= radius <= rotation_radius[-1]:
            raise ValueError("Launch radii must lie within observed rotation support and below R_in")
        # Match the C++ tabulated rotation's linear interpolation, without
        # refitting the observations or changing the receiving disk boundary.
        overrides.update({
            "v_c_nucl_kms": float(np.interp(radius, rotation_radius, rotation_speed)),
            "launch_velocity_origin": "saved_observed_rotation_profile_at_selected_launch_radius",
        })
    commands = []
    if not args.plot_only:
        models.run_forward_cases(args.run_directory, args.runner, "NGC2403", args.base_input, definitions, commands)
    checks = {}
    for row in rows:
        name = row["case"]
        directory = args.run_directory / "inputs/NGC2403" / name
        settings = models.parameters(directory / "parameters.csv")
        expected = {key: float(base[key]) for key in
                    ("mu", "eta", "beta", "Mdot_nuc_Msun_yr", "yield_y", "Z_nucl", "Z_outer_boundary",
                     "R_nucl_kpc", "R_in_kpc", "R_out_kpc", "v_c_nucl_kms")}
        expected.update({key: value for key, value in definitions[name].items() if isinstance(value, (int, float))})
        for key, value in expected.items():
            np.testing.assert_allclose(float(settings[key]), value, rtol=1e-14, atol=0)
        for filename in ("rotation_profiles.csv", "gas_profiles.csv", "profiles.csv"):
            assert models.sha256(directory / filename) == models.sha256(args.base_input / filename)
        profiles, summary = figures.load_case(args.run_directory, "NGC2403", name)
        radius = figures.values(profiles, "R_kpc")
        np.testing.assert_allclose(radius[[0, -1]],
                                   [float(settings["R_in_kpc"]), float(settings["R_out_kpc"])],
                                   rtol=1e-13, atol=0)
        flux = figures.values(profiles, "Mdot_acc_Msun_yr")
        velocity = figures.values(profiles, "v_R_kms")
        landing = figures.values(profiles, "Sigmadot_land_Msun_Gyr_kpc2")
        abundance = figures.values(profiles, "Z")
        launch_j = float(settings["R_nucl_kpc"]) * float(settings["v_c_nucl_kms"])
        np.testing.assert_allclose(figures.values(profiles, "j_land_kpc_kms"), launch_j, rtol=1e-13)
        np.testing.assert_allclose(figures.values(profiles, "Z_land_required"),
                                   float(settings["Z_nucl"]), rtol=1e-10, atol=1e-14)
        assert np.all(landing >= 0) and np.all(flux > 0) and np.all(velocity < 0)
        assert np.all(np.isfinite(abundance)) and np.all(abundance > 0) and np.all(abundance < 1)
        np.testing.assert_allclose(abundance[-1], float(settings["Z_outer_boundary"]), rtol=1e-13)
        target = float(settings["eta"]) * float(settings["Mdot_nuc_Msun_yr"])
        np.testing.assert_allclose(float(summary["total_landing_Msun_yr"]), target, rtol=1e-8)
        checks[name] = {"mu": float(settings["mu"]), "eta": float(settings["eta"]),
                        "Mdot_nuc_Msun_yr": float(settings["Mdot_nuc_Msun_yr"]),
                        "R_nucl_kpc": float(settings["R_nucl_kpc"]),
                        "R_in_kpc": float(settings["R_in_kpc"]),
                        "R_out_kpc": float(settings["R_out_kpc"]),
                        "v_c_nucl_kms": float(settings["v_c_nucl_kms"]),
                        "j_land_kpc_kms": launch_j,
                        "Z_land": float(settings["Z_nucl"]),
                        "Z_out": float(settings["Z_outer_boundary"]),
                        "Mdot_land_Msun_yr": float(summary["total_landing_Msun_yr"]),
                        "Mdot_acc_out_Msun_yr": float(summary["Mdot_out_Msun_yr"]),
                        "nonnegative_landing": True, "inward_flow": True,
                        "metallicity_solved": summary["metallicity_solved"]}
    args.figure_directory.mkdir(parents=True, exist_ok=True)
    figure = figures.predictive_comparison(args.observation_directory, args.figure_directory,
                                 case_table=args.cases, model_root=args.run_directory,
                                 title=TITLE, output_stem=STEM, expected_mu=0,
                                 gradient_limits=(-0.2, 0), angular_panel_top_right=True,
                                 landing_limits=(1e5, 1e8), stellar_legend_anchor=(1, 1),
                                 stellar_legend_loc="upper right")
    sources = [args.cases, args.runner, Path(__file__), Path(figures.__file__), Path(models.__file__),
               ROOT / "scripts/plot/color_palette.py",
               ROOT / "scripts/plot/section3_mixed_parameter_scans.py",
               ROOT / "scripts/plot/observed_metallicity_spline.py",
               ROOT / "scripts/plot/six_panel_layout.py",
               *args.base_input.glob("*.csv"), args.observation_directory / "observations.csv",
               args.observation_directory / "abundance_fits.csv"]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "model": "Observed disk; unmixed incident angular momentum and metallicity",
                "closure": "mu=0; j_land=j_nuc; Z_land=Z_nuc; predict landing, flow and chemistry",
                "normalization": "Fixed nuclear source; total landing=eta*Mdot_nuc; outer accretion is derived",
                "radius_scan": "Fixed receiving disk and nuclear source; launch speed interpolated from the saved observed rotation profile",
                "inner_boundary_scan": "Vary R_in only; retain saved observed profiles, R_nuc, R_out and nuclear source without renormalizing the source",
                "base_input": str(args.base_input.resolve()), "profiles_reused": args.plot_only,
                "commands": commands, "cases": checks,
                "figure": {"path": str((args.figure_directory / f"{STEM}.pdf").resolve()),
                           "layout": "3x2", "size_inches": figure.get_size_inches().tolist(), "legend_rows": 2,
                           "inter_row_text_clearance_pt": 4, "panel_sizes_preserved": True,
                           "font_sizes_pt": {"axis_labels": 9, "ticks": 8, "legends": 8, "title": 10},
                           "radius_major_ticks_kpc": [0, 5, 10, 15], "radius_minor_step_kpc": 1,
                           "inset_count": 0,
                           "panels": ["landing", "angular_deficit", "accretion", "radial_velocity",
                                      "metallicity", "metallicity_gradient"],
                           "landing_limits_Msun_Gyr_kpc2": [1e5, 1e8],
                           "gradient_minimum_view_dex_per_kpc": [-0.2, 0],
                           "abundance_comparison": "weighted cubic OH12 spline with GCV smoothing",
                           "abundance_spline_legend": "shared only",
                           "gradient_comparison": "exact spline derivative; upper limit zero"},
                "source_sha256": {str(p.resolve()): models.sha256(p) for p in sources}}
    (args.run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
