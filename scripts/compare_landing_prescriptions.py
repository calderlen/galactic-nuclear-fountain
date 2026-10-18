#!/usr/bin/env python3
"""Compare power-law and smooth deposition with the existing C++ steady core.

Run from the repository root:
    conda run -n malca python scripts/compare_landing_prescriptions.py

The illustrative Section 2 disk comes from config/idealized/analytic.csv. The Section 3
NGC 2403 disk uses the saved, already preprocessed fiducial inputs. Both pairs
hold the disk, total landing rate, receiving annulus, and outer mass flux fixed.
All model calculations, including normalization and chemistry, remain in C++.
This script copies inputs, invokes the runner, and plots its CSV outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "plot"))
from axis_labels import quantity_unit_label
from surface_rate_units import read_rate_rows, YEARS_PER_GYR
RUNNER = ROOT / "build/galactic-nuclear-fountain-model"
SOLAR_Z = 0.0139
PRESCRIPTIONS = ("power_law", "smooth_broken_power_law")


def csv_rows(path: Path) -> list[dict[str, str]]:
    return read_rate_rows(path)


def parameters(path: Path) -> dict[str, str]:
    return {row["name"]: row["value"] for row in csv_rows(path)}


def write_parameters(path: Path, settings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("name", "value"))
        writer.writerows((name, format(value, ".17g") if isinstance(value, float)
                         else value) for name, value in settings.items())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], log: Path, commands: list[list[str]]) -> None:
    commands.append(command)
    log.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    log.write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log}: "
                           + (completed.stderr or completed.stdout)[-1000:])


def selected_rows(path: Path, source: str | None = None) -> list[dict[str, str]]:
    rows = csv_rows(path)
    return rows if source is None else [row for row in rows if row["source"] == source]


def profile(path: Path, source: str) -> dict[str, np.ndarray]:
    rows = selected_rows(path, source)
    if not rows:
        raise ValueError(f"No {source} profiles in {path}")
    return {name: np.array([float(row[name]) for row in rows]) for name in rows[0] if name not in ("model", "source")}


def receiving_domain(settings: dict[str, str], support: tuple[float, float],
                     input_directory: Path | None = None) -> tuple[float, float]:
    lower = max(float(settings["R_in_kpc"]), support[0])
    upper = min(float(settings["R_out_kpc"]), support[1])
    if input_directory is not None:
        for filename in ("gas_profiles.csv", "profiles.csv", "rotation_profiles.csv"):
            rows = csv_rows(input_directory / filename)
            if filename == "rotation_profiles.csv":
                rows = [row for row in rows if row["source"] == "SPARC_spline"]
            radii = [float(row["R_kpc"]) for row in rows]
            lower, upper = max(lower, min(radii)), min(upper, max(radii))
    if lower >= upper:
        raise ValueError("Disk support and fitted support do not share a receiving annulus")
    return lower, upper


def outer_flux(saved_result: Path, radius: float, source: str) -> tuple[float, dict]:
    rows = selected_rows(saved_result / "summary.csv", source)
    if len(rows) != 1 or rows[0]["model"] != "forward":
        raise ValueError(f"Expected one saved forward result for {source}: {saved_result}")
    saved_outer = float(rows[0]["R_out_kpc"])
    if radius == saved_outer:
        return float(rows[0]["Mdot_out_Msun_yr"]), {
            "method": "saved_forward_summary_at_unchanged_outer_radius",
            "path": str(saved_result / "summary.csv"), "source": source,
            "sha256": sha256(saved_result / "summary.csv"), "R_kpc": radius,
        }
    # This adapts saved CSV data; no continuity or angular-momentum calculation
    # is reimplemented. Record the interpolation rather than claiming a new run.
    saved = profile(saved_result / "profiles.csv", source)
    radii = saved["R_kpc"]
    if not radii[0] <= radius <= radii[-1]:
        raise ValueError("New outer boundary is outside saved forward-profile support")
    index = int(np.searchsorted(radii, radius))
    flux = float(np.interp(radius, radii, saved["Mdot_acc_Msun_yr"]))
    return flux, {
        "method": "linear_interpolation_of_saved_forward_accretion_at_new_outer_radius",
        "path": str(saved_result / "profiles.csv"), "source": source,
        "sha256": sha256(saved_result / "profiles.csv"), "R_kpc": radius,
        "bracket_R_kpc": [float(radii[max(index - 1, 0)]), float(radii[min(index, len(radii) - 1)])],
    }


def prepare_pair(output: Path, name: str, settings: dict[str, str], smooth: dict[str, str],
                 provenance: dict, base: Path | None = None) -> None:
    for prescription in PRESCRIPTIONS:
        directory = output / "inputs" / name / prescription
        directory.mkdir(parents=True, exist_ok=True)
        if base is not None:
            for source in base.glob("*.csv"):
                shutil.copyfile(source, directory / source.name)
        else:
            shutil.copyfile(ROOT / "input/analytic/fiducial/rotation_curves.csv", directory / "rotation_curves.csv")
        updates = dict(settings)
        updates.update(smooth if prescription == PRESCRIPTIONS[1] else {"landing_profile": "power_law"})
        updates["landing_comparison"] = name
        updates["landing_normalization_origin"] = "same_total_rate_for_both_shapes_on_comparison_annulus"
        write_parameters(directory / "parameters.csv", updates)
        (directory / "comparison_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def diagnostics(rows: dict[str, np.ndarray], summary: dict[str, str]) -> dict:
    landing = rows["Sigmadot_land_Msun_Gyr_kpc2"]
    radius = rows["R_kpc"]
    total = float(summary["Mdot_land_Msun_yr"])
    integrated = float(np.trapezoid(2 * math.pi * radius * landing, radius)) / YEARS_PER_GYR
    values = {"n_samples": len(radius), "R_min_kpc": float(radius[0]),
              "R_max_kpc": float(radius[-1]), "Mdot_land_Msun_yr": total,
              "cumulative_outer_Msun_yr": float(rows["cumulative_landing_Msun_yr"][-1]),
              "sampled_trapezoid_Mdot_land_Msun_yr": integrated,
              "sampled_trapezoid_relative_error": (integrated - total) / total,
              "Mdot_out_Msun_yr": float(summary["Mdot_out_Msun_yr"]),
              "outer_profile_accretion_Msun_yr": float(rows["Mdot_acc_Msun_yr"][-1]),
              "metallicity_solved": summary["metallicity_solved"] == "1"}
    for field in ("v_R_kms", "j_land_kpc_kms", "j_land_over_j_disk", "mu_j", "Z", "Z_land_required", "mu_Z"):
        finite = rows[field][np.isfinite(rows[field])]
        values[field] = {"finite_samples": len(finite),
                         "negative_samples": int(np.count_nonzero(finite < 0)),
                         "min": float(np.min(finite)) if len(finite) else None,
                         "max": float(np.max(finite)) if len(finite) else None}
    return values


def plot_pair(output: Path, name: str, profiles: dict[str, dict[str, np.ndarray]],
              observed_chemistry: bool = False, Z_CGM: float = 0.003) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    fields = ("Sigmadot_land_Msun_Gyr_kpc2", "cumulative_landing_Msun_yr", "v_R_kms",
              "j_land_over_j_disk", "mu_j", "mu_Z" if observed_chemistry else "Z")
    labels = (r"$\dot\Sigma_{\rm land}\ [M_\odot\,\mathrm{Gyr}^{-1}\,\mathrm{kpc}^{-2}]$",
              r"$\dot M_{\rm land}(<R)\ [M_\odot\,\mathrm{yr}^{-1}]$",
              r"$v_R\ [\mathrm{km\,s}^{-1}]$", r"Required $j_{\rm land}/j_{\rm disk}$",
              r"Required $\mu_j$", r"Required $\mu_Z$" if observed_chemistry
              else r"Predicted $Z/Z_\odot$")
    for prescription, color, label in ((PRESCRIPTIONS[0], "black", "Power law"),
                                      (PRESCRIPTIONS[1], "#ce552c", "Smooth broken power law")):
        rows = profiles[prescription]
        for axis, field, ylabel in zip(axes.flat, fields, labels):
            values = rows[field] / SOLAR_Z if field == "Z" else rows[field]
            plot_radii = rows["R_kpc"]
            if field in ("mu_j", "mu_Z"):
                denominator = (rows["j_CGM_kpc_kms"] - rows["j_land_kpc_kms"]
                               if field == "mu_j" else Z_CGM - rows["Z_land_required"])
                finite_pairs = np.isfinite(denominator[:-1]) & np.isfinite(denominator[1:])
                poles = np.flatnonzero(finite_pairs & (np.signbit(denominator[:-1]) != np.signbit(denominator[1:]))) + 1
                values = np.where(denominator == 0, np.nan, values)
                # Insert breaks, retaining all sampled values on both sides of
                # a pole. Raw output and numerical diagnostics remain intact.
                plot_radii = np.insert(plot_radii, poles, np.nan)
                values = np.insert(values, poles, np.nan)
            axis.plot(plot_radii, values, color=color, lw=1.7, label=label)
            axis.set_xlabel(r"$R\ [\mathrm{kpc}]$")
            axis.set_ylabel(quantity_unit_label(ylabel))
            axis.tick_params(direction="in", top=True, right=True)
    axes[0, 0].set_yscale("log")
    axes[1, 0].set_yscale("symlog", linthresh=1)
    axes[1, 1].set_yscale("symlog", linthresh=1)
    if observed_chemistry:
        axes[1, 2].set_yscale("symlog", linthresh=1)
        axes[1, 2].axhline(0, color="0.65", lw=0.7, zorder=0)
    for axis in (axes[0, 2], axes[1, 0], axes[1, 1]):
        axis.axhline(0, color="0.65", lw=0.7, zorder=0)
    axes[1, 0].axhline(1, color="0.65", lw=0.7, ls="--", zorder=0)
    axes[0, 0].legend(frameon=False, fontsize=9)
    if not any(np.any(np.isfinite(rows["Z"])) for rows in profiles.values()):
        axes[1, 2].text(0.5, 0.5, "Chemistry unavailable:\nmissing inflow abundance boundary",
                        ha="center", va="center", transform=axes[1, 2].transAxes,
                        fontsize=10)
    figure.suptitle("Section 2: illustrative exponential disk" if name == "section2"
                   else "Section 3: NGC 2403 observed chemistry constraint" if observed_chemistry
                   else "Section 3: NGC 2403 fiducial disk")
    destination = output / "plots"
    destination.mkdir(parents=True, exist_ok=True)

    figure.savefig(destination / f"{name}.pdf", format="pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "output/smooth_landing_comparison")
    parser.add_argument("--analytic-config", type=Path, default=ROOT / "config/idealized/analytic.csv")
    parser.add_argument("--analytic-outer-flux", type=float, default=0.2,
                        help="Declared illustrative inward outer flux in Msun/yr (default: 0.2)")
    parser.add_argument("--landing-config", type=Path, default=ROOT / "config/galaxies/NGC2403/smooth_landing.csv")
    parser.add_argument("--observed-input", type=Path, default=ROOT / "output/section3/inputs/NGC2403/fiducial")
    parser.add_argument("--observed-forward", type=Path, default=ROOT / "output/section3/NGC2403/fiducial")
    parser.add_argument("--observed-chemistry-mode", choices=("predicted", "observed"),
                        default="predicted",
                        help="Predict NGC 2403 chemistry, or infer composition from its saved abundance fit")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    commands: list[list[str]] = []
    output.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        run(["make", "-j4", str(RUNNER.relative_to(ROOT))], output / "logs/build.log", commands)
    if not RUNNER.exists():
        raise RuntimeError("Build the model first with make -j4")
    smooth = parameters(args.landing_config)
    if smooth.get("landing_profile") != "smooth_broken_power_law":
        raise ValueError("Landing config must select smooth_broken_power_law")
    support = (float(smooth["landing_fit_R_min_kpc"]), float(smooth["landing_fit_R_max_kpc"]))

    analytic = parameters(args.analytic_config)
    analytic["profile_type"] = "analytic"
    analytic["metallicity_mode"] = "predicted"
    analytic["R_in_kpc"], analytic["R_out_kpc"] = receiving_domain(analytic, support)
    analytic["landing_profile"] = "power_law"
    if not math.isfinite(args.analytic_outer_flux):
        raise ValueError("The declared analytic outer flux must be finite")
    analytic["Mdot_out_Msun_yr"] = args.analytic_outer_flux
    analytic["Mdot_out_origin"] = "declared_illustrative_comparison_boundary"
    inner_abundance_assumed = not analytic.get("Z_inner_boundary")
    if inner_abundance_assumed:
        analytic["Z_inner_boundary"] = analytic["Z_nucl"]
        analytic["Z_inner_boundary_origin"] = "assumed_nuclear_abundance_for_gas_entering_from_inner_disk"
    source = csv_rows(ROOT / "input/analytic/fiducial/rotation_curves.csv")[0]["source"]
    cases = {"section2": {"settings": analytic, "source": source}}
    provenance = {"analytic_config": str(args.analytic_config.resolve()),
                  "analytic_config_sha256": sha256(args.analytic_config),
                  "boundary": {"method": "declared_illustrative_comparison_boundary",
                               "Mdot_out_Msun_yr": args.analytic_outer_flux,
                               "default_value_origin": "legacy_illustrative_case_value_not_a_current_forward_inference"},
                  "inner_abundance_boundary": {"Z": float(analytic["Z_inner_boundary"]),
                                               "assumed_equal_to_nuclear_abundance": inner_abundance_assumed},
                  "fitted_support_kpc": support,
                  "closure": "prescribed_deposition_infer_flow_and_landing_angular_momentum",
                  "shape_scope": "returned_nuclear_shape_applied_to_total_nuclear_plus_CGM_landing"}
    prepare_pair(output, "section2", analytic, smooth, provenance)

    observed = parameters(args.observed_input / "parameters.csv")
    observed["metallicity_mode"] = args.observed_chemistry_mode
    observed["R_in_kpc"], observed["R_out_kpc"] = receiving_domain(observed, support, args.observed_input)
    observed["Mdot_out_Msun_yr"], boundary = outer_flux(args.observed_forward,
                                                       float(observed["R_out_kpc"]), "SPARC_spline")
    cases["section3"] = {"settings": observed, "source": "SPARC_spline"}
    provenance = {"saved_input_directory": str(args.observed_input.resolve()),
                  "metallicity_mode": args.observed_chemistry_mode,
                  "saved_input_parameters_sha256": sha256(args.observed_input / "parameters.csv"),
                  "copied_csv_sha256": {path.name: sha256(path) for path in sorted(args.observed_input.glob("*.csv"))},
                  "boundary": boundary, "fitted_support_kpc": support,
                  "closure": "prescribed_deposition_infer_flow_and_landing_angular_momentum",
                  "shape_scope": "returned_nuclear_shape_applied_to_total_nuclear_plus_CGM_landing"}
    prepare_pair(output, "section3", observed, smooth, provenance, args.observed_input)

    results = {}
    for name, case in cases.items():
        profiles = {}
        results[name] = {}
        for prescription in PRESCRIPTIONS:
            source_dir = output / "inputs" / name / prescription
            destination = output / "models" / name / prescription
            print(f"Running {name}: {prescription}", flush=True)
            run([str(RUNNER), "inverse", str(source_dir), str(destination)],
                output / "logs" / f"{name}_{prescription}.log", commands)
            profiles[prescription] = profile(destination / "profiles.csv", case["source"])
            summary = selected_rows(destination / "summary.csv", case["source"])[0]
            results[name][prescription] = diagnostics(profiles[prescription], summary)
            if not (destination / "landing_profile.csv").exists():
                raise RuntimeError("Runner did not emit landing_profile.csv; rebuild the current C++ runner")
            prescribed = csv_rows(destination / "landing_prescription.csv")
            fixed_ratio = np.array([float(row["fixed_j_flat_stellar_sink_ratio"]) for row in prescribed])
            finite = fixed_ratio[np.isfinite(fixed_ratio)]
            results[name][prescription]["fixed_j_unmixed_flat_diagnostic"] = {
                "interpretation": "unmixed_flat_rotation_j_land_equals_j_nucl_only_not_the_inverse_supplied_SFR",
                "min_stellar_sink_over_landing": float(np.min(finite)) if len(finite) else None,
                "negative_samples": int(np.count_nonzero(finite < 0)),
                "negative_R_min_kpc": min((float(row["R_kpc"]) for row, ratio
                                           in zip(prescribed, fixed_ratio) if ratio < 0), default=None),
            }
        for prescription in PRESCRIPTIONS:
            row = results[name][prescription]
            chemistry_mask = np.ones(row["n_samples"], dtype=bool)
            observed_chemistry = name == "section3" and args.observed_chemistry_mode == "observed"
            if observed_chemistry:
                observed_radii = [float(point["R_kpc"]) for point
                                  in csv_rows(args.observed_input / "metallicity_profiles.csv")]
                radii = profiles[prescription]["R_kpc"]
                chemistry_mask = (radii >= min(observed_radii)) & (radii <= max(observed_radii))
                row["observed_chemistry_support_kpc"] = [min(observed_radii), max(observed_radii)]
            if not row["metallicity_solved"] or not np.any(chemistry_mask) or not np.all(
                    np.isfinite(profiles[prescription]["Z"][chemistry_mask])):
                raise RuntimeError(f"{name} {prescription} chemistry is not finite on its supported domain")
        plot_pair(output, name, profiles,
                  observed_chemistry=name == "section3" and args.observed_chemistry_mode == "observed",
                  Z_CGM=float(case["settings"]["Z_CGM"]))

    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "runner_sha256": sha256(RUNNER), "landing_config": str(args.landing_config.resolve()),
              "landing_config_sha256": sha256(args.landing_config), "commands": commands,
              "observed_chemistry_mode": args.observed_chemistry_mode,
              "diagnostics": results}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "README.txt").write_text(
        f"Run: conda run -n malca python scripts/compare_landing_prescriptions.py --output {output} "
        f"--observed-chemistry-mode {args.observed_chemistry_mode}\n\n"
        "Each inverse pair uses identical gas, stellar sink, radial annulus, total landing rate,\n"
        "source composition, and explicit outer mass flux. The C++ runner owns all physics.\n"
        "Section 2: current illustrative analytic gas/SFR config; declared outer inward flux\n"
        "defaults to 0.2 Msun/yr (the old illustrative-case boundary, not a new forward result).\n"
        "Choose another with --analytic-outer-flux. An absent inner abundance boundary is\n"
        "assumed equal to Z_nucl in BOTH inputs, to specify gas entering from the inner disk.\n"
        "Section 3: saved NGC 2403 fiducial disk; outer flux from its saved forward result.\n"
        f"Section 3 chemistry mode: {args.observed_chemistry_mode}. In observed mode the saved\n"
        "abundance fit constrains required landing composition only within its radial support;\n"
        "the last plot panel displays required mu_Z instead of predicted Z.\n"
        "Input provenance records any boundary interpolation and copied-file hashes.\n\n"
        "The fitted shape came from returned nuclear material on 1-15 kpc. Applying that\n"
        "shape to total nuclear-plus-CGM deposition is a controlled alternative prescription.\n"
        "It does not use the orbit catalogue or delay kernel. This is the prescribed-deposition\n"
        "closure: radial flow and required landing angular momentum are inferred.\n"
        "The specialized fixed-angular-momentum analytic power-law solution is not this test.\n"
        "Negative required angular momentum or mixing ratios are retained as viability diagnostics.\n"
        "landing_prescription.csv also records the implied stellar sink for the separate\n"
        "unmixed flat-rotation closure j_land=j_nucl. Its negative values do not replace\n"
        "the supplied stellar sink in these inverse comparisons. The NGC 2403 rotation is\n"
        "observed and nonflat, so this extra column is a flat-rotation diagnostic only.\n"
        "See comparison.json for sample extrema, conservation readouts, and exact commands.\n"
        "The sampled trapezoid integral is only a coarse output-grid check; the runner's\n"
        "normalized cumulative integral defines the exact configured total landing rate.\n"
    )
    print(f"Wrote paired inputs, runs, diagnostics, and PNG plots to {output}")


if __name__ == "__main__":
    main()
