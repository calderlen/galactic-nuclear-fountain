#!/usr/bin/env python3
"""Combine the unmixed analytic flow and chemistry variations in one PDF.

One explicit case table controls every panel. The existing C++ power-law
runner computes all quantities. Only analytic cases appear in this figure.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator
import numpy as np

import analytic_metallicity_profiles as chemistry
from axis_labels import quantity_unit_label
from power_law_model import power_law_profiles, RUNNER
from color_palette import apply_case_palette, color_cycle
from six_panel_layout import compact_six_panel_rows

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("R_nucl_kpc", "Mdot_land_Msun_yr", "lambda", "Z_land", "yield_y", "Z_outer_boundary")
TITLE = "Unmixed fountain, analytic solution"


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def calculate(config, cases):
    base = {row["name"]: float(row["value"]) for row in read_rows(config) if row["name"] != "surface_rate_time_unit"}
    profiles, parameters = {}, {}
    for case in cases:
        trial = {**base, **{key: float(case[key]) for key in FIELDS if case[key]}}
        if case["R_nucl_kpc"]:
            trial["R_in_kpc"] = trial["R_nucl_kpc"]
        trial["KS_A_net"] = (1 - trial.pop("return_fraction")) * trial.pop("KS_A_SFR")
        samples = int(trial.pop("n_points"))
        inner, outer = trial["R_in_kpc"], trial["R_out_kpc"]
        # Include the finite inner-edge values and sample the divergent
        # chemical derivative just outside that edge, as in the original plot.
        radius = np.r_[inner, np.linspace(inner + 1e-4 * (outer - inner), outer, samples)]
        profiles[case["case"]] = power_law_profiles(radius, trial)
        parameters[case["case"]] = trial
    return base, parameters, profiles


def make_figure(base, cases, profiles):
    chemistry.smplotlib.set_style(usetex=False, fontsize=9, dpi=144)
    plt.rcParams.update({"axes.prop_cycle": color_cycle(), "axes.grid": False, "legend.frameon": False,
                         "pdf.fonttype": 42, "savefig.bbox": None})
    figure, axes = plt.subplots(3, 2, figsize=(7.2, 4.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]})
    rates, angular, accretion, velocity, metal, gradient = axes.ravel()
    # Draw every case, including physically coincident chemical-only flow
    # curves. Keep the black fiducial visible where cases coincide.
    ordered = [case for case in cases if case["case"] != "fiducial"] + [cases[0]]
    for case in ordered:
        profile = profiles[case["case"]]
        radius = profile["R_kpc"]
        style = dict(color=case["color"], linewidth=1.3 if case["case"] == "fiducial" else .9,
                     linestyle=case.get("linestyle") or "-", zorder=3)
        rates.plot(radius, profile["Sigmadot_land_Msun_Gyr_kpc2"], **style)
        rates.plot(radius, profile["Sigmadot_star_Msun_Gyr_kpc2"],
                   linestyle=(0, (4, 2.5)),
                   **{**{key: value for key, value in style.items() if key != "linestyle"},
                      "linewidth": .65 if case["case"] == "fiducial" else .45})
        angular.plot(radius, 1 - profile["j_land_over_j_disk"], **style)
        velocity.plot(radius, profile["v_R_kms"], **style)
        accretion.plot(radius, profile["Mdot_acc_Msun_yr"], **style)
        metal.plot(radius, profile["Z"] / chemistry.SOLAR_METALLICITY, **style)
        valid = np.isfinite(profile["dZ_dR_per_kpc"]) & (profile["Z"] > 0)
        chemical_gradient = np.full_like(radius, np.nan)
        chemical_gradient[valid] = profile["dZ_dR_per_kpc"][valid] / (profile["Z"][valid] * np.log(10))
        gradient.plot(radius, chemical_gradient, **style)

    rates.set_yscale("log")
    rates.set_ylim(1e4, 1e7)
    rates.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    rates.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=100))
    angular.set_ylim(0, 1)
    angular.set_yticks([0, .25, .5, .75, 1])
    angular.yaxis.set_minor_locator(AutoMinorLocator(5))
    velocity.set_ylim(-4, 0)
    velocity.set_yticks([-4, -3, -2, -1, 0])
    accretion.set_ylim(0, .3)
    metal.set_ylim(0, 3)
    gradient.set_ylim(-.1, 0)
    labels = (
        r"$\dot\Sigma\;[M_\odot\,\mathrm{Gyr}^{-1}\,\mathrm{kpc}^{-2}]$",
        r"$1-j_{\rm land}/j$",
        r"$\dot M_{\rm acc}\;[M_\odot\,\mathrm{yr}^{-1}]$",
        r"$v_R\;[\mathrm{km\,s}^{-1}]$",
        r"$Z/Z_\odot$", r"$d\log_{10}Z/dR\;[\mathrm{dex\,kpc}^{-1}]$")
    for axis, label in zip(axes.ravel(), labels):
        axis.set_ylabel(quantity_unit_label(label), fontsize=9, labelpad=2)
        axis.set_xlim(0, base["R_out_kpc"])
        axis.set_xticks([0, 5, 10, 15], labels=["0", "5", "10", "15"])
        axis.xaxis.set_minor_locator(AutoMinorLocator(5))
        axis.tick_params(which="both", direction="in", top=True, right=True, labelsize=8)
    for axis in axes[-1]:
        axis.set_xlabel(r"$R\;[\mathrm{kpc}]$", fontsize=9, labelpad=2)
    top_axes = [axis.secondary_xaxis("top") for axis in axes[0]]
    for axis in top_axes:
        axis.set_xticks([0, 5, 10, 15], labels=["0", "5", "10", "15"])
        axis.set_xlabel(r"$R\;[\mathrm{kpc}]$", fontsize=9, labelpad=7)
        axis.tick_params(which="both", direction="in", labelsize=8, pad=2)
        axis.xaxis.set_minor_locator(AutoMinorLocator(5))
    rates.legend(handles=[Line2D([], [], color="black", label=r"$\dot\Sigma_{\rm land}$"),
                          Line2D([], [], color="black", linewidth=.65,
                                 linestyle=(0, (4, 2.5)), label=r"$\dot\Sigma_\star$")],
                 loc="lower left", fontsize=8, handlelength=2, labelspacing=.2)
    heading = figure.suptitle(TITLE, fontsize=10, y=.99)
    handles = [Line2D([], [], color=case["color"], linestyle=case.get("linestyle") or "-",
                      label=case["label"]) for case in cases]
    # Keep parameter families together in one shared legend row.
    leading_names={"fiducial","landing_low","landing_high","slope_low","slope_mid","slope_high"}
    order=[index for index,case in enumerate(cases) if case["case"] in leading_names]
    order.extend(index for index,case in enumerate(cases) if case["case"] not in leading_names)
    legends=[figure.legend(handles=[handles[index] for index in order], ncol=len(handles),
                           loc="lower center", bbox_to_anchor=(.5,.01), fontsize=8,
                           frameon=False, borderpad=0, borderaxespad=0,
                           handlelength=1, handletextpad=.25, columnspacing=.4)]
    figure.subplots_adjust(left=.095, right=.985, bottom=.13, top=.88, hspace=.16, wspace=.23)

    # Fit margins and gutters to the actual rendered labels, keeping four
    # points between neighboring panels and the title/legend.
    for _ in range(3):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        gap = 4 * figure.dpi / 72
        boxes = [axis.get_tightbbox(renderer) for axis in axes.ravel()]
        panel_top = max(axis.get_tightbbox(renderer).y1 for axis in top_axes)
        panel_bottom = min(box.y0 for box in boxes[-2:])
        legend_top = max(legend.get_window_extent(renderer).y1 for legend in legends)
        figure.subplots_adjust(
            left=figure.subplotpars.left + (gap - min(box.x0 for box in boxes)) / figure.bbox.width,
            right=figure.subplotpars.right + (figure.bbox.width - gap - max(box.x1 for box in boxes)) / figure.bbox.width,
            top=figure.subplotpars.top + (heading.get_window_extent(renderer).y0 - gap - panel_top) / figure.bbox.height,
            bottom=figure.subplotpars.bottom + (legend_top + gap - panel_bottom) / figure.bbox.height)
        column_gap = max(boxes[i].x1 - axes.ravel()[i].bbox.x1 +
                         axes.ravel()[i+1].bbox.x0 - boxes[i+1].x0 for i in (0, 2, 4)) + gap
        row_gap = max(axes.ravel()[i].bbox.y0 - boxes[i].y0 +
                      boxes[i+2].y1 - axes.ravel()[i+2].bbox.y1 for i in range(4)) + gap
        width = (figure.subplotpars.right - figure.subplotpars.left) * figure.bbox.width
        height = (figure.subplotpars.top - figure.subplotpars.bottom) * figure.bbox.height
        figure.subplots_adjust(wspace=2 * column_gap / (width - column_gap),
                               hspace=3 * row_gap / (height - 2 * row_gap))
    compact_six_panel_rows(figure,axes.ravel())
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/idealized/power_law.csv")
    parser.add_argument("--cases", type=Path, default=ROOT / "config/idealized/section2_unmixed_analytic_combined_cases.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/section2_unmixed_analytic_combined_scans.pdf")
    args = parser.parse_args()
    cases = apply_case_palette(read_rows(args.cases))
    base, parameters, profiles = calculate(args.config, cases)
    figure = make_figure(base, cases, profiles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    plt.close(figure)
    run_directory = ROOT / "output/section2_unmixed_analytic_combined"
    run_directory.mkdir(parents=True, exist_ok=True)
    for name, profile in profiles.items():
        with (run_directory / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(profile)
            writer.writerows(zip(*profile.values()))
    manifest = {"figure": str(args.output), "layout": "3x2", "row_height_ratios": [1, 1, 1],
                "panels": ["rates", "angular_deficit", "accretion", "radial_velocity", "metallicity", "metallicity_gradient"],
                "size_inches": figure.get_size_inches().tolist(), "all_cases_in_every_panel": True,
                "inter_row_text_clearance_pt": 4, "panel_sizes_preserved": True,
                "font_sizes_pt": {"axis_labels": 9, "ticks": 8,
                                  "legends": figure.legends[0].get_texts()[0].get_fontsize(), "title": 10},
                "radius_major_ticks_kpc": [0, 5, 10, 15], "radius_minor_step_kpc": 1,
                "inset_count": 0,
                "gradient_limits_dex_kpc": [-.1, 0], "gradient_inset": False,
                "angular_deficit_limits": [0, 1],
                "angular_deficit_role": "Prescribed j_land=j_nuc; core-exported j_land/j_disk; all cases except launch-radius variations coincide",
                "velocity_limits_kms": [-4, 0],
                "metallicity_limits_Z_sun": [0, 3],
                "observational_overlays": False,
                "yield_variation_role": "y=0 is a no-enrichment control, not a physical yield estimate",
                "surface_rate_limits_Msun_Gyr_kpc2": [1e4, 1e7],
                "legend_rows": ["fiducial, landing rate, lambda, nuclear radius, yield, incident and outer abundance"],
                "parameters": parameters, "Z_sun": chemistry.SOLAR_METALLICITY,
                "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (args.config, args.cases, Path(__file__),
                                               ROOT / "scripts/plot/color_palette.py",
                                               ROOT / "scripts/plot/six_panel_layout.py", RUNNER)}}
    (run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output}: {len(cases)} cases in every panel")


if __name__ == "__main__":
    main()
