#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from hershey_fonts import register_hershey_weight_aliases


register_hershey_weight_aliases()
import smplotlib


COMPARISON_FIGURE_SIZE = (4.3, 8.4)


smplotlib.set_style(
    usetex=False,
    fontsize=10,
    figsize=COMPARISON_FIGURE_SIZE,
    dpi=144,
)
plt.rcParams.update(
    {
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "legend.frameon": False,
    }
)


CURVE_STYLES = {
    "forward_leroy": {
        "color": "royalblue",
        "linestyle": "-",
        "marker": "o",
    },
    "forward_sparc_fit": {
        "color": "darkorange",
        "linestyle": "--",
        "marker": "D",
    },
    "inverse_dynamics": {
        "color": "purple",
        "linestyle": "-",
        "marker": "o",
    },
    "inverse_local_equilibrium": {
        "color": "0.45",
        "linestyle": "--",
        "marker": "o",
    },
}

FAMILY_TITLES = {
    "forward": "forward mixing model",
    "inverse": "inverse-dynamics model",
}

LEGEND_STYLE = {
    "fontsize": 5.5,
    "borderpad": 0.3,
    "labelspacing": 0.2,
    "handlelength": 1.6,
    "handletextpad": 0.5,
    "borderaxespad": 1.0,
    "frameon": False,
}


def read_csv(path, required_columns):
    if not path.is_file():
        raise FileNotFoundError(f"Required comparison output is missing: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = set(required_columns) - columns
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(sorted(missing))}"
            )
        return list(reader)


def number(row, name):
    value = row[name]
    if value == "":
        return math.nan
    return float(value)


def boolean(row, name):
    value = row[name].lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Invalid Boolean value for {name}: {row[name]}")


def group_rows(rows, key_names):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in key_names)].append(row)
    return groups


def sorted_numeric(rows, name):
    return sorted(rows, key=lambda row: number(row, name))


def make_plot(galaxy, family, observations, curves, residuals, summary, output_root):
    figure, (profile_axis, residual_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=COMPARISON_FIGURE_SIZE,
        gridspec_kw={"height_ratios": (1.0, 1.0), "hspace": 0.08},
    )
    profile_axis.set_box_aspect(1)
    residual_axis.set_box_aspect(1)
    profile_axis.set_title(f"{galaxy} - {FAMILY_TITLES[family]}")
    profile_axis.set_ylabel(r"$Z$")
    residual_axis.set_xlabel(r"$R/R_{\mathrm{e}}$")
    residual_axis.set_ylabel(r"$\Delta\log_{10} Z$ [dex]")
    residual_axis.axhline(0.0, color="0.55", linestyle=":", linewidth=1.0)

    curve_values = []
    curve_radii = []
    for curve_id, curve_rows in sorted(curves.items()):
        style = CURVE_STYLES[curve_id]
        ordered = sorted_numeric(curve_rows, "R_Reff")
        radius = [number(row, "R_Reff") for row in ordered]
        metallicity = [number(row, "Z") for row in ordered]
        profile_axis.plot(
            radius,
            metallicity,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0 if ordered[0]["curve_role"] == "prediction" else 1.5,
            label=ordered[0]["label"],
        )
        curve_radii.extend(radius)
        curve_values.extend(metallicity)

    ordered_observations = sorted_numeric(observations, "R_Reff")
    observation_radius = [number(row, "R_Reff") for row in ordered_observations]
    observation_Z = [number(row, "Z") for row in ordered_observations]
    observation_error_low = [number(row, "e_Z_lo") for row in ordered_observations]
    observation_error_high = [number(row, "e_Z_hi") for row in ordered_observations]
    profile_axis.errorbar(
        observation_radius,
        observation_Z,
        yerr=(observation_error_low, observation_error_high),
        color="black",
        alpha=0.45,
        linestyle="none",
        marker="o",
        markersize=4.0,
        capsize=2.0,
        label="SINGS KK04",
    )

    for curve_id, residual_rows in sorted(residuals.items()):
        style = CURVE_STYLES[curve_id]
        ordered = sorted_numeric(residual_rows, "R_Reff")
        radius = [number(row, "R_Reff") for row in ordered]
        values = [number(row, "residual_dex") for row in ordered]
        errors = [number(row, "sigma_dex") for row in ordered]
        residual_axis.errorbar(
            radius,
            values,
            yerr=errors,
            color=style["color"],
            alpha=0.65,
            linestyle="none",
            marker=style["marker"],
            markersize=3.5,
            capsize=1.5,
        )
    if not residuals:
        residual_axis.set_ylim(-0.5, 0.5)

    all_radii = observation_radius + curve_radii
    if all_radii:
        residual_axis.set_xlim(0.0, 1.03 * max(all_radii))
    all_metallicity = [
        value + error
        for value, error in zip(observation_Z, observation_error_high)
    ] + curve_values
    if all_metallicity:
        profile_axis.set_ylim(0.0, 1.08 * max(all_metallicity))
    profile_axis.legend(loc="upper right", **LEGEND_STYLE)

    representative = summary[0]
    if "stagnation" in representative["fit_status"]:
        annotation = "inverse Z not solved: radial-flow stagnation"
    else:
        slope = number(representative, "observed_slope_dex_per_Reff")
        slope_error = number(representative, "e_observed_slope_dex_per_Reff")
        if math.isfinite(slope) and math.isfinite(slope_error):
            annotation = f"observed slope = {slope:.3f} ± {slope_error:.3f} dex/Re"
        else:
            annotation = "observed slope unavailable"
        if not boolean(representative, "gradient_sample_reliable"):
            annotation += " (descriptive sample)"
    annotation += "; residuals include 0.05 dex scatter"
    figure.text(0.5, 0.015, annotation, ha="center", va="bottom", fontsize=7.5)
    figure.subplots_adjust(left=0.17, right=0.97, top=0.95, bottom=0.09)

    galaxy_directory = output_root / galaxy
    galaxy_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(galaxy_directory / f"{family}_KK04_comparison.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Plot KK04 metallicity comparisons computed by galacticwind_compare"
    )
    parser.add_argument(
        "comparison_output",
        type=Path,
        help="directory containing fit_summary.csv and the plot-ready comparison CSVs",
    )
    arguments = parser.parse_args()
    output_root = arguments.comparison_output

    observations = read_csv(
        output_root / "observations.csv",
        {"galaxy", "calibration", "source_seq", "R_Reff", "Z", "e_Z_lo", "e_Z_hi"},
    )
    curves = read_csv(
        output_root / "model_curves.csv",
        {"galaxy", "model_family", "model_curve", "curve_role", "label", "R_Reff", "Z"},
    )
    residuals = read_csv(
        output_root / "residuals.csv",
        {
            "galaxy",
            "calibration",
            "model_family",
            "model_curve",
            "R_Reff",
            "residual_dex",
            "sigma_dex",
        },
    )
    summary = read_csv(
        output_root / "fit_summary.csv",
        {
            "galaxy",
            "calibration",
            "model_family",
            "model_curve",
            "fit_status",
            "gradient_sample_reliable",
            "observed_slope_dex_per_Reff",
            "e_observed_slope_dex_per_Reff",
        },
    )

    kk04_observations = [row for row in observations if row["calibration"] == "KK04"]
    kk04_residuals = [row for row in residuals if row["calibration"] == "KK04"]
    kk04_summary = [row for row in summary if row["calibration"] == "KK04"]
    observation_groups = group_rows(kk04_observations, ("galaxy",))
    curve_groups = group_rows(curves, ("galaxy", "model_family", "model_curve"))
    residual_groups = group_rows(
        kk04_residuals, ("galaxy", "model_family", "model_curve")
    )
    summary_groups = group_rows(kk04_summary, ("galaxy", "model_family"))

    galaxies = sorted(key[0] for key in observation_groups)
    for galaxy in galaxies:
        for family in ("forward", "inverse"):
            family_curves = {
                key[2]: rows
                for key, rows in curve_groups.items()
                if key[0] == galaxy and key[1] == family
            }
            family_residuals = {
                key[2]: rows
                for key, rows in residual_groups.items()
                if key[0] == galaxy and key[1] == family
            }
            family_summary = summary_groups.get((galaxy, family), [])
            if not family_curves or not family_summary:
                raise ValueError(f"Incomplete comparison plot data for {galaxy}/{family}")
            make_plot(
                galaxy,
                family,
                observation_groups[(galaxy,)],
                family_curves,
                family_residuals,
                family_summary,
                output_root,
            )

    print(f"Wrote {2 * len(galaxies)} KK04 comparison figures to: {output_root}")


if __name__ == "__main__":
    main()
