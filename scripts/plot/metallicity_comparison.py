#!/usr/bin/env python3

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from hershey_fonts import register_hershey_weight_aliases
from paper_galaxies import PAPER_GALAXIES, PAPER_GALAXY_SET, galaxy_label


register_hershey_weight_aliases()
import smplotlib


COMPARISON_FIGURE_SIZE = (3.5, 3.5)


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
    "forward_sparc_spline": {
        "color": "black",
        "linestyle": "-",
        "marker": "o",
    },
    "inverse_dynamics": {
        "color": "black",
        "linestyle": "-",
        "marker": "o",
    },
}

PLOTTED_CURVES = {
    "forward": {"forward_sparc_spline"},
    "inverse": {"inverse_dynamics"},
}

FAMILY_TITLES = {
    "forward": "forward",
    "inverse": "inverse",
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


def group_rows(rows, key_names):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in key_names)].append(row)
    return groups


def sorted_numeric(rows, name):
    return sorted(rows, key=lambda row: number(row, name))


def make_plot(
    galaxy,
    family,
    observations,
    curves,
    output_root,
    show_title=True,
):
    figure, profile_axis = plt.subplots(figsize=COMPARISON_FIGURE_SIZE)
    profile_axis.set_box_aspect(1)
    if show_title:
        profile_axis.set_title(f"{galaxy_label(galaxy)} - {FAMILY_TITLES[family]}")
    profile_axis.set_xlabel(r"$R\;[\mathrm{kpc}]$")
    profile_axis.set_ylabel(r"$12+\log_{10}(\mathrm{O/H})$")

    curve_values = []
    curve_radii = []
    for curve_id, curve_rows in sorted(curves.items()):
        style = CURVE_STYLES[curve_id]
        ordered = sorted_numeric(curve_rows, "R_kpc")
        radius = [number(row, "R_kpc") for row in ordered]
        model_oh12 = [number(row, "model_OH12") for row in ordered]
        profile_axis.plot(
            radius,
            model_oh12,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0 if ordered[0]["curve_role"] == "prediction" else 1.5,
            label="SPARC spline",
        )
        curve_radii.extend(radius)
        curve_values.extend(model_oh12)

    ordered_observations = sorted_numeric(observations, "R_kpc")
    observation_radius = [number(row, "R_kpc") for row in ordered_observations]
    observation_oh12 = [number(row, "OH12") for row in ordered_observations]
    observation_error = [number(row, "e_OH12") for row in ordered_observations]
    profile_axis.errorbar(
        observation_radius,
        observation_oh12,
        yerr=observation_error,
        color="black",
        ecolor="black",
        markerfacecolor="black",
        markeredgecolor="black",
        alpha=1.0,
        linestyle="none",
        marker="o",
        markersize=4.0,
        capsize=2.0,
        label="SINGS KK04",
    )

    all_radii = observation_radius + curve_radii
    if all_radii:
        profile_axis.set_xlim(0.0, 1.03 * max(all_radii))
    all_abundances = [
        bound
        for value, error in zip(observation_oh12, observation_error)
        for bound in (value - error, value + error)
    ] + curve_values
    if all_abundances:
        abundance_min = min(all_abundances)
        abundance_max = max(all_abundances)
        padding = max(0.08, 0.06 * (abundance_max - abundance_min))
        profile_axis.set_ylim(abundance_min - padding, abundance_max + padding)
    figure.subplots_adjust(left=0.20, right=0.97, top=0.95, bottom=0.17)

    galaxy_directory = output_root / galaxy
    galaxy_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(galaxy_directory / f"{family}_KK04_comparison.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Plot KK04 metallicity comparisons computed by galactic-nuclear-fountain-compare"
    )
    parser.add_argument(
        "comparison_output",
        type=Path,
        help="directory containing fit_summary.csv and the plot-ready comparison CSVs",
    )
    parser.add_argument(
        "plot_output",
        nargs="?",
        type=Path,
        help="plot directory; defaults to COMPARISON_OUTPUT",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="suppress per-galaxy titles for embedding in grouped panels",
    )
    arguments = parser.parse_args()
    comparison_output = arguments.comparison_output
    output_root = arguments.plot_output or comparison_output

    observations = read_csv(
        comparison_output / "observations.csv",
        {"galaxy", "calibration", "source_seq", "R_kpc", "OH12", "e_OH12"},
    )
    curves = read_csv(
        comparison_output / "model_curves.csv",
        {
            "galaxy",
            "model_family",
            "model_curve",
            "curve_role",
            "label",
            "R_kpc",
            "model_OH12",
        },
    )
    summary = read_csv(
        comparison_output / "fit_summary.csv",
        {
            "galaxy",
            "calibration",
            "model_family",
            "model_curve",
            "fit_status",
        },
    )

    kk04_observations = [row for row in observations if row["calibration"] == "KK04"]
    kk04_summary = [row for row in summary if row["calibration"] == "KK04"]
    observation_groups = group_rows(kk04_observations, ("galaxy",))
    curve_groups = group_rows(curves, ("galaxy", "model_family", "model_curve"))
    summary_groups = group_rows(kk04_summary, ("galaxy", "model_family"))

    galaxies = [
        galaxy for galaxy in PAPER_GALAXIES if (galaxy,) in observation_groups
    ]
    for galaxy in galaxies:
        for family in ("forward", "inverse"):
            family_curves = {
                key[2]: rows
                for key, rows in curve_groups.items()
                if key[0] == galaxy
                and key[1] == family
                and key[2] in PLOTTED_CURVES[family]
            }
            family_summary = [
                row
                for row in summary_groups.get((galaxy, family), [])
                if row["model_curve"] in PLOTTED_CURVES[family]
            ]
            if not family_summary:
                raise ValueError(f"Incomplete comparison plot data for {galaxy}/{family}")
            if not family_curves and not any(
                "stagnation" in row["fit_status"] for row in family_summary
            ):
                raise ValueError(f"Missing SPARC-spline curve for {galaxy}/{family}")
            make_plot(
                galaxy,
                family,
                observation_groups[(galaxy,)],
                family_curves,
                output_root,
                show_title=not arguments.no_title,
            )

    for stale in output_root.glob("*/*_KK04_comparison.pdf"):
        if stale.parent.name not in PAPER_GALAXY_SET:
            stale.unlink()

    print(f"Wrote {2 * len(galaxies)} KK04 comparison figures to: {output_root}")


if __name__ == "__main__":
    main()
