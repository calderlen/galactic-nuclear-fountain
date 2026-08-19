#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from hershey_fonts import register_hershey_weight_aliases


register_hershey_weight_aliases()
import smplotlib


MODEL_FIGURE_SIZE = (3.5, 3.5)


smplotlib.set_style(
    usetex=False,
    fontsize=11,
    figsize=MODEL_FIGURE_SIZE,
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


TEXT_COLUMNS = {
    "model",
    "profile_type",
    "galaxy",
    "source",
    "kind",
    "landing_nonnegative",
    "radial_H2_available",
    "H2_treatment",
}

SOURCE_COLORS = {
    "Leroy_THINGS": "royalblue",
    "SPARC_corrected_fit": "darkorange",
}

SOURCE_LABELS = {
    "Leroy_THINGS": "Leroy/THINGS",
    "SPARC_corrected_fit": "SPARC fit",
}

LEGEND_STYLE = {
    "fontsize": 7.5,
    "borderpad": 0.3,
    "labelspacing": 0.2,
    "handlelength": 1.6,
    "handletextpad": 0.5,
    "borderaxespad": 1.0,
    "frameon": False,
}


def parse_number(value):
    if value is None or value.strip() == "":
        return math.nan
    return float(value)


def read_rows(path, required_columns=()):
    if not path.is_file():
        raise FileNotFoundError(f"Required model output is missing: {path}")

    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        missing = set(required_columns) - fieldnames
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(sorted(missing))}"
            )

        rows = []
        for raw_row in reader:
            row = {}
            for name, value in raw_row.items():
                row[name] = value if name in TEXT_COLUMNS else parse_number(value)
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def read_profiles(path):
    rows = read_rows(path, {"model", "source", "R_kpc"})
    profiles = {}
    for row in rows:
        profiles.setdefault(row["source"], []).append(row)
    for source_rows in profiles.values():
        source_rows.sort(key=lambda row: row["R_kpc"])
    return profiles


def read_summary(path):
    return read_rows(path, {"profile_type", "Mdot_land_Msun_yr", "mu"})


def read_rotations(path):
    return read_rows(
        path,
        {"source", "kind", "Vflat_kms", "lflat_kpc", "chi2", "dof", "reduced_chi2"},
    )


def read_metadata(path):
    rows = read_rows(path, {"galaxy"})
    if len(rows) != 1:
        raise ValueError(f"Expected one metadata row in {path}, found {len(rows)}")
    return rows[0]


def read_sparc(path):
    return read_rows(
        path,
        {"R_adopted_kpc", "Vobs_adopted_kms", "e_Vobs_adopted_kms"},
    )


def load_model_run(output_directory):
    profiles = read_profiles(output_directory / "profiles.csv")
    summary = read_summary(output_directory / "summary.csv")
    rotations = read_rotations(output_directory / "rotation_curves.csv")

    profile_types = {row["profile_type"] for row in summary}
    if len(profile_types) != 1:
        raise ValueError(
            f"Expected one profile_type in {output_directory / 'summary.csv'}, "
            f"found {sorted(profile_types)}"
        )

    if profile_types.pop() == "tabulated":
        metadata = read_metadata(output_directory / "metadata.csv")
        sparc = read_sparc(output_directory / "sparc_corrected.csv")
    else:
        metadata = None
        sparc = None

    return profiles, summary, rotations, metadata, sparc


def source_label(source):
    return SOURCE_LABELS.get(source, source.replace("_", " "))


def source_style(source, index=0, mode="color"):
    if mode == "black":
        color = "black"
    elif source in SOURCE_COLORS:
        color = SOURCE_COLORS[source]
    else:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color = colors[index % len(colors)]

    distinguish_sparc = mode in {"black", "color_dashed"}
    linestyle = "--" if distinguish_sparc and source == "SPARC_corrected_fit" else "-"
    return {"color": color, "linestyle": linestyle}


def finite_xy(rows, x_name, y_name, positive=False):
    points = [
        (row[x_name], row[y_name])
        for row in rows
        if math.isfinite(row[x_name])
        and math.isfinite(row[y_name])
        and (not positive or row[y_name] > 0)
    ]
    return [point[0] for point in points], [point[1] for point in points]


def finite_values(profiles, column):
    return [
        row[column]
        for rows in profiles.values()
        for row in rows
        if math.isfinite(row[column])
    ]


def least_data_legend_position(axis, plotted_points):
    if not plotted_points:
        return "upper right"

    x0, x1 = axis.get_xlim()
    y0, y1 = axis.get_ylim()
    if axis.get_xscale() == "log":
        x0, x1 = math.log10(x0), math.log10(x1)
    if axis.get_yscale() == "log":
        y0, y1 = math.log10(y0), math.log10(y1)
    if x1 == x0 or y1 == y0:
        return "best"

    points = []
    for x_values, y_values in plotted_points:
        for x, y in zip(x_values, y_values):
            if axis.get_xscale() == "log":
                if x <= 0:
                    continue
                x = math.log10(x)
            if axis.get_yscale() == "log":
                if y <= 0:
                    continue
                y = math.log10(y)
            points.append(((x - x0) / (x1 - x0), (y - y0) / (y1 - y0)))

    candidates = {
        "upper right": (0.75, 1.0, 0.72, 1.0),
        "upper left": (0.0, 0.25, 0.72, 1.0),
        "lower right": (0.75, 1.0, 0.0, 0.28),
        "lower left": (0.0, 0.25, 0.0, 0.28),
    }
    best_location = "upper right"
    best_score = math.inf
    for location, (xmin, xmax, ymin, ymax) in candidates.items():
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        overlap = sum(
            xmin <= x <= xmax and ymin <= y <= ymax for x, y in points
        )
        proximity = sum(
            max(0.0, 0.35 - math.hypot(x - center_x, y - center_y))
            for x, y in points
        )
        score = 1000.0 * overlap + proximity
        if score < best_score:
            best_location = location
            best_score = score
    return best_location


def finish_figure(
    figure,
    axis,
    path,
    ylabel,
    plotted_points,
    log_y=False,
    legend=True,
    title=None,
):
    axis.set_box_aspect(1)
    axis.set_xlabel(r"$R\;[\mathrm{kpc}]$")
    axis.set_ylabel(ylabel)
    if log_y:
        axis.set_yscale("log")
    if title:
        axis.set_title(title)
    if legend:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(
                handles,
                labels,
                loc=least_data_legend_position(axis, plotted_points),
                **LEGEND_STYLE,
            )
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def add_reference(axis, plotted_points, value, label=None, color="0.45", style=":"):
    if value is None or not math.isfinite(value):
        return
    x_limits = axis.get_xlim()
    axis.axhline(value, color=color, linestyle=style, linewidth=0.9, label=label)
    plotted_points.append(([x_limits[0], x_limits[1]], [value, value]))


def pad_nearly_constant_y_axis(axis, values):
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return
    lower = min(values)
    upper = max(values)
    scale = max(abs(lower), abs(upper), 1.0)
    if upper - lower <= 1e-8 * scale:
        midpoint = 0.5 * (lower + upper)
        margin = 0.05 * scale
        axis.set_ylim(midpoint - margin, midpoint + margin)


def plot_profile_column(
    profiles,
    output_directory,
    filename,
    column,
    ylabel,
    references=(),
    log_y=False,
    style_mode="black",
):
    figure, axis = plt.subplots()
    plotted_points = []
    for index, (source, rows) in enumerate(profiles.items()):
        radius, values = finite_xy(rows, "R_kpc", column, positive=log_y)
        if not radius:
            continue
        style = source_style(source, index, style_mode)
        axis.plot(radius, values, label=source_label(source), **style)
        plotted_points.append((radius, values))
    for reference in references:
        add_reference(axis, plotted_points, **reference)
    finish_figure(
        figure,
        axis,
        output_directory / filename,
        ylabel,
        plotted_points,
        log_y=log_y,
    )


def rotation_velocity(rotation, radius):
    if rotation["kind"] == "flat":
        return rotation["Vflat_kms"]
    if rotation["kind"] == "rising":
        return rotation["Vflat_kms"] * (
            1.0 - math.exp(-radius / rotation["lflat_kpc"])
        )
    raise ValueError(
        f"Unknown rotation type {rotation['kind']!r} for {rotation['source']}"
    )


def plot_rotation_curves(profiles, rotations, sparc, metadata, output_directory):
    figure, axis = plt.subplots()
    plotted_points = []
    maximum_radius = max(
        row["R_kpc"] for rows in profiles.values() for row in rows
    )

    if sparc is not None:
        sparc_points = [
            (
                row["R_adopted_kpc"],
                row["Vobs_adopted_kms"],
                row["e_Vobs_adopted_kms"],
            )
            for row in sparc
            if math.isfinite(row["R_adopted_kpc"])
            and math.isfinite(row["Vobs_adopted_kms"])
            and math.isfinite(row["e_Vobs_adopted_kms"])
        ]
        if not sparc_points:
            raise ValueError("sparc_corrected.csv has no finite corrected measurements")
        sparc_radius = [point[0] for point in sparc_points]
        sparc_velocity = [point[1] for point in sparc_points]
        errors = [point[2] for point in sparc_points]
        maximum_radius = max(maximum_radius, max(sparc_radius))
        axis.errorbar(
            sparc_radius,
            sparc_velocity,
            yerr=errors,
            fmt="o",
            markersize=2.8,
            linewidth=0.8,
            capsize=1.5,
            color="0.25",
            ecolor="0.55",
            label="corrected SPARC",
            zorder=2,
        )
        plotted_points.append((sparc_radius, sparc_velocity))
        plotted_points.append(
            (sparc_radius, [value - error for value, error in zip(sparc_velocity, errors)])
        )
        plotted_points.append(
            (sparc_radius, [value + error for value, error in zip(sparc_velocity, errors)])
        )

    curve_radius = [maximum_radius * index / 499.0 for index in range(500)]
    for index, rotation in enumerate(rotations):
        velocities = [rotation_velocity(rotation, radius) for radius in curve_radius]
        style = source_style(rotation["source"], index, "black")
        axis.plot(
            curve_radius,
            velocities,
            label=source_label(rotation["source"]),
            zorder=3,
            **style,
        )
        plotted_points.append((curve_radius, velocities))

    title = metadata["galaxy"] if metadata is not None else None
    finish_figure(
        figure,
        axis,
        output_directory / "rotation_curves.pdf",
        r"$v_c\;[\mathrm{km\,s^{-1}}]$",
        plotted_points,
        title=title,
    )


def plot_surface_rates(profiles, output_directory):
    figure, axis = plt.subplots()
    plotted_points = []
    values = finite_values(profiles, "Sigmadot_star_Msun_yr_kpc2")
    values += finite_values(profiles, "Sigmadot_land_Msun_yr_kpc2")
    log_y = bool(values) and all(value > 0 for value in values)

    first_rows = next(iter(profiles.values()))
    radius, star = finite_xy(
        first_rows, "R_kpc", "Sigmadot_star_Msun_yr_kpc2", positive=log_y
    )
    axis.plot(
        radius,
        star,
        color="black",
        linestyle="-",
        label=r"$\dot{\Sigma}_\star$",
    )
    plotted_points.append((radius, star))

    for index, (source, rows) in enumerate(profiles.items()):
        radius, landing = finite_xy(
            rows, "R_kpc", "Sigmadot_land_Msun_yr_kpc2", positive=log_y
        )
        style = source_style(source, index, "color_dashed")
        axis.plot(
            radius,
            landing,
            label=rf"$\dot{{\Sigma}}_{{\rm land}}$: {source_label(source)}",
            **style,
        )
        plotted_points.append((radius, landing))

    add_reference(axis, plotted_points, 0.0 if not log_y else None)
    finish_figure(
        figure,
        axis,
        output_directory / "surface_rates.pdf",
        r"$\dot{\Sigma}\;[M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}]$",
        plotted_points,
        log_y=log_y,
    )


def plot_cumulative_landing(profiles, summary, output_directory):
    targets = {
        row["Mdot_land_Msun_yr"]
        for row in summary
        if math.isfinite(row["Mdot_land_Msun_yr"]) and row["mu"] != 0.0
    }
    references = []
    for target in sorted(targets):
        references.append(
            {
                "value": target,
                "label": rf"landing target $={target:g}$",
                "color": "0.35",
                "style": ":",
            }
        )
    plot_profile_column(
        profiles,
        output_directory,
        "cumulative_landing_rate.pdf",
        "cumulative_landing_Msun_yr",
        r"$\dot{M}_{\rm land}(<R)\;[M_\odot\,\mathrm{yr}^{-1}]$",
        references=references,
    )


def plot_timescales(profiles, output_directory):
    figure, axis = plt.subplots()
    plotted_points = []
    first_rows = next(iter(profiles.values()))
    radius, depletion = finite_xy(
        first_rows, "R_kpc", "t_depletion_Gyr", positive=True
    )
    axis.plot(
        radius,
        depletion,
        color="black",
        linestyle="--",
        label=r"$t_{\rm depletion}$",
    )
    plotted_points.append((radius, depletion))

    for index, (source, rows) in enumerate(profiles.items()):
        radius, inflow = finite_xy(rows, "R_kpc", "t_inflow_Gyr", positive=True)
        style = source_style(source, index, "color")
        axis.plot(
            radius,
            inflow,
            label=rf"$t_{{\rm inflow}}$: {source_label(source)}",
            **style,
        )
        plotted_points.append((radius, inflow))

    finish_figure(
        figure,
        axis,
        output_directory / "timescales.pdf",
        r"$t\;[\mathrm{Gyr}]$",
        plotted_points,
        log_y=True,
    )


def plot_angular_momentum(profiles, output_directory):
    plot_profile_column(
        profiles,
        output_directory,
        "angular_momentum_ratio.pdf",
        "j_land_over_j_disk",
        r"$j_{\rm land}/j_{\rm disk}$",
        references=(
            {"value": 1.0, "label": "unity", "color": "0.35", "style": ":"},
        ),
    )
    plot_profile_column(
        profiles,
        output_directory,
        "landing_to_nuclear_angular_momentum_ratio.pdf",
        "j_land_over_j_nucl",
        r"$j_{\rm land}/j_{\rm nucl}$",
        references=(
            {"value": 1.0, "label": "unity", "color": "0.35", "style": ":"},
        ),
        style_mode="color",
    )


def plot_metallicity(profiles, output_directory):
    figure, axis = plt.subplots()
    plotted_points = []
    for index, (source, rows) in enumerate(profiles.items()):
        style = source_style(source, index, "color")
        radius, metallicity = finite_xy(rows, "R_kpc", "Z")
        axis.plot(
            radius,
            metallicity,
            label=rf"$Z$: {source_label(source)}",
            **style,
        )
        plotted_points.append((radius, metallicity))

        radius, equilibrium = finite_xy(rows, "R_kpc", "Z_eq")
        axis.plot(
            radius,
            equilibrium,
            color=style["color"],
            linestyle="--",
            label=rf"$Z_{{\rm eq}}$: {source_label(source)}",
        )
        plotted_points.append((radius, equilibrium))

    finish_figure(
        figure,
        axis,
        output_directory / "metallicity.pdf",
        r"$Z$",
        plotted_points,
    )


def plot_mu(profiles, summary, output_directory):
    figure, axis = plt.subplots()
    plotted_points = []
    input_values = {
        row["mu"] for row in summary if math.isfinite(row["mu"])
    }
    for index, (source, rows) in enumerate(profiles.items()):
        style = source_style(source, index, "color")
        radius, mu_j = finite_xy(rows, "R_kpc", "mu_j")
        axis.plot(
            radius,
            mu_j,
            label=rf"$\mu_j$: {source_label(source)}",
            **style,
        )
        plotted_points.append((radius, mu_j))

        radius, mu_z = finite_xy(rows, "R_kpc", "mu_Z")
        axis.plot(
            radius,
            mu_z,
            color=style["color"],
            linestyle=":",
            label=rf"$\mu_Z$: {source_label(source)}",
        )
        plotted_points.append((radius, mu_z))

    for value in sorted(input_values):
        add_reference(
            axis,
            plotted_points,
            value,
            label=rf"input $\mu={value:g}$",
            color="0.35",
            style="-.",
        )
    plotted_values = [
        value
        for _, values in plotted_points
        for value in values
    ]
    pad_nearly_constant_y_axis(axis, plotted_values)
    finish_figure(
        figure,
        axis,
        output_directory / "mu_parameters.pdf",
        r"$\mu$",
        plotted_points,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot analytic or observational GalacticWind C++ output"
    )
    parser.add_argument(
        "model_output",
        type=Path,
        help="directory containing profiles.csv, summary.csv, and rotation_curves.csv",
    )
    parser.add_argument(
        "plot_output",
        nargs="?",
        type=Path,
        help="plot directory; defaults to MODEL_OUTPUT/plots",
    )
    arguments = parser.parse_args()
    output_directory = arguments.plot_output or arguments.model_output / "plots"
    output_directory.mkdir(parents=True, exist_ok=True)

    profiles, summary, rotations, metadata, sparc = load_model_run(
        arguments.model_output
    )

    plot_rotation_curves(
        profiles, rotations, sparc, metadata, output_directory
    )
    plot_profile_column(
        profiles,
        output_directory,
        "radial_velocity.pdf",
        "v_R_kms",
        r"$v_R\;[\mathrm{km\,s^{-1}}]$",
        references=({"value": 0.0},),
    )
    plot_surface_rates(profiles, output_directory)
    plot_cumulative_landing(profiles, summary, output_directory)
    plot_profile_column(
        profiles,
        output_directory,
        "accretion_rate.pdf",
        "Mdot_acc_Msun_yr",
        r"$\dot{M}_{\rm acc}\;[M_\odot\,\mathrm{yr}^{-1}]$",
        references=({"value": 0.0},),
    )
    plot_timescales(profiles, output_directory)
    plot_angular_momentum(profiles, output_directory)
    plot_metallicity(profiles, output_directory)
    plot_mu(profiles, summary, output_directory)
    print(f"Wrote model figures to: {output_directory}")


if __name__ == "__main__":
    main()
