#!/usr/bin/env python3

import argparse
import csv
import math
from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from hershey_fonts import register_hershey_weight_aliases
from paper_galaxies import galaxy_label, require_paper_galaxy


register_hershey_weight_aliases()
import smplotlib


MODEL_FIGURE_SIZE = (3.5, 3.5)
FIDUCIAL_PLOT_R_MAX_KPC = 20.0
OBSERVATIONAL_SOURCE = "SPARC_spline"
SURFACE_RATE_PLOT_SCALE = 1e3  # Msun/yr/kpc^2 -> Msun/Gyr/pc^2
SURFACE_RATE_LABEL = r"$\dot{\Sigma}\;[M_\odot\,\mathrm{Gyr}^{-1}\,\mathrm{pc}^{-2}]$"


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
    "row_type",
    "landing_nonnegative",
    "radial_H2_available",
    "H2_source",
    "H2_treatment",
    "bigiel2010_available",
    "profile_sources",
}

SOURCE_COLORS = {
    "SPARC_spline": "black",
}

SOURCE_LABELS = {
    "SPARC_spline": "SPARC spline",
}

LEGEND_STYLE = {
    "fontsize": 9.0,
    "borderpad": 0.3,
    "labelspacing": 0.2,
    "handlelength": 1.6,
    "handletextpad": 0.5,
    "borderaxespad": 1.0,
    "frameon": False,
}


TIME_FIGURE_STYLE = {
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "savefig.bbox": None,
    "pdf.fonttype": 42,
}
TIME_LEGEND_STYLE = {
    "fontsize": 8,
    "frameon": False,
    "handlelength": 1.6,
    "handletextpad": 0.4,
    "labelspacing": 0.3,
    "columnspacing": 1.0,
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


def read_sfr_diagnostics(path):
    if not path.is_file():
        return None
    return read_rows(
        path,
        {
            "source",
            "row_type",
            "R_kpc",
            "Sigmadot_star_obs_Msun_yr_kpc2",
            "e_Sigmadot_star_obs_Msun_yr_kpc2",
        },
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
        if OBSERVATIONAL_SOURCE not in profiles:
            raise ValueError(
                f"{output_directory / 'profiles.csv'} has no "
                f"{OBSERVATIONAL_SOURCE} rows"
            )
        profiles = {OBSERVATIONAL_SOURCE: profiles[OBSERVATIONAL_SOURCE]}
        summary = [
            row for row in summary if row["source"] == OBSERVATIONAL_SOURCE
        ]
        rotations = [
            rotation
            for rotation in rotations
            if rotation["source"] == OBSERVATIONAL_SOURCE
        ]
        if len(rotations) != 1:
            raise ValueError(
                f"Expected one {OBSERVATIONAL_SOURCE} rotation in "
                f"{output_directory / 'rotation_curves.csv'}, found {len(rotations)}"
            )
        metadata = read_metadata(output_directory / "metadata.csv")
        sparc = read_sparc(output_directory / "sparc_corrected.csv")
        sfr_diagnostics = read_sfr_diagnostics(
            output_directory / "sfr_spline_diagnostics.csv"
        )
    else:
        metadata = None
        sparc = None
        sfr_diagnostics = None

    return profiles, summary, rotations, metadata, sparc, sfr_diagnostics


def source_label(source):
    return SOURCE_LABELS.get(source, source.replace("_", " "))


def source_suffix(profiles, source):
    if len(profiles) == 1 and source == OBSERVATIONAL_SOURCE:
        return ""
    return f": {source_label(source)}"


def source_only_label(profiles, source):
    if len(profiles) == 1 and source == OBSERVATIONAL_SOURCE:
        return None
    return source_label(source)


def source_style(source, index=0, mode="color"):
    if mode == "black":
        color = "black"
    elif source in SOURCE_COLORS:
        color = SOURCE_COLORS[source]
    else:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color = colors[index % len(colors)]

    return {"color": color, "linestyle": "-"}


def finite_xy(rows, x_name, y_name, positive=False):
    points = [
        (
            row[x_name],
            row[y_name]
            if math.isfinite(row[y_name]) and (not positive or row[y_name] > 0)
            else math.nan,
        )
        for row in rows
        if math.isfinite(row[x_name])
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
            if not math.isfinite(x) or not math.isfinite(y):
                continue
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


def add_reference(
    axis,
    plotted_points,
    value,
    label=None,
    color="0.45",
    style=":",
    linewidth=0.9,
):
    if value is None or not math.isfinite(value):
        return
    x_limits = axis.get_xlim()
    axis.axhline(
        value,
        color=color,
        linestyle=style,
        linewidth=linewidth,
        label=label,
    )
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
        axis.plot(
            radius,
            values,
            label=source_only_label(profiles, source),
            **style,
        )
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
    if rotation["kind"] == "tabulated":
        raise ValueError("Tabulated rotation curves must be read from profiles.csv")
    raise ValueError(
        f"Unknown rotation type {rotation['kind']!r} for {rotation['source']}"
    )


def plot_rotation_curves(
    profiles, rotations, sparc, metadata, output_directory, show_title=True
):
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
            color="black",
            ecolor="black",
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

    for index, rotation in enumerate(rotations):
        if rotation["kind"] == "tabulated":
            curve_radius, velocities = finite_xy(
                profiles[rotation["source"]], "R_kpc", "v_c_kms"
            )
        else:
            curve_radius = [maximum_radius * point / 499.0 for point in range(500)]
            velocities = [
                rotation_velocity(rotation, radius) for radius in curve_radius
            ]
        style = source_style(rotation["source"], index, "black")
        axis.plot(
            curve_radius,
            velocities,
            label=(
                "SPARC spline"
                if rotation["source"] == OBSERVATIONAL_SOURCE
                else source_label(rotation["source"])
            ),
            zorder=3,
            **style,
        )
        plotted_points.append((curve_radius, velocities))

    title = (
        galaxy_label(metadata["galaxy"])
        if show_title and metadata is not None
        else None
    )
    finish_figure(
        figure,
        axis,
        output_directory / "v_c.pdf",
        r"$v_c\;[\mathrm{km\,s^{-1}}]$",
        plotted_points,
        legend=False,
        title=title,
    )


def plot_surface_rates(profiles, output_directory, sfr_diagnostics=None):
    figure, axis = plt.subplots()
    plotted_points = []
    values = finite_values(profiles, "Sigmadot_star_Msun_yr_kpc2")
    values += finite_values(profiles, "Sigmadot_land_Msun_yr_kpc2")
    if sfr_diagnostics is not None:
        values += [
            row["Sigmadot_star_obs_Msun_yr_kpc2"]
            for row in sfr_diagnostics
            if row["row_type"] == "measurement"
            and math.isfinite(row["Sigmadot_star_obs_Msun_yr_kpc2"])
        ]
    log_y = bool(values) and all(value > 0 for value in values)

    if sfr_diagnostics is not None:
        measurement_styles = {
            "Leroy2008": {
                "label": "Leroy et al. (2008)",
                "markerfacecolor": "black",
            },
            "Bigiel2010_crosscal": {
                "label": "Bigiel et al. (2010)",
                "markerfacecolor": "none",
            },
        }
        for source, marker_style in measurement_styles.items():
            points = [
                (
                    row["R_kpc"],
                    row["Sigmadot_star_obs_Msun_yr_kpc2"] * SURFACE_RATE_PLOT_SCALE,
                    row["e_Sigmadot_star_obs_Msun_yr_kpc2"] * SURFACE_RATE_PLOT_SCALE,
                )
                for row in sfr_diagnostics
                if row["row_type"] == "measurement"
                and row["source"] == source
                and math.isfinite(row["R_kpc"])
                and math.isfinite(row["Sigmadot_star_obs_Msun_yr_kpc2"])
                and math.isfinite(row["e_Sigmadot_star_obs_Msun_yr_kpc2"])
                and (
                    not log_y
                    or row["Sigmadot_star_obs_Msun_yr_kpc2"] > 0
                )
            ]
            if not points:
                continue
            radius = [point[0] for point in points]
            star_observed = [point[1] for point in points]
            errors = [point[2] for point in points]
            axis.errorbar(
                radius,
                star_observed,
                yerr=errors,
                fmt="o",
                markersize=3.0,
                markerfacecolor=marker_style["markerfacecolor"],
                markeredgecolor="black",
                markeredgewidth=0.7,
                color="black",
                ecolor="0.45",
                elinewidth=0.6,
                capsize=1.2,
                label=marker_style["label"],
                zorder=2,
            )
            plotted_points.append((radius, star_observed))

    first_rows = next(iter(profiles.values()))
    radius, star = finite_xy(
        first_rows, "R_kpc", "Sigmadot_star_Msun_yr_kpc2", positive=log_y
    )
    star = [value * SURFACE_RATE_PLOT_SCALE for value in star]
    axis.plot(
        radius,
        star,
        color="black",
        linestyle="-",
        label=r"$\dot{\Sigma}_\star$",
        zorder=3,
    )
    plotted_points.append((radius, star))

    for index, (source, rows) in enumerate(profiles.items()):
        radius, landing = finite_xy(
            rows, "R_kpc", "Sigmadot_land_Msun_yr_kpc2", positive=log_y
        )
        landing = [value * SURFACE_RATE_PLOT_SCALE for value in landing]
        style = source_style(source, index, "color_dashed")
        style["color"] = "red"
        axis.plot(
            radius,
            landing,
            label=r"$\dot{\Sigma}_{\rm land}$" + source_suffix(profiles, source),
            **style,
        )
        plotted_points.append((radius, landing))

    add_reference(axis, plotted_points, 0.0 if not log_y else None)
    finish_figure(
        figure,
        axis,
        output_directory / "surface_rates.pdf",
        SURFACE_RATE_LABEL,
        plotted_points,
        log_y=log_y,
    )


def plot_cumulative_landing(profiles, summary, output_directory):
    evolving = any(row["profile_type"] == "evolving" for row in summary)
    targets = {
        row["Mdot_land_Msun_yr"]
        for row in summary
        if math.isfinite(row["Mdot_land_Msun_yr"]) and (evolving or row["mu"] != 0.0)
    }
    references = []
    for target in sorted(targets):
        references.append(
            {
                "value": target,
                "label": ("total landing" if evolving else "landing target") + rf" $={target:.3g}$",
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
            label=r"$t_{\rm inflow}$" + source_suffix(profiles, source),
            **style,
        )
        plotted_points.append((radius, inflow))

    finish_figure(
        figure,
        axis,
        output_directory / "t.pdf",
        r"$t\;[\mathrm{Gyr}]$",
        plotted_points,
        log_y=True,
    )


def plot_angular_momentum(profiles, rotations, summary, output_directory):
    output_path = output_directory / "j.pdf"
    profile_source = "SPARC_spline" if "SPARC_spline" in profiles else next(
        (rotation["source"] for rotation in rotations if rotation["kind"] == "flat"),
        next(iter(profiles), None),
    )
    if profile_source is None:
        raise ValueError("No angular-momentum profile is available")

    rows = profiles[profile_source]
    angular_radius = "R_j_kpc" if math.isfinite(rows[0].get("R_j_kpc", math.nan)) else "R_kpc"
    figure, axis = plt.subplots()
    plotted_points = []
    nuclear_values = [row["j_nucl_kpc_kms"] for row in rows if math.isfinite(row.get("j_nucl_kpc_kms", math.nan))] or [
        row["j_land_kpc_kms"] / row["j_land_over_j_nucl"]
        for row in rows
        if math.isfinite(row["j_land_kpc_kms"])
        and math.isfinite(row["j_land_over_j_nucl"])
        and row["j_land_over_j_nucl"] != 0.0
    ]
    if not nuclear_values or nuclear_values[0] == 0.0:
        raise ValueError("Cannot normalize angular momentum by j_nuc")
    j_nuc = nuclear_values[0]
    if not all(math.isclose(value, j_nuc, rel_tol=1.0e-9) for value in nuclear_values):
        raise ValueError("j_nuc is not constant across the angular-momentum profile")

    mixing_mu_values = {
        row["mu"]
        for row in summary
        if row["source"] == profile_source and math.isfinite(row["mu"])
    }
    if len(mixing_mu_values) != 1:
        raise ValueError("Expected exactly one configured mu for the selected profile")
    mixing_mu = next(iter(mixing_mu_values))
    if math.isclose(mixing_mu, -1.0):
        raise ValueError("Cannot compute j_land,CGM for mu = -1")

    quantities = (
        ("j_disk_kpc_kms", r"$j(R)/j_{\rm nuc}$", "tab:blue"),
        (
            "j_land_kpc_kms",
            r"$j_{\rm land}(R)/j_{\rm nuc}$" if rows[0]["model"] in {"disk-evolution", "forward-time"} else r"$j_{\rm land,req}(R)/j_{\rm nuc}$",
            "tab:orange",
        ),
    )
    for column, label, color in quantities:
        radius, values = finite_xy(rows, angular_radius, column)
        values = [value / j_nuc for value in values]
        axis.plot(radius, values, color=color, linestyle="-", label=label)
        plotted_points.append((radius, values))

    radius, cgm = finite_xy(rows, angular_radius, "j_CGM_kpc_kms")
    mixing = [
        (j_nuc + mixing_mu * value) / ((1.0 + mixing_mu) * j_nuc)
        for value in cgm
    ]
    axis.plot(
        radius,
        mixing,
        color="tab:green",
        linestyle="--",
        label=r"$j_{\rm land,CGM}(R)/j_{\rm nuc}$",
    )
    plotted_points.append((radius, mixing))

    cgm = [value / j_nuc for value in cgm]
    axis.plot(
        radius,
        cgm,
        color="tab:red",
        linestyle="-",
        label=r"$j_{\rm CGM}(R)/j_{\rm nuc}$",
    )
    plotted_points.append((radius, cgm))

    finish_figure(
        figure,
        axis,
        output_path,
        r"$j/j_{\rm nuc}$",
        plotted_points,
    )


def plot_metallicity(profiles, summary, output_directory):
    figure, axis = plt.subplots()
    plotted_points = []
    line_width = plt.rcParams["lines.linewidth"]
    for index, (source, rows) in enumerate(profiles.items()):
        label_suffix = source_suffix(profiles, source)
        radius, metallicity = finite_xy(rows, "R_kpc", "Z")
        axis.plot(
            radius,
            metallicity,
            color="black",
            linestyle="-",
            linewidth=line_width,
            label=r"$Z(R)$" + label_suffix,
        )
        plotted_points.append((radius, metallicity))

        radius, required = finite_xy(rows, "R_kpc", "Z_land_required")
        axis.plot(
            radius,
            required,
            color="red",
            linestyle="--",
            linewidth=line_width,
            label=r"$Z_{\rm land}^{\rm req}(R)$" + label_suffix,
        )
        plotted_points.append((radius, required))

    for value in sorted(
        {row["Z_nucl"] for row in summary if math.isfinite(row["Z_nucl"])}
    ):
        add_reference(
            axis,
            plotted_points,
            value,
            label=r"$Z_{\rm nuc}$",
            color="blue",
            style="--",
            linewidth=line_width,
        )
    for value in sorted(
        {row["Z_CGM"] for row in summary if math.isfinite(row["Z_CGM"])}
    ):
        add_reference(
            axis,
            plotted_points,
            value,
            label=r"$Z_{\rm CGM}$",
            color="green",
            style="--",
            linewidth=line_width,
        )

    finish_figure(
        figure,
        axis,
        output_directory / "Z.pdf",
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
        angular_radius = "R_j_kpc" if math.isfinite(rows[0].get("R_j_kpc", math.nan)) else "R_kpc"
        radius, mu_j = finite_xy(rows, angular_radius, "mu_j")
        axis.plot(
            radius,
            mu_j,
            label=r"$\mu_j$" + source_suffix(profiles, source),
            **style,
        )
        plotted_points.append((radius, mu_j))

        radius, mu_z = finite_xy(rows, "R_kpc", "mu_Z")
        axis.plot(
            radius,
            mu_z,
            color=style["color"],
            linestyle=":",
            label=r"$\mu_Z$" + source_suffix(profiles, source),
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
        output_directory / "mu.pdf",
        r"$\mu$",
        plotted_points,
    )


def render_model_run(model_output, output_directory, show_title=True):
    profiles, summary, rotations, metadata, sparc, sfr_diagnostics = load_model_run(
        model_output
    )
    if metadata is not None:
        require_paper_galaxy(metadata["galaxy"])

    output_directory.mkdir(parents=True, exist_ok=True)

    plot_rotation_curves(
        profiles,
        rotations,
        sparc,
        metadata,
        output_directory,
        show_title=show_title,
    )
    plot_profile_column(
        profiles,
        output_directory,
        "v_R.pdf",
        "v_R_kms",
        r"$v_R\;[\mathrm{km\,s^{-1}}]$",
    )
    plot_surface_rates(profiles, output_directory, sfr_diagnostics)
    plot_cumulative_landing(profiles, summary, output_directory)
    plot_profile_column(
        profiles,
        output_directory,
        "Mdot_acc.pdf",
        "Mdot_acc_Msun_yr",
        r"$\dot{M}_{\rm acc}\;[M_\odot\,\mathrm{yr}^{-1}]$",
    )
    plot_timescales(profiles, output_directory)
    plot_angular_momentum(profiles, rotations, summary, output_directory)
    plot_metallicity(profiles, summary, output_directory)
    plot_mu(profiles, summary, output_directory)


def load_time_profiles(model_output):
    """Read the exported times for one disk evolution or reconstruction."""
    with (model_output / "snapshots.csv").open(newline="") as stream:
        snapshots = sorted(csv.DictReader(stream), key=lambda row: float(row["t_Myr"]))
    if not snapshots:
        raise ValueError("snapshots.csv contains no saved times")
    runs = []
    burst_peaks = set()
    for snapshot in snapshots:
        directory = model_output / snapshot["directory"]
        profiles, summary, *_ = load_model_run(directory)
        if len(profiles) != 1 or len(summary) != 1 or summary[0]["profile_type"] != "evolving":
            raise ValueError("Time overlays require one evolving disk profile per snapshot")
        runs.append((float(snapshot["t_Myr"]), next(iter(profiles.values())), summary[0]))
        with (directory / "parameters.csv").open(newline="") as stream:
            parameters = {row["parameter"]: row["value"] for row in csv.DictReader(stream)}
        burst_peaks.add(float(parameters["t_b_Myr"])
                        if parameters.get("launch_history") in {
                            "baseline_plus_gaussian", "baseline_plus_periodic_gaussians"
                        } else None)
    if len(burst_peaks) != 1:
        raise ValueError("Snapshots must share the same burst time reference")
    models = {rows[0]["model"] for _, rows, _ in runs}
    if len(models) != 1:
        raise ValueError("Plot disk evolution and reconstruction in separate snapshot directories")
    evolution = models.pop() in {"disk-evolution", "forward-time"}
    title = "Disk evolution" if evolution else "Reconstruction"
    return runs, title, burst_peaks.pop()


def same_time_curve(first, other):
    """Match radii and missing values, allowing numerical roundoff in ordinates."""
    return first[0] == other[0] and len(first[1]) == len(other[1]) and all(
        (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-10, abs_tol=0.0)
        for a, b in zip(first[1], other[1])
    )


def mark_launch_radius(axis, summary):
    radius = summary["R_nucl_kpc"]
    lower, upper = axis.get_xlim()
    if lower <= radius <= upper:
        axis.plot([radius, radius], [-0.01, 0.025], transform=axis.get_xaxis_transform(),
                  color="black", linewidth=0.8, scalex=False, scaley=False, clip_on=False)
        axis.annotate(r"$R_{\rm nuc}$", xy=(radius, -0.01), xycoords=axis.get_xaxis_transform(),
                      xytext=(0, -2), textcoords="offset points",
                      ha="center", va="top", fontsize=7, annotation_clip=False)


def load_evolution_history(model_output):
    with (model_output / "snapshots.csv").open(newline="") as stream:
        snapshot = next(csv.DictReader(stream))
    directory = model_output / snapshot["directory"]
    with (directory / "snapshot.csv").open(newline="") as stream:
        provenance = {row["parameter"]: row["value"] for row in csv.DictReader(stream)}
    source = Path(provenance["input_directory"])
    with (directory / "parameters.csv").open(newline="") as stream:
        parameters = {row["parameter"]: row["value"] for row in csv.DictReader(stream)}
    history = read_rows(source / "landing_history.csv")
    initial_gas = {}
    masses = []
    # Aggregate one saved time at a time rather than retaining the full disk CSV.
    with (source / "gas_evolution.csv").open(newline="") as stream:
        for time, group in groupby(csv.DictReader(stream), key=lambda row: row["t_Myr"]):
            rows = list(group)
            if not initial_gas:
                initial_gas = {(float(row["R_lo_kpc"]), float(row["R_hi_kpc"])):
                               float(row["Sigma_g_Msun_kpc2"]) for row in rows}
            enclosed = [sum(float(row["M_g_Msun"]) for row in rows
                            if float(row["R_hi_kpc"]) <= radius)
                        for radius in (1.0, 5.0, math.inf)]
            masses.append((float(time), enclosed))
    return history, masses, initial_gas, parameters


def add_evolution_diagnostics(runs, initial_gas):
    for _, rows, summary in runs:
        for row in rows:
            area = math.pi * (row["R_hi_kpc"]**2 - row["R_lo_kpc"]**2)
            gas, Z = row["Sigma_g_Msun_kpc2"], row["Z"]
            landing, star = row["Sigmadot_land_Msun_yr_kpc2"], row["Sigmadot_star_Msun_yr_kpc2"]
            transport = (row["mass_flux_inner_Msun_yr"] - row["mass_flux_outer_Msun_yr"]) / area
            metal_transport = (row["metal_flux_inner_Msun_yr"] - row["metal_flux_outer_Msun_yr"]) / area
            initial = initial_gas[(row["R_lo_kpc"], row["R_hi_kpc"])]
            row["gas_density"] = gas / 1e6  # Msun/pc^2
            row["gas_ratio"] = gas / initial if initial > 0 else math.nan
            row["gas_landing"] = landing * SURFACE_RATE_PLOT_SCALE
            row["gas_star"] = -star * SURFACE_RATE_PLOT_SCALE
            row["gas_transport"] = transport * SURFACE_RATE_PLOT_SCALE
            row["gas_net"] = row["dSigma_g_dt_Msun_kpc2_Myr"] / 1e3
            # dZ/dt = (dSigma_Z/dt - Z*dSigma_g/dt)/Sigma_g,
            # retaining the solver's discrete face fluxes in both terms.
            row["Z_landing"] = (row["Z_land_mixing"] - Z) * landing / gas * 1e9 if gas > 0 else math.nan
            row["Z_star"] = summary["yield_y"] * star / gas * 1e9 if gas > 0 else math.nan
            row["Z_transport"] = (metal_transport - Z * transport) / gas * 1e9 if gas > 0 else math.nan
            row["Z_net"] = row["dZ_dt_per_Myr"] * 1e3


def plot_evolution_budgets(runs, colors, time_handles, epoch_title, output_directory, fiducial,
                           prefixes=("gas", "Z")):
    dotted, dashed = (0, (0.1, 3.6)), (0, (4.0, 3.0))
    for prefix in prefixes:
        ylabel = SURFACE_RATE_LABEL if prefix == "gas" else r"$\dot{Z}\;[\mathrm{Gyr}^{-1}]$"
        figure, axes = plt.subplots(2, 1, figsize=(3.5, 4.6), sharex=True)
        panels = [[("landing", "Landing", "-"), ("star", "Star formation", dashed)],
                  [("net", "Net", "-"), ("transport", "Transport", dotted)]]
        for axis, quantities in zip(axes, panels):
            handles = []
            for suffix, label, style in quantities:
                curves = [finite_xy(rows, "R_kpc", f"{prefix}_{suffix}") for _, rows, _ in runs]
                fixed = len(curves) > 1 and all(same_time_curve(curves[0], curve) for curve in curves[1:])
                width = 1.3 if style == dotted else 1.0
                for i, (radius, values) in enumerate(curves[:1] if fixed else curves):
                    axis.plot(radius, values, color="black" if fixed else colors[i],
                              linestyle=style, linewidth=width, dash_capstyle="round")
                handles.append(Line2D([], [], color="black", linestyle=style, linewidth=width,
                                      dash_capstyle="round", label=label))
            axis.set_ylabel(ylabel)
            axis.legend(handles=handles, loc="best", **TIME_LEGEND_STYLE)
            if fiducial:
                axis.set_xlim(0, FIDUCIAL_PLOT_R_MAX_KPC)
                axis.set_xticks([0, 5, 10, 15, 20])
        axes[-1].set_xlabel(r"$R\;[\mathrm{kpc}]$")
        mark_launch_radius(axes[-1], runs[0][2])
        figure.legend(handles=time_handles, loc="upper center", bbox_to_anchor=(0.59, 0.995),
                      ncol=min(5, len(runs)), title=epoch_title, title_fontsize=8, **TIME_LEGEND_STYLE)
        figure.subplots_adjust(left=0.20, right=0.96, bottom=0.12, top=0.89, hspace=0.12)
        figure.savefig(output_directory / f"{prefix}_budget.pdf", bbox_inches=None)
        plt.close(figure)


def plot_evolution_histories(history, masses, parameters, burst_peak, output_directory):
    period = float(parameters.get("burst_period_Myr", 0.0))
    relative = burst_peak is not None and period == 0.0
    reference = burst_peak if relative else 0.0
    xlabel = r"$t-t_{\rm peak}\;[\mathrm{Myr}]$" if relative else r"$t\;[\mathrm{Myr}]$"
    time = [row["t_Myr"] - reference for row in history]
    mixed = parameters["landing_mass_origin"] == "nuclear_plus_cgm"
    factor = 1.0 + float(parameters["mu"]) if mixed else 1.0
    curves = [([row["Mdot_launch_Msun_yr"] for row in history], r"$\dot{M}_{\rm launch}$", (0, (0.1, 3.6)), 1.3),
              ([row["Mdot_land_Msun_yr"] / factor for row in history], r"$\dot{M}_{\rm ret,nuc}$", (0, (4.0, 3.0)), 1.0)]
    if mixed:
        curves.append(([row["Mdot_land_Msun_yr"] for row in history], r"$\dot{M}_{\rm land}$", "-", 1.0))
    recurrent = period > 0.0 and time[-1] - time[0] > 4.0 * period
    if recurrent:
        figure, (axis, detail) = plt.subplots(2, 1, figsize=(3.5, 4.6), sharey=True)
    else:
        figure, axis = plt.subplots(figsize=MODEL_FIGURE_SIZE)
    for values, label, style, width in curves:
        axis.plot(time, values, color="black", linestyle=style, linewidth=width,
                  dash_capstyle="round", label=label)
    axis.set_xlim(time[0], time[-1])
    axis.set_ylim(bottom=0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r"$\dot{M}\;[M_\odot\,\mathrm{yr}^{-1}]$")
    if recurrent:
        # Separate the absolute history and one resolved cycle; an inset would hide
        # the dense repeated signal and its legend.
        cycle_start = burst_peak + math.ceil((time[0] - burst_peak) / period) * period
        for values, _, style, width in curves:
            selected = [(t - cycle_start, value) for t, value in zip(time, values)
                        if cycle_start <= t <= cycle_start + period]
            detail.plot(*zip(*selected), color="black", linestyle=style, linewidth=width,
                        dash_capstyle="round")
        detail.set_xlim(0.0, period)
        detail.set_xlabel(r"Time since peak $[\mathrm{Myr}]$")
        detail.set_ylabel(r"$\dot{M}\;[M_\odot\,\mathrm{yr}^{-1}]$")
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.59, 0.98),
                      ncol=3, **TIME_LEGEND_STYLE)
    else:
        axis.legend(loc="upper right", **TIME_LEGEND_STYLE)
    if not recurrent and burst_peak is not None and time[-1] > 300:
        inset = axis.inset_axes([0.43, 0.35, 0.52, 0.36])
        for values, _, style, width in curves:
            selected = [(t, value) for t, value in zip(time, values) if -30 <= t <= 100]
            inset.plot(*zip(*selected), color="black", linestyle=style, linewidth=width,
                       dash_capstyle="round")
        inset.set_xlim(-30, 100)
        inset.set_ylim(bottom=0)
        inset.tick_params(labelsize=7)
    if recurrent:
        figure.subplots_adjust(left=0.20, right=0.94, bottom=0.12, top=0.91, hspace=0.55)
    else:
        figure.subplots_adjust(left=0.20, right=0.94, bottom=0.15, top=0.87)
    figure.savefig(output_directory / "launch_landing_history.pdf", bbox_inches=None)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=MODEL_FIGURE_SIZE)
    time = [t - reference for t, _ in masses]
    for index, label in enumerate((r"$R<1\;\mathrm{kpc}$", r"$R<5\;\mathrm{kpc}$", "Total")):
        values = [enclosed[index] for _, enclosed in masses]
        axis.plot(time, values, color="black", linewidth=1.0)
        axis.annotate(label, xy=(time[-1], values[-1]), xytext=(-4, 5),
                      textcoords="offset points", ha="right", va="bottom", fontsize=8)
    axis.set_xlim(time[0], time[-1])
    axis.set_yscale("log")
    axis.margins(y=0.15)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r"$M_{\rm g}(<R)\;[M_\odot]$")
    figure.subplots_adjust(left=0.20, right=0.94, bottom=0.15, top=0.87)
    figure.savefig(output_directory / "enclosed_gas_history.pdf", bbox_inches=None)
    plt.close(figure)


def epoch_legend(times, title, signed=False):
    colors = ["#D62728", "#0072B2", "#2CA02C", "#E69F00", "#7B2CBF"] if len(times) <= 5 else [
        plt.get_cmap("viridis")(0.1 + 0.8*i/(len(times)-1)) for i in range(len(times))
    ]
    labels = [f"{t:+g}" if signed and t != 0.0 else f"{t:g}" for t in times]
    handles = [Line2D([], [], color=colors[i], linewidth=1.0, label=rf"${label}$")
               for i, label in enumerate(labels)]
    return colors, handles, title


def render_time_profiles(model_output, output_directory, phase_snapshots=None):
    """One single-column figure per quantity, with saved times overlaid."""
    runs, title, burst_peak = load_time_profiles(model_output)
    history, masses, initial_gas, parameters = load_evolution_history(model_output)
    add_evolution_diagnostics(runs, initial_gas)
    fiducial = all(summary["galaxy"] == "illustrative" for _, _, summary in runs)
    if fiducial:
        # Limit plotted samples so both axes scale to the displayed disk.
        runs = [(t, [row for row in rows if row["R_kpc"] <= FIDUCIAL_PLOT_R_MAX_KPC], summary)
                for t, rows, summary in runs]
    period = float(parameters.get("burst_period_Myr", 0.0))
    relative = burst_peak is not None and period == 0.0
    epoch_times = [t - burst_peak if relative else t for t, _, _ in runs]
    epoch_title = r"$t-t_{\rm peak}\;[\mathrm{Myr}]$" if relative else r"$t\;[\mathrm{Myr}]$"
    age_runs = runs
    age_style = epoch_legend(epoch_times, epoch_title, signed=relative)
    phase_runs, phase_style = age_runs, age_style
    if phase_snapshots is not None:
        if period <= 0.0:
            raise ValueError("Phase snapshots require a recurrent burst history")
        phase_runs, phase_title, phase_peak = load_time_profiles(phase_snapshots)
        with (phase_snapshots / "snapshots.csv").open(newline="") as stream:
            first = next(csv.DictReader(stream))
        with (phase_snapshots / first["directory"] / "parameters.csv").open(newline="") as stream:
            phase_parameters = {row["parameter"]: row["value"] for row in csv.DictReader(stream)}
        if phase_title != title or phase_peak != burst_peak or phase_parameters != parameters:
            raise ValueError("Age and phase snapshots must use the same evolution parameters and model")
        add_evolution_diagnostics(phase_runs, initial_gas)
        if fiducial:
            phase_runs = [(t, [row for row in rows if row["R_kpc"] <= FIDUCIAL_PLOT_R_MAX_KPC], summary)
                          for t, rows, summary in phase_runs]
        phases = [(t - burst_peak) % period for t, _, _ in phase_runs]
        if len(set(phases)) != len(phases):
            raise ValueError("Select distinct phases within the burst cycle")
        phase_style = epoch_legend(phases, r"Time since peak $[\mathrm{Myr}]$")
    # Short strokes with round caps render as dots, with visible gaps even
    # when the figure is reduced. Metallicity and timescales use dashes.
    dotted = (0, (0.1, 3.6))
    wide_dotted = (0, (0.1, 5.2))
    dashed = (0, (4.0, 3.0))
    short_dashed = (0, (2.0, 3.0))
    long_dashed = (0, (7.0, 3.0))
    # Surface rates are converted for display. Angular columns retain face radii.
    specifications = [
        ("Sigma_g", r"$\Sigma_{\rm g}\;[M_\odot\,\mathrm{pc}^{-2}]$", [("gas_density", None, "-")]),
        ("Sigma_g_change", r"$\Sigma_{\rm g}/\Sigma_{\rm g,0}$", [("gas_ratio", None, "-")]),
        ("v_c", r"$v_c\;[\mathrm{km\,s^{-1}}]$", [("v_c_kms", None, "-")]),
        ("v_R", r"$v_R\;[\mathrm{km\,s^{-1}}]$", [("v_R_kms", None, "-")]),
        ("surface_rates", SURFACE_RATE_LABEL, [
            ("Sigmadot_land_Msun_yr_kpc2", r"$\dot{\Sigma}_{\rm land}$", "-"),
            ("Sigmadot_star_Msun_yr_kpc2", r"$\dot{\Sigma}_\star$", dotted)]),
        ("cumulative_landing_rate", r"$\dot{M}_{\rm land}(<R)\;[M_\odot\,\mathrm{yr}^{-1}]$", [
            ("cumulative_landing_Msun_yr", None, "-")]),
        ("Mdot_acc", r"$\dot{M}_{\rm acc}\;[M_\odot\,\mathrm{yr}^{-1}]$", [("Mdot_acc_Msun_yr", None, "-")]),
        ("t", r"$t\;[\mathrm{Gyr}]$", [
            ("t_inflow_Gyr", r"$t_{\rm inflow}$", "-"),
            ("t_depletion_Gyr", r"$t_{\rm depletion}$", dashed)]),
        ("j", r"$j/j_{\rm nuc}$", [
            ("j_land_kpc_kms", r"$j_{\rm land}$", "-")]),
        ("Z", r"$Z$", [("Z", r"$Z_{\rm disk}$", "-"),
                          ("Z_land_required", r"$Z_{\rm land}$", dashed)]),
        ("mu", r"$\mu$", [("mu_j", r"$\mu_j$", "-"), ("mu_Z", r"$\mu_Z$", dotted)]),
    ]
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, ylabel, quantities in specifications:
        use_phase = name in {"surface_rates", "cumulative_landing_rate", "Mdot_acc"}
        runs = phase_runs if use_phase else age_runs
        colors, time_handles, epoch_title = phase_style if use_phase else age_style
        figure, axis = plt.subplots(figsize=MODEL_FIGURE_SIZE)
        if name == "j":
            label = r"$j_{\rm land}$" if title == "Disk evolution" else r"$j_{\rm land,req}$"
            quantities = [("j_land_kpc_kms", label, "-")]
        quantity_handles = [Line2D([], [], color="black", linestyle=style,
                                   linewidth=1.3 if style in (dotted, wide_dotted) else 1.0,
                                   dash_capstyle="round", label=label)
                            for _, label, style in quantities if label]
        all_values = []
        has_time_variation = False
        for column, curve_label, style in quantities:
            curves = []
            for _, rows, _ in runs:
                radius_column = "R_j_kpc" if name == "j" or column == "mu_j" else "R_kpc"
                radius, values = finite_xy(rows, radius_column, column, positive=name in {"t", "surface_rates", "Sigma_g"})
                if name == "j":
                    values = [value/rows[0]["j_nucl_kpc_kms"] for value in values]
                elif name == "surface_rates":
                    values = [value * SURFACE_RATE_PLOT_SCALE for value in values]
                curves.append((radius, values))
            fixed = (len(curves) > 1 or column in {"v_c_kms", "Sigmadot_star_Msun_yr_kpc2"}) and all(
                same_time_curve(curves[0], curve) for curve in curves[1:]
            )
            has_time_variation |= not fixed
            for index, (radius, values) in enumerate(curves[:1] if fixed else curves):
                axis.plot(radius, values, color="black" if fixed else colors[index], linestyle=style,
                          linewidth=1.3 if style in (dotted, wide_dotted) else 1.0,
                          dash_capstyle="round")
                all_values.extend(values)
                if name == "j" and (fixed or index == len(curves)-1):
                    points = [(r, value) for r, value in zip(radius, values) if math.isfinite(value)]
                    if points:
                        axis.annotate(curve_label, xy=points[-1], xytext=(-3, -9),
                                      textcoords="offset points", ha="right", va="top", fontsize=8)
        if name == "cumulative_landing_rate":
            totals = [summary["Mdot_land_Msun_yr"] for _, _, summary in runs]
            fixed = len(totals) > 1 and all(math.isclose(totals[0], total, rel_tol=1e-10, abs_tol=0.0)
                                          for total in totals[1:])
            has_time_variation |= not fixed
            for index, total in enumerate(totals[:1] if fixed else totals):
                axis.axhline(total, color="black" if fixed else colors[index], linestyle=dotted,
                             linewidth=0.9, dash_capstyle="round")
            quantity_handles.append(Line2D([], [], color="black", linestyle=dotted, linewidth=0.9,
                                           dash_capstyle="round", label=r"$\dot{M}_{\rm land,tot}$"))

        # Fixed references are shared by the snapshots; verify this before
        # drawing them once instead of concealing any temporal variation.
        reference_columns = {
            "j": [("j_disk_kpc_kms", r"$j_{\rm disk}$", "-"),
                  ("j_CGM_kpc_kms", r"$j_{\rm CGM}$", "-")],
            "Z": [("Z_nucl", r"$Z_{\rm nuc}$", short_dashed), ("Z_CGM", r"$Z_{\rm CGM}$", long_dashed)],
            "mu": [("mu", rf"$\mu={runs[0][2]['mu']:g}$", wide_dotted)],
        }.get(name, [])
        for column, label, style in reference_columns:
            linewidth = 1.0 if style in (dotted, wide_dotted) else 0.8
            if name == "j":
                reference = [(row["R_j_kpc"], row[column]/row["j_nucl_kpc_kms"]) for row in runs[0][1]]
                if any([(row["R_j_kpc"], row[column]/row["j_nucl_kpc_kms"]) for row in rows] != reference
                       for _, rows, _ in runs[1:]):
                    raise ValueError(f"Expected a fixed {column} reference across snapshots")
                radius, values = zip(*reference)
                axis.plot(radius, values, color="black", linestyle=style, linewidth=linewidth,
                          dash_capstyle="round", zorder=1)
                anchor = int(0.8 * (len(radius)-1))
                axis.annotate(label, xy=(radius[anchor], values[anchor]), xytext=(0, 5),
                              textcoords="offset points", ha="right", va="bottom", fontsize=8)
            else:
                values = [summary[column] for _, _, summary in runs]
                if len(set(values)) != 1:
                    raise ValueError(f"Expected a fixed {column} reference across snapshots")
                axis.axhline(values[0], color="black", linestyle=style, linewidth=linewidth,
                             dash_capstyle="round", zorder=1)
            quantity_handles.append(Line2D([], [], color="black", linestyle=style, linewidth=linewidth,
                                           dash_capstyle="round", label=label))

        axis.set_box_aspect(1)
        axis.set_xlabel(r"$R\;[\mathrm{kpc}]$")
        axis.set_ylabel(ylabel)
        if fiducial:
            axis.set_xlim(0.0, FIDUCIAL_PLOT_R_MAX_KPC)
        if name in {"t", "surface_rates", "Sigma_g"}:
            axis.set_yscale("log")
        if quantity_handles and name != "j":
            axis.legend(handles=quantity_handles, loc="best", ncol=1, **TIME_LEGEND_STYLE)
        if name == "mu":
            pad_nearly_constant_y_axis(axis, all_values)
        if has_time_variation:
            figure.legend(handles=time_handles, loc="upper center", bbox_to_anchor=(0.59, 0.995),
                          ncol=min(5, len(runs)), title=epoch_title, title_fontsize=8, **TIME_LEGEND_STYLE)
        mark_launch_radius(axis, runs[0][2])
        figure.subplots_adjust(left=0.20, right=0.98, bottom=0.15, top=0.87)
        figure.savefig(output_directory / f"{name}.pdf", bbox_inches=None)
        plt.close(figure)
    plot_evolution_budgets(phase_runs, *phase_style, output_directory, fiducial, prefixes=("gas",))
    plot_evolution_budgets(age_runs, *age_style, output_directory, fiducial, prefixes=("Z",))
    plot_evolution_histories(history, masses, parameters, burst_peak, output_directory)


def main():
    parser = argparse.ArgumentParser(
        description="Plot steady model output or overlay selected times from an evolving disk"
    )
    parser.add_argument(
        "model_output",
        type=Path,
        help="directory containing profiles.csv/summary.csv/rotation_curves.csv, or a snapshots.csv manifest",
    )
    parser.add_argument(
        "plot_output",
        nargs="?",
        type=Path,
        help="plot directory; defaults to MODEL_OUTPUT/plots",
    )
    parser.add_argument(
        "--phase-snapshots",
        type=Path,
        help="recurrent-run snapshots at distinct burst phases, used for landing, mass flow and gas budgets",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="suppress the galaxy title on steady model plots",
    )
    arguments = parser.parse_args()
    output_directory = arguments.plot_output or arguments.model_output / "plots"
    manifest = arguments.model_output / "snapshots.csv"
    if manifest.is_file():
        with plt.rc_context(TIME_FIGURE_STYLE):
            render_time_profiles(arguments.model_output, output_directory, arguments.phase_snapshots)
    else:
        render_model_run(
            arguments.model_output,
            output_directory,
            show_title=not arguments.no_title,
        )
    print(f"Wrote model figures to: {output_directory}")


if __name__ == "__main__":
    main()
