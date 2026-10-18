#!/usr/bin/env python3
"""Render the Section 3 mixing tests from controlled core-model CSV runs.

The runner calculates the dynamics and chemistry. This file only selects the
paper cases, converts abundance units, and renders their exported profiles.
"""

from __future__ import annotations

from axis_labels import quantity_unit_label

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LogLocator, MaxNLocator
import numpy as np

from hershey_fonts import register_hershey_weight_aliases
from surface_rate_units import canonical_column, canonical_rate_row
from paper_galaxies import PAPER_GALAXIES, galaxy_label
from color_palette import color_cycle, palette_case_styles, BLUE, ORANGE
from observed_metallicity_spline import fit_metallicity_spline, plot_metallicity_spline, include_spline_gradient
from observed_disk_comparison import fit_legend_handle


register_hershey_weight_aliases()
import smplotlib


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SOLAR_METALLICITY = 0.0139
SOURCE = "SPARC_spline"
LANDING_TITLE = r"Mixed fountain, observed $v_c$, $\Sigma_g$, and $\dot\Sigma_{\rm SFR}$"
CASE_STYLES = {
    "unmixed": ("", "-", "unmixed"),
    "fiducial": ("", "-", "fiducial"),
    "mu0p5": ("", "-", r"$\mu=0.5$"),
    "mu2": ("", "-", r"$\mu=2$"),
    "beta0p5": ("", "-", r"$\beta=0.5$"),
}
CASE_STYLES = palette_case_styles(CASE_STYLES)
smplotlib.set_style(usetex=False, fontsize=10, figsize=(7.0, 5.5), dpi=144)
plt.rcParams.update({"axes.prop_cycle": color_cycle(),
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.top": False,
    "ytick.right": False,
    "legend.frameon": False,
    "lines.linewidth": 1.35,
    "pdf.fonttype": 42,
    "savefig.bbox": None,
})
LEGEND_STYLE = {
    "fontsize": 8.5,
    "frameon": False,
    "handlelength": 2.0,
    "handletextpad": 0.5,
    "columnspacing": 1.3,
    "labelspacing": 0.5,
}


def read_csv(path: Path, required=()) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - {canonical_column(name)[0] for name in (reader.fieldnames or ())}
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")
        rows = [canonical_rate_row(row) for row in reader]
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def values(rows, column):
    return np.asarray([float(row[column]) if row[column] else np.nan for row in rows])


def load_case(output_root: Path, galaxy: str, case: str):
    directory = output_root / galaxy / case
    rows = read_csv(directory / "profiles.csv", (
        "source", "R_kpc", "Sigmadot_land_Msun_Gyr_kpc2", "Mdot_acc_Msun_yr",
        "v_R_kms", "Z", "dZ_dR_per_kpc", "Sigmadot_star_Msun_Gyr_kpc2",
        "j_disk_kpc_kms", "j_land_kpc_kms", "j_CGM_kpc_kms",
        "Z_land_required", "mu_j", "mu_Z",
    ))
    rows = sorted((row for row in rows if row["source"] == SOURCE), key=lambda row: float(row["R_kpc"]))
    if not rows:
        raise ValueError(f"{directory}: no {SOURCE} profiles")
    summary = [row for row in read_csv(directory / "summary.csv", ("source", "mu")) if row["source"] == SOURCE]
    if len(summary) != 1:
        raise ValueError(f"{directory}: expected one {SOURCE} summary")
    return rows, summary[0]


def observed_abundances(output_root: Path):
    rows = read_csv(output_root / "observations.csv", ("galaxy", "calibration", "R_kpc", "Z", "e_Z_low", "e_Z_high"))
    return {
        galaxy: sorted((row for row in rows if row["galaxy"] == galaxy and row["calibration"] == "KK04"),
                       key=lambda row: float(row["R_kpc"]))
        for galaxy in PAPER_GALAXIES
    }


def plot_measurements(axis, rows, label=None, *, zorder=3):
    if not rows:
        raise ValueError("No SINGS KK04 measurements for the requested galaxy")
    axis.errorbar(values(rows, "R_kpc"), values(rows, "Z") / SOLAR_METALLICITY,
                  yerr=np.vstack((values(rows, "e_Z_low"), values(rows, "e_Z_high"))) / SOLAR_METALLICITY,
                  linestyle="none", marker="o", markersize=3.0,
                  markerfacecolor="black", markeredgecolor="black", markeredgewidth=0.5,
                  color="black", ecolor="black", elinewidth=0.5, capsize=1.5, capthick=0.5,
                  zorder=zorder, label=label)


def panel_label(axis, letter, compact=False):
    axis.text(0.025 if compact else 0.04, 1.006 if compact else 1.015,
              f"({letter})", transform=axis.transAxes, ha="left",
              va="bottom", fontsize=6.5 if compact else 10)


def save_figure(figure, path: Path):
    figure.savefig(path)
    plt.close(figure)
    print(f"Wrote {path}")


def predictive_comparison(output_root: Path, output_directory: Path, *, case_table=None,
                          model_root=None, title=None, output_stem="section3_predictive_mixing",
                          expected_mu=None, gradient_limits=None, angular_panel_top_right=True,
                          landing_limits=(1e4, 1e8), stellar_legend_anchor=(0.06, 0.10),
                          stellar_legend_loc="lower left"):
    from section3_mixed_parameter_scans import layout
    from section3_models import predictive_perturbations

    model_root = output_root if model_root is None else model_root
    if case_table is None:
        perturbations = predictive_perturbations("NGC2403")
        case_styles = dict(CASE_STYLES)
    else:
        case_rows = read_csv(case_table)
        perturbations = [row for row in case_rows if row["parameter"]]
        case_styles = {row["case"]: (row["color"], row.get("linestyle") or "-", row.get("label") or row["case"])
                       for row in case_rows if not row["parameter"]}
    symbols = {"mu": r"\mu", "beta": r"\beta", "eta": r"\eta", "yield_y": "y", "Z_CGM": r"Z_{\rm CGM}",
               "Z_nucl": r"Z_{\rm nuc}", "Z_outer_boundary": r"Z_{\rm out}",
               "R_nucl_kpc": r"R_{\rm nuc}", "R_in_kpc": r"R_{\rm in}",
               "Mdot_nuc_Msun_yr": r"\dot M_{\rm nuc}"}
    for row in perturbations:
        unit = (r"\,\mathrm{kpc}" if row["parameter"] in ("R_nucl_kpc", "R_in_kpc") else
                r"\,M_\odot\,\mathrm{yr}^{-1}" if row["parameter"] == "Mdot_nuc_Msun_yr" else "")
        label = rf"${symbols[row['parameter']]}={float(row['value']):g}{unit}$"
        case_styles[row["case"]] = (row["color"], row.get("linestyle") or "-", label)
    case_styles = palette_case_styles(case_styles)
    cases = {case: load_case(model_root, "NGC2403", case) for case in case_styles}
    source_values = {}
    for case in cases:
        parameters = {row["name"]: row["value"] for row in read_csv(
            model_root / "inputs" / "NGC2403" / case / "parameters.csv", ("name", "value"))}
        if expected_mu is not None and float(parameters["mu"]) != expected_mu:
            raise ValueError(f"Expected mu={expected_mu} in {case}")
        source_values[case] = float(parameters["Mdot_nuc_Msun_yr"])
    source_perturbations = {row["case"]: float(row["value"]) for row in perturbations
                            if row["parameter"] == "Mdot_nuc_Msun_yr"}
    for case, source in source_values.items():
        expected_source = source_perturbations.get(case, source_values["fiducial"])
        if not np.isclose(source, expected_source, rtol=1e-12, atol=0):
            raise ValueError(f"Unexpected nuclear source in predictive case {case}")
    for row in perturbations:
        if row["parameter"] not in ("mu", "eta", "beta", "R_nucl_kpc", "R_in_kpc", "Mdot_nuc_Msun_yr"):
            for column in ("R_kpc", "Sigma_g_Msun_kpc2", "Sigmadot_star_Msun_Gyr_kpc2",
                           "Sigmadot_land_Msun_Gyr_kpc2", "Mdot_acc_Msun_yr", "v_R_kms",
                           "j_disk_kpc_kms", "j_land_kpc_kms"):
                np.testing.assert_allclose(values(cases[row["case"]][0], column),
                                           values(cases["fiducial"][0], column), rtol=1e-12, atol=0)
    figure, axes = plt.subplots(3, 2, figsize=(7.2, 4.5), sharex=True)
    panel_axes = axes.ravel()
    # Retain the quantity-indexed drawing code while placing angular deficit
    # above radial velocity in the shared comparison layout.
    axes = panel_axes[[0, 3, 2, 1, 4, 5]] if angular_panel_top_right else panel_axes
    columns = ("Sigmadot_land_Msun_Gyr_kpc2", "v_R_kms", "Mdot_acc_Msun_yr", "angular_deficit", "Z", "dZ_dR_per_kpc")
    labels = (
        r"$\dot\Sigma_{\rm land}\;[M_\odot\,\mathrm{Gyr}^{-1}\,\mathrm{kpc}^{-2}]$",
        r"$v_R\;[\mathrm{km}\,\mathrm{s}^{-1}]$",
        r"$\dot M_{\rm acc}\;[M_\odot\,\mathrm{yr}^{-1}]$",
        r"$1-j_{\rm land}/j$",
        r"$Z/Z_\odot$",
        r"$d\log_{10}Z/dR\;[\mathrm{dex}\,\mathrm{kpc}^{-1}]$",
    )
    handles = {}
    for case, (rows, _) in cases.items():
        radius = values(rows, "R_kpc")
        color, linestyle, label = case_styles[case]
        style = dict(color=color, linestyle=linestyle,
                     linewidth=2.0 if case == "fiducial" else 1.2 if case == "unmixed" else 0.8,
                     alpha=1.0,
                     zorder=2 if case == "fiducial" else 3 if case == "unmixed" else 4)
        handles[case] = Line2D([], [], label=label, **style)
        for axis, column in zip(axes.flat, columns):
            profile = (1 - values(rows, "j_land_kpc_kms") / values(rows, "j_disk_kpc_kms")
                       if column == "angular_deficit" else values(rows, column))
            if column == "Z":
                profile = profile / SOLAR_METALLICITY
            elif column == "dZ_dR_per_kpc":
                metallicity = values(rows, "Z")
                gradient = np.full_like(metallicity, np.nan)
                np.divide(profile, metallicity * np.log(10.0), out=gradient,
                          where=np.isfinite(metallicity) & (metallicity > 0))
                profile = gradient
            axis.plot(radius, profile, **style)
    fiducial = cases["fiducial"][0]
    stellar_style = dict(color="black", linestyle="--", linewidth=0.8, zorder=5)
    axes[0].plot(values(fiducial, "R_kpc"),
                 values(fiducial, "Sigmadot_star_Msun_Gyr_kpc2"),
                 **stellar_style, label=r"$\dot\Sigma_\star$")
    axes[0].legend(loc=stellar_legend_loc, bbox_to_anchor=stellar_legend_anchor, fontsize=8, frameon=False,
                   handlelength=1.7, handletextpad=0.4, borderpad=0.2)
    abundance_comparison = fit_metallicity_spline(observed_abundances(output_root)["NGC2403"])
    plot_metallicity_spline(axes[4:], abundance_comparison, "NGC 2403, SINGS KK04 spline")
    axes[0].set_yscale("log")
    axes[0].set_ylim(*landing_limits)
    axes[0].yaxis.set_major_locator(LogLocator(base=10,numticks=12))
    axes[0].yaxis.set_minor_locator(LogLocator(base=10,subs=range(2,10),numticks=100))
    axes[1].set_ylim(-0.3, 0)
    axes[1].set_yticks([-0.3, -0.2, -0.1, 0])
    axes[2].set_ylim(0, 0.1)
    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([0, 0.25, 0.5, 0.75, 1])
    axes[3].yaxis.set_minor_locator(AutoMinorLocator(5))
    axes[4].set_ylim(0, 3)
    gradient_lower = axes[5].dataLim.y0
    axes[5].set_ylim(*(gradient_limits if gradient_limits is not None else
                       (gradient_lower - axes[5].get_ymargin() * abs(gradient_lower), 0)))
    axes[5].yaxis.set_major_locator(MaxNLocator(4))
    include_spline_gradient(axes[5], abundance_comparison)
    axes[1].axhline(0, color="0.6", linewidth=0.5, zorder=0)
    for axis, label in zip(axes.flat, labels):
        axis.set_ylabel(quantity_unit_label(label), fontsize=9, labelpad=4)
        axis.set_xlim(0, 15)
        axis.tick_params(which="both", direction="in", top=True, right=True, labelsize=8)
        axis.tick_params(axis="x", which="both", pad=2)

    plot_cases = [dict(case=case, group="all", color=color, linestyle=linestyle, label=label)
                  for case, (color, linestyle, label) in case_styles.items()]
    # Pair each parameter vertically, retaining one ordinary handle per case.
    if case_table is None:
        legend_handles = [handles["unmixed"], handles["fiducial"], handles["mu0p5"], handles["mu2"],
                          handles["beta0p5"], handles["beta1"],
                          *(handles[row["case"]] for row in perturbations if row["parameter"] != "beta")]
    elif "unmixed" in handles:
        legend_handles = [handles[row["case"]] for row in case_rows]
    else:
        legend_handles = [handles["fiducial"], fit_legend_handle("KK04 spline"),
                          *(handles[row["case"]] for row in perturbations)]
    if case_table is None or "unmixed" in handles:
        legend_handles.extend([fit_legend_handle("KK04 spline"), Line2D([], [], linestyle="none", label=" ")])
    legend_spec = {"handles": legend_handles,
                   "labels": [handle.get_label() for handle in legend_handles],
                   "ncol": len(legend_handles)//2, "handlelength": 1.3, "fontsize": 8}
    layout(figure, panel_axes, plot_cases, vertical=True, panel_columns=2,
           title=title or f"{LANDING_TITLE} for NGC 2403", legend_position="bottom",
           top_radius_label=True, legend_spec=legend_spec, axis_fontsize=9, tick_fontsize=8)
    save_figure(figure, output_directory / f"{output_stem}.pdf")
    return figure


def shade_invalid(axis, radius, mask, color, hatch=None):
    """Shade intervals centered on samples failing the exported mixing test."""
    if len(radius) < 2:
        return
    edges = np.concatenate(([radius[0]], (radius[:-1] + radius[1:]) / 2, [radius[-1]]))
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    ends = np.flatnonzero(mask & np.concatenate((~mask[1:], [True])))
    for start, end in zip(starts, ends):
        axis.axvspan(edges[start], edges[end + 1], facecolor=color, alpha=0.15,
                     edgecolor=color if hatch else "none", hatch=hatch,
                     linewidth=0, zorder=0)


def plot_without_poles(axis, radius, profile, denominator, **style):
    """Keep a mixing-ratio pole from becoming a line crossing through zero."""
    x, y = [], []
    for index, (r, value, divisor) in enumerate(zip(radius, profile, denominator)):
        x.append(r)
        y.append(value if np.isfinite(divisor) and divisor != 0 else np.nan)
        if index + 1 < len(radius):
            following = denominator[index + 1]
            if (np.isfinite(divisor) and np.isfinite(following)
                    and (divisor == 0 or following == 0
                         or np.signbit(divisor) != np.signbit(following))):
                x.append(np.nan)
                y.append(np.nan)
    axis.plot(x, y, **style)


def required_entrainment(output_root: Path, output_directory: Path):
    directory = output_root / "NGC2403" / "required_entrainment"
    solver = read_csv(directory / "metallicity_solver.csv", ("method",))
    if len(solver) != 1 or solver[0]["method"] != "observed_profile":
        raise ValueError("Required-entrainment chemistry must come from observed_profile")
    inverse, inverse_summary = load_case(output_root, "NGC2403", "required_entrainment")
    forward, summary = load_case(output_root, "NGC2403", "fiducial")
    # The observed-mode runner leaves chemistry undefined outside measured fit support.
    support = np.isfinite(values(inverse, "Z")) & np.isfinite(values(inverse, "dZ_dR_per_kpc"))
    inverse = [row for row, keep in zip(inverse, support) if keep]
    if len(inverse) < 2:
        raise ValueError("Observed abundance fit has no resolved radial support")
    radius = values(inverse, "R_kpc")
    reference_radius = values(forward, "R_kpc")
    required_j = values(inverse, "j_land_kpc_kms") / values(inverse, "j_disk_kpc_kms")
    prescribed_j = np.interp(radius, reference_radius,
                             values(forward, "j_land_kpc_kms") / values(forward, "j_disk_kpc_kms"))
    required_z = values(inverse, "Z_land_required") / SOLAR_METALLICITY
    prescribed_z = np.interp(radius, reference_radius, values(forward, "Z_land_required") / SOLAR_METALLICITY)
    mu_j, mu_z = values(inverse, "mu_j"), values(inverse, "mu_Z")
    invalid_j = (~np.isfinite(mu_j)) | (mu_j < 0)
    invalid_z = (~np.isfinite(mu_z)) | (mu_z < 0)
    dynamic_color, chemical_color = BLUE, ORANGE
    figure, axes = plt.subplots(3, 1, figsize=(7.0, 6.2), sharex=True)
    axes[0].plot(radius, required_j, color=dynamic_color, label=r"required $j_{\rm land}/j_{\rm disk}$")
    axes[0].plot(radius, prescribed_j, color="black", linestyle="--", label=r"prescribed $j_{\rm mix}/j_{\rm disk}$")
    axes[1].plot(radius, required_z, color=chemical_color, label=r"required $Z_{\rm land}/Z_\odot$ (observed disk)")
    axes[1].plot(radius, prescribed_z, color="black", linestyle="--", label=r"prescribed $Z_{\rm mix}/Z_\odot$")
    plot_without_poles(axes[2], radius, mu_j,
                       values(inverse, "j_CGM_kpc_kms") - values(inverse, "j_land_kpc_kms"),
                       color=dynamic_color, label=r"$\mu_j$")
    plot_without_poles(axes[2], radius, mu_z,
                       float(inverse_summary["Z_CGM"]) - values(inverse, "Z_land_required"),
                       color=chemical_color, label=r"$\mu_Z$")
    axes[2].axhline(float(summary["mu"]), color="black", linestyle="--", label=r"adopted $\mu$")
    axes[2].set_yscale("symlog", linthresh=1.0, linscale=0.7)
    axes[0].axhline(0, color="0.6", linewidth=0.6, zorder=0)
    axes[1].axhline(0, color="0.6", linewidth=0.6, zorder=0)
    axes[2].axhline(0, color="0.6", linewidth=0.6, zorder=0)
    shade_invalid(axes[0], radius, invalid_j, dynamic_color)
    shade_invalid(axes[1], radius, invalid_z, chemical_color, hatch="///")
    shade_invalid(axes[2], radius, invalid_j, dynamic_color)
    shade_invalid(axes[2], radius, invalid_z, chemical_color, hatch="///")
    for letter, axis, label in zip("abc", axes, (r"$j_{\rm land}/j_{\rm disk}$", r"$Z_{\rm land}/Z_\odot$", r"$\mu$")):
        panel_label(axis, letter)
        axis.set_ylabel(quantity_unit_label(label))
        axis.legend(loc="lower right" if axis is axes[0] else "upper right", **LEGEND_STYLE)
        axis.set_xlim(radius[0], radius[-1])
    axes[2].set_xlabel(r"$R\;[\mathrm{kpc}]$")
    figure.suptitle(f"{galaxy_label('NGC2403')}: required entrainment", y=0.985, fontsize=11)
    invalid_handles = [
        Patch(facecolor=dynamic_color, alpha=0.15, label=r"no finite $\mu_j\geq0$ solution"),
        Patch(facecolor=chemical_color, alpha=0.15, hatch="///", label=r"no finite $\mu_Z\geq0$ solution"),
    ]
    figure.legend(handles=invalid_handles, loc="upper center", bbox_to_anchor=(0.52, 0.955), ncol=2, **LEGEND_STYLE)
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.09, top=0.88, hspace=0.14)
    save_figure(figure, output_directory / "section3_required_entrainment.pdf")


def galaxy_comparison(output_root: Path, output_directory: Path):
    observations = observed_abundances(output_root)
    figure = plt.figure(figsize=(7.0, 8.0))
    galaxy_grid = figure.add_gridspec(3, 2, left=0.09, right=0.91, bottom=0.09, top=0.951, wspace=0.06, hspace=0.28)
    metallicity_axes, gradient_axes = [], []
    metallicity_maximum = 0.0
    shared_handles = {}
    plotted_cases, excluded_cases, fixed_sources = {}, {}, {}
    abundance_spline_checks = {}
    for index, galaxy in enumerate(PAPER_GALAXIES):
        case_styles = dict(CASE_STYLES)
        loading_table = output_root / "configs" / galaxy / "galaxy_comparison_loading_cases.csv"
        loading_cases = read_csv(loading_table, ("case", "parameter", "value", "color", "kind")) \
            if loading_table.exists() else []
        for row in loading_cases:
            recovery = row["kind"] == "old_supply_recovery"
            label = (r"previous-supply recovery" if recovery else rf"$\eta={float(row['value']):g}$")
            case_styles[row["case"]] = (row["color"], "-." if recovery else "--", label)
        if loading_cases:
            color, linestyle, _ = case_styles["fiducial"]
            case_styles["fiducial"] = (color, linestyle, r"fiducial ($\eta=1$)")
        case_styles = palette_case_styles(case_styles)
        records = {row["case"]: row for row in read_csv(output_root / "model_cases.csv") if row["galaxy"] == galaxy}
        plotted_cases[galaxy], excluded_cases[galaxy] = [], []
        pair_grid = galaxy_grid[index // 2, index % 2].subgridspec(2, 1, height_ratios=(1.0, 1.0), hspace=0.10)
        metallicity_axis = figure.add_subplot(pair_grid[0], sharey=metallicity_axes[0] if metallicity_axes else None)
        gradient_axis = figure.add_subplot(pair_grid[1], sharex=metallicity_axis)
        metallicity_axes.append(metallicity_axis)
        gradient_axes.append(gradient_axis)
        disk_parameters = {row["name"]: row["value"] for row in read_csv(
            output_root / "inputs" / galaxy / "observed_disk" / "parameters.csv", ("name", "value"))}
        fixed_sources[galaxy] = float(disk_parameters["Mdot_nuc_Msun_yr"])
        maximum_radius = float(disk_parameters["R_out_kpc"])
        for case, (color, linestyle, label) in case_styles.items():
            row = next((row for row in loading_cases if row["case"] == case), None)
            if row:
                case_parameters = {item["name"]: item["value"] for item in read_csv(
                    output_root / "inputs" / galaxy / case / "parameters.csv", ("name", "value"))}
                for parameter in ("Mdot_nuc_Msun_yr", "mu", "beta"):
                    if not np.isclose(float(case_parameters[parameter]), float(disk_parameters[parameter]),
                                      rtol=1e-12, atol=0):
                        raise ValueError(f"Loading comparison changed {parameter}: {galaxy}/{case}")
                if not np.isclose(float(case_parameters["eta"]), float(row["value"]), rtol=1e-12, atol=0):
                    raise ValueError(f"Loading case disagrees with its saved table: {galaxy}/{case}")
            legend_key = "loading_recovery" if row and row["kind"] == "old_supply_recovery" else case
            linewidth = 1.6 if case == "fiducial" else 1.15 if row else 1.35
            shared_handles.setdefault(legend_key, Line2D(
                [], [], color=color, linestyle=linestyle, linewidth=linewidth, label=label))
            status_path = output_root / galaxy / case / "case_status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else records.get(case, {})
            if status.get("status", "valid") != "valid":
                excluded_cases[galaxy].append({"case": case, "eta": row["value"] if row else None,
                                               "status": status.get("status"),
                                               "reason": status.get("status_reason", "")})
                continue
            rows, summary = load_case(output_root, galaxy, case)
            if summary.get("landing_nonnegative") != "true" or summary.get("metallicity_solved") not in ("1", "true"):
                raise ValueError(f"Unrecorded inadmissible comparison case: {galaxy}/{case}")
            radius = values(rows, "R_kpc")
            metallicity = values(rows, "Z")
            abundance = np.where(np.isfinite(metallicity) & (metallicity > 0), metallicity / SOLAR_METALLICITY, np.nan)
            if not np.any(np.isfinite(abundance)):
                raise ValueError(f"No positive finite metallicity: {galaxy}/{case}")
            plotted_cases[galaxy].append(case)
            maximum_radius = max(maximum_radius, float(np.max(radius[np.isfinite(abundance)])))
            # Use the core-model derivative: d log10(Z/Z_sun)/dR = Z'/(Z ln 10).
            gradient = np.full_like(metallicity, np.nan)
            np.divide(values(rows, "dZ_dR_per_kpc"), metallicity * np.log(10.0),
                      out=gradient, where=np.isfinite(metallicity) & (metallicity > 0))
            metallicity_maximum = max(metallicity_maximum, np.nanmax(abundance))
            for axis, profile in ((metallicity_axis, abundance), (gradient_axis, gradient)):
                axis.plot(radius, profile, color=color, linestyle=linestyle,
                          linewidth=linewidth, label=label, zorder=3 if case == "fiducial" else 2)
        recovery = next((row for row in loading_cases if row["kind"] == "old_supply_recovery"), None)
        if recovery:
            metallicity_axis.text(0.97, 0.95,
                                  rf"recovery: $\eta={float(recovery['value']):.4g}$",
                                  transform=metallicity_axis.transAxes, ha="right", va="top",
                                  fontsize=6.5, color=recovery["color"])
        if excluded_cases[galaxy]:
            excluded_lines = []
            for status, prefix in (("invalid", "inadmissible"), ("failed", "unresolved"), ("running", "unfinished")):
                excluded = [row for row in excluded_cases[galaxy] if row["status"] == status]
                base_cases = [row for row in excluded if row["eta"] is None]
                excluded_labels = [rf"$\eta={float(row['eta']):.3g}$" for row in excluded if row["eta"] is not None]
                excluded_labels += ([r"$\eta=1$ references"] if len(base_cases) == len(CASE_STYLES)
                                    else [case_styles[row["case"]][2] for row in base_cases])
                for start in range(0, len(excluded_labels), 5):
                    excluded_lines.append((prefix + ": " if start == 0 else "    ")
                                          + ", ".join(excluded_labels[start:start + 5]))
            metallicity_axis.annotate("\n".join(excluded_lines), xy=(0.5, 1),
                                      xycoords="axes fraction", xytext=(0, 4),
                                      textcoords="offset points", ha="center", va="bottom", fontsize=6)
        plot_measurements(metallicity_axis, observations[galaxy], label="SINGS KK04")
        abundance_comparison = fit_metallicity_spline(observations[galaxy])
        _, fit_handle = plot_metallicity_spline(
            (metallicity_axis, gradient_axis), abundance_comparison,
            "SINGS KK04 spline",
        )
        abundance_spline_checks[galaxy] = {
            "measurement_count": abundance_comparison.measurement_count,
            "distinct_radius_count": abundance_comparison.distinct_radius_count,
            "radial_support_kpc": abundance_comparison.radius[[0, -1]].tolist(),
            "gradient_range_dex_per_kpc": [float(abundance_comparison.gradient_dex_kpc.min()),
                                           float(abundance_comparison.gradient_dex_kpc.max())],
        }
        observation_upper = (values(observations[galaxy], "Z")
                             + values(observations[galaxy], "e_Z_high")) / SOLAR_METALLICITY
        metallicity_maximum = max(metallicity_maximum, np.nanmax(observation_upper))
        metallicity_axis.set_title(galaxy_label(galaxy), loc="center", fontsize=10,
                                   pad=16 if excluded_cases[galaxy] else 4)
        metallicity_axis.tick_params(axis="x", labelbottom=False)
        metallicity_axis.set_xlim(0.0, maximum_radius)
        for axis in (metallicity_axis, gradient_axis):
            axis.tick_params(which="both", direction="in", top=True, right=True, labelsize=9, pad=2)
        metallicity_axis.yaxis.set_major_locator(MaxNLocator(4))
        metallicity_axis.set_ylabel(r"$Z/Z_\odot$", fontsize=9, labelpad=3)
        gradient_axis.set_ylabel(
            quantity_unit_label(r"$d\log_{10}Z/dR$" + "\n" + r"$[\mathrm{dex\,kpc}^{-1}]$"),
            fontsize=9, labelpad=3,
        )
        if index % 2 == 1:
            for axis in (metallicity_axis, gradient_axis):
                axis.yaxis.set_label_position("right")
                axis.tick_params(axis="y", labelleft=False, labelright=True)
    metallicity_axes[0].set_ylim(0, metallicity_maximum * 1.06)
    for axis in gradient_axes:
        # Pad only the displayed negative range; clipped positive nuclear
        # peaks must not enlarge the whitespace below the curves.
        lower = axis.dataLim.y0
        axis.set_ylim(bottom=lower - axis.get_ymargin() * abs(lower), top=0.0)
        axis.yaxis.set_major_locator(AutoLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())
    # Keep boundary tick labels inside their panels in the compact stack.
    for axis in metallicity_axes:
        for label in axis.get_yticklabels():
            if np.isclose(label.get_position()[1], 0.0):
                label.set_verticalalignment("bottom")
    for axis in gradient_axes:
        for label in axis.get_yticklabels():
            if np.isclose(label.get_position()[1], 0.0):
                label.set_verticalalignment("top")
    handles = list(shared_handles.values())
    handles += [Line2D([], [], color="black", linestyle="none", marker="o", markersize=3,
                       label="SINGS KK04"), fit_handle]
    labels = [handle.get_label() for handle in handles]
    heading = figure.suptitle(f"{LANDING_TITLE} for six galaxies", x=0.5, y=0.99, fontsize=12)
    ncol = len(handles)
    legend = figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.012),
                           ncol=ncol, borderaxespad=0, borderpad=0,
                           fontsize=7.5, frameon=False, handlelength=1.8,
                           handletextpad=0.4, columnspacing=1.0, labelspacing=0.4)
    radius_label = figure.supxlabel(r"$R\;[\mathrm{kpc}]$", fontsize=9, y=0.04)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    gap = 4.0 * figure.dpi / 72.0
    radius_label.set_y((legend.get_window_extent(renderer).y1 + gap) / figure.bbox.height)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    panel_top = max(axis.get_tightbbox(renderer).y1 for axis in metallicity_axes[:2])
    panel_bottom = min(axis.get_tightbbox(renderer).y0 for axis in gradient_axes[-2:])
    upper_shift = (heading.get_window_extent(renderer).y0 - gap - panel_top) / figure.bbox.height
    lower_shift = (radius_label.get_window_extent(renderer).y1 + gap - panel_bottom) / figure.bbox.height
    galaxy_grid.update(top=galaxy_grid.top + upper_shift, bottom=galaxy_grid.bottom + lower_shift)
    # Pack galaxy groups against their rendered titles/ticks, then remove
    # the saved space from the page while retaining each panel's height.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    row_gap = max(
        gradient_axes[index].bbox.y0
        - min(axis.get_tightbbox(renderer).y0 for axis in gradient_axes[index:index + 2])
        + max(axis.get_tightbbox(renderer).y1 for axis in metallicity_axes[index + 2:index + 4])
        - metallicity_axes[index + 2].bbox.y1
        for index in (0, 2)
    ) + gap
    block_height = metallicity_axes[0].bbox.y1 - gradient_axes[0].bbox.y0
    previous_gap = gradient_axes[0].bbox.y0 - metallicity_axes[2].bbox.y1
    top_margin = figure.bbox.height - metallicity_axes[0].bbox.y1
    bottom_margin = gradient_axes[-1].bbox.y0
    target_height = figure.bbox.height - 2 * max(previous_gap - row_gap, 0)
    figure.set_size_inches(figure.get_size_inches()[0], np.ceil(target_height) / figure.dpi)
    height = figure.bbox.height
    # Let whole-pixel page rounding add to the margin, not to panel heights.
    bottom_margin += height - target_height
    galaxy_grid.update(top=1 - top_margin / height, bottom=bottom_margin / height, hspace=row_gap / block_height)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    radius_label.set_y((legend.get_window_extent(renderer).y1 + gap) / figure.bbox.height)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    (output_root / "galaxy_comparison_publication_checks.json").write_text(json.dumps({
        "figure_size_inches": list(figure.get_size_inches()), "legend_entries": labels,
        "legend_columns": ncol, "plotted_cases": plotted_cases, "excluded_cases": excluded_cases,
        "fixed_nuclear_SFR_Msun_yr": fixed_sources,
        "abundance_comparison": "inverse-variance weighted cubic OH12 smoothing spline; GCV smoothing",
        "gradient_comparison": "exact logarithmic abundance spline derivative on measured support",
        "abundance_spline_checks": abundance_spline_checks,
        "axis_count": len(figure.axes), "inset_count": sum(len(axis.child_axes) for axis in figure.axes),
        "shared_radius_label_count": 1,
        "title_to_panel_gap_pt": (heading.get_window_extent(renderer).y0
                                  - max(axis.get_tightbbox(renderer).y1 for axis in metallicity_axes[:2]))
                                  * 72 / figure.dpi,
        "legend_to_radius_label_gap_pt": (radius_label.get_window_extent(renderer).y0
                                          - legend.get_window_extent(renderer).y1) * 72 / figure.dpi,
    }, indent=2) + "\n")
    save_figure(figure, output_directory / "section3_galaxy_comparison.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", nargs="?", type=Path, default=ROOT / "output" / "section3")
    parser.add_argument("output_directory", nargs="?", type=Path, default=ROOT / "output" / "pdf")
    parser.add_argument("--figure", choices=("all", "predictive", "required", "comparison"), default="all")
    parser.add_argument("--predictive-cases", type=Path,
                        help="Explicit predictive case table for an existing saved model run")
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    for name, renderer in (("predictive", predictive_comparison),
                           ("required", required_entrainment),
                           ("comparison", galaxy_comparison)):
        if arguments.figure in ("all", name):
            options = {"case_table": arguments.predictive_cases} if name == "predictive" else {}
            renderer(arguments.output_root, arguments.output_directory, **options)


if __name__ == "__main__":
    main()
