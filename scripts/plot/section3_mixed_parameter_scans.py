#!/usr/bin/env python3
"""Plot the prescribed power-law landing and j_land=j_mix disk solution.

The C++ core derives transport and the stellar sink from conservation, then
gas density from the net KS law. Chemistry uses the same mixed reservoirs.
Saved gas/SFR spline fits are comparison overlays only. Flow profiles are analytic;
the metallicity ODE is integrated numerically with an outer abundance boundary.
"""

from __future__ import annotations

from axis_labels import quantity_unit_label

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, LogLocator, MaxNLocator
import numpy as np

from hershey_fonts import register_hershey_weight_aliases
from paper_galaxies import galaxy_label
from observed_disk_comparison import load_disk_fit, plot_disk_fit, fit_legend_handle, flow_legends
from observed_metallicity_spline import fit_metallicity_spline, plot_metallicity_spline, include_spline_gradient
from six_panel_layout import compact_six_panel_rows
from surface_rate_units import read_rate_rows, YEARS_PER_GYR
from color_palette import apply_case_palette, color_cycle

register_hershey_weight_aliases()
import smplotlib


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "build/galactic-nuclear-fountain-model"
SOLAR_Z = 0.0139
MODEL_TITLE = "Mixed fountain, analytic solution"
DEFICIT_Y_LIMITS = (-1, 2)
GROUPS = {
    "landing_slope": "Landing slope",
    "source_rate": r"Nuclear source $[M_\odot\,\mathrm{yr}^{-1}]$",
    "loading_factor": "Nuclear loading factor",
    "entrainment": "CGM entrainment",
    "cgm_rotation": "CGM rotation fraction",
    "outer_metallicity": "Outer boundary metallicity",
    "nucleosynthetic_yield": "Nucleosynthetic yield",
    "reference": "Unmixed comparison",
}
CHEMISTRY_GROUPS = {"outer_metallicity", "nucleosynthetic_yield"}
OVERRIDES = ("lambda", "Mdot_nuc_Msun_yr", "eta", "mu", "beta", "Z_outer_boundary", "yield_y")


def read_csv(path):
    rows = read_rate_rows(path)
    return apply_case_palette(rows) if rows and "case" in rows[0] and "color" in rows[0] else rows


def parameters(path):
    return {row["name"]: row["value"] for row in read_csv(path)}


def write_parameters(path, settings):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("name", "value"))
        writer.writerows(settings.items())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, log):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Command failed; see {log}: {result.stderr[-1000:]}")


def load_profile(path):
    rows = read_csv(path)
    if not rows or any(row["source"] != "analytic_flat" or row["model"] != "mixed-power-law" for row in rows):
        raise ValueError(f"Expected mixed-power-law analytic_flat profiles: {path}")
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0] if key not in ("source", "model")}


def validate(profile, settings, directory):
    """Check the imposed mixture, KS closure and differential conservation."""
    radius, gas = profile["R_kpc"], profile["Sigma_g_Msun_kpc2"]
    accretion, landing = profile["Mdot_acc_Msun_yr"], profile["Sigmadot_land_Msun_Gyr_kpc2"]
    total = float(settings["eta"]) * (1 + float(settings["mu"])) * float(settings["Mdot_nuc_Msun_yr"])
    for key in ("Sigma_g_Msun_kpc2", "Sigmadot_star_Msun_Gyr_kpc2", "Sigmadot_land_Msun_Gyr_kpc2",
                "Mdot_acc_Msun_yr", "v_R_kms", "Z", "dZ_dR_per_kpc", "j_land_kpc_kms"):
        if not np.all(np.isfinite(profile[key])):
            raise ValueError(f"Nonfinite {key} in {directory}")
    if not (np.all(gas > 0) and np.all(landing > 0) and np.all(profile["Z"] > 0)):
        raise ValueError(f"Nonpositive gas, landing or metallicity in {directory}")
    np.testing.assert_allclose(landing / landing[0], (radius / radius[0]) ** -float(settings["lambda"]), rtol=1e-12)
    np.testing.assert_allclose(profile["cumulative_landing_Msun_yr"][-1], total, rtol=1e-12)
    np.testing.assert_allclose(profile["v_R_kpc_yr"], -accretion / (2 * np.pi * radius * gas), rtol=1e-12)
    np.testing.assert_allclose(profile["Z"][-1], float(settings["Z_outer_boundary"]), rtol=1e-10)
    np.testing.assert_allclose(profile["j_CGM_kpc_kms"] / profile["j_disk_kpc_kms"],
                               float(settings["beta"]), rtol=1e-12)
    mu = float(settings["mu"])
    nuclear_j = float(settings["R_nucl_kpc"]) * float(settings["v_c_nucl_kms"])
    np.testing.assert_allclose(profile["j_land_kpc_kms"],
        (nuclear_j + mu * profile["j_CGM_kpc_kms"]) / (1 + mu), rtol=1e-12)
    np.testing.assert_allclose(profile["Z_land"],
        (float(settings["Z_nucl"]) + mu * float(settings["Z_CGM"])) / (1 + mu), rtol=1e-12)
    np.testing.assert_allclose(profile["Sigmadot_star_Msun_Gyr_kpc2"],
        (1-float(settings["return_fraction"])) * float(settings["KS_A_SFR"]) * gas**float(settings["KS_N"]), rtol=1e-12)
    # Angular momentum uses the SAME mixture that supplies the chemistry.
    angular_balance = gas * profile["v_R_kpc_yr"] * YEARS_PER_GYR * profile["v_c_kms"]
    angular_source = landing * (profile["j_land_kpc_kms"] - profile["j_disk_kpc_kms"])
    np.testing.assert_allclose(angular_balance, angular_source, rtol=1e-12)
    # Independent output-grid checks of mass and metal-flux derivatives.
    mass_derivative = np.gradient(accretion, radius, edge_order=2)
    mass_rhs = 2 * np.pi * radius * (profile["Sigmadot_star_Msun_Gyr_kpc2"] - landing) / YEARS_PER_GYR
    mass_error = float(np.max(np.abs(mass_derivative - mass_rhs)) / max(np.max(np.abs(mass_rhs)), 1e-30))
    if mass_error > 0.002:
        raise ValueError(f"Output-grid continuity error {mass_error:g} in {directory}")
    metal_derivative = np.gradient(-accretion * profile["Z"], radius, edge_order=2)
    metal_rhs = 2 * np.pi * radius * (landing * profile["Z_land"] +
        (float(settings["yield_y"]) - profile["Z"]) * profile["Sigmadot_star_Msun_Gyr_kpc2"]) / YEARS_PER_GYR
    metal_error = float(np.max(np.abs(metal_derivative-metal_rhs)) / max(np.max(np.abs(metal_rhs)), 1e-30))
    if metal_error > 0.003:
        raise ValueError(f"Output-grid metal conservation error {metal_error:g} in {directory}")
    return {
        "Mdot_land_Msun_yr": total,
        "mass_equation_output_grid_relative_error": mass_error,
        "metal_equation_output_grid_relative_error": metal_error,
        "all_flow_inward": bool(np.all(accretion > 0)),
        "v_R_range_kms": [float(np.min(profile["v_R_kms"])), float(np.max(profile["v_R_kms"]))],
        "Z_solar_range": [float(np.min(profile["Z"]) / SOLAR_Z), float(np.max(profile["Z"]) / SOLAR_Z)],
        "derived_outer_accretion_Msun_yr": float(accretion[-1]),
        "samples": len(radius),
        "metallicity_solver": read_csv(directory / "metallicity_solver.csv")[0]["method"],
        "interpretation": "Power-law landing and j_mix are prescribed; gas, stellar sink and flow are predictions. Chemistry uses the same imposed mu.",
    }


def calculate_cases(args, base, cases):
    profiles, diagnostics, resolved = {}, {}, {}
    args.run_directory.mkdir(parents=True, exist_ok=True)
    for case in cases:
        name = case["case"]
        if not name or Path(name).name != name or name in profiles:
            raise ValueError(f"Invalid or duplicate case name: {name}")
        if case["group"] not in (*GROUPS, "all"):
            raise ValueError(f"Unknown scan group: {case['group']}")
        settings = {**base, **{key: case[key] for key in OVERRIDES if case.get(key)}}
        if (settings.get("closure") != "j_mix" or settings.get("landing_profile") != "power_law"
                or settings.get("cgm_model") != "constant_beta" or settings.get("rotation_model") != "flat"):
            raise ValueError("Require j_mix closure, prescribed power law, constant-beta CGM and flat rotation")
        if any(key in settings for key in ("R_g_kpc", "M_g_Msun", "Mdot_out_Msun_yr", "profile_type", "metallicity_mode")):
            raise ValueError("Gas shape/mass and outer mass flux are derived, not input parameters")
        source = args.run_directory / "inputs" / name
        destination = args.run_directory / "models" / name
        if args.plot_only:
            if parameters(destination / "parameters.csv") != settings:
                raise ValueError(f"Saved parameters differ for {name}; rerun without --plot-only")
        else:
            source.mkdir(parents=True, exist_ok=True)
            # Only physical parameters enter the solver; observations stay in overlays.
            write_parameters(source / "parameters.csv", settings)
            run([str(RUNNER), "mixed-power-law", str(source), str(destination)],
                args.run_directory / "logs" / f"{name}.log")
        profiles[name] = load_profile(destination / "profiles.csv")
        diagnostics[name] = validate(profiles[name], settings, destination)
        resolved[name] = settings
        action = "Loaded" if args.plot_only else "Computed"
        print(f"{action} {name}: landing {diagnostics[name]['Mdot_land_Msun_yr']:.4g} Msun/yr", flush=True)
    if "fiducial" not in profiles:
        raise ValueError("Case table must contain fiducial")
    for profile in profiles.values():
        for key in ("R_kpc", "v_c_kms"):
            np.testing.assert_array_equal(profile[key], profiles["fiducial"][key])
    for case in cases:
        if case["group"] == "cgm_rotation":
            profile = profiles[case["case"]]
            for key in ("Mdot_acc_Msun_yr", "v_R_kms", "Z", "j_land_kpc_kms"):
                if np.allclose(profile[key], profiles["fiducial"][key], rtol=1e-10, atol=1e-14):
                    raise ValueError(f"Beta scan {case['case']} did not change {key}")
    return profiles, diagnostics, resolved


def observations(args, base):
    directory = args.observation_directory / "inputs" / args.galaxy / "observed_disk"
    disk_fit = load_disk_fit(directory, base["return_fraction"])
    abundance_path, fit_path = args.observation_directory / "observations.csv", args.observation_directory / "abundance_fits.csv"
    abundance = [row for row in read_csv(abundance_path) if row["galaxy"] == args.galaxy and row["calibration"] == "KK04"]
    fits = [row for row in read_csv(fit_path) if row["galaxy"] == args.galaxy and row["calibration"] == "KK04"]
    if not abundance or len(fits) != 1:
        raise ValueError(f"Missing unique KK04 comparison for {args.galaxy}")
    return disk_fit, abundance, fits[0], [*disk_fit.input_paths, abundance_path, fit_path]


def case_style(case):
    fiducial=case.get("is_fiducial",case["case"]=="fiducial")
    style = dict(color=case["color"], linestyle=case.get("linestyle") or "-",
                 linewidth=1.5 if fiducial else 1.1,
                 zorder=3.5 if case["group"] == "cgm_rotation" else
                 (3 if fiducial or case["group"] == "reference" else 2))
    style.update(case.get("plot_style",{}))
    # Keep the scan patterns visible in the compact legend handles.
    if style["linestyle"] == "--":
        style["dashes"] = (3, 2)
    elif style["linestyle"] == ":":
        style.update(dashes=(0.5, 2), dash_capstyle="round")
    elif style["linestyle"] == "-.":
        style["dashes"] = (3, 1.4, 0.7, 1.4)
    return style


def layout(figure, axes, cases, vertical, extra_handles=(), *, legend_position="top",
           title=None, top_radius_label=False, panel_columns=1, legend_single_row=False,
           legend_parameter_columns=False, legend_spec=None,
           axis_fontsize=None, tick_fontsize=None):
    if vertical:
        if panel_columns==2:
            figure.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.86, hspace=0.18, wspace=0.25)
        else:
            figure.subplots_adjust(left=0.23, right=0.97, bottom=0.07, top=0.90, hspace=0.10)
    else:
        figure.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.84 if len(cases)>9 else 0.88, wspace=0.30)
    handles = [Line2D([], [], label=case["label"], **case_style(case)) for case in cases]
    if legend_spec:
        fontsize=legend_spec.get("fontsize",8)
        minimum_fontsize=fontsize if "fontsize" in legend_spec else 6.5
        while fontsize>=minimum_fontsize:
            legend=figure.legend(handles=legend_spec["handles"],labels=legend_spec["labels"],
                                 ncol=legend_spec["ncol"],handler_map=legend_spec.get("handler_map"),
                                 loc="lower center",bbox_to_anchor=(.5,.01),fontsize=fontsize,
                                 frameon=False,borderpad=0,borderaxespad=0,labelspacing=.2,
                                 handlelength=legend_spec.get("handlelength",2.4),
                                 handletextpad=legend_spec.get("handletextpad",.3),
                                 columnspacing=legend_spec.get("columnspacing",.5))
            figure.canvas.draw()
            if legend.get_window_extent(figure.canvas.get_renderer()).width<=.98*figure.bbox.width:
                break
            legend.remove()
            fontsize-=.25
        else:
            raise ValueError("Combined model legend cannot fit at a readable size")
        legends=[legend]
    elif legend_position=="bottom" and legend_single_row:
        legend_handles=[*handles,*extra_handles]
        fontsize=8.0 if vertical else 7.5
        minimum_fontsize=6.0 if legend_single_row else 6.5
        while fontsize>=minimum_fontsize:
            legend=figure.legend(handles=legend_handles,loc="lower center",
                                 bbox_to_anchor=(0.5,0.01),ncol=len(legend_handles),
                                 fontsize=fontsize,frameon=False,borderpad=0,borderaxespad=0,
                                 handlelength=1.0,handletextpad=0.25,columnspacing=0.45)
            figure.canvas.draw()
            if legend.get_window_extent(figure.canvas.get_renderer()).width<=0.98*figure.bbox.width:
                break
            legend.remove()
            fontsize-=0.25
        else:
            raise ValueError("Model legend cannot fit one readable row")
        legends=[legend]
    elif extra_handles:
        columns=None
        if legend_parameter_columns:
            groups={}
            for case,handle in zip(cases[1:],handles[1:]):
                groups.setdefault(case["group"],[]).append(handle)
            columns=[[handles[0],*extra_handles],*groups.values()]
        legends=flow_legends(figure,[*handles,*extra_handles],
                             fontsize=8 if panel_columns==2 else 6.5,position=legend_position,
                             columns=columns)
    else:
        legend = figure.legend(handles=handles,
                  loc="lower center" if legend_position=="bottom" else "upper center",
                  bbox_to_anchor=(0.5, 0.01 if legend_position=="bottom" else 0.99),
                  ncol=(len(cases)+1)//2 if vertical or len(cases)>9 else len(cases), fontsize=6.5 if vertical else 7.4, frameon=False,
                  borderpad=0, borderaxespad=0,
                  columnspacing=0.45 if vertical else 0.8,
                  handlelength=1.0 if vertical else 1.3,
                  handletextpad=0.25 if vertical else 0.35)
        legends=[legend]
    for index, axis in enumerate(axes):
        axis.tick_params(which="both", direction="in", top=True, right=True,
                         labelsize=tick_fontsize if tick_fontsize is not None else (8.5 if panel_columns==2 else 9))
        if vertical:
            axis.set_xticks([0, 5, 10, 15], labels=["0", "5", "10", "15"])
            axis.tick_params(axis="x", which="both", labeltop=index < panel_columns, pad=3.5)
        else:
            axis.set_xticks([0, 5, 10, 15])
        if vertical and index < len(axes) - panel_columns:
            axis.tick_params(labelbottom=False)
        else:
            axis.set_xlabel(r"$R\;[\mathrm{kpc}]$", fontsize=axis_fontsize if axis_fontsize is not None else (8.5 if panel_columns==2 else 9), labelpad=2)
    if vertical:
        figure.align_ylabels(axes)
        # Numeric top labels need no extra space for font descenders.
        for axis in axes[:panel_columns]:
            for tick in axis.xaxis.get_major_ticks():
                tick.label2.set_verticalalignment("baseline")
            if top_radius_label:
                axis.set_xlabel(r"$R\;[\mathrm{kpc}]$",fontsize=axis_fontsize if axis_fontsize is not None else (8.5 if panel_columns==2 else 9),labelpad=2)
                axis.xaxis.set_label_position("top")
    heading=figure.suptitle(title,x=(figure.subplotpars.left+figure.subplotpars.right)/2,
                           y=.99,fontsize=10) if title else None
    # Reserve a fixed four-point gap using rendered labels, rather than a
    # fraction of the figure height. Extra space goes back to the panels.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    if heading and vertical and panel_columns==1:
        heading.set_x(0.5)
        title_width=heading.get_window_extent(renderer).width
        if title_width>0.98*figure.bbox.width:
            heading.set_fontsize(heading.get_fontsize()*0.98*figure.bbox.width/title_width)
        figure.canvas.draw()
        renderer=figure.canvas.get_renderer()
    gap=4.0*figure.dpi/72
    if heading and legend_position=="top":
        next_top=(heading.get_window_extent(renderer).y0-gap)/figure.bbox.height
        row_gap=1.5*figure.dpi/72
        for legend in legends:
            legend.set_bbox_to_anchor((0.5,next_top),transform=figure.transFigure)
            next_top=(legend.get_window_extent(renderer).y0-row_gap)/figure.bbox.height
        figure.canvas.draw()
        renderer=figure.canvas.get_renderer()
    panel_top = max(axis.get_tightbbox(renderer).y1 for axis in (axes[:panel_columns] if vertical else axes))
    if legend_position=="bottom":
        legend_top=max(legend.get_window_extent(renderer).y1 for legend in legends)
        panel_bottom=min(axis.get_tightbbox(renderer).y0 for axis in axes[-panel_columns:])
        lower_shift=(legend_top+gap-panel_bottom)/figure.bbox.height
        upper_shift=(heading.get_window_extent(renderer).y0-gap-panel_top)/figure.bbox.height if heading else 0
        figure.subplots_adjust(bottom=figure.subplotpars.bottom+lower_shift, top=figure.subplotpars.top+upper_shift)
    else:
        legend_bottom = min(legend.get_window_extent(renderer).y0 for legend in legends)
        shift = (legend_bottom - gap - panel_top) / figure.bbox.height
        figure.subplots_adjust(top=figure.subplotpars.top + shift)
    if vertical and panel_columns==2:
        # Fit margins and gutters to the rendered labels, leaving four points
        # of clearance instead of reserving a large fraction of the page.
        for _ in range(2):
            figure.canvas.draw()
            renderer=figure.canvas.get_renderer()
            boxes=[axis.get_tightbbox(renderer) for axis in axes]
            left_indices=range(0,len(axes),2)
            right_indices=range(1,len(axes),2)
            rows=len(axes)//2
            left=(max(axes[i].bbox.x0-boxes[i].x0 for i in left_indices)+gap)/figure.bbox.width
            right=1-(max(boxes[i].x1-axes[i].bbox.x1 for i in right_indices)+gap)/figure.bbox.width
            column_gap=max(boxes[i].x1-axes[i].bbox.x1+axes[i+1].bbox.x0-boxes[i+1].x0 for i in left_indices)+gap
            row_gap=max(axes[i].bbox.y0-boxes[i].y0+boxes[i+2].y1-axes[i+2].bbox.y1 for i in range(len(axes)-2))+gap
            width=(right-left)*figure.bbox.width
            height=(figure.subplotpars.top-figure.subplotpars.bottom)*figure.bbox.height
            figure.subplots_adjust(left=left,right=right,
                                   wspace=2*column_gap/(width-column_gap),
                                   hspace=rows*row_gap/(height-(rows-1)*row_gap))
            if heading:
                heading.set_x((left+right)/2)
        if len(axes)==6 and legend_position=="bottom":
            compact_six_panel_rows(figure,axes)


def save(figure, args, stem):
    args.figure_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure_directory / f"{stem}.pdf", format="pdf")
    plt.close(figure)
    print(f"Wrote {args.figure_directory / (stem + '.pdf')}", flush=True)


def plot_abundance_measurements(axis, abundance, galaxy):
    obs = lambda key: np.array([float(row[key]) for row in abundance])
    return axis.errorbar(obs("R_kpc"), obs("Z") / SOLAR_Z,
        yerr=np.vstack((obs("e_Z_low"), obs("e_Z_high"))) / SOLAR_Z,
        linestyle="none", marker="o", markersize=2.5, color="black", elinewidth=0.5,
        capsize=1.5, capthick=0.5, zorder=4,
        label=f"{galaxy_label(galaxy)}, SINGS KK04")


def required_j_display(profile, mode="physical", nuclear_j=None):
    """Convert the core's required landing j to the requested display quantity."""
    required=profile["j_land_required_kpc_kms"]
    if mode=="physical":
        return required,r"$j_{\rm land}\;[\mathrm{kpc\,km\,s}^{-1}]$"
    if mode=="nuclear":
        if nuclear_j is None or not np.isfinite(nuclear_j) or nuclear_j<=0:
            raise ValueError("Nuclear normalization requires explicit positive j_nuc")
        return required/nuclear_j,r"$j_{\rm land}/j_{\rm nuc}$"
    disk=profile["j_disk_kpc_kms"]
    if not np.all(np.isfinite(disk)) or not np.all(disk>0):
        raise ValueError("Local normalization requires finite positive disk j")
    ratio=required/disk
    if mode=="disk":
        return ratio,r"$j_{\rm land}/j_{\rm disk}$"
    if mode=="deficit":
        return 1-ratio,r"$1-j_{\rm land}/j$"
    raise ValueError(f"Unknown required-j display: {mode}")


def draw_flow(axes, args, cases, profiles, disk_fit, *, required_j=False,
              j_display="physical", nuclear_j=None, angular_comparison_label=None,
              combined_rates=False):
    """Draw core-exported flow profiles; None omits a panel from the display."""
    cases = [case for case in cases if case["group"] not in CHEMISTRY_GROUPS]
    radius = profiles["fiducial"]["R_kpc"]
    for case in cases:
        profile = profiles[case["case"]]
        style=case_style(case)
        if required_j and case["group"]=="cgm_rotation":
            # Beta is inactive for the prescribed-gas inversion. Leave its
            # exact fiducial overlap under the black fiducial stroke.
            style["zorder"]=2.5
        fields=[(0, "Sigmadot_land_Msun_Gyr_kpc2", 1),
                                 (1, "Sigmadot_star_Msun_Gyr_kpc2", 1),
                                 (2, "Sigma_g_Msun_kpc2", 1e-6),
                                 (3, "v_R_kms", 1), (4, "Mdot_acc_Msun_yr", 1)]
        for row, field, scale in fields:
            if axes[row] is None:
                continue
            plot_style=style
            if combined_rates and row<2:
                # Case colors identify the trials; rate styles distinguish
                # landing from the stellar sink when they share an axis.
                plot_style={key:value for key,value in style.items()
                            if key not in ("linestyle","dashes","dash_capstyle")}
                plot_style["linestyle"]="-" if row==0 else "--"
                if row==1:
                    plot_style["dashes"]=(2,1.25)
            axes[row].plot(radius, profile[field] * scale, **plot_style)
        if required_j:
            values,j_label=required_j_display(profile,j_display,nuclear_j)
            axes[5].plot(radius,values,**style)
            if angular_comparison_label:
                reference={**profile,"j_land_required_kpc_kms":profile["j_land_comparison_kpc_kms"]}
                values,_=required_j_display(reference,j_display,nuclear_j)
                axes[5].plot(radius,values,color=case["color"],linestyle="--",linewidth=.8,
                             dashes=(3,2),zorder=3 if case["case"]=="fiducial" else 1)
    for row,quantity in ((1,"star"),(2,"gas")):
        if axes[row] is not None:
            plot_disk_fit(axes[row],disk_fit,quantity,f"{galaxy_label(args.galaxy)} spline fit")
    for axis in axes[:2]:
        if axis is None:
            continue
        axis.set_yscale("log")
        axis.set_ylim(1e3, 1e7)
        axis.yaxis.set_major_locator(LogLocator(base=10,numticks=12))
        axis.yaxis.set_minor_locator(LogLocator(base=10,subs=range(2,10),numticks=100))
    if required_j:
        axes[0].set_ylim(1e4, 1e7)
    if axes[2] is not None:
        axes[2].set_yscale("log")
        axes[2].set_ylim(0.1, 10)
    gas_input_cases=[profiles[case["case"]] for case in cases if case["group"] in ("gas_scale","gas_mass")]
    if gas_input_cases:
        for row,field,scale in ((1,"Sigmadot_star_Msun_Gyr_kpc2",1), (2,"Sigma_g_Msun_kpc2",1e-6)):
            if axes[row] is None:
                continue
            lower,upper=axes[row].get_ylim()
            minimum=min(np.min(profile[field])*scale for profile in gas_input_cases)
            maximum=max(np.max(profile[field])*scale for profile in gas_input_cases)
            axes[row].set_ylim(min(lower,minimum/1.2),max(upper,maximum*1.2))
    labels = (r"$\dot\Sigma_{\rm land}\;[M_\odot\,\mathrm{Gyr}^{-1}\,\mathrm{kpc}^{-2}]$",
              r"$\dot\Sigma_\star\;[M_\odot\,\mathrm{Gyr}^{-1}\,\mathrm{kpc}^{-2}]$",
              r"$\Sigma_g\;[M_\odot\,\mathrm{pc}^{-2}]$",
              r"$v_R\;[\mathrm{km}\,\mathrm{s}^{-1}]$",
              r"$\dot M_{\rm acc}\;[M_\odot\,\mathrm{yr}^{-1}]$")
    for axis, label in zip(axes, labels):
        if axis is not None:
            axis.set_ylabel(quantity_unit_label(label), fontsize=8.5 if required_j else 9, labelpad=3)
    if required_j:
        from matplotlib.ticker import MaxNLocator
        axes[5].set_ylabel(quantity_unit_label(j_label), fontsize=8.5, labelpad=3)
        axes[5].yaxis.set_major_locator(MaxNLocator(nbins=3))
        if j_display=="deficit":
            axes[5].set_ylim(*DEFICIT_Y_LIMITS)
            axes[5].set_yticks([-1, 0, 1, 2], labels=["-1", "0", "1", "2"])
            axes[5].yaxis.set_minor_locator(AutoMinorLocator(5))
    if all(np.max(profile["v_R_kms"]) <= 0 for profile in profiles.values()):
        axes[3].set_ylim(top=0)
    if all(np.min(profile["Mdot_acc_Msun_yr"]) >= 0 for profile in profiles.values()):
        axes[4].set_ylim(bottom=0)
    axes[4].set_xlim(0, float(radius[-1]))
    return cases


def plot_flow(args, cases, profiles, disk_fit, *, required_j=False, title=None,
              legend_position="top", top_radius_label=False, j_display="physical", nuclear_j=None,
              stem="section3_mixed_analytic_flow_scans", legend_parameter_columns=False,
              angular_comparison_label=None):
    panel_columns=2 if required_j else 1
    figure, axes = plt.subplots(3 if required_j else 5, panel_columns,
                               figsize=(7.2, 4.6) if required_j else (3.5, 7.8), sharex=True)
    axes=axes.ravel()
    cases=draw_flow(axes,args,cases,profiles,disk_fit,required_j=required_j,
                    j_display=j_display,nuclear_j=nuclear_j,
                    angular_comparison_label=angular_comparison_label)
    layout(figure, axes, cases, vertical=True,
           extra_handles=[fit_legend_handle(f"{galaxy_label(args.galaxy)} spline fit")],
           title=title,legend_position=legend_position,top_radius_label=top_radius_label,
           panel_columns=panel_columns,legend_parameter_columns=legend_parameter_columns)
    if angular_comparison_label:
        axes[5].legend(handles=[Line2D([],[],color="black",label="required"),
                               Line2D([],[],color="black",linestyle="--",label=angular_comparison_label)],
                       loc="lower right",fontsize=6.5,frameon=False,handlelength=1.6)
    save(figure, args, stem)


def draw_metallicity(axes, args, cases, profiles, abundance, fit, *, inactive_beta=False,
                     gradient_min=-0.2, metallicity_max=3, fontsize=9, abundance_spline=False):
    """Draw predicted abundances and gradients with their measured comparisons."""
    radius = profiles["fiducial"]["R_kpc"]
    support = [float(fit["R_min_kpc"]), float(fit["R_max_kpc"])]
    slope = float(fit["OH12_slope_dex_per_kpc"])
    for case in cases:
        profile = profiles[case["case"]]
        style=case_style(case)
        if inactive_beta and case["group"]=="cgm_rotation":
            style["zorder"]=2.5
        axes[0].plot(radius, profile["Z"] / SOLAR_Z, **style)
        axes[1].plot(radius, profile["dZ_dR_per_kpc"] / (profile["Z"] * math.log(10)), **style)
    if abundance_spline:
        comparison = fit_metallicity_spline(abundance)
        observation_handle, gradient_handle = plot_metallicity_spline(
            axes, comparison, f"{galaxy_label(args.galaxy)}, SINGS KK04 spline")
    else:
        observation_handle = plot_abundance_measurements(axes[0],abundance,args.galaxy)
        gradient_handle,=axes[1].plot(support, [slope, slope], color="black", linestyle="--", linewidth=0.9,
                                    label=f"{galaxy_label(args.galaxy)} gradient")
    axes[0].set_ylabel(quantity_unit_label(r"$Z/Z_\odot$"), fontsize=fontsize)
    axes[0].set_ylim(0, metallicity_max)
    axes[1].set_ylabel(quantity_unit_label(r"$d\log_{10}Z/dR\;[\mathrm{dex\,kpc}^{-1}]$"), fontsize=fontsize)
    axes[1].set_ylim(bottom=gradient_min, top=0)
    if gradient_min is not None:
        axes[1].set_yticks([tick for tick in [-0.20, -0.15, -0.10, -0.05, 0.00] if tick >= gradient_min])
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[1].yaxis.set_minor_locator(AutoMinorLocator(5))
    axes[1].set_xlim(0, float(radius[-1]))
    if abundance_spline:
        include_spline_gradient(axes[1], comparison)
    return observation_handle,gradient_handle


def plot_metallicity(args, cases, profiles, abundance, fit, *,
                     stem="section3_mixed_analytic_metallicity_scans", inactive_beta=False,
                     gradient_min=-0.2, title=None, legend_position="top", legend_single_row=False,
                     legend_parameter_columns=False, metallicity_max=3):
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharex=True)
    observation_handle,gradient_handle=draw_metallicity(
        axes,args,cases,profiles,abundance,fit,inactive_beta=inactive_beta,
        gradient_min=gradient_min,metallicity_max=metallicity_max)
    layout(figure, axes, cases, vertical=False, title=title,
           legend_position=legend_position, legend_single_row=legend_single_row,
           extra_handles=[gradient_handle] if legend_parameter_columns else (),
           legend_parameter_columns=legend_parameter_columns,
           panel_columns=2 if legend_parameter_columns else 1)
    axes[0].legend(handles=[observation_handle], loc="upper right", fontsize=8, frameon=False,
                   handlelength=1.8, borderaxespad=0.7, markerscale=1.2)
    save(figure, args, stem)


def add_velocity_inset(axis,cases,profiles,limits,radius_limits=None):
    """Zoom the saved radial velocities while retaining the full parent view."""
    inset=axis.inset_axes([.10,.12,.48,.50],zorder=5)
    for case in cases:
        if case["group"] in CHEMISTRY_GROUPS:
            continue
        profile=profiles[case["case"]]
        style=case_style(case)
        style["linewidth"]*=.7
        inset.plot(profile["R_kpc"],profile["v_R_kms"],**style)
    inset.set_xlim(*(axis.get_xlim() if radius_limits is None else radius_limits))
    inset.set_ylim(*limits)
    inset.set_xticks(np.linspace(*inset.get_xlim(),num=5))
    inset.set_yticks([limits[0],(limits[0]+limits[1])/2,limits[1]])
    inset.minorticks_off()
    inset.tick_params(which="major",direction="in",top=True,right=True, labelsize=6,length=2.5,width=.5,pad=1.5)
    for spine in inset.spines.values():
        spine.set_linewidth(.6)
    return inset


def plot_flow_metallicity(args, cases, profiles, disk_fit, abundance, fit, *, stem,
                          title, gradient_min=-0.2, metallicity_max=3, inactive_beta=False,
                          legend_parameter_columns=False, j_display="deficit", nuclear_j=None,
                          legend_spec=None, figure_size=(7.2,7.2),velocity_inset_limits=None,
                          velocity_inset_radius_limits=None,flow_y_limits=None,gradient_legend=True,
                          angular_comparison_label=None,show_disk_inputs=True,
                          legend_position="bottom",chemistry_cases=None,angular_panel_top_right=False,
                          axis_fontsize=None,tick_fontsize=None,angular_legend_fontsize=6.5,
                          angular_legend_loc="lower right",
                          abundance_spline=False, show_abundance_measurements=False):
    """Combine flow and chemistry, optionally omitting gas and stellar-sink panels."""
    if show_disk_inputs:
        figure,axes=plt.subplots(4,2,figsize=figure_size,sharex=True, gridspec_kw={"height_ratios":[1,1,1,2]})
    else:
        figure,axes=plt.subplots(3,2,figsize=figure_size,sharex=True)
    axes=axes.ravel()
    if show_disk_inputs:
        flow_axes=axes[:6]
    elif angular_panel_top_right:
        flow_axes=[axes[0],None,None,axes[3],axes[2],axes[1]]
    else:
        flow_axes=[axes[0],None,None,axes[1],axes[2],axes[3]]
    chemistry_axes=axes[-2:]
    draw_flow(flow_axes,args,cases,profiles,disk_fit,required_j=True,
              j_display=j_display,nuclear_j=nuclear_j,
              angular_comparison_label=angular_comparison_label)
    observation_handle,gradient_handle=draw_metallicity(
        chemistry_axes,args,cases if chemistry_cases is None else chemistry_cases,
        profiles,abundance,fit,inactive_beta=inactive_beta,
        gradient_min=gradient_min,metallicity_max=metallicity_max,fontsize=8.5,
        abundance_spline=abundance_spline)
    if abundance_spline and show_abundance_measurements:
        observation_handle = plot_abundance_measurements(chemistry_axes[0], abundance, args.galaxy)
    for index,limits in (flow_y_limits or {}).items():
        if flow_axes[index] is not None:
            flow_axes[index].set_ylim(*limits)
            if index==5 and j_display=="deficit":
                lower,upper=flow_axes[index].get_ylim()
                if (lower,upper)==(0,1):
                    ticks=[0,.25,.5,.75,1]
                elif (lower,upper)==DEFICIT_Y_LIMITS:
                    ticks=[-1,0,1,2]
                else:
                    ticks=MaxNLocator(nbins=4,steps=[1,2,5,10]).tick_values(lower,upper)
                    ticks=np.unique(np.r_[lower,ticks[(ticks>lower)&(ticks<upper)],upper])
                flow_axes[index].set_yticks(ticks,labels=[f"{tick:g}" for tick in ticks])
    # Include these local keys before fitting margins and gutters to their extents.
    if angular_comparison_label:
        flow_axes[5].legend(handles=[Line2D([],[],color="black",label="required"),
                               Line2D([],[],color="black",linestyle="--",label=angular_comparison_label)],
                       loc=angular_legend_loc,
                       fontsize=angular_legend_fontsize,frameon=False,handlelength=1.6)
    if not abundance_spline or show_abundance_measurements:
        chemistry_axes[0].legend(handles=[observation_handle],loc="upper right",fontsize=8,
                       frameon=False,handlelength=1.8,borderaxespad=0.5)
    if gradient_legend and not abundance_spline:
        chemistry_axes[1].legend(handles=[gradient_handle],
                       loc="upper right" if show_disk_inputs else "lower right",fontsize=8,
                       frameon=False,handlelength=1.8,borderaxespad=0.5)
    # A blank slot under the fiducial keeps each parameter pair in one column.
    extra_handles=([fit_legend_handle(f"{galaxy_label(args.galaxy)} spline fit")] if show_disk_inputs
                   else [Line2D([],[],linestyle="none",label=" ")])
    if axis_fontsize is not None:
        for axis in axes:
            axis.yaxis.label.set_fontsize(axis_fontsize)
    layout(figure,axes,cases,vertical=True,panel_columns=2,title=title,
           legend_position=legend_position,top_radius_label=True,extra_handles=extra_handles,
           legend_parameter_columns=legend_parameter_columns,legend_spec=legend_spec,
           axis_fontsize=axis_fontsize,tick_fontsize=tick_fontsize)
    if velocity_inset_limits is not None:
        add_velocity_inset(flow_axes[3],cases,profiles,velocity_inset_limits,velocity_inset_radius_limits)
    save(figure,args,stem)
    return figure


def plot_analytic_flow_metallicity(args, cases, profiles, disk_fit, abundance, fit):
    """Four flow panels and two chemistry panels, all with equal height."""
    figure,axes=plt.subplots(3,2,figsize=(7.2,4.5),sharex=True, gridspec_kw={"height_ratios":[1,1,1]})
    axes=axes.ravel()
    rates,angular,accretion,velocity=axes[:4]
    draw_flow([rates,rates,None,velocity,accretion],args,cases,profiles,disk_fit, combined_rates=True)
    rates.set_ylim(1e4,1e7)
    for case in cases:
        profile=profiles[case["case"]]
        angular.plot(profile["R_kpc"], 1-profile["j_land_kpc_kms"]/profile["j_disk_kpc_kms"], **case_style(case))
    angular.set_ylabel(quantity_unit_label(r"$1-j_{\rm land}/j$"),fontsize=9,labelpad=3)
    angular.set_ylim(0,1)
    angular.set_yticks([0,.25,.5,.75,1])
    angular.yaxis.set_minor_locator(AutoMinorLocator(5))
    velocity.set_ylim(-2,0)
    accretion.set_ylim(0,.5)
    rates.set_ylabel(quantity_unit_label(r"$\dot\Sigma\;[M_\odot\,\mathrm{Gyr}^{-1}\,\mathrm{kpc}^{-2}]$"),
        fontsize=9,labelpad=3)
    rates.legend(handles=[Line2D([],[],color="black",label=r"$\dot\Sigma_{\rm land}$"),
                          Line2D([],[],color="black",linestyle="--",dashes=(2,1.25), label=r"$\dot\Sigma_\star$")],
                 loc="lower left",fontsize=8,handlelength=2,labelspacing=.2)
    draw_metallicity(axes[4:],args,cases,profiles,abundance,fit,fontsize=9,gradient_min=-.1, abundance_spline=True)
    # Fixed columns align fiducial/reference and each lower/higher pair.
    case_handles=[Line2D([],[],label=case["label"],**case_style(case)) for case in cases]
    legend_handles=[*case_handles,fit_legend_handle(f"{galaxy_label(args.galaxy)} spline fit"),
                    Line2D([],[],color="none",label=" ")]
    legend_spec={"handles":legend_handles,"labels":[handle.get_label() for handle in legend_handles],
                 "ncol":6,"fontsize":8}
    layout(figure,axes,cases,vertical=True,panel_columns=2,title=MODEL_TITLE,
           legend_position="bottom",top_radius_label=True,
           axis_fontsize=9,tick_fontsize=8,
           legend_spec=legend_spec,
           extra_handles=[fit_legend_handle(f"{galaxy_label(args.galaxy)} spline fit")])
    save(figure,args,"section3_mixed_analytic_combined_scans")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/idealized/section3_mixed_scans.csv")
    parser.add_argument("--cases", type=Path, default=ROOT / "config/idealized/section3_mixed_cases.csv")
    parser.add_argument("--run-directory", type=Path, default=ROOT / "output/section3_mixed_analytic")
    parser.add_argument("--figure-directory", type=Path, default=ROOT / "output/pdf")
    parser.add_argument("--observation-directory", type=Path, default=ROOT / "output/section3")
    parser.add_argument("--galaxy", default="NGC2403", help="Comparison galaxy only; never a solver input")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="Reuse matching saved model profiles")
    args = parser.parse_args()
    if not args.skip_build and not args.plot_only:
        run(["make", "-j4", "build/galactic-nuclear-fountain-model"], args.run_directory / "logs/build.log")
    if not RUNNER.is_file():
        raise RuntimeError("Build the C++ model runner first")
    base, cases = parameters(args.config), read_csv(args.cases)
    profiles, diagnostics, resolved = calculate_cases(args, base, cases)
    disk_fit, abundance, fit, observation_paths = observations(args, base)
    core_sources = [ROOT / "model" / name for name in (
        "units.h",
        "physics.cpp", "physics.h", "models.cpp", "models.h", "rk4.cpp", "rk4.h", "model_runner.cpp")]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "mixed-power-law; prescribed landing and j_land=j_mix; derived stellar sink, KS gas and flow; flat rotation",
        "chemistry": "Numerical integration with assumed mixed incident composition and explicit outer abundance",
        "observations": "Saved gas/SFR fits and metallicity measurements are overlays only; no observed profile enters the model runner",
        "comparison_galaxy": args.galaxy,
        "profiles_reused": args.plot_only,
        "source_sha256": {str(path.resolve()): sha256(path) for path in
                           [args.config, args.cases, Path(__file__), ROOT / "scripts/plot/color_palette.py",
                            ROOT / "scripts/plot/observed_disk_comparison.py",
                            ROOT / "scripts/plot/observed_metallicity_spline.py",
                            ROOT / "scripts/plot/six_panel_layout.py",
                            ROOT / "scripts/plot/axis_labels.py",
                            ROOT / "scripts/plot/surface_rate_units.py",
                           RUNNER, *core_sources, *observation_paths]},
        "cases": resolved, "diagnostics": diagnostics,
        "figure": {"output":"section3_mixed_analytic_combined_scans.pdf",
                   "layout":"3x2","size_inches":[7.2,4.5],"row_height_ratios":[1,1,1],
                   "panels":["landing_and_stellar_sink","angular_deficit","accretion",
                             "radial_velocity","metallicity","metallicity_gradient"],
                   "angular_deficit_limits":[0,1],
                   "font_sizes_pt":{"axis_labels":9,"ticks":8,"legends":8,"title":10},
                   "radius_major_ticks_kpc":[0,5,10,15],"radius_minor_step_kpc":1,
                   "inset_count":0,
                   "gradient_minimum_view_dex_kpc":[-.1,0],
                   "gradient_reference":{"galaxy":args.galaxy,"calibration":"KK04",
                                         "line":"exact derivative of weighted cubic OH12 spline",
                                         "smoothing":"GCV","band":None,
                                         "view_expands_to_negative_spline":True,
                                         "upper_limit_dex_kpc":0,"local_legend":False},
                   "velocity_limits_kms":[-2,0],"accretion_limits_Msun_yr":[0,.5],
                   "surface_rate_limits_Msun_Gyr_kpc2":[1e4,1e7],
                   "legend_columns":["fiducial and unmixed reference","lambda","nuclear source","mu","beta","observed fit"],
                   "abundance_local_legend":False,
                   "all_cases_in_every_panel":True,
                   "rate_styles":{"landing":"solid","stellar_sink":"dashed"}},
    }
    (args.run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.run_directory / "figure_notes.txt").write_text(
        "SECTION 3.1 MIXED POWER-LAW DISK SCANS\n\n"
        "All lambda, nuclear-source, mu and beta scans are overlaid together on each quantity's axis.\n"
        "All other inputs are fixed; each case has its own color in one shared legend.\n"
        "The fiducial is black. Case colors use the shared Petroff rank-3 ten-color set.\n"
        "Beta=0.5 is dashed and beta=1 is dotted; the unmixed reference is solid gray.\n"
        "All flow and metallicity panels include an unmixed fiducial with mu=0 and all other\n"
        "fiducial inputs fixed, including the nuclear source. Its total landing is eta*Mdot_nuc.\n"
        "Blank overrides in the cases CSV use the explicitly selected baseline config.\n"
        "Total landing is eta*(1+mu)*Mdot_nuc; the mu scan changes mass supply and composition.\n"
        "Prescribed: normalized power-law landing, j_CGM=beta*j_disk, constant mu,\n"
        "j_land=(j_nuc+mu*j_CGM)/(1+mu), flat rotation, net KS law and mixed incident Z.\n"
        "Derived: stellar sink, gas density, radial velocity and mass flux at every radius,\n"
        "including the outer edge. No gas mass/scale length or outer mass flux is supplied.\n"
        "No observed spline enters the solver; saved NGC 2403 gas/SFR fits are comparison overlays.\n"
        "One combined 3x2 figure: landing and net stellar sink together; angular deficit;\n"
        "radial accretion; radial velocity; metallicity and gradient.\n"
        "The angular deficit uses core-exported j_land/j_disk and spans 0 to 1.\n"
        "All flow and metallicity/gradient panels have equal height.\n"
        "All ten cases are drawn in every panel, including the gray unmixed reference.\n"
        "The rate panel uses case colors, solid landing and dashed stellar-sink curves.\n"
        "The saved stellar-sink fit is a dotted black comparison; the two-row case legend is below.\n"
        "Fixed legend columns align fiducial/unmixed reference, lambda, nuclear source, mu, beta and observed fit.\n"
        "Beta enters dynamics and chemistry. At lambda=2 the beta-only stellar/gas profiles\n"
        "coincide, because the beta term cancels from the stellar-sink ratio for this slope.\n"
        "Source normalization cancels from the metallicity equation, so source-rate cases\n"
        "share the fiducial Z and gradient up to numerical roundoff.\n"
        "Bottom row: predicted Z/Z_sun on the left and its core-equation gradient on the right.\n"
        "Dotted black comparisons show the uncertainty-weighted NGC 2403 KK04\n"
        "cubic smoothing spline and its exact dlog10(Z)/dR derivative on measured support.\n"
        "GCV selects smoothing; adopted abundance uncertainties include the saved scatter term.\n"
        "The gradient view includes the negative derivative and stays capped at zero; no population band is shown.\n"
        "The disk/source/boundary parameters are illustrative and not fitted to these data.\n"
        "Gas, stellar sink and flow are analytic. Metallicity is integrated with the shared C++\n"
        "RK4 solver and an outer abundance boundary. Mass, angular momentum, KS and metal\n"
        "conservation are checked for every case.\n\n"
        "FIDUCIAL CONFIGURATION\n" + "".join(f"{key}: {value}\n" for key, value in resolved["fiducial"].items())
    )
    smplotlib.set_style(usetex=False, fontsize=9, dpi=144)
    plt.rcParams.update({"axes.prop_cycle": color_cycle(), "axes.grid": False, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": None})
    figure=plot_analytic_flow_metallicity(args, cases, profiles, disk_fit, abundance, fit)
    manifest["figure"].update(size_inches=figure.get_size_inches().tolist(),
                              inter_row_text_clearance_pt=4,panel_sizes_preserved=True)
    (args.run_directory / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
