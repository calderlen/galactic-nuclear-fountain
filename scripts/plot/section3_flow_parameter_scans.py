#!/usr/bin/env python3
"""Plot the analytic and observational no-CGM models used in Section 3.

The calculation follows the manuscript assumptions in this order:

    dotSigma_land = C R^-lambda, normalized on [R_in, R_out]
    dotSigma_star = dotSigma_land [3-lambda + (lambda-1) R_nuc/R]
    dotSigma_star = A Sigma_g^N
    v_R = -dotSigma_land (R - R_nuc) / Sigma_g
    dotM_acc = 2 pi R Sigma_g |v_R|

There is no entrained CGM in this model (mu=0), so beta is not a parameter.
The analytic gas profile is inferred from the Kennicutt-Schmidt closure.  The
observational calculation instead inserts the measured gas and star-formation
profiles, first testing a prescribed R^-2 wind and then inferring the wind
profile required for exact steady state.
"""

from __future__ import annotations

from axis_labels import quantity_unit_label

import argparse
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
from color_palette import color_cycle
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
import numpy as np

from hershey_fonts import register_hershey_weight_aliases
from power_law_model import power_law_profiles
from core_calculations import run_calculation
from observed_disk_comparison import load_disk_fit, plot_disk_fit, fit_legend_handle, flow_legends
from surface_rate_units import read_rate_rows, YEARS_PER_GYR


register_hershey_weight_aliases()
import smplotlib


SINGLE_COLUMN_FIGURE_SIZE = (3.5, 7.0)


smplotlib.set_style(usetex=False, fontsize=9, figsize=SINGLE_COLUMN_FIGURE_SIZE, dpi=144)
plt.rcParams.update(
    {"axes.prop_cycle": color_cycle(),
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": True,
        "legend.frameon": False,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "savefig.bbox": None,
    }
)


SURFACE_RATE_PLOT_SCALE = 1.0  # Retain native Msun/Gyr/kpc^2 units.
SURFACE_DENSITY_PLOT_SCALE = 1.0e-6  # Msun/kpc^2 -> Msun/pc^2
FIDUCIAL_RAW_KS_A = 9.95267926383743e-4
FIDUCIAL_RETURN_FRACTION = 0.4
FIDUCIAL_NET_KS_A = (1.0 - FIDUCIAL_RETURN_FRACTION) * FIDUCIAL_RAW_KS_A
SOLAR_METALLICITY = 0.0139
MODEL_CURVE_ZORDER = 2
STAR_FORMATION_LINESTYLE = (0, (2.0, 1.25))
STAR_FORMATION_LINEWIDTH = 0.75
ROOT = Path(__file__).resolve().parents[2]
NGC2403_INPUT = ROOT / "input" / "model_inputs" / "NGC2403"
NGC2403_COMPARISON_INPUT = ROOT / "output/section3/inputs/NGC2403/observed_disk"


@dataclass(frozen=True)
class FlowParameters:
    """Parameters for one steady, non-mixing Section 3 profile."""

    label: str
    color: str
    R_w_kpc: float = 0.5
    R_out_kpc: float = 15.0
    Mdot_w_Msun_yr: float = 0.5
    KS_A_net: float = FIDUCIAL_NET_KS_A
    KS_N: float = 1.4
    Z_w: float = SOLAR_METALLICITY
    yield_y: float = 0.01
    Z_outer: float = 0.3 * SOLAR_METALLICITY
    landing_slope: float = 2.0
    R_in_kpc: float | None = None  # None follows R_nuc, preserving the original cases.


@dataclass(frozen=True)
class ObservedProfile:
    """Continuous observed disk profiles used by Sections 3.2 and 3.3."""

    radius_kpc: np.ndarray
    sigma_g_Msun_kpc2: np.ndarray
    sigmadot_star_Msun_Gyr_kpc2: np.ndarray
    legacy_return_correction_applied: bool


FIDUCIAL_COLOR = "black"
MDOTW_MID_COLOR = "#0072B2"
MDOTW_HIGH_COLOR = "#009E73"
RW_LOW_COLOR = "#E69F00"
RW_HIGH_COLOR = "#D55E00"
FIDUCIAL = FlowParameters(label="fiducial", color=FIDUCIAL_COLOR)


MODEL_CASES = (
    FIDUCIAL,
    replace(
        FIDUCIAL,
        label=r"$\dot{M}_{\rm land}=0.1\,M_\odot\,\mathrm{yr}^{-1}$",
        color=MDOTW_MID_COLOR,
        Mdot_w_Msun_yr=0.1,
    ),
    replace(
        FIDUCIAL,
        label=r"$\dot{M}_{\rm land}=1.0\,M_\odot\,\mathrm{yr}^{-1}$",
        color=MDOTW_HIGH_COLOR,
        Mdot_w_Msun_yr=1.0,
    ),
    replace(FIDUCIAL, label=r"$R_{\rm nuc}=0.1\,\mathrm{kpc}$", color=RW_LOW_COLOR, R_w_kpc=0.1),
    replace(FIDUCIAL, label=r"$R_{\rm nuc}=1\,\mathrm{kpc}$", color=RW_HIGH_COLOR, R_w_kpc=1.0),
    replace(FIDUCIAL, label=r"$\lambda=3$", color="#882255", landing_slope=3.0),
    replace(FIDUCIAL, label=r"$\lambda=2.5$", color="#AA4499", landing_slope=2.5),
    replace(FIDUCIAL, label=r"$\lambda=1.5$", color="#332288", landing_slope=1.5),
)


def inner_radius(parameters: FlowParameters) -> float:
    return parameters.R_w_kpc if parameters.R_in_kpc is None else parameters.R_in_kpc


def analytic_profiles(radius_kpc, parameters: FlowParameters) -> dict[str, np.ndarray]:
    """Use the core solution rather than maintaining equations in the renderer."""
    return power_law_profiles(radius_kpc, {
        "R_nucl_kpc": parameters.R_w_kpc,
        "R_in_kpc": inner_radius(parameters),
        "R_out_kpc": parameters.R_out_kpc,
        "Mdot_land_Msun_yr": parameters.Mdot_w_Msun_yr,
        "KS_A_net": parameters.KS_A_net,
        "KS_N": parameters.KS_N,
        "lambda": parameters.landing_slope,
        "Z_land": parameters.Z_w,
        "yield_y": parameters.yield_y,
        "Z_outer_boundary": parameters.Z_outer,
    })


def read_csv(path: Path) -> list[dict[str, str]]:
    return read_rate_rows(path)


def read_parameter_values(path: Path) -> dict[str, str]:
    return {row["name"]: row["value"] for row in read_csv(path)}


def load_ngc2403_profile(parameters: FlowParameters) -> tuple[ObservedProfile, FlowParameters]:
    """Load the exported dense gas/SFR splines on their supported radial range."""
    rows = read_csv(NGC2403_INPUT / "profiles.csv")
    radius = np.asarray([float(row["R_kpc"]) for row in rows])
    gas = np.asarray([float(row["Sigma_g_Msun_kpc2"]) for row in rows])
    star = np.asarray([float(row["Sigmadot_star_Msun_Gyr_kpc2"]) for row in rows])

    if np.any(~np.isfinite(radius)) or np.any(np.diff(radius) <= 0.0):
        raise ValueError("Observed profile radii must be finite and strictly increasing")
    if np.any(~np.isfinite(gas)) or np.any(gas <= 0.0):
        raise ValueError("Observed gas surface densities must be finite and positive")
    if np.any(~np.isfinite(star)) or np.any(star <= 0.0):
        raise ValueError("Observed stellar sinks must be finite and positive")
    if not radius[0] <= parameters.R_w_kpc < radius[-1]:
        raise ValueError("R_w must lie inside the observed profile support")

    saved_parameters = read_parameter_values(NGC2403_INPUT / "parameters.csv")
    legacy_correction = saved_parameters.get("stellar_sink") != "net_locked_mass"
    if legacy_correction:
        star = (1.0 - FIDUCIAL_RETURN_FRACTION) * star

    R_out = min(parameters.R_out_kpc, radius[-1])
    effective_parameters = replace(parameters, R_out_kpc=R_out)
    selected = (radius > parameters.R_w_kpc) & (radius <= R_out)
    model_radius = np.concatenate(([parameters.R_w_kpc], radius[selected]))
    profile = ObservedProfile(
        radius_kpc=model_radius,
        sigma_g_Msun_kpc2=np.interp(model_radius, radius, gas),
        sigmadot_star_Msun_Gyr_kpc2=np.interp(model_radius, radius, star),
        legacy_return_correction_applied=legacy_correction,
    )
    return profile, effective_parameters


def validate(parameters: FlowParameters) -> None:
    if not 0.0 < parameters.R_w_kpc <= inner_radius(parameters) < parameters.R_out_kpc:
        raise ValueError("Require 0 < R_nuc <= R_in < R_out")
    if parameters.Mdot_w_Msun_yr <= 0.0:
        raise ValueError("Mdot_w must be positive")
    if parameters.KS_A_net <= 0.0 or parameters.KS_N <= 0.0:
        raise ValueError("Kennicutt-Schmidt parameters must be positive")


def radius_grid(parameters: FlowParameters, samples: int) -> np.ndarray:
    validate(parameters)
    return np.linspace(inner_radius(parameters), parameters.R_out_kpc, samples)


def sigmadot_wind(radius_kpc, parameters: FlowParameters) -> np.ndarray:
    """Wind landing surface density whose annular integral is Mdot_w."""
    return analytic_profiles(radius_kpc, parameters)["Sigmadot_land_Msun_Gyr_kpc2"]


def sigmadot_star(radius_kpc, parameters: FlowParameters) -> np.ndarray:
    """Net stellar mass-lockup sink implied by steady mass conservation."""
    return analytic_profiles(radius_kpc, parameters)["Sigmadot_star_Msun_Gyr_kpc2"]


def sigma_gas(radius_kpc, parameters: FlowParameters) -> np.ndarray:
    """Gas density inferred from dotSigma_star = A Sigma_g^N."""
    return analytic_profiles(radius_kpc, parameters)["Sigma_g_Msun_kpc2"]


def radial_velocity_kms(radius_kpc, parameters: FlowParameters) -> np.ndarray:
    return analytic_profiles(radius_kpc, parameters)["v_R_kms"]


def accretion_rate(radius_kpc, parameters: FlowParameters) -> np.ndarray:
    """Positive inward mass flux through radius R."""
    return analytic_profiles(radius_kpc, parameters)["Mdot_acc_Msun_yr"]


def verify_profiles(parameters: FlowParameters, samples: int) -> None:
    """Check normalization and algebraic closure before plotting."""
    radius = np.geomspace(inner_radius(parameters), parameters.R_out_kpc, max(samples, 10_001))
    values = analytic_profiles(radius, parameters)
    integrated_wind = np.trapezoid(2.0 * np.pi * radius * values["Sigmadot_land_Msun_Gyr_kpc2"], radius) / YEARS_PER_GYR
    if not math.isclose(integrated_wind, parameters.Mdot_w_Msun_yr, rel_tol=3.0e-7):
        raise RuntimeError("Wind surface-density profile failed its mass normalization")

    gas = values["Sigma_g_Msun_kpc2"]
    star = values["Sigmadot_star_Msun_Gyr_kpc2"]
    if not np.allclose(parameters.KS_A_net * gas**parameters.KS_N, star, rtol=1.0e-12):
        raise RuntimeError("Gas profile failed the Kennicutt-Schmidt closure")

    velocity_kpc_per_year = values["v_R_kpc_yr"]
    flux_from_velocity = 2.0 * np.pi * radius * gas * np.abs(velocity_kpc_per_year)
    if not np.allclose(flux_from_velocity, values["Mdot_acc_Msun_yr"], rtol=1.0e-12):
        raise RuntimeError("Accretion profile failed the radial-flux identity")


def core_parameters(parameters):
    return {"R_nucl_kpc": parameters.R_w_kpc, "R_in_kpc": inner_radius(parameters),
            "R_out_kpc": parameters.R_out_kpc, "Mdot_land_Msun_yr": parameters.Mdot_w_Msun_yr,
            "lambda": parameters.landing_slope, "Z_land": parameters.Z_w,
            "yield_y": parameters.yield_y, "Z_outer_boundary": parameters.Z_outer}


def metallicity_profile(radius_kpc, sigmadot_star, sigmadot_wind_profile, parameters):
    """Use the core metallicity ODE with the original outer and regular-inner limits."""
    rows = run_calculation("metallicity",
        ("R_kpc", "Sigmadot_star_Msun_Gyr_kpc2", "Sigmadot_land_Msun_Gyr_kpc2"),
        zip(radius_kpc, sigmadot_star, sigmadot_wind_profile), core_parameters(parameters))["values"]
    return np.asarray([float(row["Z"]) for row in rows])


def observational_calculations(profile: ObservedProfile, parameters: FlowParameters):
    """Run the original flat-rotation, regular unmixed experiment in the C++ core."""
    tables = run_calculation("unmixed-observed",
        ("R_kpc", "Sigma_g_Msun_kpc2", "Sigmadot_star_Msun_Gyr_kpc2"),
        zip(profile.radius_kpc, profile.sigma_g_Msun_kpc2, profile.sigmadot_star_Msun_Gyr_kpc2),
        core_parameters(parameters))
    rows = tables["values"]
    values = {"R_kpc": profile.radius_kpc, "Sigma_g_obs_Msun_kpc2": profile.sigma_g_Msun_kpc2,
              "Sigmadot_star_obs_Msun_Gyr_kpc2": profile.sigmadot_star_Msun_Gyr_kpc2,
              **{key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}}
    return values, float(tables["summary"][0]["Mdot_land_required_Msun_yr"])


def write_observational_csv(path: Path, values: dict[str, np.ndarray]) -> None:
    columns = tuple(values)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*(values[column] for column in columns)))


def make_observational_figure(values: dict[str, np.ndarray], parameters: FlowParameters, required_total: float):
    """Plot the prescribed-wind test and steady observational inversion."""
    radius = values["R_kpc"]
    figure, axes = plt.subplots(4, 1, figsize=SINGLE_COLUMN_FIGURE_SIZE, sharex=True, gridspec_kw={"hspace": 0.08})
    rate_axis, velocity_axis, residual_axis, metallicity_axis = axes

    rate_axis.plot(
        radius,
        values["Sigmadot_star_obs_Msun_Gyr_kpc2"] * SURFACE_RATE_PLOT_SCALE,
        color="0.45",
        linestyle=STAR_FORMATION_LINESTYLE,
        label=r"observed $\dot{\Sigma}_\star$",
    )
    rate_axis.plot(
        radius,
        values["Sigmadot_w_prescribed_Msun_Gyr_kpc2"] * SURFACE_RATE_PLOT_SCALE,
        color="#0072B2",
        label=r"prescribed $R^{-2}$",
    )
    rate_axis.plot(
        radius,
        values["Sigmadot_w_required_Msun_Gyr_kpc2"] * SURFACE_RATE_PLOT_SCALE,
        color="#D55E00",
        label="required",
    )
    rate_axis.plot(
        radius,
        values["Sigmadot_w_Rm2_same_total_Msun_Gyr_kpc2"]
        * SURFACE_RATE_PLOT_SCALE,
        color="black",
        linestyle=":",
        label=r"$R^{-2}$, same total",
    )
    rate_axis.set_yscale("log")
    rate_axis.set_ylabel(quantity_unit_label(r"$\dot{\Sigma}\;[M_\odot\,\mathrm{Gyr}^{-1}\,\mathrm{kpc}^{-2}]$"))
    rate_axis.legend(fontsize=6.7, ncol=2, columnspacing=0.8, handlelength=1.8)

    velocity_axis.plot(radius, values["v_R_prescribed_kms"], color="#0072B2", label="prescribed")
    velocity_axis.plot(radius, values["v_R_required_kms"], color="#D55E00", label="required")
    velocity_axis.axhline(0.0, color="0.65", linewidth=0.6)
    velocity_axis.set_ylabel(quantity_unit_label(r"$v_R\;[\mathrm{km\,s}^{-1}]$"))
    velocity_axis.legend(fontsize=7.0)

    residual_axis.plot(radius, values["residual_Msun_Gyr_kpc2"] * SURFACE_RATE_PLOT_SCALE, color="#0072B2")
    residual_axis.set_yscale("log")
    residual_axis.set_ylabel(
        quantity_unit_label("Gas accumulation rate\n" r"$[M_\odot\,\mathrm{Gyr}^{-1}\,\mathrm{kpc}^{-2}]$")
    )

    metallicity_axis.plot(radius, values["Z_prescribed"] / SOLAR_METALLICITY, color="#0072B2", label="prescribed")
    metallicity_axis.plot(radius, values["Z_required"] / SOLAR_METALLICITY, color="#D55E00", label="required")
    metallicity_axis.set_ylabel(quantity_unit_label(r"$Z\;[Z_\odot]$"))
    metallicity_axis.set_xlabel(r"$R\;[\mathrm{kpc}]$")
    metallicity_axis.legend(fontsize=7.0)

    for axis in axes:
        axis.set_xlim(parameters.R_w_kpc, parameters.R_out_kpc)
        axis.tick_params(axis="y", which="both", right=True, labelright=False)
    figure.suptitle(
        rf"NGC 2403; inferred $\dot{{M}}_{{\rm land}}={required_total:.2f}\,M_\odot\,\mathrm{{yr}}^{{-1}}$",
        fontsize=8.0,
        y=0.99,
    )
    figure.subplots_adjust(left=0.21, right=0.95, bottom=0.075, top=0.945, hspace=0.08)

    figure.align_ylabels(axes)
    return figure


def make_figure(samples: int):
    cases = MODEL_CASES
    figure, axes = plt.subplots(4, 1, figsize=SINGLE_COLUMN_FIGURE_SIZE, sharex=True, gridspec_kw={"hspace": 0.08})
    rates_axis, gas_axis, velocity_axis, accretion_axis = axes
    rates_axis.set_yscale("log")
    gas_axis.set_yscale("log")
    velocity_axis.set_yscale("linear")

    disk_fit = load_disk_fit(NGC2403_COMPARISON_INPUT, FIDUCIAL_RETURN_FRACTION)
    plot_disk_fit(rates_axis,disk_fit,"star","spline fit",scale=SURFACE_RATE_PLOT_SCALE)
    plot_disk_fit(gas_axis,disk_fit,"gas","spline fit")

    for parameters in cases:
        verify_profiles(parameters, samples)
        radius = radius_grid(parameters, samples)
        values = analytic_profiles(radius, parameters)
        rates_axis.plot(
            radius,
            values["Sigmadot_land_Msun_Gyr_kpc2"] * SURFACE_RATE_PLOT_SCALE,
            color=parameters.color,
            linestyle="-",
            zorder=MODEL_CURVE_ZORDER,
        )
        rates_axis.plot(
            radius,
            values["Sigmadot_star_Msun_Gyr_kpc2"] * SURFACE_RATE_PLOT_SCALE,
            color=parameters.color,
            linestyle=STAR_FORMATION_LINESTYLE,
            linewidth=STAR_FORMATION_LINEWIDTH,
            zorder=MODEL_CURVE_ZORDER,
        )
        gas_axis.plot(
            radius,
            values["Sigma_g_Msun_kpc2"] * SURFACE_DENSITY_PLOT_SCALE,
            color=parameters.color,
            zorder=MODEL_CURVE_ZORDER,
        )
        velocity = values["v_R_kms"]
        velocity_axis.plot(radius, velocity, color=parameters.color, zorder=MODEL_CURVE_ZORDER)
        accretion_axis.plot(radius, values["Mdot_acc_Msun_yr"], color=parameters.color, zorder=MODEL_CURVE_ZORDER)

    rates_axis.set_ylim(bottom=1.0e3, top=1.0e7)
    rates_axis.yaxis.set_major_locator(LogLocator(base=10,numticks=12))
    rates_axis.yaxis.set_minor_locator(LogLocator(base=10,subs=range(2,10),numticks=100))
    gas_axis.set_ylim(bottom=1.0, top=10.0)
    velocity_axis.set_ylim(bottom=-4.0, top=0.0)
    accretion_axis.set_ylim(bottom=0.0, top=0.3)
    rates_axis.set_ylabel(quantity_unit_label(r"$\dot{\Sigma}\;[M_\odot\,\mathrm{Gyr}^{-1}\,\mathrm{kpc}^{-2}]$"))
    gas_axis.set_ylabel(quantity_unit_label(r"$\Sigma_g\;[M_\odot\,\mathrm{pc}^{-2}]$"))
    velocity_axis.set_ylabel(quantity_unit_label(r"$v_R\;[\mathrm{km\,s}^{-1}]$"))
    accretion_axis.set_ylabel(quantity_unit_label(r"$\dot{M}_{\rm acc}\;[M_\odot\,\mathrm{yr}^{-1}]$"))
    accretion_axis.set_xlabel(r"$R\;[\mathrm{kpc}]$")

    maximum_radius = max(parameters.R_out_kpc for parameters in cases)
    for axis in axes:
        axis.set_xlim(0.0, maximum_radius)
        axis.tick_params(axis="y", which="both", right=True, labelright=False)
    radius_ticks = np.linspace(0.0, maximum_radius, 7)
    accretion_axis.set_xticks(radius_ticks)
    top_radius_axis = rates_axis.secondary_xaxis("top")
    top_radius_axis.set_xticks(radius_ticks)
    top_radius_axis.set_xlabel(r"$R\;[\mathrm{kpc}]$", labelpad=9.0)
    top_radius_axis.tick_params(axis="x", which="both", direction="in", pad=2.0)
    top_radius_axis.minorticks_on()
    rates_axis.set_zorder(2)

    model_legend = rates_axis.legend(
        handles=(
            Line2D([], [], color=FIDUCIAL_COLOR, linestyle="-", label=r"$\dot{\Sigma}_{\rm land}$"),
            Line2D(
                [],
                [],
                color=FIDUCIAL_COLOR,
                linestyle=STAR_FORMATION_LINESTYLE,
                linewidth=STAR_FORMATION_LINEWIDTH,
                label=r"$\dot{\Sigma}_\star$",
            ),
        ),
        loc="upper right",
        fontsize=7.0,
        handlelength=2.2,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    rates_axis.add_artist(model_legend)
    handles=[Line2D([],[],color=case.color,label=case.label) for case in cases]
    handles.append(fit_legend_handle("NGC 2403 spline fit"))
    heading=figure.suptitle("Unmixed fountain, analytic solution",x=0.5,y=0.995,fontsize=10)
    parameter_legends=flow_legends(figure,handles,fontsize=8 if len(cases)>5 else 9,
                                 position="bottom",handlelength=1.1,handletextpad=.3)
    figure.subplots_adjust(left=0.205, right=0.95, bottom=0.075, top=0.88, hspace=0.08)
    figure.align_ylabels(axes)

    # Fit the title and bottom legend around the rendered radius labels.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    title_width=heading.get_window_extent(renderer).width
    if title_width>0.98*figure.bbox.width:
        heading.set_fontsize(heading.get_fontsize()*0.98*figure.bbox.width/title_width)
        figure.canvas.draw()
        renderer=figure.canvas.get_renderer()
    legend_axis_gap = 4.0 * figure.dpi / 72.0
    label_bounds = top_radius_axis.xaxis.label.get_window_extent(renderer)
    top_shift = (heading.get_window_extent(renderer).y0 - legend_axis_gap - label_bounds.y1) / figure.bbox.height
    legend_top = max(legend.get_window_extent(renderer).y1 for legend in parameter_legends)
    panel_bottom = accretion_axis.get_tightbbox(renderer).y0
    bottom_shift = (legend_top + legend_axis_gap - panel_bottom) / figure.bbox.height
    figure.subplots_adjust(top=figure.subplotpars.top + top_shift, bottom=figure.subplotpars.bottom + bottom_shift)
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the Section 3 analytic parameter variations and NGC 2403 "
            "forward/inverse observational tests."
        )
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("output/pdf"),
        help="directory for the combined parameter-variation PDF",
    )
    parser.add_argument("--samples", type=int, default=1000, help="radial samples per curve")
    arguments = parser.parse_args()
    if arguments.samples < 2:
        parser.error("--samples must be at least 2")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    figure = make_figure(arguments.samples)
    path = arguments.output_directory / "section3_flow_parameter_variations.pdf"

    figure.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(f"Wrote {path}")

    observed_profile, observed_parameters = load_ngc2403_profile(FIDUCIAL)
    observational_values, required_total = observational_calculations(observed_profile, observed_parameters)
    observational_csv = (arguments.output_directory / "section3_observational_tests.csv")
    write_observational_csv(observational_csv, observational_values)
    print(f"Wrote {observational_csv}")

    observational_figure = make_observational_figure(observational_values, observed_parameters, required_total)
    observational_pdf = (arguments.output_directory / "section3_observational_tests.pdf")

    observational_figure.savefig(observational_pdf, bbox_inches=None)
    plt.close(observational_figure)
    print(f"Wrote {observational_pdf}")
    print(f"Required steady landing rate: {required_total:.8g} Msun/yr")
    if observed_profile.legacy_return_correction_applied:
        print("Applied the 0.6 locked-mass correction to the legacy saved " "NGC 2403 SFR spline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
