#!/usr/bin/env python3
"""Plot the analytic and observational no-CGM models used in Section 3.

The calculation follows the manuscript assumptions in this order:

    dotSigma_w = dotM_w / [2 pi R^2 ln(R_out/R_w)]
    dotSigma_star = dotSigma_w (1 + R_w/R)
    dotSigma_star = A Sigma_g^N
    v_R = -dotSigma_w (R - R_w) / Sigma_g
    dotM_acc = 2 pi R Sigma_g |v_R|

There is no entrained CGM in this model (mu=0), so beta is not a parameter.
The analytic gas profile is inferred from the Kennicutt-Schmidt closure.  The
observational calculation instead inserts the measured gas and star-formation
profiles, first testing a prescribed R^-2 wind and then inferring the wind
profile required for exact steady state.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from hershey_fonts import register_hershey_weight_aliases


register_hershey_weight_aliases()
import smplotlib


SINGLE_COLUMN_FIGURE_SIZE = (3.5, 7.0)


smplotlib.set_style(
    usetex=False,
    fontsize=9,
    figsize=SINGLE_COLUMN_FIGURE_SIZE,
    dpi=144,
)
plt.rcParams.update(
    {
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


KPC_PER_YEAR_TO_KM_PER_SECOND = 3.0856775814913673e16 / 31557600.0
SURFACE_RATE_PLOT_SCALE = 1.0e3  # Msun/yr/kpc^2 -> Msun/Gyr/pc^2
SURFACE_DENSITY_PLOT_SCALE = 1.0e-6  # Msun/kpc^2 -> Msun/pc^2
FIDUCIAL_RAW_KS_A = 9.95267926383743e-13
FIDUCIAL_RETURN_FRACTION = 0.4
FIDUCIAL_NET_KS_A = (1.0 - FIDUCIAL_RETURN_FRACTION) * FIDUCIAL_RAW_KS_A
SOLAR_METALLICITY = 0.014
ERRORBAR_ZORDER = 1
MODEL_CURVE_ZORDER = 2
OBSERVATION_MARKER_ZORDER = 3
STAR_FORMATION_LINESTYLE = (0, (2.0, 1.25))
STAR_FORMATION_LINEWIDTH = 0.75
ROOT = Path(__file__).resolve().parents[2]
NGC2403_INPUT = ROOT / "input" / "model_inputs" / "NGC2403"


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


@dataclass(frozen=True)
class ObservationalSeries:
    label: str
    marker: str
    marker_face_color: str
    radius_kpc: np.ndarray
    sigma_g_Msun_kpc2: np.ndarray
    e_sigma_g_Msun_kpc2: np.ndarray
    sigmadot_star_Msun_yr_kpc2: np.ndarray
    e_sigmadot_star_Msun_yr_kpc2: np.ndarray


@dataclass(frozen=True)
class ObservedProfile:
    """Continuous observed disk profiles used by Sections 3.2 and 3.3."""

    radius_kpc: np.ndarray
    sigma_g_Msun_kpc2: np.ndarray
    sigmadot_star_Msun_yr_kpc2: np.ndarray
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
        label=r"$\dot{M}_w=0.1\,M_\odot\,\mathrm{yr}^{-1}$",
        color=MDOTW_MID_COLOR,
        Mdot_w_Msun_yr=0.1,
    ),
    replace(
        FIDUCIAL,
        label=r"$\dot{M}_w=1.0\,M_\odot\,\mathrm{yr}^{-1}$",
        color=MDOTW_HIGH_COLOR,
        Mdot_w_Msun_yr=1.0,
    ),
    replace(
        FIDUCIAL,
        label=r"$R_w=0.1\,\mathrm{kpc}$",
        color=RW_LOW_COLOR,
        R_w_kpc=0.1,
    ),
    replace(
        FIDUCIAL,
        label=r"$R_w=1\,\mathrm{kpc}$",
        color=RW_HIGH_COLOR,
        R_w_kpc=1.0,
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_parameter_values(path: Path) -> dict[str, str]:
    return {row["name"]: row["value"] for row in read_csv(path)}


def load_ngc2403_observations() -> tuple[ObservationalSeries, ...]:
    definitions = (
        (
            "Leroy+2008",
            "o",
            "black",
            NGC2403_INPUT / "leroy_profiles_used.csv",
            "e_SigmaSFR_Msun_yr_kpc2",
        ),
        (
            "Bigiel+2010",
            "o",
            "white",
            NGC2403_INPUT / "bigiel_profiles_used.csv",
            "e_SigmaSFR_Msun_yr_kpc2",
        ),
    )
    lockup_fraction = 1.0 - FIDUCIAL_RETURN_FRACTION
    result = []
    for label, marker, face_color, path, sfr_error_column in definitions:
        rows = read_csv(path)
        gas_scale = 1.36 if label == "Bigiel+2010" else 1.0
        result.append(
            ObservationalSeries(
                label=label,
                marker=marker,
                marker_face_color=face_color,
                radius_kpc=np.asarray([float(row["R_kpc"]) for row in rows]),
                sigma_g_Msun_kpc2=gas_scale * np.asarray(
                    [float(row["Sigma_g_Msun_kpc2"]) for row in rows]
                ),
                e_sigma_g_Msun_kpc2=gas_scale * np.asarray(
                    [float(row["e_Sigma_g_Msun_kpc2"]) for row in rows]
                ),
                sigmadot_star_Msun_yr_kpc2=lockup_fraction
                * np.asarray([float(row["SigmaSFR_Msun_yr_kpc2"]) for row in rows]),
                e_sigmadot_star_Msun_yr_kpc2=lockup_fraction
                * np.asarray([float(row[sfr_error_column]) for row in rows]),
            )
        )
    return tuple(result)


def load_ngc2403_profile(parameters: FlowParameters) -> tuple[ObservedProfile, FlowParameters]:
    """Load the exported dense gas/SFR splines on their supported radial range."""
    rows = read_csv(NGC2403_INPUT / "profiles.csv")
    radius = np.asarray([float(row["R_kpc"]) for row in rows])
    gas = np.asarray([float(row["Sigma_g_Msun_kpc2"]) for row in rows])
    star = np.asarray([float(row["Sigmadot_star_Msun_yr_kpc2"]) for row in rows])

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
        sigmadot_star_Msun_yr_kpc2=np.interp(model_radius, radius, star),
        legacy_return_correction_applied=legacy_correction,
    )
    return profile, effective_parameters


def validate(parameters: FlowParameters) -> None:
    if not 0.0 < parameters.R_w_kpc < parameters.R_out_kpc:
        raise ValueError("Require 0 < R_w < R_out")
    if parameters.Mdot_w_Msun_yr <= 0.0:
        raise ValueError("Mdot_w must be positive")
    if parameters.KS_A_net <= 0.0 or parameters.KS_N <= 0.0:
        raise ValueError("Kennicutt-Schmidt parameters must be positive")


def radius_grid(parameters: FlowParameters, samples: int) -> np.ndarray:
    validate(parameters)
    return np.linspace(parameters.R_w_kpc, parameters.R_out_kpc, samples)


def sigmadot_wind(radius_kpc, parameters: FlowParameters) -> np.ndarray:
    """Wind landing surface density whose annular integral is Mdot_w."""
    radius = np.asarray(radius_kpc, dtype=float)
    normalization = 2.0 * np.pi * np.log(parameters.R_out_kpc / parameters.R_w_kpc)
    return parameters.Mdot_w_Msun_yr / (normalization * radius**2)


def sigmadot_star(radius_kpc, parameters: FlowParameters) -> np.ndarray:
    """Net stellar mass-lockup sink implied by steady mass conservation."""
    radius = np.asarray(radius_kpc, dtype=float)
    return sigmadot_wind(radius, parameters) * (1.0 + parameters.R_w_kpc / radius)


def sigma_gas(radius_kpc, parameters: FlowParameters) -> np.ndarray:
    """Gas density inferred from dotSigma_star = A Sigma_g^N."""
    return (sigmadot_star(radius_kpc, parameters) / parameters.KS_A_net) ** (
        1.0 / parameters.KS_N
    )


def radial_velocity_kms(radius_kpc, parameters: FlowParameters) -> np.ndarray:
    radius = np.asarray(radius_kpc, dtype=float)
    velocity_kpc_per_year = -(
        sigmadot_wind(radius, parameters)
        * (radius - parameters.R_w_kpc)
        / sigma_gas(radius, parameters)
    )
    return velocity_kpc_per_year * KPC_PER_YEAR_TO_KM_PER_SECOND


def accretion_rate(radius_kpc, parameters: FlowParameters) -> np.ndarray:
    """Positive inward mass flux through radius R."""
    radius = np.asarray(radius_kpc, dtype=float)
    return (
        2.0
        * np.pi
        * radius
        * sigmadot_wind(radius, parameters)
        * (radius - parameters.R_w_kpc)
    )


def verify_profiles(parameters: FlowParameters, samples: int) -> None:
    """Check normalization and algebraic closure before plotting."""
    radius = np.geomspace(
        parameters.R_w_kpc,
        parameters.R_out_kpc,
        max(samples, 10_001),
    )
    integrated_wind = np.trapezoid(
        2.0 * np.pi * radius * sigmadot_wind(radius, parameters), radius
    )
    if not math.isclose(
        integrated_wind, parameters.Mdot_w_Msun_yr, rel_tol=3.0e-7
    ):
        raise RuntimeError("Wind surface-density profile failed its mass normalization")

    gas = sigma_gas(radius, parameters)
    star = sigmadot_star(radius, parameters)
    if not np.allclose(parameters.KS_A_net * gas**parameters.KS_N, star, rtol=1.0e-12):
        raise RuntimeError("Gas profile failed the Kennicutt-Schmidt closure")

    velocity_kpc_per_year = radial_velocity_kms(
        radius, parameters
    ) / KPC_PER_YEAR_TO_KM_PER_SECOND
    flux_from_velocity = 2.0 * np.pi * radius * gas * np.abs(velocity_kpc_per_year)
    if not np.allclose(flux_from_velocity, accretion_rate(radius, parameters), rtol=1.0e-12):
        raise RuntimeError("Accretion profile failed the radial-flux identity")


def metallicity_profile(
    radius_kpc: np.ndarray,
    sigmadot_star: np.ndarray,
    sigmadot_wind_profile: np.ndarray,
    parameters: FlowParameters,
) -> np.ndarray:
    """Integrate the shared steady metallicity equation inward from R_out."""
    radius = np.asarray(radius_kpc, dtype=float)
    offset = radius - parameters.R_w_kpc
    source = parameters.Z_w + parameters.yield_y * sigmadot_star / sigmadot_wind_profile

    # With Sigma_g*v_R = -dotSigma_w*(R-R_w), the metallicity equation has
    # integrating factor 1/(R-R_w).  Evaluate its inward integral directly.
    metallicity = np.empty_like(radius)
    metallicity[0] = source[0]
    positive_offset = offset[1:]
    tail_integral = np.zeros_like(positive_offset)
    if positive_offset.size > 1:
        source_slope = np.diff(source[1:]) / np.diff(positive_offset)
        source_intercept = source[1:-1] - source_slope * positive_offset[:-1]
        intervals = (
            source_intercept
            * (1.0 / positive_offset[:-1] - 1.0 / positive_offset[1:])
            + source_slope
            * np.log(positive_offset[1:] / positive_offset[:-1])
        )
        tail_integral[:-1] = np.cumsum(intervals[::-1])[::-1]
    metallicity[1:] = positive_offset * (
        parameters.Z_outer / offset[-1] + tail_integral
    )
    return metallicity


def observational_calculations(
    profile: ObservedProfile,
    parameters: FlowParameters,
) -> tuple[dict[str, np.ndarray], float]:
    """Calculate the Section 3.2 forward test and Section 3.3 inversion."""
    radius = profile.radius_kpc
    gas = profile.sigma_g_Msun_kpc2
    star = profile.sigmadot_star_Msun_yr_kpc2
    offset = radius - parameters.R_w_kpc

    prescribed_wind = sigmadot_wind(radius, parameters)
    prescribed_velocity = -prescribed_wind * offset / gas
    residual = prescribed_wind * (1.0 + parameters.R_w_kpc / radius) - star

    inversion_integrand = radius * offset * star
    inversion_integral = np.zeros_like(radius)
    interval_width = np.diff(radius)
    interval_midpoint = 0.5 * (radius[:-1] + radius[1:])
    midpoint_star = 0.5 * (star[:-1] + star[1:])
    midpoint_integrand = (
        interval_midpoint
        * (interval_midpoint - parameters.R_w_kpc)
        * midpoint_star
    )
    inversion_integral[1:] = np.cumsum(
        interval_width
        * (inversion_integrand[:-1] + 4.0 * midpoint_integrand + inversion_integrand[1:])
        / 6.0
    )
    required_wind = np.empty_like(radius)
    required_wind[0] = 0.5 * star[0]
    required_wind[1:] = inversion_integral[1:] / (
        radius[1:] * offset[1:] ** 2
    )
    required_velocity = -required_wind * offset / gas
    required_total = np.trapezoid(2.0 * np.pi * radius * required_wind, radius)
    matched_r2_wind = sigmadot_wind(
        radius,
        replace(parameters, Mdot_w_Msun_yr=required_total),
    )

    if not np.allclose(
        gas * prescribed_velocity,
        -prescribed_wind * offset,
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise RuntimeError("The prescribed-wind velocity identity failed")
    if not math.isclose(required_wind[0], 0.5 * star[0], rel_tol=1.0e-12):
        raise RuntimeError("The inferred wind failed its regular inner boundary")
    if not np.all(np.isfinite(required_wind)) or np.any(required_wind <= 0.0):
        raise RuntimeError("The inferred wind profile must be finite and positive")

    values = {
        "R_kpc": radius,
        "Sigma_g_obs_Msun_kpc2": gas,
        "Sigmadot_star_obs_Msun_yr_kpc2": star,
        "Sigmadot_w_prescribed_Msun_yr_kpc2": prescribed_wind,
        "v_R_prescribed_kpc_yr": prescribed_velocity,
        "v_R_prescribed_kms": prescribed_velocity * KPC_PER_YEAR_TO_KM_PER_SECOND,
        "residual_Msun_yr_kpc2": residual,
        "external_source_required_Msun_yr_kpc2": -residual,
        "Sigmadot_w_required_Msun_yr_kpc2": required_wind,
        "v_R_required_kpc_yr": required_velocity,
        "v_R_required_kms": required_velocity * KPC_PER_YEAR_TO_KM_PER_SECOND,
        "Sigmadot_w_Rm2_same_total_Msun_yr_kpc2": matched_r2_wind,
        "Z_prescribed": metallicity_profile(
            radius, star, prescribed_wind, parameters
        ),
        "Z_required": metallicity_profile(radius, star, required_wind, parameters),
    }
    return values, required_total


def write_observational_csv(path: Path, values: dict[str, np.ndarray]) -> None:
    columns = tuple(values)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*(values[column] for column in columns)))


def make_observational_figure(
    values: dict[str, np.ndarray],
    parameters: FlowParameters,
    required_total: float,
):
    """Plot the prescribed-wind test and steady observational inversion."""
    radius = values["R_kpc"]
    figure, axes = plt.subplots(
        4,
        1,
        figsize=SINGLE_COLUMN_FIGURE_SIZE,
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    rate_axis, velocity_axis, residual_axis, metallicity_axis = axes

    rate_axis.plot(
        radius,
        values["Sigmadot_star_obs_Msun_yr_kpc2"] * SURFACE_RATE_PLOT_SCALE,
        color="0.45",
        linestyle=STAR_FORMATION_LINESTYLE,
        label=r"observed $\dot{\Sigma}_\star$",
    )
    rate_axis.plot(
        radius,
        values["Sigmadot_w_prescribed_Msun_yr_kpc2"] * SURFACE_RATE_PLOT_SCALE,
        color="#0072B2",
        label=r"prescribed $R^{-2}$",
    )
    rate_axis.plot(
        radius,
        values["Sigmadot_w_required_Msun_yr_kpc2"] * SURFACE_RATE_PLOT_SCALE,
        color="#D55E00",
        label="required",
    )
    rate_axis.plot(
        radius,
        values["Sigmadot_w_Rm2_same_total_Msun_yr_kpc2"]
        * SURFACE_RATE_PLOT_SCALE,
        color="black",
        linestyle=":",
        label=r"$R^{-2}$, same total",
    )
    rate_axis.set_yscale("log")
    rate_axis.set_ylabel(
        r"$\dot{\Sigma}\;[M_\odot\,\mathrm{Gyr}^{-1}\,\mathrm{pc}^{-2}]$"
    )
    rate_axis.legend(fontsize=6.7, ncol=2, columnspacing=0.8, handlelength=1.8)

    velocity_axis.plot(
        radius,
        values["v_R_prescribed_kms"],
        color="#0072B2",
        label="prescribed",
    )
    velocity_axis.plot(
        radius,
        values["v_R_required_kms"],
        color="#D55E00",
        label="required",
    )
    velocity_axis.axhline(0.0, color="0.65", linewidth=0.6)
    velocity_axis.set_ylabel(r"$v_R\;[\mathrm{km\,s}^{-1}]$")
    velocity_axis.legend(fontsize=7.0)

    residual_axis.plot(
        radius,
        values["residual_Msun_yr_kpc2"] * SURFACE_RATE_PLOT_SCALE,
        color="#0072B2",
    )
    residual_axis.set_yscale("log")
    residual_axis.set_ylabel(
        "Gas accumulation rate\n"
        r"$[M_\odot\,\mathrm{Gyr}^{-1}\,\mathrm{pc}^{-2}]$"
    )

    metallicity_axis.plot(
        radius,
        values["Z_prescribed"] / SOLAR_METALLICITY,
        color="#0072B2",
        label="prescribed",
    )
    metallicity_axis.plot(
        radius,
        values["Z_required"] / SOLAR_METALLICITY,
        color="#D55E00",
        label="required",
    )
    metallicity_axis.set_ylabel(r"$Z\;[Z_\odot]$")
    metallicity_axis.set_xlabel(r"$R\;[\mathrm{kpc}]$")
    metallicity_axis.legend(fontsize=7.0)

    for axis in axes:
        axis.set_xlim(parameters.R_w_kpc, parameters.R_out_kpc)
        axis.tick_params(axis="y", which="both", right=True, labelright=False)
    figure.suptitle(
        rf"NGC 2403; inferred $\dot{{M}}_w={required_total:.2f}\,M_\odot\,\mathrm{{yr}}^{{-1}}$",
        fontsize=8.0,
        y=0.99,
    )
    figure.subplots_adjust(
        left=0.21,
        right=0.95,
        bottom=0.075,
        top=0.945,
        hspace=0.08,
    )
    figure.align_ylabels(axes)
    return figure


def make_figure(cases: tuple[FlowParameters, ...], samples: int):
    figure, axes = plt.subplots(
        4,
        1,
        figsize=SINGLE_COLUMN_FIGURE_SIZE,
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    rates_axis, gas_axis, velocity_axis, accretion_axis = axes
    rates_axis.set_yscale("log")

    observations = load_ngc2403_observations()
    for series in observations:
        errorbar_style = {
            "fmt": "none",
            "ecolor": "black",
            "elinewidth": 0.35,
            "capsize": 1.5,
            "capthick": 0.35,
            "zorder": ERRORBAR_ZORDER,
        }
        marker_style = {
            "linestyle": "none",
            "marker": series.marker,
            "markersize": 3.6,
            "markerfacecolor": series.marker_face_color,
            "markeredgecolor": "black",
            "markeredgewidth": 0.35,
            "zorder": OBSERVATION_MARKER_ZORDER,
        }
        observations_to_plot = (
            (
                rates_axis,
                series.sigmadot_star_Msun_yr_kpc2 * SURFACE_RATE_PLOT_SCALE,
                series.e_sigmadot_star_Msun_yr_kpc2 * SURFACE_RATE_PLOT_SCALE,
            ),
            (
                gas_axis,
                series.sigma_g_Msun_kpc2 * SURFACE_DENSITY_PLOT_SCALE,
                series.e_sigma_g_Msun_kpc2 * SURFACE_DENSITY_PLOT_SCALE,
            ),
        )
        for axis, values, uncertainties in observations_to_plot:
            axis.errorbar(
                series.radius_kpc,
                values,
                yerr=uncertainties,
                **errorbar_style,
            )
            axis.plot(series.radius_kpc, values, **marker_style)

    # Preserve the automatic observational scaling before model curves are added.
    gas_observation_limits = gas_axis.get_ylim()

    for parameters in cases:
        verify_profiles(parameters, samples)
        radius = radius_grid(parameters, samples)
        rates_axis.plot(
            radius,
            sigmadot_wind(radius, parameters) * SURFACE_RATE_PLOT_SCALE,
            color=parameters.color,
            linestyle="-",
            zorder=MODEL_CURVE_ZORDER,
        )
        rates_axis.plot(
            radius,
            sigmadot_star(radius, parameters) * SURFACE_RATE_PLOT_SCALE,
            color=parameters.color,
            linestyle=STAR_FORMATION_LINESTYLE,
            linewidth=STAR_FORMATION_LINEWIDTH,
            zorder=MODEL_CURVE_ZORDER,
        )
        gas_axis.plot(
            radius,
            sigma_gas(radius, parameters) * SURFACE_DENSITY_PLOT_SCALE,
            color=parameters.color,
            zorder=MODEL_CURVE_ZORDER,
        )
        velocity_axis.plot(
            radius,
            radial_velocity_kms(radius, parameters),
            color=parameters.color,
            zorder=MODEL_CURVE_ZORDER,
        )
        accretion_axis.plot(
            radius,
            accretion_rate(radius, parameters),
            color=parameters.color,
            zorder=MODEL_CURVE_ZORDER,
        )

    rates_axis.set_ylim(1.0e-3, 10.0)
    gas_axis.set_ylim(gas_observation_limits)
    velocity_axis.set_ylim(top=0.0)
    accretion_axis.set_ylim(bottom=0.0)
    rates_axis.set_ylabel(
        r"$\dot{\Sigma}\;[M_\odot\,\mathrm{Gyr}^{-1}\,\mathrm{pc}^{-2}]$"
    )
    gas_axis.set_ylabel(r"$\Sigma_g\;[M_\odot\,\mathrm{pc}^{-2}]$")
    velocity_axis.set_ylabel(r"$v_R\;[\mathrm{km\,s}^{-1}]$")
    accretion_axis.set_ylabel(
        r"$\dot{M}_{\rm acc}\;[M_\odot\,\mathrm{yr}^{-1}]$"
    )
    accretion_axis.set_xlabel(r"$R\;[\mathrm{kpc}]$")

    maximum_radius = max(parameters.R_out_kpc for parameters in cases)
    for axis in axes:
        axis.set_xlim(0.0, maximum_radius)
        axis.tick_params(axis="y", which="both", right=True, labelright=False)
    radius_ticks = np.linspace(0.0, maximum_radius, 7)
    accretion_axis.set_xticks(radius_ticks)
    top_radius_axis = rates_axis.secondary_xaxis("top")
    top_radius_axis.set_xticks(radius_ticks)
    top_radius_axis.set_xlabel(r"$R\;[\mathrm{kpc}]$", labelpad=4.0)
    top_radius_axis.tick_params(
        axis="x",
        which="both",
        direction="in",
        pad=2.0,
    )
    top_radius_axis.minorticks_on()
    rates_axis.set_zorder(2)

    observation_handles = [
        Line2D(
            [],
            [],
            color="0.5",
            linestyle="none",
            marker=series.marker,
            markersize=4.6,
            markerfacecolor=series.marker_face_color,
            markeredgecolor="black",
            markeredgewidth=0.35,
            label=series.label,
        )
        for series in observations
    ]
    model_legend = rates_axis.legend(
        handles=(
            Line2D(
                [],
                [],
                color=FIDUCIAL_COLOR,
                linestyle="-",
                label=r"$\dot{\Sigma}_w$",
            ),
            Line2D(
                [],
                [],
                color=FIDUCIAL_COLOR,
                linestyle=STAR_FORMATION_LINESTYLE,
                linewidth=STAR_FORMATION_LINEWIDTH,
                label=r"$\dot{\Sigma}_\star$",
            ),
        ),
        loc="lower left",
        fontsize=7.0,
        handlelength=2.2,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    rates_axis.add_artist(model_legend)
    rates_axis.legend(
        handles=observation_handles,
        loc="lower left",
        bbox_to_anchor=(0.22, 0.0),
        title="NGC 2403",
        title_fontsize=7.5,
        fontsize=7.0,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    gas_axis.legend(
        handles=observation_handles,
        loc="upper right",
        title="NGC 2403",
        title_fontsize=7.5,
        fontsize=7.0,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    legend_order = (0, 3, 1, 4, 2)
    parameter_handles = [
        Line2D([], [], color=cases[index].color, label=cases[index].label)
        for index in legend_order
    ]
    parameter_legend = figure.legend(
        handles=parameter_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=3,
        fontsize=6.7,
        handlelength=1.4,
        columnspacing=0.7,
        handletextpad=0.3,
        labelspacing=0.25,
        frameon=False,
    )
    parameter_legend.set_zorder(1)
    figure.subplots_adjust(
        left=0.205,
        right=0.95,
        bottom=0.075,
        top=0.88,
        hspace=0.08,
    )
    figure.align_ylabels(axes)
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
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="radial samples per curve",
    )
    arguments = parser.parse_args()
    if arguments.samples < 2:
        parser.error("--samples must be at least 2")
    return arguments


def main() -> int:
    arguments = parse_arguments()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)

    figure = make_figure(MODEL_CASES, arguments.samples)
    path = arguments.output_directory / "section3_flow_parameter_variations.pdf"
    figure.savefig(path, bbox_inches=None)
    plt.close(figure)
    print(f"Wrote {path}")

    observed_profile, observed_parameters = load_ngc2403_profile(FIDUCIAL)
    observational_values, required_total = observational_calculations(
        observed_profile,
        observed_parameters,
    )
    observational_csv = (
        arguments.output_directory / "section3_observational_tests.csv"
    )
    write_observational_csv(observational_csv, observational_values)
    print(f"Wrote {observational_csv}")

    observational_figure = make_observational_figure(
        observational_values,
        observed_parameters,
        required_total,
    )
    observational_pdf = (
        arguments.output_directory / "section3_observational_tests.pdf"
    )
    observational_figure.savefig(observational_pdf, bbox_inches=None)
    plt.close(observational_figure)
    print(f"Wrote {observational_pdf}")
    print(f"Required steady landing rate: {required_total:.8g} Msun/yr")
    if observed_profile.legacy_return_correction_applied:
        print(
            "Applied the 0.6 locked-mass correction to the legacy saved "
            "NGC 2403 SFR spline"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
