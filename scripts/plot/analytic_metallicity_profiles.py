#!/usr/bin/env python3
"""Plot the analytic metallicity solution from the metallicity-gradient model.

The adopted solution assumes

    Sigma_dot_star / Sigma_dot_w = 1 + R_w / R

and imposes Z(R_out) = Z_out.  Metallicities are plotted in units of Z_sun,
while the total-metal yield y_Z is specified directly as a dimensionless mass
fraction.  Edit PARAMETER_SETS below to change the curves in the figure.
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hershey_fonts import register_hershey_weight_aliases


register_hershey_weight_aliases()
import smplotlib


smplotlib.set_style(
    usetex=False,
    fontsize=10,
    figsize=(7.1, 5.6),
    dpi=144,
)
plt.rcParams.update(
    {
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "legend.frameon": False,
    }
)


SOLAR_METALLICITY = 0.014
ABUNDANCE_SCATTER_DEX = 0.05
FIDUCIAL_GAS_MASS_MSUN = 1.0e9
FIDUCIAL_WIND_RATE_MSUN_PER_YEAR = 0.1
KPC_PER_YEAR_TO_KM_PER_SECOND = 3.0856775814913673e16 / 31557600.0
ROOT = Path(__file__).resolve().parents[2]
MOUSTAKAS_CATALOG = ROOT / "input" / "sings_hii_regions.csv"
MOUSTAKAS_RAW_REGIONS = ROOT / "input" / "sings" / "table10.dat"
MOUSTAKAS_RAW_GALAXIES = ROOT / "input" / "sings" / "table1.dat"

Q1_MOUSTAKAS_GALAXIES = (
    "NGC2403",
    "NGC2841",
    "NGC3198",
    "NGC3521",
    "NGC5055",
    "NGC6946",
    "NGC7331",
    "NGC7793",
)

# Moustakas et al. (2010) gradient-quality sample: at least five published
# H II-region spectra spanning at least 0.1 R25.  Names are normalized to the
# convention used by this project.
MOUSTAKAS_GRADIENT_GALAXIES = (
    "NGC0628",
    "NGC0925",
    "NGC1097",
    "NGC2403",
    "NGC2841",
    "NGC3031",
    "NGC3184",
    "NGC3198",
    "NGC3351",
    "NGC3521",
    "NGC3621",
    "NGC4254",
    "NGC4321",
    "NGC4559",
    "NGC4736",
    "NGC5033",
    "NGC5055",
    "NGC5194",
    "NGC6946",
    "NGC7331",
    "NGC7793",
)


@dataclass(frozen=True)
class ProfileParameters:
    """Parameters for one analytic total-metallicity curve."""

    label: str
    rw_kpc: float
    rout_kpc: float
    zw_solar: float
    yield_mass_fraction: float
    zout_solar: float
    color: str
    linestyle: str = "-"


@dataclass(frozen=True)
class ObservationalGradientBand:
    """Equal-galaxy summary of the Q=1 Moustakas gradient fits."""

    calibration: str
    lower: float
    median: float
    upper: float
    galaxy_slopes: tuple[float, ...]


FIDUCIAL = ProfileParameters(
    label="fiducial",
    rw_kpc=1.0,
    rout_kpc=15.0,
    zw_solar=1.0,
    yield_mass_fraction=0.01,
    zout_solar=0.3,
    color="black",
)

# Each non-fiducial curve changes just one parameter.  These are the values to
# edit when exploring another parameter selection.
PARAMETER_SETS = (
    FIDUCIAL,
    replace(
        FIDUCIAL,
        label=r"$Z_w=0.6\,Z_\odot$",
        zw_solar=0.6,
        color="#0072B2",
    ),
    replace(
        FIDUCIAL,
        label=r"$y_Z=0.02$",
        yield_mass_fraction=0.02,
        color="#D55E00",
    ),
    replace(
        FIDUCIAL,
        label=r"$Z_{\rm out}=0.6\,Z_\odot$",
        zout_solar=0.6,
        color="#C43C39",
    ),
    replace(
        FIDUCIAL,
        label=r"$R_w=0.1\,\mathrm{kpc}$",
        rw_kpc=0.1,
        color="#78CBCD",
    ),
    replace(
        FIDUCIAL,
        label=r"$R_w=2\,\mathrm{kpc}$",
        rw_kpc=2.0,
        color="#4E9B70",
    ),
)


OBSERVATIONAL_BAND_STYLES = {
    "KK04": {
        "facecolor": "0.35",
        "alpha": 0.12,
        "color": "0.35",
        "linestyle": (0, (3.0, 1.5)),
    },
    "PT05": {
        "facecolor": "0.65",
        "alpha": 0.12,
        "color": "0.50",
        "linestyle": (0, (3.0, 1.5)),
    },
}


def csv_true(value):
    return value.strip().lower() == "true"


def fixed_width_field(line, first_byte, last_byte):
    return line.ljust(last_byte)[first_byte - 1 : last_byte].strip()


def optional_float_field(line, first_byte, last_byte):
    value = fixed_width_field(line, first_byte, last_byte)
    return None if not value else float(value)


def normalized_galaxy_name(name):
    return name.replace(" ", "")


def load_moustakas_gradient_catalog():
    """Read the full CDS tables and put all 21 disks on the SINGS kpc scale."""
    galaxy_r25_kpc = {}
    for line in MOUSTAKAS_RAW_GALAXIES.read_text(encoding="utf-8").splitlines():
        name = normalized_galaxy_name(fixed_width_field(line, 1, 8))
        r25_arcmin = optional_float_field(line, 26, 30)
        distance_mpc = optional_float_field(line, 64, 69)
        if name and r25_arcmin is not None and distance_mpc is not None:
            galaxy_r25_kpc[name] = (
                1_000.0
                * distance_mpc
                * np.tan(np.deg2rad(r25_arcmin / 60.0))
            )

    catalog = []
    for line in MOUSTAKAS_RAW_REGIONS.read_text(encoding="utf-8").splitlines():
        sequence = fixed_width_field(line, 1, 3)
        if not sequence:
            continue
        galaxy = normalized_galaxy_name(fixed_width_field(line, 5, 13))
        normalized_radius = optional_float_field(line, 33, 36)
        radius_kpc = (
            None
            if normalized_radius is None or galaxy not in galaxy_r25_kpc
            else normalized_radius * galaxy_r25_kpc[galaxy]
        )

        offset_ra = optional_float_field(line, 118, 121)
        offset_dec = optional_float_field(line, 123, 126)
        if offset_ra == -999:
            offset_ra = None
        if offset_dec == -999:
            offset_dec = None

        row = {
            "galaxy": galaxy,
            "source_seq": sequence,
            "R_sings_kpc": "" if radius_kpc is None else str(radius_kpc),
            "offRA_arcsec": "" if offset_ra is None else str(offset_ra),
            "offDE_arcsec": "" if offset_dec is None else str(offset_dec),
        }
        for calibration, abundance_bytes, error_bytes, flag_byte in (
            ("KK04", (70, 73), (75, 78), 80),
            ("PT05", (82, 85), (87, 90), 92),
        ):
            abundance = optional_float_field(line, *abundance_bytes)
            error = optional_float_field(line, *error_bytes)
            flag = fixed_width_field(line, flag_byte, flag_byte)
            row[f"OH12_{calibration}_dex"] = (
                "" if abundance is None else str(abundance)
            )
            row[f"e_OH12_{calibration}_dex"] = (
                "" if error is None else str(error)
            )
            row[f"usable_OH12_{calibration}"] = str(
                abundance is not None and error is not None and not flag
            )
        catalog.append(row)
    return catalog


def collapse_repeated_regions(rows, calibration, radius_column):
    """Collapse repeated spectra using the comparison pipeline's weighting."""
    grouped = defaultdict(list)
    error_column = f"e_OH12_{calibration}_dex"
    abundance_column = f"OH12_{calibration}_dex"

    for row in rows:
        has_position = row["offRA_arcsec"] and row["offDE_arcsec"]
        key = (
            ("position", row["offRA_arcsec"], row["offDE_arcsec"])
            if has_position
            else ("sequence", row["source_seq"])
        )
        grouped[key].append(
            (
                float(row[radius_column]),
                float(row[abundance_column]),
                float(row[error_column]),
            )
        )

    collapsed = []
    for group in grouped.values():
        errors = np.asarray([measurement[2] for measurement in group])
        weights = 1.0 / errors**2
        collapsed.append(
            (
                np.average(
                    [measurement[0] for measurement in group], weights=weights
                ),
                np.average(
                    [measurement[1] for measurement in group], weights=weights
                ),
                np.sqrt(1.0 / np.sum(weights)),
            )
        )
    return collapsed


def weighted_gradient(observations):
    """Fit 12 + log10(O/H) = a + b R and return b in dex/kpc."""
    radius = np.asarray([observation[0] for observation in observations])
    abundance = np.asarray([observation[1] for observation in observations])
    measurement_error = np.asarray(
        [observation[2] for observation in observations]
    )
    sigma = np.sqrt(measurement_error**2 + ABUNDANCE_SCATTER_DEX**2)
    design = np.column_stack((np.ones_like(radius), radius))
    weighted_design = design / sigma[:, np.newaxis]
    weighted_abundance = abundance / sigma
    intercept, slope = np.linalg.lstsq(
        weighted_design, weighted_abundance, rcond=None
    )[0]
    del intercept
    return slope


def observational_gradient_bands(sample, catalog_path=MOUSTAKAS_CATALOG):
    """Return KK04 and PT05 P14/median/P86 bands for one galaxy sample."""
    if sample == "q1":
        with catalog_path.open(newline="", encoding="utf-8") as catalog_file:
            catalog = list(csv.DictReader(catalog_file))
        galaxies = Q1_MOUSTAKAS_GALAXIES
        radius_column = "R_adopted_kpc"
    elif sample == "moustakas21":
        catalog = load_moustakas_gradient_catalog()
        galaxies = MOUSTAKAS_GRADIENT_GALAXIES
        radius_column = "R_sings_kpc"
    else:
        raise ValueError(f"Unknown observational sample: {sample}")

    bands = []
    for calibration in ("KK04", "PT05"):
        abundance_column = f"OH12_{calibration}_dex"
        error_column = f"e_OH12_{calibration}_dex"
        usable_column = f"usable_OH12_{calibration}"
        slopes = []
        for galaxy in galaxies:
            rows = [
                row
                for row in catalog
                if row["galaxy"] == galaxy
                and csv_true(row[usable_column])
                and row[radius_column]
                and row[abundance_column]
                and row[error_column]
            ]
            observations = collapse_repeated_regions(
                rows, calibration, radius_column
            )
            if len(observations) < 2:
                raise ValueError(
                    f"{galaxy} has fewer than two usable {calibration} regions"
                )
            slopes.append(weighted_gradient(observations))

        lower, median, upper = np.percentile(slopes, (14.0, 50.0, 86.0))
        bands.append(
            ObservationalGradientBand(
                calibration=calibration,
                lower=lower,
                median=median,
                upper=upper,
                galaxy_slopes=tuple(slopes),
            )
        )
    return tuple(bands)


def draw_observational_gradient_bands(axis, bands):
    """Draw quiet, unlabeled reference bands behind the model curves."""
    for band in bands:
        style = OBSERVATIONAL_BAND_STYLES[band.calibration]
        axis.axhspan(
            band.lower,
            band.upper,
            facecolor=style["facecolor"],
            alpha=style["alpha"],
            linewidth=0.0,
            zorder=0,
        )
        axis.axhline(
            band.median,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=0.7,
            zorder=1,
        )


def validate(parameters):
    if parameters.rw_kpc <= 0.0:
        raise ValueError("R_w must be positive")
    if parameters.rout_kpc <= parameters.rw_kpc:
        raise ValueError("R_out must be larger than R_w")
    if parameters.yield_mass_fraction < 0.0:
        raise ValueError("y_Z must be nonnegative")


def metallicity(radius_kpc, parameters):
    """Return Z/Z_sun, including its finite limiting value at R = R_w."""
    validate(parameters)
    radius = np.asarray(radius_kpc, dtype=float)
    if np.any(radius < parameters.rw_kpc) or np.any(
        radius > parameters.rout_kpc
    ):
        raise ValueError("R must lie between R_w and R_out")

    rw = parameters.rw_kpc
    rout = parameters.rout_kpc
    yield_solar = parameters.yield_mass_fraction / SOLAR_METALLICITY
    radial_offset = radius - rw
    boundary_slope = (
        parameters.zout_solar
        - parameters.zw_solar
        - 2.0 * yield_solar
    ) / (rout - rw)

    # x ln(x) -> 0 as x -> 0.  Evaluating the product piecewise avoids
    # taking log(0) at the inner endpoint.
    logarithmic_term = np.zeros_like(radius)
    positive = radial_offset > 0.0
    log_argument = (
        rout
        * radial_offset[positive]
        / (radius[positive] * (rout - rw))
    )
    logarithmic_term[positive] = (
        yield_solar
        * radial_offset[positive]
        / rw
        * np.log(log_argument)
    )

    return (
        parameters.zw_solar
        + 2.0 * yield_solar
        + radial_offset * boundary_slope
        + logarithmic_term
    )


def logarithmic_gradient(radius_kpc, parameters):
    """Return d(Z/Z_sun)/d ln(R); this diverges as R approaches R_w."""
    validate(parameters)
    radius = np.asarray(radius_kpc, dtype=float)
    if np.any(radius <= parameters.rw_kpc) or np.any(
        radius > parameters.rout_kpc
    ):
        raise ValueError("The gradient requires R_w < R <= R_out")

    rw = parameters.rw_kpc
    rout = parameters.rout_kpc
    yield_solar = parameters.yield_mass_fraction / SOLAR_METALLICITY
    boundary_slope = (
        parameters.zout_solar
        - parameters.zw_solar
        - 2.0 * yield_solar
    ) / (rout - rw)
    log_argument = rout * (radius - rw) / (radius * (rout - rw))
    return (
        radius * boundary_slope
        + yield_solar * radius / rw * np.log(log_argument)
        + yield_solar
    )


def metallicity_gradient_dex_per_kpc(radius_kpc, parameters):
    """Return the observationally standard d log10(Z/Z_sun) / dR."""
    radius = np.asarray(radius_kpc, dtype=float)
    metallicity_solar = metallicity(radius, parameters)
    return logarithmic_gradient(radius, parameters) / (
        radius * metallicity_solar * np.log(10.0)
    )


def gas_surface_density_msun_per_pc2(radius_kpc):
    """Return the illustrative Sigma_g = M_gas / (2 pi R^2) profile."""
    radius = np.asarray(radius_kpc, dtype=float)
    if np.any(radius <= 0.0):
        raise ValueError("The gas surface-density profile requires R > 0")
    sigma_msun_per_kpc2 = FIDUCIAL_GAS_MASS_MSUN / (
        2.0 * np.pi * radius**2
    )
    return sigma_msun_per_kpc2 / 1.0e6


def radial_velocity_kms(radius_kpc, parameters):
    """Return v_R for matching R^-2 gas and wind surface-density profiles."""
    validate(parameters)
    radius = np.asarray(radius_kpc, dtype=float)
    if np.any(radius < parameters.rw_kpc) or np.any(
        radius > parameters.rout_kpc
    ):
        raise ValueError("R must lie between R_w and R_out")
    velocity_kpc_per_year = -(
        FIDUCIAL_WIND_RATE_MSUN_PER_YEAR
        / FIDUCIAL_GAS_MASS_MSUN
        * (radius - parameters.rw_kpc)
    )
    return velocity_kpc_per_year * KPC_PER_YEAR_TO_KM_PER_SECOND


def radius_samples(parameters, sample_count, inner_offset_fraction):
    """Return profile samples and nonsingular gradient samples."""
    span = parameters.rout_kpc - parameters.rw_kpc
    profile_radius = np.linspace(
        parameters.rw_kpc, parameters.rout_kpc, sample_count
    )
    gradient_radius = np.linspace(
        parameters.rw_kpc + inner_offset_fraction * span,
        parameters.rout_kpc,
        sample_count,
    )
    return profile_radius, gradient_radius


def make_figure(
    parameter_sets,
    sample_count,
    inner_offset_fraction,
    observational_bands,
):
    figure, axes = plt.subplots(
        2, 2, figsize=(7.1, 5.6)
    )
    metallicity_axis, gradient_axis = axes[0]
    gas_axis, velocity_axis = axes[1]
    gradient_inset = gradient_axis.inset_axes([0.37, 0.19, 0.53, 0.42])

    draw_observational_gradient_bands(gradient_axis, observational_bands)
    draw_observational_gradient_bands(gradient_inset, observational_bands)

    for parameters in parameter_sets:
        profile_radius, gradient_radius = radius_samples(
            parameters, sample_count, inner_offset_fraction
        )
        style = {
            "color": parameters.color,
            "linestyle": parameters.linestyle,
            "linewidth": 1.0,
            "zorder": 3 if parameters == FIDUCIAL else 2,
        }
        metallicity_axis.plot(
            profile_radius,
            metallicity(profile_radius, parameters),
            label=parameters.label,
            **style,
        )
        gradient_axis.plot(
            gradient_radius,
            metallicity_gradient_dex_per_kpc(gradient_radius, parameters),
            **style,
        )
        gradient_inset.plot(
            gradient_radius,
            metallicity_gradient_dex_per_kpc(gradient_radius, parameters),
            **style,
        )

    shared_radius = np.linspace(
        min(parameters.rw_kpc for parameters in parameter_sets),
        max(parameters.rout_kpc for parameters in parameter_sets),
        sample_count,
    )
    gas_axis.plot(
        shared_radius,
        gas_surface_density_msun_per_pc2(shared_radius),
        color=FIDUCIAL.color,
        linewidth=1.0,
    )

    plotted_launch_radii = set()
    for parameters in parameter_sets:
        if parameters.rw_kpc in plotted_launch_radii:
            continue
        plotted_launch_radii.add(parameters.rw_kpc)
        profile_radius, _ = radius_samples(
            parameters, sample_count, inner_offset_fraction
        )
        velocity_axis.plot(
            profile_radius,
            radial_velocity_kms(profile_radius, parameters),
            color=parameters.color,
            linestyle=parameters.linestyle,
            linewidth=1.0,
            zorder=3 if parameters == FIDUCIAL else 2,
        )

    metallicity_axis.set_xlabel(r"$R\;[\mathrm{kpc}]$")
    metallicity_axis.set_ylabel(r"$Z\;[Z_\odot]$")
    metallicity_axis.set_ylim(0.0, 4.0)
    gradient_axis.set_xlabel(r"$R\;[\mathrm{kpc}]$")
    gradient_axis.set_ylabel(
        r"$d\log_{10}Z/dR\;[\mathrm{dex\,kpc}^{-1}]$"
    )
    gradient_axis.set_ylim(-0.4, 0.0)
    gradient_inset.set_xlim(2.5, 10.0)
    gradient_inset.set_ylim(-0.06, -0.02)
    gradient_inset.set_xticks([2.5, 5.0, 7.5, 10.0])
    gradient_inset.set_yticks([-0.06, -0.04, -0.02])
    gradient_inset.tick_params(axis="both", which="both", labelsize=7)
    gradient_axis.indicate_inset_zoom(
        gradient_inset, edgecolor="0.4", linewidth=0.6
    )
    gas_axis.set_xlabel(r"$R\;[\mathrm{kpc}]$")
    gas_axis.set_ylabel(
        r"$\Sigma_{\rm g}\;[M_\odot\,\mathrm{pc}^{-2}]$"
    )
    gas_axis.set_yscale("log")
    gas_axis.set_ylim(1.0, 1.0e4)
    velocity_axis.set_xlabel(r"$R\;[\mathrm{kpc}]$")
    velocity_axis.set_ylabel(r"$v_R\;[\mathrm{km\,s}^{-1}]$")
    velocity_axis.set_ylim(-1.5, 0.0)
    legend_handles, legend_labels = metallicity_axis.get_legend_handles_labels()
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=6,
        fontsize=8.0,
        handlelength=2.5,
        columnspacing=1.4,
        handletextpad=0.6,
        labelspacing=0.45,
    )

    for axis in (metallicity_axis, gradient_axis, gas_axis, velocity_axis):
        axis.set_xlim(
            0.0,
            max(parameters.rout_kpc for parameters in parameter_sets),
        )

    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.10,
        top=0.93,
        wspace=0.30,
        hspace=0.36,
    )
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot the analytic Z(R) profile and its logarithmic gradient."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/analytic_metallicity_profiles.pdf"),
        help="output figure path (default: figures/analytic_metallicity_profiles.pdf)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="number of samples per curve (default: 1000)",
    )
    parser.add_argument(
        "--inner-offset-fraction",
        type=float,
        default=1.0e-4,
        help=(
            "fraction of R_out-R_w omitted from the inner edge of the gradient "
            "panel, where the analytic derivative diverges (default: 1e-4)"
        ),
    )
    parser.add_argument(
        "--observational-calibration",
        choices=("KK04", "PT05"),
        default="KK04",
        help=(
            "Moustakas abundance calibration used for the quiet reference "
            "band (default: KK04)"
        ),
    )
    parser.add_argument(
        "--observational-sample",
        choices=("q1", "moustakas21"),
        default="q1",
        help=(
            "galaxy sample used for the observational band: the eight-galaxy "
            "Q=1 overlap or the 21 Moustakas gradient-quality disks "
            "(default: q1)"
        ),
    )
    arguments = parser.parse_args()
    if arguments.samples < 2:
        parser.error("--samples must be at least 2")
    if not 0.0 < arguments.inner_offset_fraction < 1.0:
        parser.error("--inner-offset-fraction must lie strictly between 0 and 1")
    return arguments


def main():
    arguments = parse_arguments()
    all_bands = observational_gradient_bands(arguments.observational_sample)
    bands = tuple(
        band
        for band in all_bands
        if band.calibration == arguments.observational_calibration
    )
    figure = make_figure(
        PARAMETER_SETS,
        arguments.samples,
        arguments.inner_offset_fraction,
        bands,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    for band in bands:
        print(
            f"{arguments.observational_sample} {band.calibration}: "
            f"P14={band.lower:.6f}, median={band.median:.6f}, "
            f"P86={band.upper:.6f} dex/kpc"
        )
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
