#!/usr/bin/env python3
"""Plot the analytic metallicity solution from the metallicity-gradient model.

The adopted solution assumes

    Sigma_dot_star / Sigma_dot_land = 3-lambda + (lambda-1) R_nuc/R

and imposes Z(R_out) = Z_out.  Metallicities are plotted in units of Z_sun,
while the total-metal yield y is specified directly as a dimensionless mass
fraction.  Edit PARAMETER_SETS below to change the curves in the figure.
The incident metallicity Z_land equals Z_nuc in this unmixed model.
The left panel overlays measured NGC 2403 SINGS H II-region oxygen abundances using
the solar-scaled convention Z/Z_sun = 10**(12 + log10(O/H) - 8.69).
"""

from axis_labels import quantity_unit_label

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
from color_palette import color_cycle
import numpy as np

from hershey_fonts import register_hershey_weight_aliases
from power_law_model import power_law_profiles
from core_calculations import collapse_abundances, fit_abundances


register_hershey_weight_aliases()
import smplotlib


smplotlib.set_style(usetex=False, fontsize=10, figsize=(7.1, 3.0), dpi=144)
plt.rcParams.update(
    {"axes.prop_cycle": color_cycle(),
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "legend.frameon": False,
    }
)


SOLAR_METALLICITY = 0.0139
SOLAR_OXYGEN_ABUNDANCE_DEX = 8.69
ABUNDANCE_SCATTER_DEX = 0.05
ROOT = Path(__file__).resolve().parents[2]
MOUSTAKAS_CATALOG = ROOT / "input" / "sings_hii_regions.csv"
MOUSTAKAS_RAW_REGIONS = ROOT / "input" / "sings" / "table10.dat"
MOUSTAKAS_RAW_GALAXIES = ROOT / "input" / "sings" / "table1.dat"

Q1_MOUSTAKAS_GALAXIES = ("NGC2403", "NGC2841", "NGC3198", "NGC3521", "NGC5055", "NGC6946", "NGC7331", "NGC7793")

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
    landing_slope: float = 2.0
    rin_kpc: float | None = None


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

NO_DISK_METAL_PRODUCTION = replace(FIDUCIAL, label=r"$y=0$", yield_mass_fraction=0.0, color="#7B3294")
NO_ABUNDANCE_CONTRAST = replace(
    FIDUCIAL,
    label=r"$Z_{\rm out}=Z_{\rm land}$",
    zout_solar=FIDUCIAL.zw_solar,
    color="#8C6D31",
)
DIAGNOSTIC_PARAMETER_SETS = (FIDUCIAL, NO_DISK_METAL_PRODUCTION, NO_ABUNDANCE_CONTRAST)

# Each non-fiducial curve changes just one parameter.  These are the values to
# edit when exploring another parameter selection.
PARAMETER_SETS = (
    *DIAGNOSTIC_PARAMETER_SETS,
    replace(FIDUCIAL, label=r"$Z_{\rm land}=0.6\,Z_\odot$", zw_solar=0.6, color="#0072B2"),
    replace(FIDUCIAL, label=r"$y=0.02$", yield_mass_fraction=0.02, color="#D55E00"),
    replace(FIDUCIAL, label=r"$Z_{\rm out}=0.6\,Z_\odot$", zout_solar=0.6, color="#C43C39"),
    replace(FIDUCIAL, label=r"$R_{\rm nuc}=0.1\,\mathrm{kpc}$", rw_kpc=0.1, color="#78CBCD"),
    replace(FIDUCIAL, label=r"$R_{\rm nuc}=2\,\mathrm{kpc}$", rw_kpc=2.0, color="#4E9B70"),
)


METALLICITY_SCANS = {
    "": PARAMETER_SETS,
    "landing_slope": (
        FIDUCIAL,
        replace(FIDUCIAL, label=r"$\lambda=1$", color="#0072B2", landing_slope=1.0),
        replace(FIDUCIAL, label=r"$\lambda=2.5$", color="#009E73", landing_slope=2.5),
    ),
    "radial_boundaries": (
        FIDUCIAL,
        replace(FIDUCIAL, label=r"$R_{\rm in}=2\,\mathrm{kpc}$", color="#0072B2", rin_kpc=2.0),
        replace(FIDUCIAL, label=r"$R_{\rm out}=10\,\mathrm{kpc}$", color="#E69F00", rout_kpc=10.0),
        replace(FIDUCIAL, label=r"$R_{\rm out}=20\,\mathrm{kpc}$", color="#D55E00", rout_kpc=20.0),
    ),
}


OBSERVATIONAL_BAND_STYLES = {
    "KK04": {"facecolor": "0.35", "alpha": 0.12, "color": "0.35", "linestyle": (0, (3.0, 1.5))},
    "PT05": {"facecolor": "0.65", "alpha": 0.12, "color": "0.50", "linestyle": (0, (3.0, 1.5))},
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
            galaxy_r25_kpc[name] = (1_000.0 * distance_mpc * np.tan(np.deg2rad(r25_arcmin / 60.0)))

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
            row[f"OH12_{calibration}_dex"] = ("" if abundance is None else str(abundance))
            row[f"e_OH12_{calibration}_dex"] = ("" if error is None else str(error))
            row[f"usable_OH12_{calibration}"] = str(abundance is not None and error is not None and not flag)
        catalog.append(row)
    return catalog


def collapse_repeated_regions(rows, calibration, radius_column):
    """Use the core repeated-sky-position weighting on the selected catalog sample."""
    selected = [{"source_seq": row["source_seq"], "offRA_arcsec": row["offRA_arcsec"],
                 "offDE_arcsec": row["offDE_arcsec"], "R_kpc": row[radius_column],
                 "OH12": row[f"OH12_{calibration}_dex"], "e_OH12": row[f"e_OH12_{calibration}_dex"]}
                for row in rows]
    return [(row["R_kpc"], row["OH12"], row["e_OH12"]) for row in collapse_abundances(selected)]


def weighted_gradient(observations):
    """Read the core inverse-variance OH12 gradient, retaining the adopted scatter."""
    fit, _ = fit_abundances(observations, scatter=ABUNDANCE_SCATTER_DEX)
    return fit["OH12_slope_dex_per_kpc"]


def observational_abundance_profiles(sample, calibration, catalog_path=MOUSTAKAS_CATALOG):
    """Return unique-region measurements on the sample's adopted kpc scale."""
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

    abundance_column = f"OH12_{calibration}_dex"
    error_column = f"e_OH12_{calibration}_dex"
    usable_column = f"usable_OH12_{calibration}"
    profiles = {}
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
        observations = collapse_repeated_regions(rows, calibration, radius_column)
        if len(observations) < 2:
            raise ValueError(f"{galaxy} has fewer than two usable {calibration} regions")
        profiles[galaxy] = observations
    return profiles


def observational_gradient_bands(sample, catalog_path=MOUSTAKAS_CATALOG):
    """Return KK04 and PT05 P14/median/P86 bands for one galaxy sample."""
    bands = []
    for calibration in ("KK04", "PT05"):
        profiles = observational_abundance_profiles(sample, calibration, catalog_path)
        slopes = [weighted_gradient(observations) for observations in profiles.values()]

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


def draw_observed_metallicity(axis, profiles, calibration, *, legend_fontsize=6.5):
    """Overlay measured oxygen abundances, assuming solar-scaled composition."""
    observations = np.asarray([observation for profile in profiles.values() for observation in profile])
    radius, oxygen_abundance, measurement_error = observations.T
    metallicity_solar = 10.0 ** (oxygen_abundance - SOLAR_OXYGEN_ABUNDANCE_DEX)
    lower = 10.0 ** (oxygen_abundance - measurement_error - SOLAR_OXYGEN_ABUNDANCE_DEX)
    upper = 10.0 ** (oxygen_abundance + measurement_error - SOLAR_OXYGEN_ABUNDANCE_DEX)
    artist = axis.errorbar(
        radius,
        metallicity_solar,
        yerr=np.array([metallicity_solar - lower, upper - metallicity_solar]),
        linestyle="none",
        marker="o",
        markersize=3.2,
        markerfacecolor="black",
        markeredgecolor="black",
        markeredgewidth=0.5,
        color="black",
        ecolor="black",
        elinewidth=0.5,
        alpha=1.0,
        capsize=2.0,
        capthick=0.5,
        zorder=1,
        label=(
            "NGC 2403"
            if set(profiles) == {"NGC2403"}
            else f"SINGS {calibration} ({len(profiles)} disks)"
        ),
    )
    axis.legend(handles=[artist], loc="upper right", fontsize=legend_fontsize, handlelength=1.0)
    return float(np.max(upper))


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
        axis.axhline(band.median, color=style["color"], linestyle=style["linestyle"], linewidth=0.7, zorder=1)


def inner_radius(parameters):
    return parameters.rw_kpc if parameters.rin_kpc is None else parameters.rin_kpc


def analytic_profiles(radius_kpc, parameters):
    """Read Z and its derivative from the shared core analytic solution."""
    return power_law_profiles(radius_kpc, {
        "R_nucl_kpc": parameters.rw_kpc,
        "R_in_kpc": inner_radius(parameters),
        "R_out_kpc": parameters.rout_kpc,
        "Mdot_land_Msun_yr": 0.5,
        "KS_A_net": 0.6 * 9.95267926383743e-4,
        "KS_N": 1.4,
        "lambda": parameters.landing_slope,
        "Z_land": parameters.zw_solar * SOLAR_METALLICITY,
        "yield_y": parameters.yield_mass_fraction,
        "Z_outer_boundary": parameters.zout_solar * SOLAR_METALLICITY,
    })


def metallicity(radius_kpc, parameters):
    """Return the core Z/Z_sun, including the finite value at launch."""
    return analytic_profiles(radius_kpc, parameters)["Z"] / SOLAR_METALLICITY


def logarithmic_gradient(radius_kpc, parameters):
    """Return d(Z/Z_sun)/d ln(R) from the core analytic derivative."""
    radius = np.asarray(radius_kpc, dtype=float)
    return radius * analytic_profiles(radius, parameters)["dZ_dR_per_kpc"] / SOLAR_METALLICITY


def metallicity_gradient_dex_per_kpc(radius_kpc, parameters):
    """Return d log10(Z)/dR in dex/kpc."""
    values = analytic_profiles(radius_kpc, parameters)
    with np.errstate(divide="ignore", invalid="ignore"):
        return values["dZ_dR_per_kpc"] / (values["Z"] * np.log(10.0))


def radius_samples(parameters, sample_count, inner_offset_fraction):
    """Return profile samples and nonsingular gradient samples."""
    start = inner_radius(parameters)
    span = parameters.rout_kpc - start
    profile_radius = np.linspace(start, parameters.rout_kpc, sample_count)
    gradient_radius = np.linspace(
        start + inner_offset_fraction * span if start == parameters.rw_kpc else start,
        parameters.rout_kpc,
        sample_count,
    )
    return profile_radius, gradient_radius


def make_figure(
    parameter_sets,
    sample_count,
    inner_offset_fraction,
    observational_bands,
    observational_profiles,
    observational_calibration,
):
    figure, (metallicity_axis, gradient_axis) = plt.subplots(1, 2, figsize=(7.1, 3.0))
    heading = figure.suptitle("Unmixed fountain, analytic solution", x=0.5, y=0.995, fontsize=12)
    gradient_inset = gradient_axis.inset_axes([0.37, 0.19, 0.53, 0.42])

    observed_upper = draw_observed_metallicity(metallicity_axis, observational_profiles, observational_calibration)
    draw_observational_gradient_bands(gradient_axis, observational_bands)
    draw_observational_gradient_bands(gradient_inset, observational_bands)

    for parameters in parameter_sets:
        profile_radius, gradient_radius = radius_samples(parameters, sample_count, inner_offset_fraction)
        style = {
            "color": parameters.color,
            "linestyle": parameters.linestyle,
            "linewidth": 1.0,
            "zorder": 4 if parameters in DIAGNOSTIC_PARAMETER_SETS else 3,
        }
        metallicity_axis.plot(profile_radius, metallicity(profile_radius, parameters), label=parameters.label, **style)
        gradient_axis.plot(gradient_radius, metallicity_gradient_dex_per_kpc(gradient_radius, parameters), **style)
        gradient_inset.plot(gradient_radius, metallicity_gradient_dex_per_kpc(gradient_radius, parameters), **style)

    metallicity_axis.set_xlabel(r"$R\;[\mathrm{kpc}]$")
    metallicity_axis.set_ylabel(r"$Z/Z_\odot$")
    metallicity_axis.set_ylim(0.0, max(4.0, 1.05 * observed_upper))
    gradient_axis.set_xlabel(r"$R\;[\mathrm{kpc}]$")
    gradient_axis.set_ylabel(quantity_unit_label(r"$d\log_{10}Z/dR\;[\mathrm{dex\,kpc}^{-1}]$"))
    gradient_axis.set_ylim(-0.4, 0.0)
    gradient_inset.set_xlim(2.5, 10.0)
    gradient_inset.set_ylim(-0.06, -0.01)
    gradient_inset.set_xticks([2.5, 5.0, 7.5, 10.0])
    gradient_inset.set_yticks([-0.06, -0.04, -0.02, -0.01])
    gradient_inset.tick_params(axis="both", which="both", labelsize=7)
    gradient_axis.indicate_inset_zoom(gradient_inset, edgecolor="0.4", linewidth=0.6)
    legend_handles, legend_labels = metallicity_axis.get_legend_handles_labels()
    model_legend = [
        (handle, label)
        for handle, label in zip(legend_handles, legend_labels)
        if label in {parameters.label for parameters in parameter_sets}
    ]
    # Keep related settings adjacent in the single horizontal legend row.
    grouped_legend = ([model_legend[index] for index in (0, 1, 4, 3, 5, 2, 6, 7)]
                      if parameter_sets == PARAMETER_SETS else model_legend)
    legend = figure.legend(
        [handle for handle, _ in grouped_legend],
        [label for _, label in grouped_legend],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        borderaxespad=0.0,
        ncol=len(grouped_legend),
        fontsize=9.5,
        handlelength=1.4,
        columnspacing=0.4,
        handletextpad=0.4,
        labelspacing=0.45,
    )

    for axis in (metallicity_axis, gradient_axis):
        axis.set_xlim(0.0, max(parameters.rout_kpc for parameters in parameter_sets))

    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.85, wspace=0.30)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    gap = 4.0 * figure.dpi / 72.0
    panel_top = max(axis.get_tightbbox(renderer).y1 for axis in (metallicity_axis, gradient_axis))
    panel_bottom = min(axis.get_tightbbox(renderer).y0 for axis in (metallicity_axis, gradient_axis))
    top_shift = (heading.get_window_extent(renderer).y0 - gap - panel_top) / figure.bbox.height
    bottom_shift = (legend.get_window_extent(renderer).y1 + gap - panel_bottom) / figure.bbox.height
    figure.subplots_adjust(top=figure.subplotpars.top + top_shift, bottom=figure.subplotpars.bottom + bottom_shift)
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot the analytic Z(R) profile and its logarithmic gradient.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/analytic_metallicity_profiles.pdf"),
        help="output figure path (default: figures/analytic_metallicity_profiles.pdf)",
    )
    parser.add_argument("--samples", type=int, default=1000, help="number of samples per curve (default: 1000)")
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
            "band and measured Z(R) points (default: KK04)"
        ),
    )
    parser.add_argument(
        "--observational-sample",
        choices=("q1", "moustakas21"),
        default="q1",
        help=(
            "galaxy sample used for the gradient band: the eight-galaxy "
            "Q=1 overlap or the 21 Moustakas gradient-quality disks "
            "(default: q1); the left Z(R) panel uses only NGC 2403"
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
    bands = tuple(band for band in all_bands if band.calibration == arguments.observational_calibration)
    observed_profiles = {
        "NGC2403": observational_abundance_profiles("q1", arguments.observational_calibration)["NGC2403"]
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for name, cases in METALLICITY_SCANS.items():
        figure = make_figure(
            cases, arguments.samples, arguments.inner_offset_fraction,
            bands, observed_profiles, arguments.observational_calibration,
        )
        path = arguments.output if not name else arguments.output.with_name(
            f"{arguments.output.stem}_{name}{arguments.output.suffix}"
        )
        figure.savefig(path, bbox_inches="tight", pad_inches=0.02)
        plt.close(figure)
        print(f"Wrote {path}")
    print(
        f"Observed {arguments.observational_calibration} Z(R): "
        f"{sum(len(profile) for profile in observed_profiles.values())} unique regions "
        f"in {len(observed_profiles)} galaxies; solar-scaled oxygen abundances"
    )
    for band in bands:
        print(
            f"{arguments.observational_sample} {band.calibration}: "
            f"P14={band.lower:.6f}, median={band.median:.6f}, "
            f"P86={band.upper:.6f} dex/kpc"
        )


if __name__ == "__main__":
    main()
