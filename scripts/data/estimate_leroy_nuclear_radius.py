#!/usr/bin/env python3
"""Test whether Leroy SFR profiles support an effective nuclear launch radius.

This is an exploratory decomposition, not a production model-input generator.
For each paper-sample galaxy, it compares

    Sigma_SFR(R) = A_disk exp(-R / L_disk)

with

    Sigma_SFR(R) = A_disk exp(-R / L_disk)
                 + A_nuc exp[-(R - R_ring)^2 / (2 width^2)].

The effective nuclear radius uses the correct annular area weighting,

    R_nuc = integral R^2 Sigma_nuc dR / integral R Sigma_nuc dR.

Only Leroy measurements at R <= R25 are fitted. Bigiel points are intentionally
excluded because they extend the outer disk rather than the nuclear region.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "plot"))

from paper_galaxies import PAPER_GALAXIES, galaxy_label  # noqa: E402


@dataclass(frozen=True)
class GalaxyData:
    galaxy: str
    radius: np.ndarray
    sigma_sfr: np.ndarray
    error: np.ndarray
    r25: float
    leroy_scale_length: float
    catalog_sfr: float


@dataclass(frozen=True)
class ProfileFit:
    galaxy: str
    fractional_error_floor: float
    point_count: int
    disk_only_amplitude: float
    disk_only_scale_length: float
    disk_only_chi2: float
    disk_amplitude: float
    disk_scale_length: float
    nuclear_amplitude: float
    ring_radius: float
    nuclear_width: float
    two_component_chi2: float
    delta_bic: float
    effective_nuclear_radius: float
    nuclear_sfr: float
    model_total_sfr_r25: float
    nuclear_fraction_model: float
    nuclear_fraction_catalog: float
    supporting_bin_count: int
    compact_for_nuclear_radius: bool
    central_extent_over_r25: float
    status: str


def finite_positive(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def load_metadata(path: Path) -> dict[str, dict[str, float]]:
    metadata: dict[str, dict[str, float]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            galaxy = row["galaxy"]
            if galaxy not in PAPER_GALAXIES:
                continue
            fields = {
                "r25": finite_positive(row["R25_leroy_kpc"]),
                "scale_length": finite_positive(row["lSFR_leroy_kpc"]),
                "catalog_sfr": finite_positive(row["SFR_leroy_Msun_yr"]),
            }
            if any(value is None for value in fields.values()):
                raise ValueError(f"Incomplete Leroy metadata for {galaxy}: {fields}")
            metadata[galaxy] = {name: float(value) for name, value in fields.items()}

    missing = set(PAPER_GALAXIES) - set(metadata)
    if missing:
        raise ValueError(f"Missing Leroy metadata for: {', '.join(sorted(missing))}")
    return metadata


def load_profiles(path: Path, metadata: dict[str, dict[str, float]]) -> list[GalaxyData]:
    rows: dict[str, list[tuple[float, float, float]]] = {
        galaxy: [] for galaxy in PAPER_GALAXIES
    }
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            galaxy = row["galaxy"]
            if galaxy not in rows:
                continue
            radius = finite_positive(row["R_kpc"])
            sigma_sfr = finite_positive(row["SigmaSFR_Msun_yr_kpc2"])
            error = finite_positive(row["e_SigmaSFR_Msun_yr_kpc2"])
            if radius is None or sigma_sfr is None or error is None:
                continue
            if radius <= metadata[galaxy]["r25"] * (1.0 + 1e-12):
                rows[galaxy].append((radius, sigma_sfr, error))

    profiles: list[GalaxyData] = []
    for galaxy in PAPER_GALAXIES:
        values = sorted(rows[galaxy])
        if len(values) < 8:
            raise ValueError(f"{galaxy} has only {len(values)} usable Leroy bins inside R25")
        array = np.asarray(values, dtype=float)
        profiles.append(
            GalaxyData(
                galaxy=galaxy,
                radius=array[:, 0],
                sigma_sfr=array[:, 1],
                error=array[:, 2],
                r25=metadata[galaxy]["r25"],
                leroy_scale_length=metadata[galaxy]["scale_length"],
                catalog_sfr=metadata[galaxy]["catalog_sfr"],
            )
        )
    return profiles


def best_disk_fit(
    radius: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    scale_grid: np.ndarray,
) -> tuple[float, float, float]:
    basis = np.exp(-radius[None, :] / scale_grid[:, None])
    denominator = np.sum(weights[None, :] * basis * basis, axis=1)
    numerator = np.sum(weights[None, :] * basis * values[None, :], axis=1)
    amplitude = np.maximum(numerator / denominator, 0.0)
    residual = values[None, :] - amplitude[:, None] * basis
    chi2 = np.sum(weights[None, :] * residual * residual, axis=1)
    index = int(np.argmin(chi2))
    return float(amplitude[index]), float(scale_grid[index]), float(chi2[index])


def fit_two_component(
    radius: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    scale_grid: np.ndarray,
    center_grid: np.ndarray,
    width_grid: np.ndarray,
    r25: float,
) -> tuple[float, float, float, float, float, float]:
    disk_basis = np.exp(-radius[None, :] / scale_grid[:, None])
    sdd = np.sum(weights[None, :] * disk_basis * disk_basis, axis=1)
    syd = np.sum(weights[None, :] * disk_basis * values[None, :], axis=1)
    ywy = float(np.sum(weights * values * values))

    best: tuple[float, float, float, float, float, float] | None = None
    best_chi2 = math.inf

    for center in center_grid:
        for width in width_grid:
            if center + 2.0 * width > 0.5 * r25:
                continue
            central_basis = np.exp(-0.5 * ((radius - center) / width) ** 2)
            scc = float(np.sum(weights * central_basis * central_basis))
            syc = float(np.sum(weights * values * central_basis))
            sdc = np.sum(weights[None, :] * disk_basis * central_basis[None, :], axis=1)
            determinant = sdd * scc - sdc * sdc

            disk_amplitude = np.divide(
                syd * scc - syc * sdc,
                determinant,
                out=np.zeros_like(syd),
                where=determinant > 0.0,
            )
            nuclear_amplitude = np.divide(
                syc * sdd - syd * sdc,
                determinant,
                out=np.zeros_like(syd),
                where=determinant > 0.0,
            )

            disk_negative = disk_amplitude < 0.0
            nuclear_negative = nuclear_amplitude < 0.0
            disk_amplitude = np.where(
                disk_negative,
                0.0,
                disk_amplitude,
            )
            nuclear_amplitude = np.where(
                disk_negative,
                max(syc / scc, 0.0),
                nuclear_amplitude,
            )
            disk_amplitude = np.where(
                nuclear_negative,
                np.maximum(syd / sdd, 0.0),
                disk_amplitude,
            )
            nuclear_amplitude = np.where(nuclear_negative, 0.0, nuclear_amplitude)

            chi2 = (
                ywy
                - 2.0 * disk_amplitude * syd
                - 2.0 * nuclear_amplitude * syc
                + disk_amplitude * disk_amplitude * sdd
                + 2.0 * disk_amplitude * nuclear_amplitude * sdc
                + nuclear_amplitude * nuclear_amplitude * scc
            )
            index = int(np.argmin(chi2))
            if float(chi2[index]) < best_chi2:
                best_chi2 = float(chi2[index])
                best = (
                    float(disk_amplitude[index]),
                    float(scale_grid[index]),
                    float(nuclear_amplitude[index]),
                    float(center),
                    float(width),
                    best_chi2,
                )

    if best is None:
        raise RuntimeError("No valid two-component grid point")
    return best


def component_integrals(
    r25: float,
    disk_amplitude: float,
    disk_scale_length: float,
    nuclear_amplitude: float,
    ring_radius: float,
    nuclear_width: float,
) -> tuple[float, float, float, float]:
    radius = np.linspace(0.0, r25, 4001)
    disk = disk_amplitude * np.exp(-radius / disk_scale_length)
    nuclear = nuclear_amplitude * np.exp(
        -0.5 * ((radius - ring_radius) / nuclear_width) ** 2
    )
    nuclear_annular = radius * nuclear
    nuclear_norm = float(np.trapezoid(nuclear_annular, radius))
    effective_radius = (
        float(np.trapezoid(radius * nuclear_annular, radius)) / nuclear_norm
        if nuclear_norm > 0.0
        else math.nan
    )
    nuclear_sfr = 2.0 * math.pi * nuclear_norm
    disk_sfr = 2.0 * math.pi * float(np.trapezoid(radius * disk, radius))
    total_sfr = disk_sfr + nuclear_sfr
    fraction = nuclear_sfr / total_sfr if total_sfr > 0.0 else math.nan
    return effective_radius, nuclear_sfr, total_sfr, fraction


def classify(
    delta_bic: float,
    supporting_bin_count: int,
    nuclear_amplitude: float,
    compact_for_nuclear_radius: bool,
) -> str:
    if nuclear_amplitude <= 0.0 or delta_bic < 2.0:
        return "no_evidence"
    if delta_bic < 10.0:
        return (
            "tentative_compact_candidate"
            if compact_for_nuclear_radius
            else "tentative_broad_inner_component"
        )
    if not compact_for_nuclear_radius:
        return "strong_broad_inner_component"
    if supporting_bin_count < 2:
        return "strong_compact_single_bin_candidate"
    return "strong_compact_multi_bin_candidate"


def fit_profile(data: GalaxyData, fractional_error_floor: float) -> ProfileFit:
    radius = data.radius
    values = data.sigma_sfr
    sigma = np.hypot(data.error, fractional_error_floor * values)
    weights = 1.0 / sigma**2

    scale_grid = np.geomspace(
        max(0.15, 0.3 * data.leroy_scale_length),
        min(2.0 * data.r25, 3.0 * data.leroy_scale_length),
        80,
    )
    minimum_spacing = float(np.min(np.diff(radius)))
    minimum_width = max(0.5 * minimum_spacing, 0.02 * data.r25)
    maximum_width = 0.20 * data.r25
    width_grid = np.geomspace(minimum_width, maximum_width, 44)
    center_grid = np.linspace(0.0, 0.30 * data.r25, 61)

    disk_only_amplitude, disk_only_scale, disk_only_chi2 = best_disk_fit(
        radius,
        values,
        weights,
        scale_grid,
    )
    (
        disk_amplitude,
        disk_scale,
        nuclear_amplitude,
        ring_radius,
        nuclear_width,
        two_component_chi2,
    ) = fit_two_component(
        radius,
        values,
        weights,
        scale_grid,
        center_grid,
        width_grid,
        data.r25,
    )

    disk_bic = disk_only_chi2 + 2.0 * math.log(len(radius))
    two_component_bic = two_component_chi2 + 5.0 * math.log(len(radius))
    delta_bic = disk_bic - two_component_bic
    central_at_bins = nuclear_amplitude * np.exp(
        -0.5 * ((radius - ring_radius) / nuclear_width) ** 2
    )
    supporting_bin_count = int(np.count_nonzero(central_at_bins >= sigma))
    effective_radius, nuclear_sfr, total_sfr, nuclear_fraction = component_integrals(
        data.r25,
        disk_amplitude,
        disk_scale,
        nuclear_amplitude,
        ring_radius,
        nuclear_width,
    )
    central_extent_over_r25 = (ring_radius + 2.0 * nuclear_width) / data.r25
    compact_for_nuclear_radius = (
        math.isfinite(effective_radius)
        and effective_radius <= 0.15 * data.r25
        and central_extent_over_r25 <= 0.25
    )

    return ProfileFit(
        galaxy=data.galaxy,
        fractional_error_floor=fractional_error_floor,
        point_count=len(radius),
        disk_only_amplitude=disk_only_amplitude,
        disk_only_scale_length=disk_only_scale,
        disk_only_chi2=disk_only_chi2,
        disk_amplitude=disk_amplitude,
        disk_scale_length=disk_scale,
        nuclear_amplitude=nuclear_amplitude,
        ring_radius=ring_radius,
        nuclear_width=nuclear_width,
        two_component_chi2=two_component_chi2,
        delta_bic=delta_bic,
        effective_nuclear_radius=effective_radius,
        nuclear_sfr=nuclear_sfr,
        model_total_sfr_r25=total_sfr,
        nuclear_fraction_model=nuclear_fraction,
        nuclear_fraction_catalog=nuclear_sfr / data.catalog_sfr,
        supporting_bin_count=supporting_bin_count,
        compact_for_nuclear_radius=compact_for_nuclear_radius,
        central_extent_over_r25=central_extent_over_r25,
        status=classify(
            delta_bic,
            supporting_bin_count,
            nuclear_amplitude,
            compact_for_nuclear_radius,
        ),
    )


def write_floor_results(path: Path, fits: list[ProfileFit]) -> None:
    fieldnames = list(ProfileFit.__dataclass_fields__)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for fit in fits:
            writer.writerow({name: getattr(fit, name) for name in fieldnames})


def write_summary(
    path: Path,
    profiles: list[GalaxyData],
    fits: list[ProfileFit],
    primary_floor: float,
) -> None:
    by_galaxy: dict[str, list[ProfileFit]] = {galaxy: [] for galaxy in PAPER_GALAXIES}
    for fit in fits:
        by_galaxy[fit.galaxy].append(fit)

    fieldnames = [
        "galaxy",
        "label",
        "leroy_point_count_inside_R25",
        "innermost_R_kpc",
        "R25_kpc",
        "catalog_SFR_Msun_yr",
        "fractional_error_floor_primary",
        "delta_BIC_primary",
        "status_primary",
        "supporting_bins_primary",
        "compact_for_R_nuc_primary",
        "central_extent_over_R25_primary",
        "R_nuc_effective_kpc_primary",
        "R_ring_kpc_primary",
        "nuclear_width_kpc_primary",
        "Mdot_nuc_Msun_yr_primary",
        "nuclear_fraction_model_primary",
        "nuclear_fraction_catalog_primary",
        "R_nuc_min_across_floors_kpc",
        "R_nuc_max_across_floors_kpc",
        "delta_BIC_min_across_floors",
        "delta_BIC_max_across_floors",
        "statuses_across_floors",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for profile in profiles:
            galaxy_fits = sorted(
                by_galaxy[profile.galaxy], key=lambda item: item.fractional_error_floor
            )
            primary = min(
                galaxy_fits,
                key=lambda item: abs(item.fractional_error_floor - primary_floor),
            )
            finite_radii = [
                fit.effective_nuclear_radius
                for fit in galaxy_fits
                if math.isfinite(fit.effective_nuclear_radius)
            ]
            writer.writerow(
                {
                    "galaxy": profile.galaxy,
                    "label": galaxy_label(profile.galaxy),
                    "leroy_point_count_inside_R25": len(profile.radius),
                    "innermost_R_kpc": profile.radius[0],
                    "R25_kpc": profile.r25,
                    "catalog_SFR_Msun_yr": profile.catalog_sfr,
                    "fractional_error_floor_primary": primary.fractional_error_floor,
                    "delta_BIC_primary": primary.delta_bic,
                    "status_primary": primary.status,
                    "supporting_bins_primary": primary.supporting_bin_count,
                    "compact_for_R_nuc_primary": primary.compact_for_nuclear_radius,
                    "central_extent_over_R25_primary": primary.central_extent_over_r25,
                    "R_nuc_effective_kpc_primary": primary.effective_nuclear_radius,
                    "R_ring_kpc_primary": primary.ring_radius,
                    "nuclear_width_kpc_primary": primary.nuclear_width,
                    "Mdot_nuc_Msun_yr_primary": primary.nuclear_sfr,
                    "nuclear_fraction_model_primary": primary.nuclear_fraction_model,
                    "nuclear_fraction_catalog_primary": primary.nuclear_fraction_catalog,
                    "R_nuc_min_across_floors_kpc": min(finite_radii, default=math.nan),
                    "R_nuc_max_across_floors_kpc": max(finite_radii, default=math.nan),
                    "delta_BIC_min_across_floors": min(
                        fit.delta_bic for fit in galaxy_fits
                    ),
                    "delta_BIC_max_across_floors": max(
                        fit.delta_bic for fit in galaxy_fits
                    ),
                    "statuses_across_floors": ";".join(
                        f"{fit.fractional_error_floor:.2f}:{fit.status}"
                        for fit in galaxy_fits
                    ),
                }
            )


def write_fitted_profiles(
    path: Path,
    profiles: list[GalaxyData],
    fits: list[ProfileFit],
    primary_floor: float,
) -> None:
    primary_fits = {
        fit.galaxy: fit
        for fit in fits
        if math.isclose(fit.fractional_error_floor, primary_floor, abs_tol=1e-12)
    }
    fieldnames = [
        "galaxy",
        "R_kpc",
        "disk_only_Msun_yr_kpc2",
        "disk_component_Msun_yr_kpc2",
        "nuclear_component_Msun_yr_kpc2",
        "total_model_Msun_yr_kpc2",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for profile in profiles:
            fit = primary_fits[profile.galaxy]
            radius = np.linspace(0.0, profile.r25, 500)
            disk_only = fit.disk_only_amplitude * np.exp(
                -radius / fit.disk_only_scale_length
            )
            disk = fit.disk_amplitude * np.exp(-radius / fit.disk_scale_length)
            nuclear = fit.nuclear_amplitude * np.exp(
                -0.5 * ((radius - fit.ring_radius) / fit.nuclear_width) ** 2
            )
            for values in zip(radius, disk_only, disk, nuclear, disk + nuclear):
                writer.writerow(dict(zip(fieldnames, (profile.galaxy, *values))))


def plot_diagnostics(
    path: Path,
    profiles: list[GalaxyData],
    fits: list[ProfileFit],
    primary_floor: float,
) -> None:
    primary_fits = {
        fit.galaxy: fit
        for fit in fits
        if math.isclose(fit.fractional_error_floor, primary_floor, abs_tol=1e-12)
    }
    figure, axes = plt.subplots(4, 2, figsize=(11.0, 13.5), constrained_layout=True)
    axes_flat = axes.ravel()

    for axis, profile in zip(axes_flat, profiles):
        fit = primary_fits[profile.galaxy]
        radius = np.linspace(0.0, profile.r25, 1000)
        disk_only = fit.disk_only_amplitude * np.exp(
            -radius / fit.disk_only_scale_length
        )
        disk = fit.disk_amplitude * np.exp(-radius / fit.disk_scale_length)
        nuclear = fit.nuclear_amplitude * np.exp(
            -0.5 * ((radius - fit.ring_radius) / fit.nuclear_width) ** 2
        )
        sigma = np.hypot(profile.error, primary_floor * profile.sigma_sfr)

        axis.errorbar(
            profile.radius,
            profile.sigma_sfr,
            yerr=sigma,
            fmt="o",
            color="black",
            markersize=3.5,
            linewidth=0.8,
            capsize=1.5,
            label="Leroy",
        )
        axis.plot(radius, disk_only, color="0.5", linestyle="--", label="disk only")
        axis.plot(radius, disk, color="#1f77b4", label="disk component")
        axis.plot(radius, nuclear, color="#d62728", label="central component")
        axis.plot(radius, disk + nuclear, color="black", linewidth=1.5, label="total")
        if math.isfinite(fit.effective_nuclear_radius):
            axis.axvline(
                fit.effective_nuclear_radius,
                color="#d62728",
                linestyle=":",
                linewidth=1.2,
            )
        axis.set_yscale("log")
        axis.set_xlim(0.0, profile.r25)
        plotted_positive = np.concatenate(
            [
                profile.sigma_sfr,
                disk_only[disk_only > 0.0],
                disk[disk > 0.0],
                nuclear[nuclear > 0.0],
                (disk + nuclear)[disk + nuclear > 0.0],
            ]
        )
        lower = max(1e-7, 0.3 * float(np.min(profile.sigma_sfr)))
        upper = 3.0 * float(np.max(plotted_positive))
        axis.set_ylim(lower, upper)
        axis.set_title(
            f"{galaxy_label(profile.galaxy)}  "
            f"Delta BIC={fit.delta_bic:.1f}\n{fit.status.replace('_', ' ')}"
        )
        axis.set_xlabel("R [kpc]")
        axis.set_ylabel(r"$\dot\Sigma_{\rm SFR}$ [$M_\odot$ yr$^{-1}$ kpc$^{-2}$]")

    for axis in axes_flat[len(profiles) :]:
        axis.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.02), ncol=2)
    figure.suptitle(
        f"Leroy central-SFR decomposition ({100.0 * primary_floor:.0f}% error floor)",
        fontsize=15,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_methodology(
    path: Path,
    floors: list[float],
    primary_floor: float,
) -> None:
    path.write_text(
        "\n".join(
            [
                "# Leroy effective nuclear-radius test",
                "",
                "This is an exploratory structural test, not a canonical model-input product.",
                "",
                "## Data",
                "",
                "- Current paper-sample galaxies from `scripts/plot/paper_galaxies.py`.",
                "- Leroy SFR annuli only, restricted to positive measurements at `R <= R25`.",
                "- Bigiel measurements are excluded because they constrain the outer disk.",
                "",
                "## Models",
                "",
                "The null model is `Sigma_SFR = A_disk exp(-R/L_disk)`. The alternative",
                "adds a nonnegative Gaussian central/ring component",
                "`A_nuc exp[-(R-R_ring)^2/(2 width^2)]`. Amplitudes are solved by",
                "weighted nonnegative least squares on a grid of disk scale length, ring",
                "radius, and component width. The central component is constrained by",
                "`R_ring + 2 width <= 0.5 R25`.",
                "",
                "The effective radius is area weighted:",
                "`R_nuc = integral R^2 Sigma_nuc dR / integral R Sigma_nuc dR`.",
                "The same component gives `Mdot_nuc = 2 pi integral R Sigma_nuc dR`.",
                "",
                "## Evidence and robustness",
                "",
                "`Delta BIC = BIC_disk_only - BIC_disk_plus_central`; positive values",
                "favor a central component. Values below 2 are classified as no evidence,",
                "2--10 as tentative, and >=10 as strong. A strong component is called",
                "multi-bin only when its modeled contribution exceeds the effective error",
                "in at least two measured annuli. This is not a beam-resolution test.",
                "",
                "For use as a nuclear-radius proxy, a component must additionally have",
                "`R_nuc <= 0.15 R25` and `R_ring + 2 width <= 0.25 R25`. Broader",
                "components are reported as inner-disk structure rather than nuclear",
                "launch-radius measurements. These compactness cuts are diagnostic choices,",
                "not externally calibrated physical thresholds.",
                "",
                f"Fits use fractional error floors {floors}; the primary report uses {primary_floor}.",
                "The across-floor range is a systematics sensitivity diagnostic, not a",
                "formal confidence interval. The inferred values remain model dependent.",
                "",
            ]
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=ROOT / "input",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "output" / "diagnostics" / "leroy_nuclear_radius",
    )
    parser.add_argument(
        "--fractional-error-floors",
        default="0.00,0.10,0.20,0.30",
        help="Comma-separated fractional systematic floors.",
    )
    parser.add_argument("--primary-floor", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    floors = sorted({float(value) for value in args.fractional_error_floors.split(",")})
    if not floors or any(value < 0.0 for value in floors):
        raise ValueError("Fractional error floors must be nonnegative")
    if not any(math.isclose(value, args.primary_floor, abs_tol=1e-12) for value in floors):
        raise ValueError("The primary floor must be included in --fractional-error-floors")

    metadata = load_metadata(args.input_directory / "galaxies.csv")
    profiles = load_profiles(args.input_directory / "leroy_radial_profiles.csv", metadata)
    fits = [
        fit_profile(profile, floor)
        for profile in profiles
        for floor in floors
    ]

    args.output_directory.mkdir(parents=True, exist_ok=True)
    write_floor_results(args.output_directory / "results_by_error_floor.csv", fits)
    write_summary(args.output_directory / "summary.csv", profiles, fits, args.primary_floor)
    write_fitted_profiles(
        args.output_directory / "fitted_profiles.csv",
        profiles,
        fits,
        args.primary_floor,
    )
    plot_diagnostics(
        args.output_directory / "leroy_nuclear_radius_diagnostic.png",
        profiles,
        fits,
        args.primary_floor,
    )
    write_methodology(
        args.output_directory / "methodology.md",
        floors,
        args.primary_floor,
    )
    print(f"Wrote Leroy nuclear-radius diagnostic to {args.output_directory}")


if __name__ == "__main__":
    main()
