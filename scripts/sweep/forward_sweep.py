#!/usr/bin/env python3
"""Run staged forward-model sweeps without retaining every radial profile."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

PLOT_SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "plot"
if str(PLOT_SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PLOT_SCRIPT_DIRECTORY))

from paper_galaxies import (  # noqa: E402
    PAPER_GALAXIES,
    PAPER_GALAXY_SET,
    canonical_galaxy_name,
    galaxy_label,
)


DEFAULT_GALAXIES = (
    "NGC3521",
    "NGC7793",
    "NGC2403",
    "NGC3198",
    "NGC5055",
    "NGC6946",
)
DEFAULT_MU_VALUES = (
    0.0,
    0.03,
    0.05,
    0.08,
    0.12,
    0.2,
    0.3,
    0.5,
    0.7,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
)
CALIBRATIONS = ("KK04", "PT05")
OH12_SUN = 8.69
Z_SUN = 0.0139
ABUNDANCE_SCATTER_DEX = 0.05
TARGET_LANDING_RTOL = 1.0e-5

RESULT_FIELDS = (
    "galaxy",
    "point_id",
    "mu",
    "beta",
    "f_land",
    "SFR_Msun_yr",
    "Mdot_land_input_Msun_yr",
    "target_landing_applies",
    "rotation_source",
    "calibration",
    "status",
    "error",
    "dynamics_finite",
    "metallicity_solved",
    "inward_flow",
    "landing_nonnegative",
    "target_landing_matched",
    "valid_for_ranking",
    "min_v_R_kpc_yr",
    "max_v_R_kpc_yr",
    "min_v_R_kms",
    "max_v_R_kms",
    "min_Sigmadot_land_Msun_yr_kpc2",
    "max_Sigmadot_land_Msun_yr_kpc2",
    "achieved_Mdot_land_Msun_yr",
    "landing_relative_error",
    "n_metallicity_profile_points",
    "n_catalog_usable",
    "n_unique_catalog_regions",
    "n_model_overlap",
    "fit_status",
    "weighted_mean_residual_dex",
    "chi2_fixed_model",
    "reduced_chi2_fixed_model",
    "chi2_after_offset",
    "reduced_chi2_after_offset",
)

BEST_FIELDS = (
    "galaxy",
    "rank",
    "point_id",
    "mu",
    "beta",
    "f_land",
    "SFR_Msun_yr",
    "Mdot_land_input_Msun_yr",
    "ranking_calibration",
    "mean_reduced_chi2_fixed_model",
    "rotation_scores",
    "output_directory",
)


@dataclass(frozen=True)
class SweepPoint:
    point_id: str
    mu: float
    beta: float | None
    f_land: float | None
    planned_status: str

    @property
    def target_landing_applies(self) -> bool:
        return self.mu != 0.0


@dataclass(frozen=True)
class Observation:
    source_sequence: int
    offset_ra_arcsec: float | None
    offset_de_arcsec: float | None
    radius_kpc: float
    oh12: float
    e_oh12: float


@dataclass(frozen=True)
class FitResult:
    n_catalog_usable: int
    n_unique_catalog_regions: int
    n_model_overlap: int
    fit_status: str
    weighted_mean_residual_dex: float | None
    chi2_fixed_model: float | None
    reduced_chi2_fixed_model: float | None
    chi2_after_offset: float | None
    reduced_chi2_after_offset: float | None


@dataclass(frozen=True)
class RankedPoint:
    point: SweepPoint
    score: float
    source_scores: dict[str, float]


def linear_space(start: float, stop: float, count: int) -> list[float]:
    if count < 1:
        raise ValueError("linear spacing requires at least one value")
    if count == 1:
        return [float(start)]
    return [
        start + (stop - start) * index / (count - 1)
        for index in range(count)
    ]


def logarithmic_space(start: float, stop: float, count: int) -> list[float]:
    if start <= 0.0 or stop <= 0.0:
        raise ValueError("logarithmic spacing requires positive bounds")
    return [math.exp(value) for value in linear_space(math.log(start), math.log(stop), count)]


def build_grid(
    mu_values: Sequence[float],
    beta_values: Sequence[float],
    f_land_values: Sequence[float],
) -> list[SweepPoint]:
    unique_mu = list(dict.fromkeys(float(value) for value in mu_values))
    if any(value < 0.0 for value in unique_mu):
        raise ValueError("mu values must be nonnegative")
    if any(value < 0.0 or value > 1.0 for value in beta_values):
        raise ValueError("beta values must lie between 0 and 1")
    if any(value <= 0.0 for value in f_land_values):
        raise ValueError("f_land values must be positive")

    points: list[SweepPoint] = []
    sequence = 1
    for mu in unique_mu:
        if mu == 0.0:
            points.append(SweepPoint(f"p{sequence:06d}", mu, None, None, "pending"))
            sequence += 1
            continue
        for beta in beta_values:
            status = "singular_boundary" if math.isclose(beta, 1.0, rel_tol=0.0, abs_tol=1.0e-12) else "pending"
            for f_land in f_land_values:
                points.append(
                    SweepPoint(
                        f"p{sequence:06d}",
                        mu,
                        float(beta),
                        float(f_land),
                        status,
                    )
                )
                sequence += 1
    return points


def parse_float(text: str | None) -> float | None:
    if text is None or text.strip() == "":
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def csv_bool(text: str | None) -> bool:
    return text is not None and text.strip().lower() in {"1", "true", "yes"}


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field)) for field in fields})


def adopted_sfrs(path: Path, galaxies: Sequence[str]) -> dict[str, float]:
    requested = set(galaxies)
    result: dict[str, float] = {}
    for row in read_csv(path):
        galaxy = row["galaxy"]
        if galaxy not in requested:
            continue
        sfr = parse_float(row.get("SFR_leroy_Msun_yr"))
        if sfr is None:
            sfr = parse_float(row.get("SFR_things_Msun_yr"))
        if sfr is None:
            raise ValueError(f"No Leroy or THINGS SFR is available for {galaxy}")
        result[galaxy] = sfr
    missing = requested - result.keys()
    if missing:
        raise ValueError(f"Galaxies missing from {path}: {', '.join(sorted(missing))}")
    return result


def rotation_sources(input_directory: Path) -> list[str]:
    sources = [row["source"] for row in read_csv(input_directory / "rotation_curves.csv")]
    if not sources:
        raise ValueError(f"No rotation curves in {input_directory / 'rotation_curves.csv'}")
    return sources


def load_observations(
    catalog_path: Path,
    galaxy: str,
    calibration: str,
) -> list[Observation]:
    selected: list[Observation] = []
    required = (
        "R_adopted_kpc",
        f"OH12_{calibration}_dex",
        f"e_OH12_{calibration}_dex",
    )
    for row in read_csv(catalog_path):
        if row["galaxy"] != galaxy or not csv_bool(row.get(f"usable_OH12_{calibration}")):
            continue
        values = {name: parse_float(row.get(name)) for name in required}
        if any(value is None for value in values.values()):
            continue
        selected.append(
            Observation(
                source_sequence=int(float(row["source_seq"])),
                offset_ra_arcsec=parse_float(row.get("offRA_arcsec")),
                offset_de_arcsec=parse_float(row.get("offDE_arcsec")),
                radius_kpc=float(values["R_adopted_kpc"]),
                oh12=float(values[f"OH12_{calibration}_dex"]),
                e_oh12=float(values[f"e_OH12_{calibration}_dex"]),
            )
        )
    return sorted(selected, key=lambda item: (item.radius_kpc, item.source_sequence))


def collapse_repeated_regions(observations: Sequence[Observation]) -> list[Observation]:
    grouped: dict[tuple[bool, float, float, int], list[Observation]] = defaultdict(list)
    for observation in observations:
        has_offset = observation.offset_ra_arcsec is not None and observation.offset_de_arcsec is not None
        key = (
            (True, float(observation.offset_ra_arcsec), float(observation.offset_de_arcsec), 0)
            if has_offset
            else (False, 0.0, 0.0, observation.source_sequence)
        )
        grouped[key].append(observation)

    collapsed: list[Observation] = []
    for key in sorted(grouped):
        group = grouped[key]
        weights = [1.0 / (item.e_oh12 * item.e_oh12) for item in group]
        weight_sum = sum(weights)

        def weighted(attribute: str) -> float:
            return sum(weight * getattr(item, attribute) for weight, item in zip(weights, group)) / weight_sum

        collapsed.append(
            Observation(
                source_sequence=group[0].source_sequence,
                offset_ra_arcsec=group[0].offset_ra_arcsec,
                offset_de_arcsec=group[0].offset_de_arcsec,
                radius_kpc=weighted("radius_kpc"),
                oh12=weighted("oh12"),
                e_oh12=math.sqrt(1.0 / weight_sum),
            )
        )
    return sorted(collapsed, key=lambda item: item.radius_kpc)


def interpolate(x: Sequence[float], y: Sequence[float], query: float) -> float:
    if query == x[-1]:
        return y[-1]
    right = bisect.bisect_right(x, query)
    left = right - 1
    fraction = (query - x[left]) / (x[right] - x[left])
    return y[left] + fraction * (y[right] - y[left])


def compare_metallicity_curve(
    observations: Sequence[Observation],
    radii_kpc: Sequence[float],
    metallicities: Sequence[float],
) -> FitResult:
    unique = collapse_repeated_regions(observations)
    curve = sorted(
        (radius, metallicity)
        for radius, metallicity in zip(radii_kpc, metallicities)
        if math.isfinite(radius) and math.isfinite(metallicity) and metallicity > 0.0
    )
    if len(curve) < 2:
        return FitResult(len(observations), len(unique), 0, "missing model curve", None, None, None, None, None)

    curve_radii = [item[0] for item in curve]
    curve_oh12 = [OH12_SUN + math.log10(item[1] / Z_SUN) for item in curve]
    overlapping = [
        item for item in unique if curve_radii[0] <= item.radius_kpc <= curve_radii[-1]
    ]
    if not overlapping:
        return FitResult(len(observations), len(unique), 0, "no model-overlap points", None, None, None, None, None)

    residuals: list[float] = []
    sigmas: list[float] = []
    for observation in overlapping:
        model_oh12 = interpolate(curve_radii, curve_oh12, observation.radius_kpc)
        residuals.append(observation.oh12 - model_oh12)
        sigmas.append(math.hypot(observation.e_oh12, ABUNDANCE_SCATTER_DEX))

    weights = [1.0 / (sigma * sigma) for sigma in sigmas]
    weight_sum = sum(weights)
    offset = sum(weight * residual for weight, residual in zip(weights, residuals)) / weight_sum
    chi2_fixed = sum((residual / sigma) ** 2 for residual, sigma in zip(residuals, sigmas))
    chi2_offset = sum(((residual - offset) / sigma) ** 2 for residual, sigma in zip(residuals, sigmas))
    count = len(residuals)
    kpc_span = max(item.radius_kpc for item in overlapping) - min(item.radius_kpc for item in overlapping)
    if count < 2:
        status = "no gradient fit"
    elif count < 8:
        status = "descriptive: fewer than 8 unique regions"
    elif kpc_span < 2.0:
        status = "descriptive: radial span below 2 kpc"
    else:
        status = "fit-grade"
    return FitResult(
        n_catalog_usable=len(observations),
        n_unique_catalog_regions=len(unique),
        n_model_overlap=count,
        fit_status=status,
        weighted_mean_residual_dex=offset,
        chi2_fixed_model=chi2_fixed,
        reduced_chi2_fixed_model=chi2_fixed / count,
        chi2_after_offset=chi2_offset,
        reduced_chi2_after_offset=chi2_offset / (count - 1) if count > 1 else None,
    )


def write_parameter_input(
    base_directory: Path,
    destination: Path,
    mu: float,
    beta: float,
    mdot_land: float,
    include_provenance: bool,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    parameter_rows = read_csv(base_directory / "parameters.csv")
    overrides = {
        "mu": format(mu, ".17g"),
        "beta": format(beta, ".17g"),
        "Mdot_land_Msun_yr": format(mdot_land, ".17g"),
    }
    found: set[str] = set()
    for row in parameter_rows:
        name = row["name"]
        if name in overrides:
            row["value"] = overrides[name]
            found.add(name)
    missing = overrides.keys() - found
    if missing:
        raise ValueError(f"Missing model parameters: {', '.join(sorted(missing))}")
    write_csv(destination / "parameters.csv", ("name", "value"), parameter_rows)

    required = ["profiles.csv", "gas_profiles.csv", "rotation_curves.csv"]
    if any(
        row["kind"] == "tabulated"
        for row in read_csv(base_directory / "rotation_curves.csv")
    ):
        required.append("rotation_profiles.csv")
    optional = (
        "metadata.csv",
        "sparc_corrected.csv",
        "leroy_profiles_used.csv",
        "bigiel_profiles_used.csv",
        "gas_spline_diagnostics.csv",
        "sfr_spline_diagnostics.csv",
        "sfr_crosscalibration.csv",
        "rotation_spline_diagnostics.csv",
    ) if include_provenance else ()
    for name in (*required, *optional):
        source = base_directory / name
        if name in required and not source.exists():
            raise FileNotFoundError(source)
        if source.exists():
            (destination / name).symlink_to(source.resolve())


def invoke_model(
    model_binary: Path,
    base_input: Path,
    output_directory: Path,
    point: SweepPoint,
    sfr: float,
    timeout_seconds: float,
    include_provenance: bool,
) -> subprocess.CompletedProcess[str]:
    beta = 0.0 if point.beta is None else point.beta
    mdot_land = sfr if point.f_land is None else point.f_land * sfr
    with tempfile.TemporaryDirectory(prefix=f"gnf-{point.point_id}-input-") as raw_input:
        input_directory = Path(raw_input)
        write_parameter_input(base_input, input_directory, point.mu, beta, mdot_land, include_provenance)
        return subprocess.run(
            [str(model_binary), "forward", str(input_directory), str(output_directory)],
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )


def rows_by_source(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[row["source"]].append(row)
    return grouped


def blank_result_row(
    galaxy: str,
    point: SweepPoint,
    sfr: float,
    source: str,
    calibration: str,
    status: str,
    error: str = "",
) -> dict[str, object]:
    mdot_input = sfr if point.f_land is None else point.f_land * sfr
    row: dict[str, object] = {field: None for field in RESULT_FIELDS}
    row.update(
        {
            "galaxy": galaxy,
            "point_id": point.point_id,
            "mu": point.mu,
            "beta": point.beta,
            "f_land": point.f_land,
            "SFR_Msun_yr": sfr,
            "Mdot_land_input_Msun_yr": mdot_input,
            "target_landing_applies": point.target_landing_applies,
            "rotation_source": source,
            "calibration": calibration,
            "status": status,
            "error": error,
            "valid_for_ranking": False,
        }
    )
    return row


def evaluate_model_output(
    galaxy: str,
    point: SweepPoint,
    sfr: float,
    output_directory: Path,
    sources: Sequence[str],
    observations: dict[str, list[Observation]],
) -> list[dict[str, object]]:
    profiles = rows_by_source(output_directory / "profiles.csv")
    summaries = {row["source"]: row for row in read_csv(output_directory / "summary.csv")}
    result_rows: list[dict[str, object]] = []
    expected_target = None if point.f_land is None else point.f_land * sfr

    for source in sources:
        if source not in profiles or source not in summaries:
            for calibration in CALIBRATIONS:
                result_rows.append(
                    blank_result_row(
                        galaxy, point, sfr, source, calibration, "model_output_missing", "Missing source rows"
                    )
                )
            continue

        source_profiles = sorted(profiles[source], key=lambda row: float(row["R_kpc"]))
        summary = summaries[source]
        radii = [float(row["R_kpc"]) for row in source_profiles]
        velocities = [float(row["v_R_kpc_yr"]) for row in source_profiles]
        velocities_kms = [float(row["v_R_kms"]) for row in source_profiles]
        landing = [float(row["Sigmadot_land_Msun_yr_kpc2"]) for row in source_profiles]
        cumulative = [float(row["cumulative_landing_Msun_yr"]) for row in source_profiles]
        metallicity = [float(row["Z"]) for row in source_profiles]
        achieved = float(summary["total_landing_Msun_yr"])
        r_nucl = float(summary["R_nucl_kpc"])

        dynamics_finite = all(
            math.isfinite(value)
            for values in (radii, velocities, velocities_kms, landing, cumulative)
            for value in values
        ) and math.isfinite(achieved)
        inward_flow = dynamics_finite and all(
            velocity < 0.0
            for radius, velocity in zip(radii, velocities)
            if not (point.mu == 0.0 and radius == r_nucl)
        )
        landing_nonnegative = csv_bool(summary.get("landing_nonnegative"))
        metallicity_solved = csv_bool(summary.get("metallicity_solved"))
        relative_error = (
            None
            if expected_target is None
            else abs(achieved - expected_target) / max(abs(expected_target), 1.0e-300)
        )
        target_matched = relative_error is None or relative_error <= TARGET_LANDING_RTOL
        physical_valid = (
            dynamics_finite
            and inward_flow
            and landing_nonnegative
            and metallicity_solved
            and target_matched
        )

        for calibration in CALIBRATIONS:
            fit = compare_metallicity_curve(observations[calibration], radii, metallicity)
            valid_for_ranking = physical_valid and fit.reduced_chi2_fixed_model is not None
            status = "ok" if valid_for_ranking else "physical_invalid" if not physical_valid else "no_model_overlap"
            row = blank_result_row(galaxy, point, sfr, source, calibration, status)
            row.update(
                {
                    "dynamics_finite": dynamics_finite,
                    "metallicity_solved": metallicity_solved,
                    "inward_flow": inward_flow,
                    "landing_nonnegative": landing_nonnegative,
                    "target_landing_matched": target_matched,
                    "valid_for_ranking": valid_for_ranking,
                    "min_v_R_kpc_yr": min(velocities) if velocities else None,
                    "max_v_R_kpc_yr": max(velocities) if velocities else None,
                    "min_v_R_kms": min(velocities_kms) if velocities_kms else None,
                    "max_v_R_kms": max(velocities_kms) if velocities_kms else None,
                    "min_Sigmadot_land_Msun_yr_kpc2": min(landing) if landing else None,
                    "max_Sigmadot_land_Msun_yr_kpc2": max(landing) if landing else None,
                    "achieved_Mdot_land_Msun_yr": achieved,
                    "landing_relative_error": relative_error,
                    "n_metallicity_profile_points": sum(
                        math.isfinite(value) and value > 0.0 for value in metallicity
                    ),
                    "n_catalog_usable": fit.n_catalog_usable,
                    "n_unique_catalog_regions": fit.n_unique_catalog_regions,
                    "n_model_overlap": fit.n_model_overlap,
                    "fit_status": fit.fit_status,
                    "weighted_mean_residual_dex": fit.weighted_mean_residual_dex,
                    "chi2_fixed_model": fit.chi2_fixed_model,
                    "reduced_chi2_fixed_model": fit.reduced_chi2_fixed_model,
                    "chi2_after_offset": fit.chi2_after_offset,
                    "reduced_chi2_after_offset": fit.reduced_chi2_after_offset,
                }
            )
            result_rows.append(row)
    return result_rows


def rank_points(
    rows: Sequence[dict[str, object]],
    points: Sequence[SweepPoint],
    galaxy: str,
    calibration: str,
    sources: Sequence[str],
) -> list[RankedPoint]:
    point_lookup = {point.point_id: point for point in points}
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if (
            row["galaxy"] == galaxy
            and row["calibration"] == calibration
            and row["valid_for_ranking"] is True
            and row["rotation_source"] in sources
            and isinstance(row["reduced_chi2_fixed_model"], float)
        ):
            scores[str(row["point_id"])][str(row["rotation_source"])] = float(
                row["reduced_chi2_fixed_model"]
            )

    ranked: list[RankedPoint] = []
    for point_id, source_scores in scores.items():
        if set(source_scores) != set(sources):
            continue
        ranked.append(
            RankedPoint(
                point=point_lookup[point_id],
                score=sum(source_scores.values()) / len(sources),
                source_scores=dict(sorted(source_scores.items())),
            )
        )
    return sorted(ranked, key=lambda item: (item.score, item.point.point_id))


def result_rows_for_failure(
    galaxy: str,
    point: SweepPoint,
    sfr: float,
    sources: Sequence[str],
    status: str,
    error: str,
) -> list[dict[str, object]]:
    return [
        blank_result_row(galaxy, point, sfr, source, calibration, status, error)
        for source in sources
        for calibration in CALIBRATIONS
    ]


def run_galaxy_sweep(
    galaxy: str,
    points: Sequence[SweepPoint],
    sfr: float,
    model_binary: Path,
    input_directory: Path,
    metallicity_catalog: Path,
    timeout_seconds: float,
    progress_every: int,
) -> tuple[list[dict[str, object]], list[str]]:
    sources = rotation_sources(input_directory)
    observations = {
        calibration: load_observations(metallicity_catalog, galaxy, calibration)
        for calibration in CALIBRATIONS
    }
    rows: list[dict[str, object]] = []
    executable_points = sum(point.planned_status == "pending" for point in points)
    completed = 0
    started = time.monotonic()

    for point in points:
        if point.planned_status != "pending":
            rows.extend(
                result_rows_for_failure(
                    galaxy, point, sfr, sources, point.planned_status, "beta=1 is singular for mu>0"
                )
            )
            continue

        completed += 1
        try:
            with tempfile.TemporaryDirectory(prefix=f"gnf-{galaxy}-{point.point_id}-output-") as raw_output:
                output_directory = Path(raw_output)
                completed_process = invoke_model(
                    model_binary,
                    input_directory,
                    output_directory,
                    point,
                    sfr,
                    timeout_seconds,
                    include_provenance=False,
                )
                if completed_process.returncode != 0:
                    error = completed_process.stderr.strip() or completed_process.stdout.strip()
                    rows.extend(
                        result_rows_for_failure(
                            galaxy, point, sfr, sources, "model_failed", error[-1000:]
                        )
                    )
                else:
                    rows.extend(
                        evaluate_model_output(
                            galaxy,
                            point,
                            sfr,
                            output_directory,
                            sources,
                            observations,
                        )
                    )
        except subprocess.TimeoutExpired:
            rows.extend(
                result_rows_for_failure(
                    galaxy,
                    point,
                    sfr,
                    sources,
                    "model_timeout",
                    f"Exceeded {timeout_seconds:g} seconds",
                )
            )
        except Exception as error:  # Keep the rest of a scientific sweep usable.
            rows.extend(
                result_rows_for_failure(
                    galaxy, point, sfr, sources, "evaluation_failed", str(error)[-1000:]
                )
            )

        if progress_every > 0 and (completed % progress_every == 0 or completed == executable_points):
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            remaining = (executable_points - completed) / rate if rate > 0.0 else math.nan
            print(
                f"{galaxy}: {completed}/{executable_points} model runs "
                f"({rate:.1f}/s, approximately {remaining:.0f}s remaining)",
                flush=True,
            )
    return rows, sources


def retain_best_outputs(
    output_root: Path,
    galaxy: str,
    ranked: Sequence[RankedPoint],
    keep_best: int,
    sfr: float,
    model_binary: Path,
    input_directory: Path,
    timeout_seconds: float,
    ranking_calibration: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for rank, ranked_point in enumerate(ranked[:keep_best], start=1):
        directory = output_root / "best" / galaxy / f"{rank:02d}_{ranked_point.point.point_id}"
        directory.mkdir(parents=True)
        completed_process = invoke_model(
            model_binary,
            input_directory,
            directory,
            ranked_point.point,
            sfr,
            timeout_seconds,
            include_provenance=True,
        )
        if completed_process.returncode != 0:
            raise RuntimeError(
                f"Failed to retain {galaxy} rank {rank}: {completed_process.stderr.strip()}"
            )
        point = ranked_point.point
        records.append(
            {
                "galaxy": galaxy,
                "rank": rank,
                "point_id": point.point_id,
                "mu": point.mu,
                "beta": point.beta,
                "f_land": point.f_land,
                "SFR_Msun_yr": sfr,
                "Mdot_land_input_Msun_yr": sfr if point.f_land is None else point.f_land * sfr,
                "ranking_calibration": ranking_calibration,
                "mean_reduced_chi2_fixed_model": ranked_point.score,
                "rotation_scores": json.dumps(ranked_point.source_scores, sort_keys=True),
                "output_directory": str(directory.relative_to(output_root)),
            }
        )
    return records


def percentile_limits(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(values)
    if not ordered:
        return (0.0, 1.0)
    lower = ordered[int(0.05 * (len(ordered) - 1))]
    upper = ordered[int(0.95 * (len(ordered) - 1))]
    if lower == upper:
        upper = lower + 1.0
    return lower, upper


def plot_sweep_maps(
    path: Path,
    galaxies: Sequence[str],
    points: Sequence[SweepPoint],
    rankings: dict[str, Sequence[RankedPoint]],
    mu_values: Sequence[float],
    beta_values: Sequence[float],
    f_land_values: Sequence[float],
) -> None:
    matplotlib_cache: tempfile.TemporaryDirectory[str] | None = None
    if "MPLCONFIGDIR" not in os.environ:
        matplotlib_cache = tempfile.TemporaryDirectory(prefix="gnf-matplotlib-")
        os.environ["MPLCONFIGDIR"] = matplotlib_cache.name
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from hershey_fonts import register_hershey_weight_aliases

    register_hershey_weight_aliases()
    import smplotlib

    smplotlib.set_style(
        usetex=False,
        fontsize=10,
        figsize=(3.5, 3.5),
        dpi=144,
    )
    plt.rcParams.update(
        {
            "axes.grid": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": False,
            "ytick.right": False,
        }
    )

    score_cmap = plt.get_cmap("RdYlBu").copy()
    score_cmap.set_bad("white")
    positive_mu = [value for value in mu_values if value > 0.0]
    point_by_coordinate = {
        (point.mu, point.beta, point.f_land): point
        for point in points
        if point.mu > 0.0
    }

    with PdfPages(path) as pdf:
        for galaxy in galaxies:
            score_by_id = {item.point.point_id: item.score for item in rankings[galaxy]}
            log_scores = [math.log10(max(value, 1.0e-300)) for value in score_by_id.values()]
            vmin, vmax = percentile_limits(log_scores)

            figure, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
            image = None
            for axis, f_land in zip(axes.flat, f_land_values):
                matrix = []
                for mu in positive_mu:
                    matrix.append(
                        [
                            math.log10(max(score_by_id[point_by_coordinate[(mu, beta, f_land)].point_id], 1.0e-300))
                            if point_by_coordinate[(mu, beta, f_land)].point_id in score_by_id
                            else math.nan
                            for beta in beta_values
                        ]
                    )
                image = axis.imshow(
                    matrix,
                    origin="lower",
                    aspect="auto",
                    cmap=score_cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                axis.set_title(rf"$f_{{\rm land}}={f_land:.3g}$")
                axis.set_xticks(range(0, len(beta_values), 3), [f"{beta_values[i]:.2g}" for i in range(0, len(beta_values), 3)])
                axis.set_yticks(range(len(positive_mu)), [f"{value:g}" for value in positive_mu])
                axis.set_xlabel(r"$\beta$")
                axis.set_ylabel(r"$\mu$")
            figure.suptitle(galaxy)
            if image is not None:
                figure.colorbar(
                    image,
                    ax=axes,
                    shrink=0.75,
                    label=r"$\log_{10}(\widetilde{\chi^2_\nu})$",
                )
            pdf.savefig(figure)
            plt.close(figure)

            best_score_matrix: list[list[float]] = []
            best_fland_matrix: list[list[float]] = []
            invalid_matrix: list[list[float]] = []
            for mu in positive_mu:
                score_row: list[float] = []
                f_row: list[float] = []
                invalid_row: list[float] = []
                for beta in beta_values:
                    candidates = [
                        (score_by_id[point_by_coordinate[(mu, beta, f_land)].point_id], f_land)
                        for f_land in f_land_values
                        if point_by_coordinate[(mu, beta, f_land)].point_id in score_by_id
                    ]
                    invalid_row.append(float(len(f_land_values) - len(candidates)))
                    if candidates:
                        score, best_f = min(candidates)
                        score_row.append(math.log10(max(score, 1.0e-300)))
                        f_row.append(best_f)
                    else:
                        score_row.append(math.nan)
                        f_row.append(math.nan)
                best_score_matrix.append(score_row)
                best_fland_matrix.append(f_row)
                invalid_matrix.append(invalid_row)

            figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
            panels = (
                (
                    best_score_matrix,
                    "Best absolute-fit score over $f_{land}$",
                    r"$\log_{10}(\widetilde{\chi^2_\nu})$",
                    vmin,
                    vmax,
                    score_cmap,
                ),
                (best_fland_matrix, "$f_{land}$ giving the best score", r"$f_{land}$", None, None, None),
                (
                    invalid_matrix,
                    "Invalid landing fractions",
                    "count",
                    0.0,
                    float(len(f_land_values)),
                    None,
                ),
            )
            for axis, (matrix, title, color_label, panel_min, panel_max, cmap) in zip(axes, panels):
                image = axis.imshow(
                    matrix,
                    origin="lower",
                    aspect="auto",
                    cmap=cmap,
                    vmin=panel_min,
                    vmax=panel_max,
                )
                axis.set_title(title)
                axis.set_xticks(range(0, len(beta_values), 3), [f"{beta_values[i]:.2g}" for i in range(0, len(beta_values), 3)])
                axis.set_yticks(range(len(positive_mu)), [f"{value:g}" for value in positive_mu])
                axis.set_xlabel(r"$\beta$")
                axis.set_ylabel(r"$\mu$")
                figure.colorbar(image, ax=axis, shrink=0.8, label=color_label)
            figure.suptitle(galaxy)
            pdf.savefig(figure)
            plt.close(figure)
    if matplotlib_cache is not None:
        matplotlib_cache.cleanup()


def parse_mu_values(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one mu value is required")
    try:
        return [float(value) for value in values]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a compact staged sweep around the C++ forward model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--galaxy", action="append", dest="galaxies", help="Galaxy to sweep; repeat for multiple galaxies")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument(
        "--mu-values",
        type=parse_mu_values,
        default=list(DEFAULT_MU_VALUES),
        help="Comma-separated mu values; mu=0 is run once",
    )
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--beta-count", type=int, default=20)
    parser.add_argument("--fland-min", type=float, default=0.1)
    parser.add_argument("--fland-max", type=float, default=5.0)
    parser.add_argument("--fland-count", type=int, default=12)
    parser.add_argument("--keep-best", type=int, default=10)
    parser.add_argument("--ranking-calibration", choices=CALIBRATIONS, default="KK04")
    parser.add_argument(
        "--model-binary",
        type=Path,
        default=repo_root / "build" / "galactic-nuclear-fountain-model",
    )
    parser.add_argument("--input-root", type=Path, default=repo_root / "input" / "model_inputs")
    parser.add_argument(
        "--galaxy-catalog",
        type=Path,
        default=repo_root / "input" / "galaxies_triple_overlap.csv",
    )
    parser.add_argument(
        "--metallicity-catalog",
        type=Path,
        default=repo_root / "input" / "sings_hii_regions.csv",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="Maximum seconds for one model run")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--skip-plots", action="store_true", help="Skip PDF generation (useful for automated tests)")
    return parser


def validate_paths(arguments: argparse.Namespace, galaxies: Sequence[str]) -> None:
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {arguments.output}")
    if not arguments.model_binary.is_file():
        raise FileNotFoundError(
            f"Model executable is missing: {arguments.model_binary}. Run `make` first."
        )
    if not arguments.galaxy_catalog.is_file():
        raise FileNotFoundError(arguments.galaxy_catalog)
    if not arguments.metallicity_catalog.is_file():
        raise FileNotFoundError(arguments.metallicity_catalog)
    for galaxy in galaxies:
        directory = arguments.input_root / galaxy
        for name in (
            "parameters.csv",
            "profiles.csv",
            "gas_profiles.csv",
            "rotation_curves.csv",
        ):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
        if any(
            row["kind"] == "tabulated"
            for row in read_csv(directory / "rotation_curves.csv")
        ):
            rotation_profile_path = directory / "rotation_profiles.csv"
            if not rotation_profile_path.is_file():
                raise FileNotFoundError(rotation_profile_path)
            parameters = {
                row["name"]: row["value"]
                for row in read_csv(directory / "parameters.csv")
            }
            r_nucl = float(parameters["R_nucl_kpc"])
            r_out = float(parameters["R_out_kpc"])
            for row in read_csv(rotation_profile_path):
                radius = float(row["R_kpc"])
                if not r_nucl <= radius <= r_out:
                    continue
                velocity = float(row["V_kms"])
                velocity_derivative = float(row["dV_dR_kms_kpc"])
                angular_momentum_derivative = velocity + radius * velocity_derivative
                if not math.isfinite(angular_momentum_derivative) or angular_momentum_derivative <= 0.0:
                    raise ValueError(
                        f"{rotation_profile_path}: d(R*v_c)/dR must be positive "
                        f"through the model domain; got {angular_momentum_derivative:.6g} "
                        f"at R={radius:.6g} kpc"
                    )
    if arguments.keep_best < 0:
        raise ValueError("--keep-best must be nonnegative")
    if arguments.timeout <= 0.0:
        raise ValueError("--timeout must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = build_parser(repo_root)
    arguments = parser.parse_args(argv)
    galaxies = list(
        dict.fromkeys(
            canonical_galaxy_name(galaxy)
            for galaxy in (arguments.galaxies or DEFAULT_GALAXIES)
        )
    )

    try:
        if not arguments.skip_plots:
            excluded = [galaxy for galaxy in galaxies if galaxy not in PAPER_GALAXY_SET]
            if excluded:
                allowed = ", ".join(galaxy_label(galaxy) for galaxy in PAPER_GALAXIES)
                raise ValueError(
                    "Sweep plots are restricted to the paper sample "
                    f"({allowed}); got {', '.join(excluded)}. Use --skip-plots "
                    "to run a non-paper galaxy without creating figures."
                )
        validate_paths(arguments, galaxies)
        beta_values = linear_space(arguments.beta_min, arguments.beta_max, arguments.beta_count)
        f_land_values = logarithmic_space(arguments.fland_min, arguments.fland_max, arguments.fland_count)
        mu_values = list(dict.fromkeys(arguments.mu_values))
        points = build_grid(mu_values, beta_values, f_land_values)
        sfrs = adopted_sfrs(arguments.galaxy_catalog, galaxies)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    arguments.output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    all_sources: dict[str, list[str]] = {}
    rankings: dict[str, list[RankedPoint]] = {}
    best_rows: list[dict[str, object]] = []
    started = time.monotonic()

    print(
        f"Sweep grid: {len(points)} canonical points per galaxy; "
        f"{sum(point.planned_status == 'pending' for point in points)} model runs per galaxy",
        flush=True,
    )
    for galaxy in galaxies:
        input_directory = arguments.input_root / galaxy
        print(f"Starting {galaxy} with SFR={sfrs[galaxy]:.6g} Msun/yr", flush=True)
        rows, sources = run_galaxy_sweep(
            galaxy,
            points,
            sfrs[galaxy],
            arguments.model_binary,
            input_directory,
            arguments.metallicity_catalog,
            arguments.timeout,
            arguments.progress_every,
        )
        all_rows.extend(rows)
        all_sources[galaxy] = sources
        rankings[galaxy] = rank_points(
            rows, points, galaxy, arguments.ranking_calibration, sources
        )
        print(
            f"{galaxy}: {len(rankings[galaxy])} independently valid, rankable points",
            flush=True,
        )

    write_csv(arguments.output / "sweep_results.csv", RESULT_FIELDS, all_rows)

    for galaxy in galaxies:
        best_rows.extend(
            retain_best_outputs(
                arguments.output,
                galaxy,
                rankings[galaxy],
                arguments.keep_best,
                sfrs[galaxy],
                arguments.model_binary,
                arguments.input_root / galaxy,
                arguments.timeout,
                arguments.ranking_calibration,
            )
        )
    write_csv(arguments.output / "best_points.csv", BEST_FIELDS, best_rows)

    if not arguments.skip_plots:
        plot_sweep_maps(
            arguments.output / "sweep_maps.pdf",
            galaxies,
            points,
            rankings,
            mu_values,
            beta_values,
            f_land_values,
        )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
        "model_binary": str(arguments.model_binary.resolve()),
        "galaxies": galaxies,
        "sfr_Msun_yr": sfrs,
        "grid": {
            "mu": mu_values,
            "beta": beta_values,
            "f_land": f_land_values,
            "raw_cartesian_points_per_galaxy": len(mu_values) * len(beta_values) * len(f_land_values),
            "canonical_points_per_galaxy": len(points),
            "executed_points_per_galaxy": sum(point.planned_status == "pending" for point in points),
            "singular_boundary_points_per_galaxy": sum(point.planned_status == "singular_boundary" for point in points),
            "mu_zero_policy": "one run; beta and requested f_land are not independent in the nonmixing limit",
            "beta_one_policy": "record singular_boundary without executing for mu>0",
        },
        "ranking": {
            "calibration": arguments.ranking_calibration,
            "statistic": "mean reduced_chi2_fixed_model across every rotation source",
            "requires": [
                "finite dynamics",
                "v_R < 0 away from the mu=0 nuclear endpoint",
                "nonnegative landing profile",
                f"target landing relative error <= {TARGET_LANDING_RTOL:g} when mu>0",
                "solved metallicity profile",
                "at least one overlapping metallicity observation",
            ],
            "keep_best_per_galaxy": arguments.keep_best,
        },
        "comparison_constants": {
            "OH12_sun": OH12_SUN,
            "Z_sun": Z_SUN,
            "abundance_scatter_dex": ABUNDANCE_SCATTER_DEX,
        },
        "elapsed_seconds": time.monotonic() - started,
        "outputs": {
            "sweep_results": "sweep_results.csv",
            "best_points": "best_points.csv",
            "sweep_maps": None if arguments.skip_plots else "sweep_maps.pdf",
            "best_profiles": "best/<galaxy>/<rank>_<point_id>/",
        },
    }
    with (arguments.output / "run_manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    print(
        f"Completed sweep in {manifest['elapsed_seconds']:.1f}s; outputs: {arguments.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
