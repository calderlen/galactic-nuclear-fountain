#!/usr/bin/env python3
"""Prepare empirical disk inputs and run per-galaxy model-comparison cases.

Physics and optional regular-unmixed reference calculations use the C++ core.
The Julia observation exporter owns gas/SFR/rotation preprocessing. This script
selects cases, sends oxygen abundances to the C++ comparison code, and records their provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "plot"))
from paper_galaxies import PAPER_GALAXIES  # noqa: E402
from surface_rate_units import read_rate_rows
from core_calculations import collapse_abundances, fit_abundances

CASES = {"unmixed": {"mu": 0.0}, "fiducial": {}, "mu0p5": {"mu": 0.5}, "mu2": {"mu": 2.0}, "beta0p5": {"beta": 0.5}}

# A CLI adapter around existing profile loading and regular unmixed physics.
# No transport or metallicity equation is reimplemented here.
NORMALIZATION_ADAPTER = r'''#include "csv.h"
#include "models.h"
#include "profiles.h"
#include "units.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
namespace csv = galactic_nuclear_fountain::csv;
LinearProfile load(const std::filesystem::path& path,const std::string& column) {
    LinearProfile p;
    for (const auto& row:csv::read(path).rows) {
        if (path.filename()=="rotation_profiles.csv" && csv::value(row,"source")!="SPARC_spline") continue;
        p.radius.push_back(csv::number(row,"R_kpc"));
        p.value.push_back(csv::number(row,column));
    }
    return p;
}
int main(int argc,char** argv) {
    try {
        if (argc!=2 && !(argc==3 && std::string(argv[2])=="--check-forward"))
            throw std::runtime_error("Expected an empirical input directory and optional --check-forward");
        const std::filesystem::path dir(argv[1]);
        std::map<std::string,double> settings;
        for (const auto& row:csv::read(dir/"parameters.csv").rows) {
            try { settings[csv::value(row,"name")]=csv::number(row,"value"); }
            catch (const std::exception&) {}
        }
        auto gas=load(dir/"gas_profiles.csv","Sigma_g_Msun_kpc2");
        auto dg=load(dir/"gas_profiles.csv","dSigma_g_dR_Msun_kpc3");
        auto star=load(dir/"profiles.csv","Sigmadot_star_Msun_Gyr_kpc2");
        auto vc=load(dir/"rotation_profiles.csv","V_kms");
        auto dvc=load(dir/"rotation_profiles.csv","dV_dR_kms_kpc");
        const RotationCurve rot{{linear_profile_value,&vc},{linear_profile_value,&dvc}};
        const double rn=settings.at("R_nucl_kpc"),ri=settings.at("R_in_kpc");
        // The exporter includes the exact launch point in its spline samples.
        const double launch=settings.at("v_c_nucl_kms");
        if (std::abs(launch-linear_profile_value(rn,&vc))>1e-12*std::abs(launch))
            throw std::runtime_error("Exported rotation curve must include the exact launch point");
        if (argc==3) {
            const double ro=settings.at("R_out_kpc");
            const ForwardModelParameters model{rn,ro,settings.at("mu"),settings.at("beta"),
                {linear_profile_value,&gas},{linear_profile_value,&dg},
                {linear_profile_value,&star},rot,ri,launch};
            const auto solution=solve_forward_model(model,settings.at("Mdot_land_Msun_yr"),
                settings.at("h_0_kpc"),settings.at("dynamics_atol"),settings.at("rtol"));
            double minimum=std::numeric_limits<double>::infinity(),maximum_absolute=0.0;
            double minimum_radius=ri,minimum_velocity=minimum,maximum_velocity=-minimum;
            const auto inspect=[&](double R) {
                const double landing=forward_landing_rate(R,model,solution);
                const double velocity=forward_radial_velocity(R,model,solution);
                if (!std::isfinite(landing) || !std::isfinite(velocity))
                    throw std::runtime_error("Nonfinite forward dynamics in landing preflight");
                if (landing<minimum) {minimum=landing;minimum_radius=R;}
                maximum_absolute=std::max(maximum_absolute,std::abs(landing));
                minimum_velocity=std::min(minimum_velocity,velocity);
                maximum_velocity=std::max(maximum_velocity,velocity);
            };
            constexpr std::size_t dense_count=20001;
            const std::size_t output_count=static_cast<std::size_t>(settings.at("n_points"));
            for (std::size_t i=0;i<dense_count;++i)
                inspect(ri+(ro-ri)*static_cast<double>(i)/static_cast<double>(dense_count-1));
            for (std::size_t i=0;i<output_count;++i)
                inspect(ri+(ro-ri)*static_cast<double>(i)/static_cast<double>(output_count-1));
            // Match model_runner.cpp::landing_nonnegative, in canonical units.
            const double tolerance=1e-10*std::max(maximum_absolute,units::per_gyr(1.0));
            std::cout << std::setprecision(17) << minimum << ',' << maximum_absolute << ','
                << tolerance << ',' << minimum_radius << ',' << minimum_velocity << ','
                << maximum_velocity << ',' << forward_accretion_rate(ro,model,solution) << ','
                << dense_count+output_count << ',' << (minimum>=-tolerance ? 1 : 0) << '\n';
            return 0;
        }
        const ForwardModelParameters model{rn,settings.at("R_out_kpc"),0.0,
            settings.at("beta"),{linear_profile_value,&gas},{linear_profile_value,&dg},
            {linear_profile_value,&star},rot,rn,launch};
        const auto s=solve_forward_nonmixing(model,settings.at("h_0_kpc"),
            settings.at("dynamics_atol"),settings.at("rtol"));
        const double excluded=forward_cumulative_landing(ri,s);
        std::cout << std::setprecision(17)
            << (s.Mdot_land-excluded)/settings.at("eta") << ','
            << s.Mdot_land << ',' << excluded << ',' << launch << '\n';
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
'''


def read_csv(path: Path) -> list[dict[str, str]]:
    return read_rate_rows(path)


def write_csv(path: Path, rows: list[dict], columns=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parameters(path: Path) -> dict[str, str]:
    return {row["name"]: row["value"] for row in read_csv(path)}


def change_parameters(path: Path, updates: dict) -> None:
    current = parameters(path)
    current.update({key: format(value, ".17g") if isinstance(value, float)
                    else str(value) for key, value in updates.items()})
    write_csv(path, [{"name": key, "value": value} for key, value in current.items()])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], log: Path, commands: list) -> str:
    commands.append(command)
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log}: "
                           + (completed.stderr or completed.stdout)[-1000:])
    return completed.stdout


def export_disks(output: Path, configs: dict[str, Path], commands: list) -> None:
    for galaxy, config in configs.items():
        destination = output / "inputs" / galaxy / "observed_disk"
        # The exporter validates the galaxy against the config's parent directory.
        galaxy_config = output / "configs" / galaxy / "model_cases.csv"
        galaxy_config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config, galaxy_config)
        copy_case_tables(config.parent, galaxy_config.parent)
        print(f"Exporting fresh measured disk: {galaxy}", flush=True)
        run(["julia", f"--project={ROOT}", str(ROOT / "scripts/data/export_model_input.jl"),
             galaxy, str(destination), str(galaxy_config)],
            output / "logs" / f"export_{galaxy}.log", commands)


def copy_case_tables(config_directory: Path, destination: Path) -> None:
    """Save the exact case selections beside the copied galaxy configuration."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("section3_predictive_perturbations.csv", "galaxy_comparison_loading_cases.csv"):
        source = config_directory / name
        if source.exists():
            shutil.copyfile(source, destination / name)


def observed_abundances(output: Path, settings: dict[str, dict[str, str]]) -> list[dict]:
    selected = {galaxy: [] for galaxy in settings}
    for row in read_csv(ROOT / "input/sings_hii_regions.csv"):
        if row["galaxy"] not in settings or row["usable_OH12_KK04"].lower() != "true":
            continue
        if not all(row.get(name) for name in ("R_adopted_kpc", "OH12_KK04_dex", "e_OH12_KK04_dex")):
            continue
        selected[row["galaxy"]].append({"source_seq": row["source_seq"], "offRA_arcsec": row["offRA_arcsec"],
            "offDE_arcsec": row["offDE_arcsec"], "R_kpc": row["R_adopted_kpc"],
            "OH12": row["OH12_KK04_dex"], "e_OH12": row["e_OH12_KK04_dex"]})
    observations = []
    fields = ("R_kpc", "OH12", "e_OH12", "sigma_OH12_dex", "Z", "e_Z_low", "e_Z_high",
              "source_sequences", "n_catalog_rows")
    for galaxy, rows in selected.items():
        for row in collapse_abundances(rows, scatter=float(settings[galaxy]["abundance_scatter_dex"])):
            observations.append({"galaxy": galaxy, "calibration": "KK04", **{key: row[key] for key in fields}})
    observations.sort(key=lambda row: (row["galaxy"], row["R_kpc"]))
    write_csv(output / "observations.csv", observations)

    fits = []
    for galaxy, galaxy_settings in settings.items():
        scatter = float(galaxy_settings["abundance_scatter_dex"])
        directory = output / "inputs" / galaxy / "observed_disk"
        p = parameters(directory / "parameters.csv")
        rows = [row for row in observations if row["galaxy"] == galaxy
                and float(p["R_in_kpc"]) <= row["R_kpc"] <= float(p["R_out_kpc"])]
        if len(rows) < 2:
            raise ValueError(f"Fewer than two unique supported KK04 regions for {galaxy}")
        result, profile = fit_abundances([(row["R_kpc"], row["OH12"], row["e_OH12"]) for row in rows],
            sample_count=int(galaxy_settings["metallicity_profile_points"]), scatter=scatter)
        fit = {"galaxy": galaxy, "calibration": "KK04", "n_unique_regions": len(rows),
            **{key: result[key] for key in ("R_min_kpc", "R_max_kpc", "R_pivot_kpc", "OH12_at_pivot",
                                          "e_OH12_at_pivot", "OH12_slope_dex_per_kpc", "e_slope_dex_per_kpc")},
            "abundance_scatter_dex": scatter, "fit_chi2": result["fit_chi2"], "fit_dof": result["fit_dof"],
            "gradient_sample_reliable": len(rows) >= 8 and result["R_max_kpc"]-result["R_min_kpc"] >= 2,
            "method": "weighted linear OH12; duplicate sky positions combined; model-domain points only"}
        fits.append(fit)
        write_csv(directory / "metallicity_fit.csv", [fit])
        write_csv(directory / "metallicity_profiles.csv", profile)
    write_csv(output / "abundance_fits.csv", fits)
    return fits


def normalization_adapter(output: Path, commands: list) -> Path:
    directory = output / "tools"
    directory.mkdir(parents=True, exist_ok=True)
    source, binary = directory / "regular_unmixed_source.cpp", directory / "regular_unmixed_source"
    source.write_text(NORMALIZATION_ADAPTER)
    run([*shlex.split(os.environ.get("CXX", "c++")), "-std=c++23", "-O2",
         f"-I{ROOT / 'model'}", f"-I{ROOT / 'io'}", str(source),
         str(ROOT / "build/libgalactic-nuclear-fountain.a"), str(ROOT / "build/io/csv.o"),
         "-o", str(binary)], output / "logs/compile_normalization_adapter.log", commands)
    return binary


def predictive_perturbations(galaxy: str, config_root: Path = ROOT / "config/galaxies") -> list[dict[str, str]]:
    path = config_root / galaxy / "section3_predictive_perturbations.csv"
    return read_csv(path) if path.exists() else []


def galaxy_comparison_loading_cases(galaxy: str, config_root: Path = ROOT / "config/galaxies") -> list[dict[str, str]]:
    """Explicit loading cases for the six-galaxy comparison, at fixed source."""
    path = config_root / galaxy / "galaxy_comparison_loading_cases.csv"
    return read_csv(path) if path.exists() else []


def forward_dynamics_check(helper: Path, destination: Path, output: Path,
                           galaxy: str, case: str, commands: list) -> dict:
    """Ask the existing C++ dynamics to test landing before solving chemistry."""
    response = run([str(helper), str(destination), "--check-forward"],
                   output / "logs" / f"{galaxy}_{case}_dynamics.log", commands)
    names = ("minimum_landing_Msun_Gyr_kpc2", "maximum_absolute_landing_Msun_Gyr_kpc2",
             "landing_nonnegative_tolerance_Msun_Gyr_kpc2", "minimum_landing_radius_kpc",
             "minimum_v_R_kpc_yr", "maximum_v_R_kpc_yr", "Mdot_out_Msun_yr",
             "dynamics_sample_points", "landing_nonnegative")
    diagnostic = dict(zip(names, map(float, response.strip().split(",")), strict=True))
    diagnostic["dynamics_sample_points"] = int(diagnostic["dynamics_sample_points"])
    diagnostic["landing_nonnegative"] = bool(diagnostic["landing_nonnegative"])
    return diagnostic


def resolve_nuclear_source(output: Path, galaxy: str, settings: dict[str, str],
                           commands: list, helper: Path | None = None) -> Path | None:
    """Use the supplied nuclear rate by default and preserve its provenance."""
    base = output / "inputs" / galaxy / "observed_disk"
    p = parameters(base / "parameters.csv")
    mode = settings.get("source_normalization", "configured")
    updates = {"source_normalization": mode}
    if mode == "configured":
        source = float(settings["Mdot_nuc_Msun_yr"])
        updates["nuclear_SFR_origin"] = settings.get("nuclear_SFR_origin", "supplied_config")
        # A resumed export may still contain diagnostics from its former mode.
        updates.update({key: "" for key in (
            "regular_unmixed_full_landing_Msun_yr", "excluded_inner_landing_Msun_yr") if key in p})
    elif mode == "regular_unmixed":
        if helper is None:
            helper = normalization_adapter(output, commands)
        response = run([str(helper), str(base)], output / "logs" / f"normalize_{galaxy}.log", commands)
        source, full, excluded, launch_velocity = map(float, response.strip().split(","))
        updates.update({
            "nuclear_SFR_origin": "inferred_regular_unmixed_annular_normalization_not_measured",
            "regular_unmixed_full_landing_Msun_yr": full,
            "excluded_inner_landing_Msun_yr": excluded,
            "v_c_nucl_kms": launch_velocity,
        })
    else:
        raise ValueError(f"Unknown source_normalization={mode!r} for {galaxy}")
    if not math.isfinite(source) or source <= 0:
        raise ValueError(f"Nonpositive or nonfinite nuclear source for {galaxy}")
    updates.update({
        "Mdot_nuc_Msun_yr": source,
        "Mdot_land_Msun_yr": float(p["eta"]) * (1 + float(p["mu"])) * source,
        "launch_velocity_origin": "observed_rotation_spline_at_exact_exported_launch_point",
    })
    change_parameters(base / "parameters.csv", updates)
    metadata_path = base / "metadata.csv"
    metadata = read_csv(metadata_path)
    metadata[0].update({key: format(value, ".17g") if isinstance(value, float) else value
                        for key, value in updates.items()})
    write_csv(metadata_path, metadata)
    return helper


def run_forward_cases(output: Path, runner: Path, galaxy: str, base: Path,
                      definitions: dict, commands: list, *,
                      allow_invalid: bool = False, case_group: str = "baseline",
                      dynamics_helper: Path | None = None) -> list[dict]:
    """Vary explicit case parameters on a fixed observed disk.

    Broad loading trials can violate the steady solution's requirements. They
    retain a status record without stopping the other comparison cases; the
    existing baseline/predictive runs keep their strict failure behavior.
    """
    p = parameters(base / "parameters.csv")
    records = []
    for case, overrides in definitions.items():
        selected = {key: float(p[key]) for key in
                    ("Mdot_nuc_Msun_yr", "mu", "beta", "eta", "yield_y", "Z_CGM", "Z_nucl", "Z_outer_boundary")}
        selected.update(overrides)
        source = selected["Mdot_nuc_Msun_yr"]
        destination = output / "inputs" / galaxy / case
        shutil.copytree(base, destination, dirs_exist_ok=True)
        case_parameters = {
            **selected, "Mdot_land_Msun_yr": selected["eta"] * (1.0 + selected["mu"]) * source,
            "section3_case": case, "metallicity_mode": "predicted",
            "nuclear_SFR_origin": "source_sensitivity_case" if "Mdot_nuc_Msun_yr" in overrides
                                  else p.get("nuclear_SFR_origin", "supplied_config"),
            "nuclear_SFR_reference_origin": p.get("nuclear_SFR_origin", "supplied_config"),
        }
        change_parameters(destination / "parameters.csv", case_parameters)
        metadata_path = destination / "metadata.csv"
        metadata = read_csv(metadata_path)
        metadata[0].update({key: format(case_parameters[key], ".17g")
                            if isinstance(case_parameters[key], float) else case_parameters[key]
                            for key in case_parameters})
        write_csv(metadata_path, metadata)
        model_output = output / galaxy / case
        print(f"Running {galaxy}/{case}: nuclear source {source:.6g} Msun/yr", flush=True)
        record = {"galaxy": galaxy, "case": case, "model": "forward",
                  "case_group": case_group, "status": "running", "status_reason": "",
                  **selected, "Mdot_land_Msun_yr": case_parameters["Mdot_land_Msun_yr"],
                  "Mdot_out_Msun_yr": "",
                  "source_normalization": p.get("source_normalization", "configured"),
                  "nuclear_SFR_origin": case_parameters["nuclear_SFR_origin"],
                  "regular_unmixed_full_landing_Msun_yr": p.get("regular_unmixed_full_landing_Msun_yr", ""),
                  "excluded_inner_landing_Msun_yr": p.get("excluded_inner_landing_Msun_yr", "")}
        model_output.mkdir(parents=True, exist_ok=True)
        status_path = model_output / "case_status.json"
        # Invalidate any previous outputs before invoking the solver, including
        # when rerunning into a directory containing an older successful case.
        status_path.write_text(json.dumps(record, indent=2) + "\n")
        try:
            if dynamics_helper is not None:
                record.update(forward_dynamics_check(dynamics_helper, destination, output, galaxy, case, commands))
                if not record["landing_nonnegative"]:
                    record["status"] = "invalid"
                    raise ValueError(
                        f"Negative landing in core dynamics before chemistry: minimum "
                        f"{record['minimum_landing_Msun_Gyr_kpc2']:.9g} Msun/Gyr/kpc^2 at "
                        f"R={record['minimum_landing_radius_kpc']:.9g} kpc; tolerance "
                        f"{record['landing_nonnegative_tolerance_Msun_Gyr_kpc2']:.9g} Msun/Gyr/kpc^2")
            run([str(runner), "forward", str(destination), str(model_output)],
                output / "logs" / f"{galaxy}_{case}.log", commands)
            summary = next(row for row in read_csv(model_output / "summary.csv") if row["source"] == "SPARC_spline")
            if allow_invalid and not (model_output / "profiles.csv").is_file():
                raise RuntimeError(f"Missing model profiles for {galaxy}/{case}")
            record.update({"Mdot_land_Msun_yr": float(summary["Mdot_land_Msun_yr"]),
                           "Mdot_out_Msun_yr": float(summary["Mdot_out_Msun_yr"])})
            if summary.get("landing_nonnegative") != "true" or summary.get("metallicity_solved") not in ("1", "true"):
                record["status"] = "invalid"
                raise ValueError(f"Invalid prediction for {galaxy}/{case}: {summary}")
            record["status"] = "valid"
        except (RuntimeError, ValueError, OSError, StopIteration) as error:
            if record["status"] != "invalid":
                record["status"] = "failed"
            record["status_reason"] = str(error)
            status_path.write_text(json.dumps(record, indent=2) + "\n")
            if not allow_invalid:
                raise
            print(f"Excluded {galaxy}/{case}: {record['status_reason']}", flush=True)
        else:
            status_path.write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
    return records


def run_models(output: Path, runner: Path, commands: list, galaxies: list[str],
               settings: dict[str, dict[str, str]], config_root: Path = ROOT / "config/galaxies", *,
               comparison_only: bool = False, record_sink: list | None = None) -> list[dict]:
    helper = None
    records = record_sink if record_sink is not None else []
    for galaxy in galaxies:
        base = output / "inputs" / galaxy / "observed_disk"
        copy_case_tables(config_root / galaxy, output / "configs" / galaxy)
        helper = resolve_nuclear_source(output, galaxy, settings[galaxy], commands, helper)
        if helper is None and (comparison_only or galaxy_comparison_loading_cases(galaxy, config_root)):
            helper = normalization_adapter(output, commands)
        definitions = dict(CASES)
        if not comparison_only:
            definitions.update({row["case"]: {row["parameter"]: float(row["value"])}
                                for row in predictive_perturbations(galaxy, config_root)})
        records.extend(run_forward_cases(output, runner, galaxy, base, definitions, commands,
                                         allow_invalid=comparison_only,
                                         dynamics_helper=helper if comparison_only else None))
        loading_definitions = {row["case"]: {row["parameter"]: float(row["value"])}
                               for row in galaxy_comparison_loading_cases(galaxy, config_root)}
        records.extend(run_forward_cases(output, runner, galaxy, base, loading_definitions,
                                         commands, allow_invalid=True,
                                         case_group="galaxy_comparison_loading", dynamics_helper=helper))

    if "NGC2403" in galaxies and not comparison_only:
        galaxy, case = "NGC2403", "required_entrainment"
        base = output / "inputs" / galaxy / "fiducial"
        destination = output / "inputs" / galaxy / case
        shutil.copytree(base, destination, dirs_exist_ok=True)
        p = parameters(base / "parameters.csv")
        summary = read_csv(output / galaxy / "fiducial" / "summary.csv")[0]
        change_parameters(destination / "parameters.csv", {
            "metallicity_mode": "observed", "Mdot_out_Msun_yr": float(summary["Mdot_out_Msun_yr"]),
            "section3_case": case,
            "outer_mass_flux_origin": "fiducial_forward_model_not_observed",
            "trial_landing_origin": f"R^-{p['lambda']} at fiducial total landing rate; not fitted",
            "observed_metallicity_origin": "deduplicated_KK04_weighted_linear_fit_on_measured_support",
        })
        print(f"Running {galaxy}/{case}: independently supplied observed chemistry", flush=True)
        run([str(runner), "inverse", str(destination), str(output / galaxy / case)],
            output / "logs" / f"{galaxy}_{case}.log", commands)
        records.append({"galaxy": galaxy, "case": case, "model": "inverse",
                        "metallicity_mode": "observed",
                        "Mdot_out_Msun_yr": float(summary["Mdot_out_Msun_yr"])})
    write_csv(output / "model_cases.csv", [
        {key: row.get(key, "") for key in dict.fromkeys(key for item in records for key in item)}
        for row in records])
    return records


def write_provenance(output: Path, configs: dict[str, Path], settings: dict, commands: list,
                     fits: list, records: list, status: str, *, comparison_only: bool = False) -> None:
    sources = [ROOT / "input" / name for name in (
        "galaxies_triple_overlap.csv", "sparc_rotation_curves.csv", "leroy_radial_profiles.csv",
        "bigiel_radial_profiles.csv", "sings_hii_regions.csv")]
    sources += [*configs.values(), ROOT / "observations/observations.jl", ROOT / "scripts/data/export_model_input.jl",
                Path(__file__), ROOT / "scripts/plot/section3_mixing_figures.py",
                ROOT / "scripts/plot/core_calculations.py", ROOT / "observations/comparison.cpp",
                ROOT / "observations/comparison.h"]
    sources += [config.parent / "section3_predictive_perturbations.csv"
                for config in configs.values()
                if (config.parent / "section3_predictive_perturbations.csv").exists()]
    sources += [config.parent / "galaxy_comparison_loading_cases.csv"
                for config in configs.values()
                if (config.parent / "galaxy_comparison_loading_cases.csv").exists()]
    sources += [ROOT / "model" / name for name in (
        "model_runner.cpp", "calculation_runner.cpp", "models.cpp", "models.h", "physics.cpp", "physics.h",
        "profiles.cpp", "profiles.h", "rk4.cpp", "rk4.h")]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": status,
        "model_scope": "galaxy_comparison_only" if comparison_only else "all_section3_cases",
        "galaxies": list(configs), "settings": settings, "commands": commands,
        "configuration_files": {galaxy: str(path) for galaxy, path in configs.items()},
        "source_sha256": {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path):
                          sha256(path) for path in sources},
        "source_normalization": "Use each supplied Mdot_nuc with its stated observational or trial provenance. Total landing is eta*(1+mu)*Mdot_nuc; choose the inner accretion rate on R_in>R_nuc to meet that total, and calculate the outer accretion rate from the solution. The older regular_unmixed reference calculation requires explicit selection.",
        "controlled_comparison": "Observed disk and R_in/R_out fixed per galaxy. Base mu/beta cases hold source, eta and chemistry fixed; explicit perturbation tables can also vary the nuclear source. Galaxy-comparison loading tables vary eta at the once-resolved nuclear source and fiducial mu/beta, with independent per-case validity status. Total landing is eta*(1+mu)*Mdot_nuc.",
        "comparison_landing_preflight": "The existing C++ forward dynamics is solved independently before chemistry for comparison cases. Landing and radial velocity are sampled on 20001 radii plus the exact export grid. Negative landing uses model_runner.cpp's 1e-10*max(max_abs_landing,1e9) Msun/Gyr/kpc^2 tolerance and is recorded invalid without attempting chemistry. Genuine numerical failures remain failed/unresolved.",
        "disk_preprocessing": "Current Julia observation loader: adopted geometry; uncertainty weighted splines; Bigiel gas helium correction; Bigiel SFR cross calibration; net stellar sink=(1-return_fraction)*SFR.",
        "chemical_constraint": "KK04 observations deduplicated by sky position with inverse variance weights; linear oxygen abundance fit on measured radii within model domain; no extrapolation. Required entrainment inversion uses this observed fit, the configured trial power-law landing and fiducial model outer mass flux.",
        "scientific_status": "Configured nuclear sources retain their per-galaxy provenance: observation-based, provisional or illustrative. No optimization of model parameters. Imposed outer abundance agreement is not independent evidence.",
        "abundance_fits": fits, "model_runs": records,
    }
    runner = ROOT / "build/galactic-nuclear-fountain-model"
    if runner.exists():
        manifest["model_binary_sha256"] = sha256(runner)
    calculations = ROOT / "build/galactic-nuclear-fountain-calculations"
    if calculations.exists():
        manifest["calculation_binary_sha256"] = sha256(calculations)
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Section 3 figure inputs and model runs\n\n"
        "These are illustrative comparisons using refreshed measured disk profiles, not fitted nuclear fountains. "
        "Each galaxy's independent parameters are in its `model_cases.csv`; exact paths, settings, commands and hashes are in `run_manifest.json`.\n\n"
        "The Julia exporter recalculates gas, stellar sink and rotation profiles from the current observation loader. "
        "The stellar sink applies each galaxy's configured returned fraction once. The supplied nuclear rate is used by default. "
        "On R_in>R_nuc, the inner accretion rate is chosen so total landing equals eta*(1+mu)*Mdot_nuc; the outer accretion rate is an output. "
        "These runs do not impose zero accretion at R_nuc or require a regular extension to that radius. "
        "The older regular-unmixed reference calculation remains available only by explicit selection. Each adopted nuclear rate and its provenance are saved with the inputs. "
        "The exported rotation table includes the exact launch point and its observed rotation-spline circular speed. "
        "At fixed nuclear source and eta, total landing changes with (1+mu). Launch and receiving-disk radii come from each galaxy's configuration.\n\n"
        "Each selected galaxy has `unmixed`, `fiducial`, `mu0p5`, `mu2`, and `beta0p5` forward profiles. "
        "Those baseline cases hold the nuclear source and chemical assumptions fixed. Each galaxy's optional "
        "`section3_predictive_perturbations.csv` supplies additional one-parameter cases, including explicit nuclear-SFR variations. "
        "`galaxy_comparison_loading_cases.csv` supplies eta=0.1, 0.3, 3 and 10 trials, together with old-supply recovery diagnostics where specified. "
        "Eta=1 is the existing fiducial. Loading trials hold the supplied nuclear rate, mu, beta and chemistry fixed. "
        "Each loading trial records valid, invalid or failed status in model_cases.csv and its case_status.json; invalid/failed trials are excluded from plotted predictions. "
        "Observed disk profiles remain fixed.\n\n"
        "`--comparison-only` runs just the five baseline comparison cases plus the loading table, omitting the separate predictive-perturbation and inverse cases. "
        "In that mode baseline failures also receive independent status records so other galaxies and loading trials can finish. "
        "The existing C++ core first solves dynamics and samples landing on 20001 radii plus the output grid. Negative landing is recorded invalid before chemistry, with its minimum, location, tolerance and velocity range retained. "
        "A failed numerical solve is not a physical exclusion; an invalid case explicitly violates the recorded landing/chemistry requirements.\n\n"
        "`observations.csv` contains deduplicated KK04 H II-region measurements, and `abundance_fits.csv` records weighted straight-line oxygen fits. "
        "Repeated sky positions are combined before adding each galaxy's configured abundance scatter in quadrature for the fit. "
        "The fitted metal fraction is 0.0139 times 10^(OH12-8.69), with its analytical derivative. "
        "This assumes oxygen tracks total metallicity at fixed abundance pattern. Fits are restricted to measured radii inside the model domain. "
        "NGC 5055 has sparse abundance coverage and its gradient is descriptive.\n\n"
        "When NGC 2403 is selected, its `required_entrainment` case uses the observed abundance fit independently of a predictive chemical solution. "
        "Its trial landing uses the configured power-law slope at the fiducial total rate, and its outer mass flux comes from the fiducial forward model. "
        "It is a conditional consistency test: the outer mass flux and trial landing are model choices, not observations. "
        "Chemistry outside the measured fitted support is undefined and excluded from the figure.\n\n"
        "Reproduce model outputs and figures:\n\n```sh\nmake\npython3 scripts/section3_models.py\n"
        "conda run -n base env MPLCONFIGDIR=/private/tmp/gnf_section3_mpl python scripts/plot/section3_mixing_figures.py\n```\n\n"
        "To refresh observational inputs without running models, use `--prepare-only`; to run from already refreshed exports, use `--skip-export`.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, default=ROOT / "config/galaxies",
                        help="Root containing GALAXY/model_cases.csv for each selected galaxy")
    parser.add_argument("--galaxies", nargs="+", choices=PAPER_GALAXIES, default=list(PAPER_GALAXIES),
                        help="Galaxies to run (default: all six)")
    parser.add_argument("--output", type=Path, default=ROOT / "output/section3")
    parser.add_argument("--runner", type=Path, default=ROOT / "build/galactic-nuclear-fountain-model")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--comparison-only", action="store_true",
                        help="Run only the six-galaxy baseline/loading comparison; record case failures and continue")
    args = parser.parse_args()
    output, config_root = args.output.resolve(), args.config_root.resolve()
    configs = {galaxy: config_root / galaxy / "model_cases.csv" for galaxy in args.galaxies}
    output.mkdir(parents=True, exist_ok=True)
    settings, commands = {galaxy: parameters(config) for galaxy, config in configs.items()}, []
    existing_manifest = output / "run_manifest.json"
    if args.skip_export and existing_manifest.exists():
        # Retain the actual Julia acquisition commands when the model stage is
        # resumed after preparing inputs, without repeating prior model runs.
        previous = json.loads(existing_manifest.read_text())
        commands.extend(command for command in previous.get("commands", []) if command and command[0] == "julia")
    records = []
    fits = []
    try:
        if not args.skip_export:
            export_disks(output, configs, commands)
        fits = observed_abundances(output, settings)
        if not args.prepare_only:
            run_models(output, args.runner.resolve(), commands, list(configs), settings, config_root,
                       comparison_only=args.comparison_only, record_sink=records)
        write_provenance(output, configs, settings, commands, fits, records,
                         "prepared" if args.prepare_only else "model_runs_complete",
                         comparison_only=args.comparison_only)
    except Exception:
        write_provenance(output, configs, settings, commands, fits, records, "failed",
                         comparison_only=args.comparison_only)
        raise
    print(f"Section 3 inputs/results: {output}", flush=True)


if __name__ == "__main__":
    main()
