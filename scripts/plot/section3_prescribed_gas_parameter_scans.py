#!/usr/bin/env python3
"""Combined flow and chemistry supply scans with analytic gas and required j_land.

The C++ prescribed-gas module owns dynamics and optional incident-composition
chemistry. The gray unmixed comparison is a separate j_land=j_nuc solution.
Beta is an inactive comparison parameter for this closure, so those cases
coincide exactly with the fiducial. Saved gas/SFR spline fits are comparison overlays.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from color_palette import color_cycle

import section3_mixed_parameter_scans as figures

ROOT=Path(__file__).resolve().parents[2]
FLOW_TITLE=r"Mixed fountain, prescribed $\Sigma_g$"
SCAN_OVERRIDES=(*figures.OVERRIDES,"R_g_kpc")
sys.path.insert(0,str(ROOT/"scripts"))
from run_prescribed_gas import run_model


def calculate_cases(args, base, cases):
    if (base.get("closure")!="required_j" or base.get("supply_model")!="nuclear_plus_cgm"
            or base.get("metallicity_mode")!="prescribed_incident"):
        raise ValueError("Scans require explicit required_j, nuclear_plus_cgm supply and prescribed_incident chemistry")
    profiles, diagnostics, resolved = {}, {}, {}
    reference=figures.parameters(args.reference_config)
    if reference.get("closure")!="j_mix" or float(reference["mu"])!=0:
        raise ValueError("The unmixed comparison must be an explicit mu=0 angular-momentum-closed model")
    for key in ("R_in_kpc","R_out_kpc","lambda","Mdot_nuc_Msun_yr","eta","KS_A_SFR","KS_N",
                "return_fraction","Z_nucl","Z_CGM","yield_y","Z_outer_boundary","n_points"):
        if float(reference[key])!=float(base[key]):
            raise ValueError(f"Reference {key} differs from the fiducial; choose matching explicit configs")
    if base.get("rotation_model")!="flat" or float(base["v_c_kms"])!=float(reference["v_c_nucl_kms"]):
        raise ValueError("This matched unmixed comparison requires the same explicit flat rotation")
    for case in cases:
        name=case["case"]
        if not name or Path(name).name!=name or name in profiles:
            raise ValueError(f"Invalid or duplicate case name: {name}")
        if case["group"] not in (*figures.GROUPS,"gas_scale","all"):
            raise ValueError(f"Unknown scan group: {case['group']}")
        destination=args.run_directory/"models"/name
        if case["group"]=="reference":
            source=args.run_directory/"inputs"/name
            if args.plot_only:
                if figures.parameters(destination/"parameters.csv")!=reference:
                    raise ValueError("Unmixed reference changed; rerun without --plot-only")
            else:
                source.mkdir(parents=True,exist_ok=True)
                figures.write_parameters(source/"parameters.csv",reference)
                figures.run([str(figures.RUNNER),"mixed-power-law",str(source),str(destination)],
                            args.run_directory/"logs"/f"{name}.log")
            profile=figures.load_profile(destination/"profiles.csv")
            diagnostics[name]=figures.validate(profile,reference,destination)
            # Same displayed quantity, but it is imposed for this comparison.
            profile["j_land_required_kpc_kms"]=profile["j_land_kpc_kms"].copy()
            resolved[name]={"model":"unmixed angular-momentum closure","parameters":reference}
        else:
            overrides={key:case[key] for key in SCAN_OVERRIDES if key!="beta" and case.get(key)}
            expected={**base,**overrides}
            if args.plot_only:
                manifest=json.loads((destination/"run_manifest.json").read_text())
                if manifest["parameters"]!=expected:
                    raise ValueError(f"Saved parameters differ for {name}; rerun without --plot-only")
                rows=figures.read_csv(destination/"profiles.csv")
                profile={key:np.array([float(row[key]) for row in rows]) for key in rows[0] if key!="model"}
            else:
                profile,manifest=run_model(args.config,destination,overrides)
            diagnostics[name]=manifest["checks"]
            resolved[name]={"model":"prescribed-gas required_j","parameters":manifest["parameters"],
                            "Mdot_land_Msun_yr":manifest["Mdot_land_Msun_yr"],
                            "inactive_beta_comparison":case.get("beta") or None}
        profiles[name]=profile
        print(f"{'Loaded' if args.plot_only else 'Computed'} {name}",flush=True)
    if "fiducial" not in profiles:
        raise ValueError("Case table must contain fiducial")
    fiducial=profiles["fiducial"]
    for case in cases:
        p=profiles[case["case"]]
        np.testing.assert_array_equal(p["R_kpc"],fiducial["R_kpc"])
        if case["group"] not in ("reference","gas_scale"):
            for field in ("Sigma_g_Msun_kpc2","Sigmadot_star_Msun_Gyr_kpc2"):
                np.testing.assert_array_equal(p[field],fiducial[field])
        if case["group"]=="gas_scale":
            for field in ("Sigmadot_land_Msun_Gyr_kpc2","cumulative_landing_Msun_yr",
                          "v_c_kms","dv_c_dR_kms_kpc","j_disk_kpc_kms","Z_land"):
                np.testing.assert_array_equal(p[field],fiducial[field])
        if case["group"]=="cgm_rotation":
            for field in fiducial:
                np.testing.assert_array_equal(p[field],fiducial[field])
        if case["group"]=="loading_factor":
            scale=float(case["eta"])/float(base["eta"])
            np.testing.assert_allclose(p["Sigmadot_land_Msun_Gyr_kpc2"],
                                       scale*fiducial["Sigmadot_land_Msun_Gyr_kpc2"],rtol=1e-12)
            np.testing.assert_array_equal(p["Z_land"],fiducial["Z_land"])
        if case["group"] in figures.CHEMISTRY_GROUPS:
            for field in ("R_kpc", "Sigma_g_Msun_kpc2", "Sigmadot_star_Msun_Gyr_kpc2",
                          "Sigmadot_land_Msun_Gyr_kpc2", "v_R_kpc_yr", "v_R_kms",
                          "Mdot_acc_Msun_yr", "j_disk_kpc_kms", "j_land_required_kpc_kms"):
                np.testing.assert_array_equal(p[field],fiducial[field])
            if np.allclose(p["Z"],fiducial["Z"],rtol=1e-10,atol=1e-14):
                raise ValueError(f"Chemistry scan {case['case']} did not change metallicity")
    return profiles,diagnostics,resolved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/idealized/section3_prescribed_gas_scans.csv")
    parser.add_argument("--cases",type=Path,default=ROOT/"config/idealized/section3_prescribed_gas_cases.csv")
    parser.add_argument("--reference-config",type=Path,default=ROOT/"config/idealized/section3_prescribed_gas_unmixed_reference.csv")
    parser.add_argument("--run-directory",type=Path,default=ROOT/"output/section3_prescribed_gas")
    parser.add_argument("--figure-directory",type=Path,default=ROOT/"output/pdf")
    parser.add_argument("--observation-directory",type=Path,default=ROOT/"output/section3")
    parser.add_argument("--galaxy",default="NGC2403",help="Comparison overlays only")
    parser.add_argument("--figure-prefix",default="section3_prescribed_gas")
    parser.add_argument("--j-display",choices=("physical","nuclear","deficit","disk"),
                        default="deficit",help="Landing-j axis; default is 1-j_land/j, with local disk j=R*v_c(R)")
    parser.add_argument("--skip-build",action="store_true")
    parser.add_argument("--plot-only",action="store_true")
    args=parser.parse_args()
    args.run_directory.mkdir(parents=True,exist_ok=True)
    if not args.skip_build and not args.plot_only:
        figures.run(["make","-j4","build/galactic-nuclear-fountain-model"],args.run_directory/"logs/build.log")
    base,cases=figures.parameters(args.config),figures.read_csv(args.cases)
    profiles,diagnostics,resolved=calculate_cases(args,base,cases)
    reference=figures.parameters(args.reference_config)
    nuclear_j=float(reference["R_nucl_kpc"])*float(reference["v_c_nucl_kms"])
    disk_fit,abundance,fit,observation_paths=figures.observations(args,base)
    files=[args.config,args.cases,args.reference_config,Path(__file__),Path(figures.__file__),
           ROOT/"scripts/run_prescribed_gas.py",ROOT/"scripts/plot/observed_disk_comparison.py",
           ROOT/"scripts/plot/axis_labels.py",

           ROOT/"scripts/plot/surface_rate_units.py",
           figures.RUNNER,*observation_paths]
    files.extend(ROOT/"model"/name for name in (
        "units.h",
        "prescribed_gas.cpp","prescribed_gas.h","prescribed_gas_io.cpp","prescribed_gas_io.h",
        "models.cpp","models.h","physics.cpp","physics.h","rk4.cpp","rk4.h","profiles.cpp","profiles.h"))
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),
              "model":"Supplied gas, KS sink and outer inward mass flux; inferred flow and required j_land",
              "supply":"Shared C++ reservoir-rate/composition helpers; no angular-momentum mixture imposed",
              "chemistry":"Conditional on assumed incident composition, yield and explicit inflow abundances",
              "observations":"Saved gas/SFR fit curves and metallicity measurements are overlays only for the default analytic gas config",
              "reference":"Actual unmixed j_land=j_nuc model with derived gas and derived outer flux",
              "inactive_parameters":["beta: no CGM angular-momentum closure is used"],
              "exact_overlaps":{"prescribed_gas_cases_except_gas_scale":["Sigma_g","KS stellar sink"],
                                "gas_scale_cases_with_fiducial":["landing","incident metallicity","rotation"],
                                "beta_cases_with_fiducial":"all exported profiles",
                                "chemistry_cases_with_fiducial":"gas, stellar sink, landing, flow and required j_land"},
              "gas_scale_scan":{"R_g_kpc":[float(base["R_g_kpc"]),
                                             *[float(case["R_g_kpc"]) for case in cases if case["group"]=="gas_scale"]],
                                "fixed_M_g_Msun":float(base["M_g_Msun"]),
                                "normalization":"C++ exponential-gas normalization holds receiving-annulus mass fixed"},
              "chemistry_scan_display":"Z_outer_boundary and yield_y cases appear only in the bottom two panels of the combined figure",
              "profiles_reused":args.plot_only,"cases":resolved,"diagnostics":diagnostics,
              "required_j_displays":[args.j_display],"required_j_deficit_limits":list(figures.DEFICIT_Y_LIMITS),
              "figure":{"title":FLOW_TITLE,"layout":"4x2","size_inches":[7.2,7.2],
                        "row_height_ratios":[1,1,1,2],
                        "output":args.figure_prefix+"_combined_scans.pdf"},
              "normalization_j_nuc_kpc_kms":nuclear_j,
              "local_j_normalization":"j_disk(R)=R*v_c(R), from each exported profile",
              "source_sha256":{str(path.resolve()):figures.sha256(path) for path in files}}
    (args.run_directory/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (args.run_directory/"figure_notes.txt").write_text(
        "PRESCRIBED-GAS / REQUIRED LANDING ANGULAR MOMENTUM SCANS\n\n"
        "Default gas is an independent exponential disk; flat rotation is supplied.\n"
        "Saved NGC 2403 gas/SFR spline fits are dotted black comparison curves on their saved support.\n"
        "Metallicity measurements remain unchanged, with capped error bars.\n"
        "Supply/composition cases have the same gas and KS stellar sink; gas-scale cases vary both.\n"
        "The gas-scale scan uses R_g=2 and 4 kpc about 3 kpc, with M_g=9e8 Msun fixed in the receiving annulus.\n"
        "C++ renormalizes the exponential gas profile for each R_g. Landing supply, rotation and outer flux stay fixed.\n"
        "Lambda, loading factor eta and mu change the incident supply, flow and required j.\n"
        "The eta scan uses 0.5 and 2 around eta=1; Mdot_nuc remains fixed at 0.125 Msun/yr.\n"
        "Z_outer_boundary and yield_y are varied independently about the same fiducial.\n"
        "Both are dimensionless mass fractions; yield_y is per mass locked in long-lived stars.\n"
        "Their cases change only chemistry and appear in the bottom two panels and shared legend.\n"
        "Their gas, landing, stellar sink, radial flow and required j exactly match the fiducial.\n"
        "C++ computes total landing = eta*(1+mu)*Mdot_nuc and incident Z from the supplied reservoirs.\n"
        "These are supply/composition assumptions; j_mix is never imposed on the required-j model.\n"
        "Beta is inactive and its two cases coincide exactly with the fiducial in every panel.\n"
        "Their dashed/dotted legend entries retain the same scan key; the black fiducial is drawn above them.\n"
        "The solid gray unmixed reference is the actual j_land=j_nuc solution, with predicted gas\n"
        "and its own derived outer mass flux. Its angular momentum is imposed, not inverted.\n"
        "This reference uses an explicitly selected comparison config; it is not a fixed-gas mu=0 inversion.\n"
        "Combined layout: four rows and two columns, 7.2 by 7.2 inches, spanning both document columns.\n"
        "The bottom metallicity/gradient row has twice the height of each flow row.\n"
        "Margins and panel gutters are fitted to rendered labels with four-point clearance.\n"
        "Row order: landing and net KS stellar sink; gas and radial velocity; accretion and required j_land; metallicity and gradient.\n"
        f"Title: {FLOW_TITLE}. Radius labels appear above and below both columns; the shared legend uses two rows below.\n"
        "Landing-rate display spans 1e4 to 1e7 Msun/Gyr/kpc^2, equivalent to 1e-5 to 1e-2 Msun/yr/kpc^2.\n"
        "Adopted display: 1-j_land/j, where j is the local disk angular momentum R*v_c(R).\n"
        "Nuclear j comes from the explicit reference R_nucl*v_c_nucl; local j comes from the core profile.\n"
        "The adopted display has fixed y limits of -1 and 2, labeled ticks at -1, 0, 1 and 2, and no horizontal reference line.\n"
        "Values outside this display range remain in the exported profiles.\n"
        "Metallicity panels assume reservoir incident composition, yield and inflow abundance boundaries.\n"
        "These are conditional predictions, not an inversion of observed metallicity or a joint mixture test.\n"
        "RK4 handles single-direction transport; shared conservative metal balance handles stagnation.\n"
        "No uncertainty shading is added. The observed gradient is dashed black with a local key.\n"
        "The metallicity-gradient panel spans -0.1 to 0 dex/kpc.\n"
        "One combined PDF places all model keys in two rows below the panels.\n"
        "Parameters are illustrative, not fitted. Per-case CSVs, input snapshots and checks are under models/.\n")
    figures.smplotlib.set_style(usetex=False,fontsize=9,dpi=144)
    plt.rcParams.update({"axes.prop_cycle": color_cycle(), "axes.grid":False,"legend.frameon":False,"pdf.fonttype":42,"savefig.bbox":None})
    figures.plot_flow_metallicity(args,cases,profiles,disk_fit,abundance,fit,
                                 stem=args.figure_prefix+"_combined_scans",inactive_beta=True,
                                 gradient_min=-0.1,title=FLOW_TITLE,
                                 j_display=args.j_display,nuclear_j=nuclear_j)
    for suffix in ("_flow_scans", "_flow_scans_j_land_over_j_nuc",
                   "_flow_scans_one_minus_j_land_over_j_disk", "_flow_scans_j_land_over_j_disk",
                   "_metallicity_scans"):
        (args.figure_directory / f"{args.figure_prefix}{suffix}.pdf").unlink(missing_ok=True)


if __name__=="__main__":
    main()
