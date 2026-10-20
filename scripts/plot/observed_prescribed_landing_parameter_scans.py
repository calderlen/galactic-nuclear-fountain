#!/usr/bin/env python3
"""Observed-disk prescribed-landing scans for manuscript models 3 and 7.

Measured gas, rotation and net stellar-sink fits enter the shared C++ solver.
Landing is a prescribed power law. Required angular momentum is compared with
the nuclear or mixed hypothesis; chemistry is conditional on incident composition.
Each model combines flow and chemistry in one full-width 3x2 figure.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from color_palette import color_cycle
import numpy as np

import section3_mixed_parameter_scans as figures
from paper_galaxies import galaxy_label

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from run_prescribed_gas import run_model

SCAN_FIELDS=("lambda","Mdot_land_Msun_yr","Mdot_out_Msun_yr","eta","mu","beta")
SNAPSHOTS={"gas_file":"gas_profiles.csv","rotation_file":"rotation_profiles.csv",
           "sfr_file":"sfr_profiles.csv","sfr_parameters_file":"sfr_parameters.csv"}


def calculate_cases(args,config,case_file,mode):
    base,cases=figures.parameters(config),figures.read_csv(case_file)
    directory=args.run_directory/mode
    profiles,resolved={},{}
    for case in cases:
        overrides={key:case[key] for key in SCAN_FIELDS if case.get(key)}
        destination=directory/"models"/case["case"]
        if args.plot_only:
            manifest=json.loads((destination/"run_manifest.json").read_text())
            expected={**base,**overrides,**{key:name for key,name in SNAPSHOTS.items() if key in base}}
            if manifest["parameters"]!=expected:
                raise ValueError(f"Saved parameters differ for {mode}/{case['case']}; rerun without --plot-only")
            for key in SNAPSHOTS:
                source=Path(base[key])
                source=source if source.is_absolute() else ROOT/source
                if manifest["inputs"][key]["sha256"]!=figures.sha256(source):
                    raise ValueError(f"Observed input changed: {source}; rerun without --plot-only")
            rows=figures.read_csv(destination/"profiles.csv")
            profile={key:np.array([float(row[key]) for row in rows]) for key in rows[0] if key!="model"}
        else:
            profile,manifest=run_model(config,destination,overrides)
        profiles[case["case"]]=profile
        difference=profile["j_land_required_minus_comparison_kpc_kms"]
        resolved[case["case"]]={"parameters":manifest["parameters"],"checks":manifest["checks"],
                               "Mdot_land_Msun_yr":manifest["Mdot_land_Msun_yr"],
                               "required_minus_comparison_j_range_kpc_kms":[float(np.min(difference)),float(np.max(difference))]}

    fiducial=profiles["fiducial"]
    for case in cases:
        profile=profiles[case["case"]]
        for key in ("R_kpc","Sigma_g_Msun_kpc2","Sigmadot_star_Msun_Gyr_kpc2",
                    "v_c_kms","dv_c_dR_kms_kpc","j_disk_kpc_kms","j_nucl_kpc_kms"):
            np.testing.assert_array_equal(profile[key],fiducial[key])
        if case["group"]=="outer_accretion":
            delta=float(case["Mdot_out_Msun_yr"])-float(base["Mdot_out_Msun_yr"])
            # Adaptive integration across the tabulated sink changes its steps
            # when the flux boundary changes; allow one ppm in the offset.
            np.testing.assert_allclose(profile["Mdot_acc_Msun_yr"]-fiducial["Mdot_acc_Msun_yr"],
                                       delta,rtol=1e-6,atol=1e-10)
        if case["group"]=="cgm_rotation":
            for key in fiducial:
                if key not in ("j_land_comparison_kpc_kms","j_land_required_minus_comparison_kpc_kms"):
                    np.testing.assert_array_equal(profile[key],fiducial[key])
        np.testing.assert_allclose(profile["j_land_required_minus_comparison_kpc_kms"],
            profile["j_land_required_kpc_kms"]-profile["j_land_comparison_kpc_kms"],rtol=1e-12,atol=1e-12)
    return base,cases,profiles,resolved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=("unmixed","mixed","both"),default="both")
    for mode,section in (("unmixed","section2"),("mixed","section3")):
        parser.add_argument(f"--{mode}-config",type=Path,default=ROOT/f"config/galaxies/NGC2403/{section}_prescribed_landing_scans.csv")
        parser.add_argument(f"--{mode}-cases",type=Path,default=ROOT/f"config/galaxies/NGC2403/{section}_prescribed_landing_cases.csv")
    parser.add_argument("--run-directory",type=Path,default=ROOT/"output/observed_prescribed_landing")
    parser.add_argument("--figure-directory",type=Path,default=ROOT/"output/pdf")
    parser.add_argument("--observation-directory",type=Path,default=ROOT/"output/section3")
    parser.add_argument("--galaxy",default="NGC2403",help="Comparison labels/data; use matching explicit galaxy configs")
    parser.add_argument("--plot-only",action="store_true")
    parser.add_argument("--skip-build",action="store_true")
    args=parser.parse_args()
    if not args.skip_build and not args.plot_only:
        figures.run(["make","-j4","build/galactic-nuclear-fountain-model"],args.run_directory/"build.log")
    figures.smplotlib.set_style(usetex=False,fontsize=9,dpi=144)
    plt.rcParams.update({"axes.prop_cycle": color_cycle(), "axes.grid":False,"legend.frameon":False,"pdf.fonttype":42,"savefig.bbox":None})
    for mode in (("unmixed","mixed") if args.model=="both" else (args.model,)):
        config=getattr(args,f"{mode}_config")
        case_file=getattr(args,f"{mode}_cases")
        base,cases,profiles,resolved=calculate_cases(args,config,case_file,mode)
        angular_limits=(0,1)
        disk_fit,abundance,fit,observation_paths=figures.observations(args,base)
        title=(f"{mode.capitalize()} fountain test, {galaxy_label(args.galaxy)} "
               r"$v_c$, $\Sigma_g$, $\dot\Sigma_{\rm SFR}$; prescribed landing")
        section="section2" if mode=="unmixed" else "section3"
        stem=f"{section}_observed_prescribed_landing"
        directory=args.run_directory/mode
        files=[config,case_file,Path(__file__),Path(figures.__file__),ROOT/"scripts/plot/color_palette.py",
               ROOT/"scripts/run_prescribed_gas.py",
               ROOT/"scripts/plot/observed_disk_comparison.py",ROOT/"scripts/plot/axis_labels.py",
               ROOT/"scripts/plot/observed_metallicity_spline.py",
               ROOT/"scripts/plot/six_panel_layout.py",
               ROOT/"scripts/plot/surface_rate_units.py",*observation_paths]
        manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"model":mode,
                  "closure":"Observed gas and stellar sink; power-law landing; derive flow and required j",
                  "comparison":"Required j is compared with the hypothesis, never replaced by it",
                  "chemistry":"Predict Z from the supplied incident composition and explicit inflow boundaries",
                  "nuclear_rate_origin":"Existing observed-model normalization; not a measured nuclear SFR",
                  "title":title,"profiles_reused":args.plot_only,"cases":resolved,
                  "beta_effect":"Changes the angular comparison only; excluded from chemistry panels",
                  "source_sha256":{str(path.resolve()):figures.sha256(path) for path in files}}
        manifest["figure"]={"layout":"3x2","size_inches":[7.2,4.5],
                            "row_height_ratios":[1,1,1],"legend_rows":2,
                            "font_sizes_pt":{"axis_labels":9,"ticks":8,"legends":8,"title":10},
                            "radius_major_ticks_kpc":[0,5,10,15],"radius_minor_step_kpc":1,
                            "inset_count":0,
                            "velocity_limits_kms":[-1 if mode=="unmixed" else -3,0],
                            "metallicity_limits_solar":[0,3],
                            "angular_deficit_limits":list(angular_limits),
                            "panels":["landing","angular_deficit","accretion","radial_velocity",
                                      "metallicity","metallicity_gradient"],
                            "output":stem+"_combined_scans.pdf"}
        manifest["figure"].update({
            "abundance_comparison":"weighted cubic OH12 spline with GCV smoothing",
            "abundance_spline_legend":"shared only",
            "gradient_comparison":"exact spline derivative; upper limit zero"})
        (directory/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        (directory/"figure_notes.txt").write_text(
            f"OBSERVED PRESCRIBED-LANDING {mode.upper()} TEST\n\n"
            "Observed v_c, gas and measured-SFR spline fits are physical inputs.\n"
            "Saved net stellar-sink convention is retained; the return fraction is applied exactly once.\n"
            "All cases use the same observed disk. Landing slope, normalization and outer flux are varied.\n"
            "Mixed cases additionally vary mu and beta. Beta changes only the comparison curve.\n"
            "The angular panel compares solid required 1-j_land/j with dashed nuclear/mixed hypotheses.\n"
            f"The angular deficit display spans {angular_limits[0]} to {angular_limits[1]}; full values remain in the CSVs.\n"
            "Metallicity is a conditional prediction; observed abundances are comparisons, not imposed profiles.\n"
            "Dotted abundance/gradient comparisons use the uncertainty-weighted cubic spline\n"
            "and its exact derivative, with the gradient view capped at zero.\n"
            "The nuclear normalization and launch properties match the existing observed-model runs.\n"
            "A required-j mismatch is retained as a consistency diagnostic, not adjusted away.\n"
            "One combined 3x2 PDF, 7.2 inches wide, with equal-height panels.\n"
            "Compact row gaps leave four points between neighboring texts, preserving panel sizes.\n"
            "Read across rows: landing and angular deficit; accretion and velocity; metallicity and gradient.\n"
            "Gas and stellar sink remain physical inputs and are omitted from the display.\n"
            "Radius labels above and below both columns; paired two-row case legend below the panels.\n")
        chemistry_cases=[case for case in cases if case["group"]!="cgm_rotation"]
        legend_handles=[figures.Line2D([],[],label=cases[0]["label"],**figures.case_style(cases[0])),
                        figures.fit_legend_handle("KK04 spline")]
        groups={}
        for case in cases[1:]:
            groups.setdefault(case["group"],[]).append(case)
        for paired_cases in groups.values():
            legend_handles.extend(figures.Line2D([],[],label=case["label"],**figures.case_style(case))
                                  for case in paired_cases)
        legend_spec={"handles":legend_handles,
                     "labels":[handle.get_label() for handle in legend_handles],
                     "ncol":len(legend_handles)//2,"handlelength":1.0,"fontsize":8,
                     "handletextpad":.25,"columnspacing":.35}
        legend_spec.update(handlelength=.8,handletextpad=.15,columnspacing=.2)
        figure=figures.plot_flow_metallicity(args,cases,profiles,disk_fit,abundance,fit,
                                     stem=stem+"_combined_scans",gradient_min=-0.2,title=title,
                                     legend_spec=legend_spec,
                                     metallicity_max=3,
                                     chemistry_cases=chemistry_cases,gradient_legend=False,
                                     abundance_spline=True,
                                     show_abundance_measurements=mode=="mixed",
                                     flow_y_limits=({3:(-1,0),4:(0,None),5:angular_limits} if mode=="unmixed"
                                                    else {3:(-3,0),5:angular_limits}),
                                     show_disk_inputs=False,
                                     angular_panel_top_right=True,
                                     axis_fontsize=9,tick_fontsize=8,angular_legend_fontsize=8,
                                     angular_legend_loc="best",
                                     figure_size=(7.2,4.5),legend_position="bottom",
                                     angular_comparison_label=(r"$j_{\rm nuc}$" if mode=="unmixed"
                                                               else r"$j_{\rm mix}$"))
        manifest["figure"].update(size_inches=figure.get_size_inches().tolist(),
                                  inter_row_text_clearance_pt=4,panel_sizes_preserved=True)
        (directory/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        for suffix in ("_flow_scans", "_metallicity_scans"):
            (args.figure_directory/f"{stem}{suffix}.pdf").unlink(missing_ok=True)


if __name__=="__main__":
    main()
