#!/usr/bin/env python3
"""Overlay every case from the Section 2 and Section 3 prescribed-gas figures.

Read and validate saved C++ profiles against the two explicit scan tables.
Mixed cases use solid strokes; all unmixed cases use dashed strokes, including
the separate gray angular-momentum-closed reference. One compact legend is shared.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from color_palette import color_cycle
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import numpy as np

import section2_prescribed_gas_parameter_scans as unmixed
import section3_prescribed_gas_parameter_scans as mixed
import section3_mixed_parameter_scans as figures

ROOT=figures.ROOT
TITLE=r"Mixed and unmixed fountains, prescribed $\Sigma_g$"
STEM="prescribed_gas_mixed_unmixed_overlay"
PARAMETER_COLORS={
    "landing_slope":("#3B6C8E","#83A7C0"),
    "gas_scale":("#C43C39","#D88C87"),
    "gas_mass":("#367B3E","#8EB68C"),
    "landing_rate":("#76508C","#B09BC2"),
    "outer_accretion":("#805B43","#B69A83"),
    "loading_factor":("#198DA2","#7FBFCA"),
    "entrainment":("#D37416","#E6B579"),
    "cgm_rotation":("#A98610","#CDBF6E"),
    "outer_metallicity":("#6F7C1D","#A6AF75"),
    "nucleosynthetic_yield":("#B45185","#D69AB8"),
}
FIDUCIAL_STYLE={"linewidth":2.1,"alpha":1.0,"zorder":3.8}
VARIATION_STYLE={"linewidth":.8,"alpha":.82,"zorder":2}
REFERENCE_STYLE={"linewidth":.9,"alpha":.9,"zorder":2.5}
VELOCITY_INSET_LIMITS=(-3,0)
VELOCITY_INSET_RADIUS_LIMITS=(2,10)
FLOW_Y_LIMITS={0:(1e5,None),1:(1e4,None),2:(.1,None),4:(-.05,.2)}
METALLICITY_MAX=3.0


def load_cases(args):
    common=dict(plot_only=True,observation_directory=args.observation_directory,galaxy=args.galaxy)
    unmixed_args=SimpleNamespace(**common,config=ROOT/"config/idealized/section2_prescribed_gas_scans.csv",
        cases=ROOT/"config/idealized/section2_prescribed_gas_cases.csv",run_directory=args.unmixed_directory)
    mixed_args=SimpleNamespace(**common,config=ROOT/"config/idealized/section3_prescribed_gas_scans.csv",
        cases=ROOT/"config/idealized/section3_prescribed_gas_cases.csv",run_directory=args.mixed_directory,
        reference_config=ROOT/"config/idealized/section3_prescribed_gas_unmixed_reference.csv")
    unmixed_base=figures.parameters(unmixed_args.config)
    mixed_base=figures.parameters(mixed_args.config)
    unmixed_cases=figures.read_csv(unmixed_args.cases)
    mixed_cases=figures.read_csv(mixed_args.cases)
    unmixed_profiles,_=unmixed.calculate_cases(unmixed_args,unmixed_base,unmixed_cases)
    mixed_profiles,_,_=mixed.calculate_cases(mixed_args,mixed_base,mixed_cases)
    np.testing.assert_array_equal(unmixed_profiles["fiducial"]["R_kpc"], mixed_profiles["fiducial"]["R_kpc"])
    assert unmixed_base["return_fraction"]==mixed_base["return_fraction"]
    profiles={"fiducial":mixed_profiles["fiducial"]}
    groups=[]
    source_paths=[Path(__file__),Path(figures.__file__),Path(unmixed.__file__),Path(mixed.__file__),
                  unmixed_args.config,unmixed_args.cases,mixed_args.config,mixed_args.cases,
                  mixed_args.reference_config]
    for family,cases,original,directory in (
        ("unmixed",unmixed_cases,unmixed_profiles,args.unmixed_directory),
        ("mixed",mixed_cases,mixed_profiles,args.mixed_directory)):
        displayed=[]
        color_offsets={}
        for case in cases:
            name=f"{family}_{case['case']}"
            model_family="unmixed" if family=="unmixed" or case["group"]=="reference" else "mixed"
            color=case["color"]
            if case["group"] in PARAMETER_COLORS:
                index=color_offsets.get(case["group"],0)
                color=PARAMETER_COLORS[case["group"]][index]
                color_offsets[case["group"]]=index+1
            style=(FIDUCIAL_STYLE if case["case"]=="fiducial" else
                   REFERENCE_STYLE if case["group"]=="reference" else VARIATION_STYLE)
            displayed.append({**case,"case":name,"is_fiducial":case["case"]=="fiducial",
                              "family":model_family,"source_family":family,
                              "color":color,"plot_style":style,
                              "label":"unmixed reference" if case["group"]=="reference" else case["label"],
                              "linestyle":"--" if model_family=="unmixed" else "-"})
            profiles[name]=original[case["case"]]
            source_paths.append(directory/"models"/case["case"]/"profiles.csv")
        groups.append(displayed)
    disk_fit,abundance,fit,observation_paths=figures.observations(args,mixed_base)
    source_paths.extend(observation_paths)
    return groups,profiles,disk_fit,abundance,fit,source_paths


def compact_legend(groups,galaxy):
    """Share matching parameter values, with solid/dashed handles side by side."""
    entries={}
    for case in [*groups[1],*groups[0]]:
        label=case["label"]
        if case["is_fiducial"]:
            label=f"{case['family']} fiducial"
        key=(label,case["color"])
        entries.setdefault(key,[]).append(Line2D([],[],**figures.case_style(case)))
    values=[(lines[0] if len(lines)==1 else tuple(lines),label) for (label,_),lines in entries.items()]
    by_label={label:(handle,label) for handle,label in values}
    order=["mixed fiducial","unmixed fiducial","unmixed reference"]
    # Four entries per column keep parameter pairs together. Put the longer
    # supply/mass labels in two columns so they do not widen every column.
    columns=[[by_label[label] for label in order]+[
        (figures.fit_legend_handle(f"{figures.galaxy_label(galaxy)} spline fit"),
         f"{figures.galaxy_label(galaxy)} spline fit")]]
    paired_groups=(("landing_slope","gas_scale"),("gas_mass","landing_rate"),
                   ("outer_accretion","loading_factor"),("entrainment","cgm_rotation"),
                   ("outer_metallicity","nucleosynthetic_yield"))
    for pair in paired_groups:
        labels=[]
        for group in pair:
            for case in [*groups[1],*groups[0]]:
                if case["group"]==group and case["label"] not in labels:
                    labels.append(case["label"])
        columns.append([by_label[label] for label in labels])
    ordered=[entry for column in columns for entry in column]
    return {"handles":[handle for handle,_ in ordered],"labels":[label for _,label in ordered],
            "ncol":len(columns),"handler_map":{tuple:HandlerTuple(ndivide=None,pad=.25)}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unmixed-directory",type=Path,default=ROOT/"output/section2_prescribed_gas")
    parser.add_argument("--mixed-directory",type=Path,default=ROOT/"output/section3_prescribed_gas")
    parser.add_argument("--observation-directory",type=Path,default=ROOT/"output/section3")
    parser.add_argument("--figure-directory",type=Path,default=ROOT/"output/pdf")
    parser.add_argument("--galaxy",default="NGC2403")
    args=parser.parse_args()
    groups,profiles,disk_fit,abundance,fit,source_paths=load_cases(args)
    cases=[case for group in groups for case in group]
    figures.smplotlib.set_style(usetex=False,fontsize=9,dpi=144)
    plt.rcParams.update({"axes.prop_cycle": color_cycle(), "axes.grid":False,"legend.frameon":False,"pdf.fonttype":42,"savefig.bbox":None})
    figures.plot_flow_metallicity(args,cases,profiles,disk_fit,abundance,fit,stem=STEM,title=TITLE,
                                  gradient_min=-.2,metallicity_max=METALLICITY_MAX,inactive_beta=True,
                                  legend_spec=compact_legend(groups,args.galaxy),figure_size=(7.2,7.2),
                                  velocity_inset_limits=VELOCITY_INSET_LIMITS,
                                  velocity_inset_radius_limits=VELOCITY_INSET_RADIUS_LIMITS,
                                  flow_y_limits=FLOW_Y_LIMITS,gradient_legend=False)
    args.figure_directory.joinpath(STEM+"_manifest.json").write_text(json.dumps({
        "created_utc":datetime.now(timezone.utc).isoformat(),"profiles_reused":True,
        "title":TITLE,"layout":"4x2","size_inches":[7.2,7.2],"row_height_ratios":[1,1,1,2],
        "flow_case_count":sum(case["group"] not in figures.CHEMISTRY_GROUPS for case in cases),
        "chemistry_case_count":len(cases),"j_limits":[-1,2],"gradient_limits":[-.2,0],
        "families":{"unmixed":"Section 2 direct-supply required-j scan; dashed curves",
                    "mixed":"Section 3 reservoir-supply required-j scan; solid curves",
                    "gray_reference":"Original Section 3 unmixed j_land=j_nuc comparison; dashed gray"},
        "legend":"One shared four-row legend; matching values combine solid and dashed handles",
        "observed_gradient_legend":False,
        "parameter_color_pairs":PARAMETER_COLORS,
        "line_hierarchy":{"fiducial":FIDUCIAL_STYLE,"variation":VARIATION_STYLE,"reference":REFERENCE_STYLE},
        "velocity_inset_limits_kms":list(VELOCITY_INSET_LIMITS),
        "velocity_inset_radius_limits_kpc":list(VELOCITY_INSET_RADIUS_LIMITS),
        "flow_y_limits":FLOW_Y_LIMITS,"metallicity_max_solar":METALLICITY_MAX,
        "observations":"One set of saved gas/SFR overlays, KK04 measurements/errors and gradient",
        "cases":cases,"source_sha256":{str(path.resolve()):figures.sha256(path) for path in source_paths}
    },indent=2)+"\n")


if __name__=="__main__":
    main()
