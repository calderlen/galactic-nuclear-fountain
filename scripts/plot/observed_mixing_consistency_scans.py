#!/usr/bin/env python3
"""Scan the mixed observational mu_j = mu_Z = mu consistency test.

The existing C++ inverse/observed-chemistry runner owns all physics. Saved
measured disk and abundance fits remain fixed while trial supply parameters vary.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from color_palette import COLOR_CYCLE, PETROFF_RANK3, apply_case_palette, color_cycle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

import section3_mixed_parameter_scans as scans
import section3_mixing_figures as diagnostic
from axis_labels import quantity_unit_label
from paper_galaxies import galaxy_label

ROOT=scans.ROOT
FIELDS=("lambda","eta","mu","beta","Mdot_out_Msun_yr","Z_CGM","Z_nucl","yield_y")
INPUTS=("profiles.csv","gas_profiles.csv","rotation_curves.csv","rotation_profiles.csv")
PANELS=(
    ("landing_slope","Landing slope","lambda",r"\lambda"),
    ("loading_factor","Loading factor","eta",r"\eta"),
    ("entrainment","Entrainment ratio","mu",r"\mu"),
    ("cgm_rotation","CGM rotation","beta",r"\beta"),
    ("outer_accretion","Outer accretion","Mdot_out_Msun_yr",r"\dot M_{\rm acc,out}"),
    ("cgm_metallicity","CGM metallicity","Z_CGM",r"Z_{\rm CGM}"),
    ("nuclear_metallicity","Nuclear metallicity","Z_nucl",r"Z_{\rm nuc}"),
    ("nucleosynthetic_yield","Yield","yield_y",r"y"))
LOW_COLOR,HIGH_COLOR="#0072B2","#D55E00"
MU_LIMITS=(-1,10)
OVERLAY_SIZE_MM=(180,110)
ADOPTED_MU_STYLE={"linestyle":":","dashes":(.1,3),"dash_capstyle":"round","linewidth":1.1}


def root_path(path):
    path=Path(path)
    return path if path.is_absolute() else ROOT/path


def compute(args,base,name,overrides):
    settings={**base,**overrides}
    source=args.run_directory/"inputs"/name
    output=args.run_directory/args.galaxy/name
    source.mkdir(parents=True,exist_ok=True)
    disk=root_path(settings.pop("observed_disk_directory"))
    metal=root_path(settings.pop("observed_metallicity_file"))
    for filename in INPUTS:
        shutil.copyfile(disk/filename,source/filename)
    shutil.copyfile(metal,source/"metallicity_profiles.csv")
    scans.write_parameters(source/"parameters.csv",settings)
    scans.run([str(scans.RUNNER),"inverse",str(source),str(output)], args.run_directory/"logs"/(name+".log"))
    method=diagnostic.read_csv(output/"metallicity_solver.csv")[0]["method"]
    if method!="observed_profile":
        raise ValueError("Consistency chemistry must use the independently observed abundance profile")
    rows,summary=diagnostic.load_case(args.run_directory,args.galaxy,name)
    profile={key:diagnostic.values(rows,key) for key in rows[0] if key not in ("model","source")}
    radius=profile["R_kpc"]
    support=np.isfinite(profile["Z"]) & np.isfinite(profile["dZ_dR_per_kpc"])
    total=float(settings["eta"])*(1+float(settings["mu"]))*float(settings["Mdot_nuc_Msun_yr"])
    np.testing.assert_allclose(profile["cumulative_landing_Msun_yr"][-1],total,rtol=1e-12)
    np.testing.assert_allclose(profile["Mdot_acc_Msun_yr"][-1],float(settings["Mdot_out_Msun_yr"]),rtol=1e-12)
    for key in ("Sigma_g_Msun_kpc2","Sigmadot_star_Msun_Gyr_kpc2","Sigmadot_land_Msun_Gyr_kpc2",
                "v_R_kms","j_land_kpc_kms"):
        if not np.all(np.isfinite(profile[key])):
            raise ValueError(f"Nonfinite {key} in {name}")
    # Check observed inputs directly, including their derivative and support.
    observed=scans.read_csv(metal)
    observed_r=np.array([float(row["R_kpc"]) for row in observed])
    np.testing.assert_array_equal(support,(radius>=observed_r[0]) & (radius<=observed_r[-1]))
    for key in ("Z","dZ_dR_per_kpc"):
        expected=np.interp(radius[support],observed_r,[float(row[key]) for row in observed])
        np.testing.assert_allclose(profile[key][support],expected,rtol=1e-13)
    mass_rhs=2*np.pi*radius*(profile["Sigmadot_star_Msun_Gyr_kpc2"]- profile["Sigmadot_land_Msun_Gyr_kpc2"])/1e9
    mass_error=float(np.max(abs(np.gradient(profile["Mdot_acc_Msun_yr"],radius,edge_order=2)-mass_rhs)) /
                     max(np.max(abs(mass_rhs)),1e-30))
    if mass_error>.005:
        raise ValueError(f"Continuity check failed for {name}: {mass_error:g}")
    valid=support & np.isfinite(profile["mu_j"]) & np.isfinite(profile["mu_Z"]) & \
          (profile["mu_j"]>=0) & (profile["mu_Z"]>=0)
    record={"parameters":settings,"metallicity_solver":method,"Mdot_land_Msun_yr":total,
            "mass_equation_output_grid_relative_error":mass_error,
            "observed_support_samples":int(np.count_nonzero(support)),
            "both_ratios_finite_nonnegative_samples":int(np.count_nonzero(valid))}
    print(f"Computed {name}: observed chemistry, landing {total:.6g} Msun/yr",flush=True)
    return profile,record


def render(args,base,cases,profiles):
    figure,axes=plt.subplots(4,2,figsize=(7.2,8.4),sharex=True,sharey=True)
    axes=axes.ravel()
    fiducial=profiles["fiducial"]
    support=np.isfinite(fiducial["Z"])
    radius=fiducial["R_kpc"][support]
    panel_legends=[]
    for axis,(group,heading,field,symbol) in zip(axes,PANELS):
        pair=sorted((case for case in cases if case["group"]==group),key=lambda case:float(case[field]))
        if len(pair)!=2:
            raise ValueError(f"Provide one lower/higher pair for {group}")
        selected=[(pair[0],LOW_COLOR),(pair[1],HIGH_COLOR),(cases[0],"black")]
        axis.axhspan(MU_LIMITS[0],0,color="0.94",zorder=0)
        axis.axhline(0,color="0.65",linewidth=.5,zorder=0)
        adopted_values={}
        for case,color in selected:
            profile=profiles[case["case"]]
            width=1.5 if case["case"]=="fiducial" else 1.1
            diagnostic.plot_without_poles(axis,radius,profile["mu_j"][support],
                (profile["j_CGM_kpc_kms"]-profile["j_land_kpc_kms"])[support],
                color=color,linestyle="-",linewidth=width)
            diagnostic.plot_without_poles(axis,radius,profile["mu_Z"][support],
                (float(case.get("Z_CGM") or base["Z_CGM"])-profile["Z_land_required"])[support],
                color=color,linestyle="--",dashes=(5,2.5),linewidth=width)
            adopted=float(case.get("mu") or base["mu"])
            # Identical adopted values share one dotted reference line.
            # The fiducial color wins where the trial mixtures coincide.
            adopted_values[adopted]=color
        for adopted,color in adopted_values.items():
            if MU_LIMITS[0]<=adopted<=MU_LIMITS[1]:
                axis.plot(radius,np.full_like(radius,adopted),color=color,zorder=6, **ADOPTED_MU_STYLE)
        axis.set_yscale("symlog",linthresh=1,linscale=.7)
        axis.set_ylim(*MU_LIMITS)
        axis.set_yticks([-1,-.5,0,.5,1,2,5,10],labels=["-1","-0.5","0","0.5","1","2","5","10"])
        axis.set_xlim(radius[0],radius[-1])
        axis.set_xticks([1,2,3,4,5,6])
        axis.tick_params(which="both",direction="in",top=True,right=True,labelsize=7.5)
        axis.set_ylabel(quantity_unit_label(r"$\mu$"),fontsize=9,labelpad=2)
        values=[float(pair[0][field]),float(base[field]),float(pair[1][field])]
        unit=""
        number_format=".5g"
        if field in ("Z_CGM","Z_nucl","yield_y"):
            values=[value/scans.SOLAR_Z for value in values]
            unit=r"\;[Z_\odot]"
            number_format=".3g"
        elif field=="Mdot_out_Msun_yr":
            unit=r"\;[M_\odot\,\mathrm{yr}^{-1}]"
            number_format=".4f"
        axis.set_title(rf"${symbol}{unit}$",fontsize=8.5,pad=0,y=1.18)
        axis.title.set_verticalalignment("bottom")
        panel_legends.append(axis.legend(handles=[
            Line2D([],[],color=color,label=rf"${format(value,number_format)}$")
            for color,value in zip((LOW_COLOR,"black",HIGH_COLOR),values)],
            loc="lower center",bbox_to_anchor=(.5,1.01),ncol=3,fontsize=8,
            frameon=False,borderpad=0,borderaxespad=0,handlelength=1.25,
            handletextpad=.35,columnspacing=1.2))
    for axis in axes[-2:]:
        axis.set_xlabel(r"$R\;[\mathrm{kpc}]$",fontsize=9,labelpad=2)
    heading=figure.suptitle(
        rf"Mixed fountain consistency, observed $v_c$, $\Sigma_g$, $\dot\Sigma_{{\rm SFR}}$ and $Z$ for {galaxy_label(args.galaxy)}",
        fontsize=10,y=.99)
    type_legend=figure.legend(handles=[
        Line2D([],[],color="black",label=r"$\mu_j$"),
        Line2D([],[],color="black",linestyle="--",dashes=(5,2.5),label=r"$\mu_Z$"),
        Line2D([],[],color="black",label=r"adopted $\mu$",**ADOPTED_MU_STYLE),
        Patch(facecolor="0.94",label=r"$\mu<0$ (inadmissible)")],
        loc="upper center",bbox_to_anchor=(.5,.965),ncol=4,fontsize=8,
        frameon=False,borderpad=0,borderaxespad=0,numpoints=3)
    figure.subplots_adjust(left=.075,right=.985,bottom=.065,top=.87,hspace=.30,wspace=.16)
    # Fit the outer legend/title spacing with the same four-point clearance.
    figure.canvas.draw()
    renderer=figure.canvas.get_renderer()
    gap=4*figure.dpi/72
    if heading.get_window_extent(renderer).width>.98*figure.bbox.width:
        heading.set_fontsize(10*.98*figure.bbox.width/heading.get_window_extent(renderer).width)
    # Remove invisible legend padding, then pack the subplot keys and titles.
    type_legend.set_bbox_to_anchor((.5,(heading.get_window_extent(renderer).y0-gap)/figure.bbox.height))
    for _ in range(3):
        for axis,legend in zip(axes,panel_legends):
            legend.set_bbox_to_anchor((.5,1+2*figure.dpi/72/axis.bbox.height),transform=axis.transAxes)
        figure.canvas.draw()
        renderer=figure.canvas.get_renderer()
        for axis,legend in zip(axes,panel_legends):
            title_bottom=legend.get_window_extent(renderer).y1+2*figure.dpi/72
            axis.title.set_position((.5,(title_bottom-axis.bbox.y0)/axis.bbox.height))
        figure.canvas.draw()
        renderer=figure.canvas.get_renderer()
        panel_top=max(axis.get_tightbbox(renderer).y1 for axis in axes[:2])
        figure.subplots_adjust(top=figure.subplotpars.top+(type_legend.get_window_extent(renderer).y0-gap-panel_top)/figure.bbox.height)
        boxes=[axis.get_tightbbox(renderer) for axis in axes]
        column_gap=max(boxes[i].x1-axes[i].bbox.x1+axes[i+1].bbox.x0-boxes[i+1].x0 for i in range(0,8,2))+gap
        row_gap=max(axes[i].bbox.y0-boxes[i].y0+boxes[i+2].y1-axes[i+2].bbox.y1 for i in range(6))+gap
        width=(figure.subplotpars.right-figure.subplotpars.left)*figure.bbox.width
        height=(figure.subplotpars.top-figure.subplotpars.bottom)*figure.bbox.height
        figure.subplots_adjust(wspace=2*column_gap/(width-column_gap), hspace=4*row_gap/(height-3*row_gap))
    stem=args.output_stem or "section3_observed_mixing_consistency_scans"
    diagnostic.save_figure(figure,args.figure_directory/(stem+".pdf"))


def render_overlay(args,base,cases,profiles):
    """Render the same exported trials together on one shared set of axes."""
    figure,axis=plt.subplots(figsize=tuple(value/25.4 for value in OVERLAY_SIZE_MM))
    fiducial=profiles["fiducial"]
    support=np.isfinite(fiducial["Z"])
    radius=fiducial["R_kpc"][support]
    palette_cases=apply_case_palette(cases)
    variations=[case for case in palette_cases if case["case"]!="fiducial"]
    colors={case["case"]:case["color"] for case in palette_cases}
    # Quantity line styles already encode mu_j, mu_Z and adopted mu. Keep
    # them intact and distinguish the second pass through the colors by
    # sparse open circles, including matching parameter-legend handles.
    markers={case["case"]:("o" if index>=len(COLOR_CYCLE) else None) for index,case in enumerate(variations)}
    markers["fiducial"]=None
    marker_style={"markersize":2.4,"markerfacecolor":"white","markeredgewidth":.6}
    axis.axhspan(MU_LIMITS[0],0,color="0.94",zorder=0)
    axis.axhline(0,color="0.65",linewidth=.5,zorder=0)
    adopted_values={}
    ordered=[*variations,next(case for case in cases if case["case"]=="fiducial")]
    for case in ordered:
        profile=profiles[case["case"]]
        color=colors[case["case"]]
        width=1.5 if case["case"]=="fiducial" else .9
        case_markers={"marker":markers[case["case"]],"markevery":.16,**marker_style}
        diagnostic.plot_without_poles(axis,radius,profile["mu_j"][support],
            (profile["j_CGM_kpc_kms"]-profile["j_land_kpc_kms"])[support],
            color=color,linestyle="-",linewidth=width,**case_markers)
        axis.lines[-1].set_label("_"+case["case"]+"_mu_j")
        diagnostic.plot_without_poles(axis,radius,profile["mu_Z"][support],
            (float(case.get("Z_CGM") or base["Z_CGM"])-profile["Z_land_required"])[support],
            color=color,linestyle="--",dashes=(5,2.5),linewidth=width,**case_markers)
        axis.lines[-1].set_label("_"+case["case"]+"_mu_Z")
        adopted_values[float(case.get("mu") or base["mu"])]=color
    for adopted,color in adopted_values.items():
        if MU_LIMITS[0]<=adopted<=MU_LIMITS[1]:
            axis.plot(radius,np.full_like(radius,adopted),color=color,zorder=6, **ADOPTED_MU_STYLE)
    axis.set_yscale("symlog",linthresh=1,linscale=.7)
    axis.set_ylim(*MU_LIMITS)
    axis.set_yticks([-1,-.5,0,.5,1,2,5,10],labels=["-1","-0.5","0","0.5","1","2","5","10"])
    axis.set_xlim(radius[0],radius[-1])
    axis.set_xticks([1,2,3,4,5,6])
    axis.set_xlabel(r"$R\;[\mathrm{kpc}]$",fontsize=12,labelpad=2)
    axis.set_ylabel(quantity_unit_label(r"$\mu$"),fontsize=12,labelpad=2)
    axis.tick_params(which="both",direction="in",top=True,right=True,labelsize=10)
    axis.tick_params(which="major",length=5,width=.7)
    axis.tick_params(which="minor",length=2.5,width=.5)
    title=figure.suptitle(
        rf"Mixed fountain consistency, observed $v_c$, $\Sigma_g$, $\dot\Sigma_{{\rm SFR}}$ and $Z$ for {galaxy_label(args.galaxy)}",
        fontsize=10,y=1-4/(figure.get_figheight()*72))
    type_legend=figure.legend(handles=[
        Line2D([],[],color="black",label=r"$\mu_j$"),
        Line2D([],[],color="black",linestyle="--",dashes=(5,2.5),label=r"$\mu_Z$"),
        Line2D([],[],color="black",label=r"adopted $\mu$",**ADOPTED_MU_STYLE),
        Patch(facecolor="0.94",label=r"$\mu<0$ (inadmissible)")],
        loc="upper center",bbox_to_anchor=(.5,.95),ncol=4,fontsize=9,
        frameon=False,borderpad=0,borderaxespad=0,numpoints=3)
    columns=[[Line2D([],[],color="black",label="fiducial"),Line2D([],[],color="none",label=" ")]]
    for group,_,field,symbol in PANELS:
        pair=sorted((case for case in cases if case["group"]==group),key=lambda case:float(case[field]))
        handles=[]
        for case in pair:
            value=float(case[field])
            unit=""
            number_format=".5g"
            if field in ("Z_CGM","Z_nucl","yield_y"):
                value/=scans.SOLAR_Z
                unit=r"\,Z_\odot"
                number_format=".3g"
            elif field=="Mdot_out_Msun_yr":
                unit=r"\,M_\odot\,\mathrm{yr}^{-1}"
                number_format=".4f"
            handles.append(Line2D([],[],color=colors[case["case"]],
                                  marker=markers[case["case"]],**marker_style,
                                  label=rf"${symbol}={format(value,number_format)}{unit}$"))
        columns.append(handles)
    # One lower/higher pair per column gives exactly two rows; compact handle
    # spacing keeps full labels and units readable at 9 points.
    legend=figure.legend(handles=[handle for column in columns for handle in column],
                         loc="lower center",bbox_to_anchor=(.5,4/(figure.get_figheight()*72)),
                         ncol=len(columns),fontsize=9,frameon=False,
                         borderpad=0,borderaxespad=0,labelspacing=.25,
                         handlelength=.9,handletextpad=.2,columnspacing=.25)
    legends=[legend]
    figure.subplots_adjust(left=.075,right=.985,bottom=.23,top=.86)
    figure.canvas.draw()
    renderer=figure.canvas.get_renderer()
    gap=4*figure.dpi/72
    if legend.get_window_extent(renderer).width>figure.bbox.width-2*gap:
        raise ValueError("Publication legend exceeds the 180 mm figure width")
    if title.get_window_extent(renderer).width>.98*figure.bbox.width:
        title.set_fontsize(10*.98*figure.bbox.width/title.get_window_extent(renderer).width)
    figure.canvas.draw()
    type_legend.set_bbox_to_anchor((.5,(title.get_window_extent(renderer).y0-gap)/figure.bbox.height))
    for _ in range(2):
        figure.canvas.draw()
        renderer=figure.canvas.get_renderer()
        box=axis.get_tightbbox(renderer)
        legend_top=max(legend.get_window_extent(renderer).y1 for legend in legends)
        figure.subplots_adjust(
            left=figure.subplotpars.left+(gap-box.x0)/figure.bbox.width,
            right=figure.subplotpars.right+(figure.bbox.width-gap-box.x1)/figure.bbox.width,
            bottom=figure.subplotpars.bottom+(legend_top+gap-box.y0)/figure.bbox.height,
            top=figure.subplotpars.top+(type_legend.get_window_extent(renderer).y0-gap-box.y1)/figure.bbox.height)
    stem=args.output_stem or "section3_observed_mixing_consistency_overlay"
    path=args.figure_directory/(stem+".pdf")
    diagnostic.save_figure(figure,path)
    notes={"figure":str(path),"layout":"one shared axis", "cases":[case["case"] for case in cases],
           "size_mm":list(OVERLAY_SIZE_MM),"font_sizes_pt":{"axis":12,"tick":10,"legend":9,"title":10},
           "parameter_legend_rows":2,
           "mu_limits":list(MU_LIMITS),"mu_j":"solid","mu_Z":"dashed","adopted_mu":"dotted line",
           "adopted_mu_style":ADOPTED_MU_STYLE,
           "palette":"Petroff rank-3 ten-color set",
           "palette_hex":list(PETROFF_RANK3),"case_markers":markers,
           "case_colors":{key:color if isinstance(color,str) else list(color) for key,color in colors.items()},
           "legend_order":"lower over higher, paired by parameter", "metallicity_and_yield_unit":"Z_sun=0.0139",
           "coincident_curves":"Identical inferred quantities remain exactly overlapped; all cases are rendered",
           "physics_manifest":str(args.run_directory/"run_manifest.json"),
           "renderer_sha256":scans.sha256(Path(__file__)),
           "palette_source_sha256":scans.sha256(ROOT/"scripts/plot/color_palette.py"),
           "profile_sha256":{case["case"]:scans.sha256(args.run_directory/args.galaxy/case["case"]/"profiles.csv") for case in cases}}
    (args.run_directory/"overlay_manifest.json").write_text(json.dumps(notes,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/galaxies/NGC2403/section3_observed_consistency_scans.csv")
    parser.add_argument("--cases",type=Path,default=ROOT/"config/galaxies/NGC2403/section3_observed_consistency_cases.csv")
    parser.add_argument("--run-directory",type=Path,default=ROOT/"output/observed_mixing_consistency")
    parser.add_argument("--figure-directory",type=Path,default=ROOT/"output/pdf")
    parser.add_argument("--output-stem",help="PDF filename without extension; defaults to the selected layout's name")
    parser.add_argument("--skip-build",action="store_true")
    parser.add_argument("--layout",choices=("panels","overlay"),default="panels")
    args=parser.parse_args()
    base=scans.parameters(args.config)
    args.galaxy=base["galaxy"]
    if base["metallicity_mode"]!="observed" or base["profile_type"]!="tabulated":
        raise ValueError("Use the explicit observed-disk, observed-chemistry configuration")
    disk=root_path(base["observed_disk_directory"])
    saved=scans.parameters(disk/"parameters.csv")
    if saved["stellar_sink"]!="net_locked_mass" or float(saved["return_fraction"])!=float(base["return_fraction"]):
        raise ValueError("The saved measured-SFR net sink must use the configured return fraction")
    if not args.skip_build:
        scans.run(["make","-j4","build/galactic-nuclear-fountain-model"],args.run_directory/"build.log")
    cases=scans.read_csv(args.cases)
    profiles,records={},{}
    for case in cases:
        overrides={key:case[key] for key in FIELDS if case.get(key)}
        profiles[case["case"]],records[case["case"]]=compute(args,base,case["case"],overrides)
    fiducial=profiles["fiducial"]
    for profile in profiles.values():
        for key in ("R_kpc","Sigma_g_Msun_kpc2","Sigmadot_star_Msun_Gyr_kpc2","v_c_kms","Z","dZ_dR_per_kpc"):
            np.testing.assert_array_equal(profile[key],fiducial[key])
    for case in cases:
        if case["group"]=="cgm_rotation":
            for key in ("mu_Z","Z_land_required","j_land_kpc_kms","v_R_kms"):
                np.testing.assert_array_equal(profiles[case["case"]][key],fiducial[key])
        if case["group"] in ("cgm_metallicity","nuclear_metallicity","nucleosynthetic_yield"):
            profile=profiles[case["case"]]
            for key in ("mu_j","j_land_kpc_kms","v_R_kms","Mdot_acc_Msun_yr"):
                np.testing.assert_array_equal(profile[key],fiducial[key])
            if np.allclose(profile["mu_Z"],fiducial["mu_Z"],rtol=1e-10,atol=1e-12,equal_nan=True):
                raise ValueError(f"Chemical scan {case['case']} did not change inferred mu_Z")
    # An independence control varies assumed mu while holding total supply fixed.
    # Neither inferred ratio may simply recover the input mixing composition.
    control_checks=[]
    total=records["fiducial"]["Mdot_land_Msun_yr"]
    for mu in (.5,2):
        control,record=compute(args,base,f"control_fixed_supply_mu{mu:g}",
                              {"mu":str(mu),"Mdot_nuc_Msun_yr":format(total/(float(base["eta"])*(1+mu)),".17g")})
        for key in ("mu_j","mu_Z","j_land_kpc_kms","Z_land_required","v_R_kms"):
            np.testing.assert_allclose(control[key],fiducial[key],rtol=1e-10,atol=1e-11,equal_nan=True)
        control_checks.append({"mu":mu,"Mdot_land_Msun_yr":record["Mdot_land_Msun_yr"],"inferred_profiles_unchanged":True})
    sources=[args.config,args.cases,Path(__file__),Path(scans.__file__),Path(diagnostic.__file__),
             scans.RUNNER,*[disk/name for name in (*INPUTS,"parameters.csv")],root_path(base["observed_metallicity_file"]),
             *[ROOT/"model"/name for name in ("model_runner.cpp","models.cpp","physics.cpp","units.h","profiles.cpp","rk4.cpp")]]
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"cases":records,
              "chemistry":"Independent observed KK04 abundance and derivative on measured support; no predicted Z is used",
              "supply":"eta*(1+mu)*Mdot_nuc; mu scans change trial total landing at fixed nuclear source",
              "comparison":"Inferred mu_j and mu_Z versus the corresponding adopted model-8 mu",
              "independence_controls":control_checks,"display":{"mu_limits":list(MU_LIMITS),"scale":"symlog","poles":"curves broken across each case's denominator sign changes","negative_mu":"inadmissible",
                  "mu_j":"solid","mu_Z":"dashed","adopted_mu":"dotted line",
                  "case_colors":{"fiducial":"black","lower":LOW_COLOR,"higher":HIGH_COLOR},
                  "subplot_key_value_order":"lower, fiducial, higher",
                  "subplot_headings":"symbols and units only",
                  "metallicity_and_yield_display_unit":"Z_sun=0.0139"},
              "source_sha256":{str(path.resolve()):scans.sha256(path) for path in sources}}
    (args.run_directory/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (args.run_directory/"figure_notes.txt").write_text(
        "MIXED OBSERVATIONAL CONSISTENCY SCANS\n\n"
        "One combined figure: paired lambda, eta, mu, beta, outer-accretion, Z_CGM, Z_nuc and yield variations.\n"
        "Each subplot's key shows colored lines beside lower, fiducial and higher values.\n"
        "Subplot headings show only parameter symbols and applicable units.\n"
        "Metallicities and yields display in Z_sun=0.0139; model inputs retain their native mass fractions.\n"
        "Solid mu_j and dashed mu_Z are independently inferred by the C++ inverse runner.\n"
        "Dotted lines show the adopted mixture for the corresponding model-8 case.\n"
        "Chemical scan values come from the explicit case table and change mu_Z while leaving mu_j and dynamics fixed.\n"
        "Measured gas, SFR, rotation and KK04 abundance fits stay fixed.\n"
        "The net stellar sink already includes the return fraction; no second factor is applied.\n"
        "Changing mu at fixed nuclear source changes the trial total landing rate.\n"
        "Separate fixed-total-supply controls show that neither inferred ratio recovers imposed mu by construction.\n"
        "Only measured abundance radial support is displayed. Negative mu is inadmissible; breaks mark poles.\n"
        "The symlog view spans -1 to 10; full ratios remain in the CSVs.\n"
        "This is a conditional consistency test, without uncertainty propagation or parameter fitting.\n")
    scans.smplotlib.set_style(usetex=False,fontsize=9,dpi=144)
    plt.rcParams.update({"axes.prop_cycle": color_cycle(), "axes.grid":False,"legend.frameon":False,"pdf.fonttype":42,"savefig.bbox":None})
    args.figure_directory.mkdir(parents=True,exist_ok=True)
    if args.layout=="overlay":
        render_overlay(args,base,cases,profiles)
    else:
        render(args,base,cases,profiles)


if __name__=="__main__":
    main()
