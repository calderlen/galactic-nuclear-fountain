#!/usr/bin/env python3
"""Plot the saved NGC 2403 observational inputs for Section 2.2.

Read exported measurements and adopted disk profiles directly. The metallicity
comparison uses the shared uncertainty-weighted cubic abundance spline.
"""

from __future__ import annotations

from axis_labels import quantity_unit_label

from pathlib import Path

import matplotlib.pyplot as plt
from color_palette import BLUE, color_cycle
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator, NullFormatter

from hershey_fonts import register_hershey_weight_aliases
from surface_rate_units import read_rate_rows
from observed_metallicity_spline import fit_metallicity_spline

register_hershey_weight_aliases()
import smplotlib


ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "output/section3/inputs/NGC2403/observed_disk"
OUTPUT = ROOT / "output/pdf/section2_ngc2403_observed_inputs"
FIT_COLOR = BLUE
SOLAR_Z = 0.0139
FIGURE_SIZE = (84.0 / 25.4, 4.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    rows = read_rate_rows(path)
    if not rows:
        raise ValueError(f"No saved input rows in {path}")
    return rows


def values(rows, column):
    return np.asarray([float(row[column]) for row in rows])


def profile(axis, rows, column, scale=1.0, label=None):
    radius = values(rows, "R_kpc")
    quantity = values(rows, column) / scale
    if not (np.all(np.isfinite(quantity)) and np.all(np.diff(radius) > 0)):
        raise ValueError(f"Invalid saved profile: {column}")
    line, = axis.plot(radius, quantity, color=FIT_COLOR, linewidth=0.6, linestyle="-", label=label, zorder=5)
    return line


def measurements(axis, rows, column, error_column, label, marker="o", scale=1.0,
                 markerfacecolor="black", markersize=1.9, zorder=2):
    if not rows:
        raise ValueError(f"Missing measurements for {label}")
    errors = (np.vstack([values(rows, name) for name in error_column])
              if isinstance(error_column, tuple) else values(rows, error_column))
    return axis.errorbar(values(rows, "R_kpc"), values(rows, column) / scale,
                  yerr=errors / scale,
                  linestyle="none", marker=marker, markersize=markersize,
                  markerfacecolor=markerfacecolor, markeredgecolor="black",
                  markeredgewidth=0.4, color="black", ecolor="black",
                  elinewidth=0.35, capsize=0.9, capthick=0.35,
                  label=label, zorder=zorder)


def main():
    parameters = {row["name"]: row["value"] for row in read_csv(INPUTS / "parameters.csv")}
    metadata = read_csv(INPUTS / "metadata.csv")[0]
    r_out = float(parameters["R_out_kpc"])
    return_fraction = float(parameters["return_fraction"])

    rotation = read_csv(INPUTS / "rotation_spline_diagnostics.csv")
    gas = read_csv(INPUTS / "gas_spline_diagnostics.csv")
    sink = read_csv(INPUTS / "sfr_spline_diagnostics.csv")
    abundance = [row for row in read_csv(ROOT / "output/section3/observations.csv")
                 if row["galaxy"] == "NGC2403" and row["calibration"] == "KK04"]
    abundance_spline = fit_metallicity_spline(abundance)

    def select(rows, row_type, source=None):
        return sorted((row for row in rows if row["row_type"] == row_type
                       and float(row["R_kpc"]) <= r_out + 1e-8
                       and (source is None or row["source"] == source)),
                      key=lambda row: float(row["R_kpc"]))

    smplotlib.set_style(usetex=False, fontsize=8, figsize=FIGURE_SIZE, dpi=180)
    plt.rcParams.update({"axes.prop_cycle": color_cycle(),
        "axes.grid": False, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.bbox": None,
        "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=FIGURE_SIZE)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.93, hspace=0.18)

    rotation_handle=measurements(axes[0], select(rotation, "measurement"), "Vobs_kms",
                                 "e_Vobs_kms", "SPARC", marker="^", markersize=1.8)
    profile(axes[0], select(rotation, "dense"), "V_spline_kms")
    axes[0].set_ylabel(quantity_unit_label(r"$v_c\ [\mathrm{km\ s^{-1}}]$"))
    axes[0].set_ylim(0, 155)
    axes[0].yaxis.set_major_locator(MaxNLocator(3))

    disk_handles=[]
    for source, label, marker, face in [("Leroy2008_THINGS", "Leroy+2008", "s", "black"),
                                       ("Bigiel2010", "Bigiel+2010", "o", "white")]:
        # Opaque open markers cover interior error bars; profiles draw above both.
        disk_handles.append(measurements(axes[1], select(gas, "measurement", source),
                     "Sigma_g_obs_Msun_kpc2", "e_Sigma_g_obs_Msun_kpc2",
                     label, marker, scale=1e6, markerfacecolor=face,
                     markersize=1.9 if face == "white" else 1.5,
                     zorder=4 if face == "white" else 2))
    profile(axes[1], select(gas, "dense"), "Sigma_g_spline_Msun_kpc2", scale=1e6)
    axes[1].set_yscale("linear")
    axes[1].set_ylim(0, 10)
    axes[1].set_yticks([0, 5, 10])
    axes[1].set_ylabel(quantity_unit_label(r"$\Sigma_g\ [M_\odot\ \mathrm{pc^{-2}}]$"))

    # These diagnostic measurements and the saved fit already include (1-R).
    # Do not apply the returned-mass correction a second time.
    for source, label, marker, face in [("Leroy2008", "Leroy+2008", "s", "black"),
                                       ("Bigiel2010_crosscal", "Bigiel+2010", "o", "white")]:
        measurements(axes[2], select(sink, "measurement", source),
                     "Sigmadot_star_obs_Msun_Gyr_kpc2",
                     "e_Sigmadot_star_obs_Msun_Gyr_kpc2", label, marker, markerfacecolor=face,
                     markersize=1.9 if face == "white" else 1.5,
                     zorder=4 if face == "white" else 2)
    profile(axes[2], select(sink, "dense"), "Sigmadot_star_spline_Msun_Gyr_kpc2")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e3, 1e7)
    axes[2].set_yticks([1e3, 1e5, 1e7])
    axes[2].set_ylabel(quantity_unit_label(r"$\dot\Sigma_\star$"+"\n"+r"$[M_\odot\ \mathrm{Gyr^{-1}\ kpc^{-2}}]$"))

    # The exported Z values and asymmetric errors already use the solar-scaled
    # abundance pattern. This renderer only converts the display to solar units.
    abundance_handle=measurements(axes[3], abundance, "Z", ("e_Z_low", "e_Z_high"),
                                  "Moustakas+2010", marker="D", markersize=1.8, scale=SOLAR_Z)
    axes[3].plot(abundance_spline.radius, abundance_spline.abundance_solar,
                 color=FIT_COLOR, linewidth=0.6, linestyle="-", zorder=5)
    axes[3].set_ylabel(r"$Z/Z_\odot$")
    axes[3].set_ylim(0, 2.6)
    axes[3].set_yticks([0, 1, 2])
    axes[3].set_xlim(0, 15)
    axes[3].set_xticks([0, 5, 10, 15])

    for index, axis in enumerate(axes):
        axis.tick_params(which="both", direction="in", top=True, right=True)
        axis.tick_params(which="major", length=5, width=0.7)
        axis.tick_params(which="minor", length=2.5, width=0.6)
        axis.minorticks_on()
        if index == 1:
            axis.yaxis.set_minor_locator(AutoMinorLocator(5))
        if index == 2:
            axis.yaxis.set_minor_locator(LogLocator(base=10, subs=range(1, 10), numticks=100))
            axis.yaxis.set_minor_formatter(NullFormatter())
        if index==3:
            axis.set_xlabel(r"$R\ [\mathrm{kpc}]$", labelpad=1)
        axis.yaxis.labelpad=2
    fig.align_ylabels(axes)
    legend=fig.legend(handles=[rotation_handle,*disk_handles,abundance_handle],
                      loc="lower center",bbox_to_anchor=(0.5,0.01),ncol=4,
                      fontsize=8,handlelength=0.8,handletextpad=0.2,
                      columnspacing=0.3,borderpad=0,borderaxespad=0)
    # Fit the figure around its 8-point labels and external legend at final size.
    gap=3*fig.dpi/72
    for _ in range(3):
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        boxes=[axis.get_tightbbox(renderer) for axis in axes]
        left=(max(axis.bbox.x0-box.x0 for axis,box in zip(axes,boxes))+gap)/fig.bbox.width
        right=1-(max(box.x1-axis.bbox.x1 for axis,box in zip(axes,boxes))+gap)/fig.bbox.width
        bottom=fig.subplotpars.bottom+(legend.get_window_extent(renderer).y1+gap-boxes[-1].y0)/fig.bbox.height
        top=1-(max(0,boxes[0].y1-axes[0].bbox.y1)+gap)/fig.bbox.height
        row_gap=max(axes[i].bbox.y0-boxes[i].y0+boxes[i+1].y1-axes[i+1].bbox.y1 for i in range(3))+gap
        height=(top-bottom)*fig.bbox.height
        fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top, hspace=4*row_gap/(height-3*row_gap))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".pdf"), format="pdf")
    plt.close(fig)

    notes = (
        "NGC 2403 observational inputs for Section 2.2.\n"
        "Panels: rotation, gas surface density, net stellar sink, and metallicity in solar units.\n"
        "Layout: title-free short 4x1 stack at 84 mm single-column width and 4 inches high.\n"
        "Axis and shared external-legend text use 8-point type at the final printed size.\n"
        "Black measurements include endcapped measurement-error bars. Blue curves are\n"
        "saved adopted rotation/gas/stellar-sink profiles and the shared weighted abundance spline.\n"
        "SPARC uses filled triangles, Leroy filled squares, Bigiel open circles, and Moustakas filled diamonds.\n"
        "The single-row legend below all panels contains only the four datasets, with capped error bars.\n"
        "The Moustakas label omits the calibration suffix; the plotted abundances still use KK04.\n"
        "Bigiel markers are slightly larger, with white centers hiding interior error-bar segments.\n"
        "Marker sizes: SPARC/Moustakas 1.8 pt, Leroy 1.5 pt, Bigiel 1.9 pt; blue fit width 0.6 pt.\n"
        "Blue fit curves draw above all measurement markers and error bars.\n"
        "Gas surface density uses a linear axis from 0 to 10 M_sun/pc^2; the stellar sink remains logarithmic.\n"
        "The stellar-sink axis spans 1e3 to 1e7 M_sun/Gyr/kpc^2.\n"
        f"Rotation, gas and stellar-sink panels show data within R <= {r_out:.6f} kpc.\n"
        "SPARC radii use the adopted Leroy geometry (3.2 Mpc, 63 deg inclination).\n"
        f"Net stellar sink = (1-return_fraction)*SFR; return_fraction={return_fraction:g}.\n"
        "The saved SFR diagnostics already contain this correction. Bigiel SFR\n"
        f"measurements already include cross-calibration factor {metadata['SFR_crosscal_factor']}.\n"
        f"Gas treatment: {metadata['H2_treatment']}.\n"
        "Gas and rotation use the saved weighted cubic smoothing splines; the\n"
        "stellar-sink spline was fitted in log space.\n"
        f"KK04 abundances: {len(abundance)} unique regions, measurement errors displayed.\n"
        "Abundance catalog: Moustakas et al. (2010), ApJS 190, 233, Table 10;\n"
        "VizieR J/ApJS/190/233, H II-region literature compilation, KK04 calibration.\n"
        "Abundance fit method: uncertainty-weighted cubic smoothing spline in OH12; GCV smoothing.\n"
        "Coincident radii are pooled by inverse variance; weights use exported sigma_OH12_dex,\n"
        "including the adopted intrinsic-scatter term. Measurement error bars remain displayed.\n"
        "Solar abundance-pattern assumption: Z = 0.0139 * 10^(OH12 - 8.69).\n"
        "The bottom panel displays Z/Z_sun with Z_sun=0.0139, using exported Z values\n"
        "and their asymmetric measurement errors. The curve is the converted logarithmic-abundance spline.\n"
        f"Abundance fit support: {abundance_spline.radius[0]:g} to {abundance_spline.radius[-1]:g} kpc; no extrapolation.\n"
        "This display comparison does not replace saved model-input profiles.\n"
        f"Saved disk inputs: {INPUTS}\n"
        f"Saved abundance measurements: {ROOT / 'output/section3/observations.csv'}\n"
    )
    OUTPUT.with_suffix(".txt").write_text(notes)
    print(f"Saved {OUTPUT.with_suffix('.pdf')}")
    for name, rows in [("rotation", rotation), ("gas", gas), ("stellar sink", sink)]:
        curve = select(rows, "dense")
        print(f"{name}: {len(select(rows, 'measurement'))} measurements; "
              f"adopted curve {curve[0]['R_kpc']} to {curve[-1]['R_kpc']} kpc")
    print(f"KK04: {len(abundance)} regions; fit "
          f"{abundance_spline.radius[0]:g} to {abundance_spline.radius[-1]:g} kpc")


if __name__ == "__main__":
    main()
