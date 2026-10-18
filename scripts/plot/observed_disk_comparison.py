"""Saved gas/SFR spline comparisons for model-flow figures.

Plot the exported adopted fits directly, on their saved radial support.
Raw measurements belong to input/fit diagnostic figures. No fits are recalculated
here, and these display curves do not become inputs to an analytic model.
"""
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations

import numpy as np
from surface_rate_units import read_rate_rows


FIT_STYLE={"color":"black","linestyle":":","linewidth":1.0,"zorder":4,"dash_capstyle":"round"}


def fit_legend_handle(label):
    from matplotlib.lines import Line2D
    return Line2D([],[],label=label,**FIT_STYLE)


def flow_legends(figure, handles, *, fontsize, top=.99, position="top", handlelength=1.0,
                 handletextpad=.25, columnspacing=.45, columns=None, first_row_indices=None):
    """Pack model and comparison keys into two rows inside the existing width."""
    style=dict(loc="upper center",frameon=False,borderpad=0,borderaxespad=0,
               handlelength=handlelength,handletextpad=handletextpad,columnspacing=columnspacing)
    if columns is not None:
        # A single legend keeps each two-entry parameter column aligned.
        ordered=[handle for column in columns for handle in column]
        paired_style={**style,"loc":"lower center" if position=="bottom" else "upper center", "labelspacing":.2}
        while fontsize>=6.5:
            legend=figure.legend(handles=ordered,ncol=len(columns),fontsize=fontsize,
                                 bbox_to_anchor=(.5,.01 if position=="bottom" else top),**paired_style)
            figure.canvas.draw()
            if legend.get_window_extent(figure.canvas.get_renderer()).width<=.98*figure.bbox.width:
                return [legend]
            legend.remove()
            fontsize=round(fontsize-.25,2)
        raise ValueError("Paired flow legend cannot fit two readable rows in the figure width")
    count=len(handles)
    preferred=set(range(0,count,2))
    while fontsize>=6.5:
        # Single-entry legend widths include the actual math/font metrics.
        samples=[figure.legend(handles=[handle],fontsize=fontsize,**style) for handle in handles]
        figure.canvas.draw()
        renderer=figure.canvas.get_renderer()
        widths=[legend.get_window_extent(renderer).width for legend in samples]
        for legend in samples:
            legend.remove()
        gap=columnspacing*fontsize*figure.dpi/72
        available=.98*figure.bbox.width
        candidates=[]
        partitions=([tuple(i for i in first_row_indices if i!=0)] if first_row_indices is not None
                    else combinations(range(1,count),(count+1)//2-1))
        for members in partitions:
            first={0,*members}
            if handles[1].get_label()=="unmixed fiducial" and 1 in first:
                continue
            rows=([i for i in range(count) if i in first],[i for i in range(count) if i not in first])
            row_widths=[sum(widths[i] for i in row)+(len(row)-1)*gap for row in rows]
            if max(row_widths)<=available:
                candidates.append((len(first.symmetric_difference(preferred)),max(row_widths),rows))
        if candidates:
            rows=min(candidates,key=lambda item:item[:2])[2]
            break
        fontsize=round(fontsize-.25,2)
    else:
        raise ValueError("Flow legend cannot fit two readable rows in the figure width")
    legends=[figure.legend(handles=[handles[i] for i in row],ncol=len(row),fontsize=fontsize,
                           bbox_to_anchor=(.5,top),**style) for row in rows]
    figure.canvas.draw()
    renderer=figure.canvas.get_renderer()
    if position not in ("top","bottom"):
        raise ValueError("Flow legend position must be top or bottom")
    row_gap=1.5*figure.dpi/72
    next_top=(.01+(sum(legend.get_window_extent(renderer).height for legend in legends)+row_gap)
              /figure.bbox.height) if position=="bottom" else top
    for legend in legends:
        legend.set_bbox_to_anchor((.5,next_top),transform=figure.transFigure)
        next_top=(legend.get_window_extent(renderer).y0-row_gap)/figure.bbox.height
    return legends


@dataclass(frozen=True)
class ObservedDiskFit:
    radius: np.ndarray
    gas: np.ndarray  # Msun/pc^2
    star: np.ndarray  # net stellar sink, Msun/Gyr/kpc^2
    input_paths: tuple[Path, ...]


def read_rows(path):
    return read_rate_rows(path)


def checked_curve(radius, values):
    radius,values=np.asarray(radius,dtype=float),np.asarray(values,dtype=float)
    if (len(radius)<2 or radius.shape!=values.shape or not np.all(np.isfinite(radius))
            or not np.all(np.diff(radius)>0) or not np.all(np.isfinite(values))
            or not np.all(values>0)):
        raise ValueError("Saved observational fits need increasing radii and finite positive values")
    return radius,values


def load_disk_fit(directory, return_fraction):
    """Read the saved dense gas and net-SFR fits, without extrapolating/refitting."""
    directory=Path(directory)
    profile_path,parameter_path=directory/"profiles.csv",directory/"parameters.csv"
    parameters={row["name"]:row["value"] for row in read_rows(parameter_path)}
    rows=read_rows(profile_path)
    radius=[float(row["R_kpc"]) for row in rows]
    radius,gas=checked_curve(radius,[float(row["Sigma_g_Msun_kpc2"])/1e6 for row in rows])
    _,star=checked_curve(radius,[float(row["Sigmadot_star_Msun_Gyr_kpc2"]) for row in rows])
    returned=float(return_fraction)
    if not np.isfinite(returned) or not 0<=returned<1:
        raise ValueError("Comparison return fraction must be in [0,1)")
    if parameters.get("stellar_sink")=="net_locked_mass":
        saved_returned=float(parameters["return_fraction"])
        if not np.isfinite(saved_returned) or not 0<=saved_returned<1:
            raise ValueError("Invalid saved fit return fraction")
        # Saved net sinks already include (1-R); identical conventions need no
        # further correction. A different model R rescales the same fitted SFR.
        star=star*((1-returned)/(1-saved_returned))
    elif "stellar_sink" not in parameters:
        star=star*(1-returned)  # Explicit legacy raw-SFR export convention.
    else:
        raise ValueError(f"Unknown saved stellar-sink convention: {parameters['stellar_sink']}")
    return ObservedDiskFit(radius,gas,star,(profile_path,parameter_path))


def plot_disk_fit(axis, fit, quantity, label, scale=1.0):
    if quantity not in ("gas","star"):
        raise ValueError("Observational disk comparisons are gas or net SFR")
    line,=axis.plot(fit.radius,getattr(fit,quantity)*scale,label=label,**FIT_STYLE)
    return line


def plot_sfr_diagnostic_fit(axis, rows, scale=1.0):
    """Use the saved dense fit in an existing model run's SFR diagnostics."""
    dense=sorted((row for row in rows if row["row_type"]=="dense"),key=lambda row:float(row["R_kpc"]))
    radius,star=checked_curve([row["R_kpc"] for row in dense],
                             [row["Sigmadot_star_spline_Msun_Gyr_kpc2"] for row in dense])
    line,=axis.plot(radius,star*scale,label="observed SFR spline fit",**FIT_STYLE)
    return line,radius,star*scale
