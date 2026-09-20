#!/usr/bin/env python3
"""Plot selected parcel trajectories in cylindrical R-z coordinates."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from hershey_fonts import register_hershey_weight_aliases

register_hershey_weight_aliases()
import smplotlib


def read_trajectories(path):
    required = {
        "parcel_id", "status", "v_k_kms", "theta_rad", "psi_rad", "t_Myr",
        "R_kpc", "z_kpc", "vR_kms", "vz_kms", "vphi_kms",
        "j_z_kpc2_per_Myr", "m_Msun",
    }
    grouped = defaultdict(list)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Trajectory CSV is missing columns: {sorted(missing)}")
        for row in reader:
            grouped[int(row["parcel_id"])].append(row)
    if not grouped:
        raise ValueError("Trajectory CSV contains no samples")

    trajectories = []
    numeric_fields = required - {"parcel_id", "status"}
    for parcel_id, rows in grouped.items():
        rows.sort(key=lambda row: float(row["t_Myr"]))
        statuses = {row["status"] for row in rows}
        if len(statuses) != 1:
            raise ValueError(f"Parcel {parcel_id} has inconsistent statuses")
        values = {name: np.array([float(row[name]) for row in rows]) for name in numeric_fields}
        if any(not np.all(np.isfinite(array)) for array in values.values()):
            raise ValueError(f"Parcel {parcel_id} has nonfinite trajectory values")
        if np.any(np.diff(values["t_Myr"]) <= 0):
            raise ValueError(f"Parcel {parcel_id} has invalid time ordering")
        trajectories.append((parcel_id, statuses.pop(), values))
    return sorted(trajectories)


def plot_trajectories(trajectories, output_path, r_max=None, z_max=None,
                      equal_aspect=False):
    all_R = np.concatenate([values["R_kpc"] for _, _, values in trajectories])
    all_z = np.concatenate([values["z_kpc"] for _, _, values in trajectories])
    if r_max is None:
        r_max = max(1.0, 1.08 * float(all_R.max()))
    if z_max is None:
        z_max = max(0.5, 1.10 * float(all_z.max()))

    # AASTeX two-column layouts provide about 3.5 inches for one column.
    smplotlib.set_style(usetex=False, fontsize=8, figsize=(3.5, 3.65), dpi=144)
    plt.rcParams.update({
        "axes.grid": False, "axes.labelsize": 8, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "legend.frameon": False,
        "savefig.bbox": None, "pdf.fonttype": 42,
    })
    figure, ax = plt.subplots(figsize=(3.5, 3.65))
    figure.subplots_adjust(left=0.16, right=0.80, bottom=0.22, top=0.97)

    speeds = np.array([values["v_k_kms"][0] for _, _, values in trajectories])
    speed_min, speed_max = float(speeds.min()), float(speeds.max())
    if speed_min == speed_max:
        speed_min -= 0.5
        speed_max += 0.5
    normalization = Normalize(speed_min, speed_max)
    color_map = plt.get_cmap("inferno")
    styles = {"returned": "-", "timed_out": "--", "integration_failed": ":"}
    labels = []
    for parcel_id, status, values in trajectories:
        color = color_map(normalization(values["v_k_kms"][0]))
        line, = ax.plot(values["R_kpc"], values["z_kpc"],
                        linestyle=styles.get(status, "-"), color=color,
                        linewidth=1.25, zorder=2)
        theta = math.degrees(values["theta_rad"][0])
        psi = math.degrees(values["psi_rad"][0])
        labels.append((line, f"$\\theta={theta:.0f}^\\circ,\\ "
                             f"\\psi={psi:.0f}^\\circ$"))

    ax.set(xlabel=r"$R$ [kpc]", ylabel=r"$z$ [kpc]",
           xlim=(0, r_max), ylim=(0, z_max))
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_box_aspect(1)

    colorbar_ax = ax.inset_axes([1.05, 0.0, 0.055, 1.0])
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=color_map),
        cax=colorbar_ax, label=r"$v_k$ [km s$^{-1}$]",
    )
    colorbar.ax.tick_params(labelsize=7)
    if len(labels) <= 12:
        legend = ax.legend(
            [item[0] for item in labels], [item[1] for item in labels],
            loc="upper center", bbox_to_anchor=(0.5, -0.11),
            ncols=1 if len(labels) <= 4 else 2,
            title="Launch angles", title_fontsize=6.5,
        )
        for text in legend.get_texts():
            text.set_math_fontfamily("cm")
    else:
        print("More than 12 trajectories: omitted per-particle launch-angle legend")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path, help="trajectory CSV from the C++ exporter")
    parser.add_argument("output", type=Path, nargs="?", help="defaults to TRAJECTORIES-stem.pdf")
    parser.add_argument("--r-max", type=float)
    parser.add_argument("--z-max", type=float)
    parser.add_argument("--equal-aspect", action="store_true",
                        help="use the same physical scale for one kpc in R and z")
    args = parser.parse_args()
    output = args.output or args.trajectories.with_name(args.trajectories.stem + "_R_z.pdf")
    plot_trajectories(read_trajectories(args.trajectories), output,
                      args.r_max, args.z_max, args.equal_aspect)


if __name__ == "__main__":
    main()
