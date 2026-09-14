#!/usr/bin/env python3
"""Plot kernel CSVs as combined pages and individual single-column PDFs."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from hershey_fonts import register_hershey_weight_aliases

register_hershey_weight_aliases()
import smplotlib


SINGLE_COLUMN_FIGURE_SIZE = (3.5, 3.5)


def diagnostic_page(nrows, ncols):
    """Lay out independent, publication-size panels on one overview page."""
    width, height = SINGLE_COLUMN_FIGURE_SIZE
    figure = plt.figure(figsize=(ncols * width, nrows * height), layout="constrained")
    panels = figure.subfigures(nrows, ncols, squeeze=False, wspace=0, hspace=0)
    axes = np.array([[panel.subplots() for panel in row] for row in panels])
    return figure, axes


def square_colorbar(mesh, ax, label):
    """Keep the data axes square and the colorbar aligned to their height."""
    ax.set_box_aspect(1)
    colorbar_ax = ax.inset_axes([1.05, 0, 0.05, 1])
    ax.figure.colorbar(mesh, cax=colorbar_ax, label=label)


def save_diagnostics(figure, output_directory, stem, panel_names):
    """Save the overview and each panel, including its labels and colorbar."""
    # Freeze the layout before hiding neighboring panels for individual exports.
    # Preserve the full 3.5-inch width when trimming unused vertical space.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    figure.set_layout_engine("none")
    destination = output_directory / f"{stem}_diagnostics.pdf"
    figure.savefig(destination, dpi=300, bbox_inches=None)
    print(f"Wrote {destination}")
    for panel, name in zip(figure.subfigs, panel_names):
        for other in figure.subfigs:
            other.set_visible(other is panel)
        bounds = panel.bbox.transformed(figure.dpi_scale_trans.inverted()).frozen()
        ax = panel.axes[0]
        if ax.get_box_aspect() == 1:
            content = ax.get_tightbbox(renderer).transformed(figure.dpi_scale_trans.inverted())
            bounds.y0 = max(bounds.y0, content.y0 - 0.04)
            bounds.y1 = min(bounds.y1, content.y1 + 0.04)
        destination = output_directory / f"{stem}_{name}.pdf"
        figure.savefig(destination, dpi=300, bbox_inches=bounds, pad_inches=0)
        print(f"Wrote {destination}")
    plt.close(figure)


def plot_log_histogram(ax, density, edges):
    """Outline the existing density bins on a log axis, leaving empty bins blank."""
    positive = density[density > 0]
    lower = float(positive.min()) / 2 if positive.size else 1e-6
    upper = float(positive.max()) * 2 if positive.size else 1.0
    ax.set_yscale("log")
    ax.stairs(np.where(density > 0, density, np.nan), edges,
              baseline=None, fill=False, color="black", linewidth=1.2)
    ax.set_ylim(lower, upper)


def potential_and_radial_gradient(R, z, parameters):
    """Mirror model/potential.cpp in kpc, Myr, and solar masses; Phi(infinity)=0."""
    G = 4.4985e-12
    Md, ad, bd, Mb, ab, rho, rs = (
        float(parameters[key]) for key in
        ("M_d_Msun", "a_d_kpc", "b_d_kpc", "M_b_Msun", "a_b_kpc",
         "rho_s_Msun_kpc3", "r_s_kpc")
    )
    r = np.hypot(R, z)
    disk_squared = R**2 + (ad + np.sqrt(z**2 + bd**2))**2
    halo_scale = 4 * np.pi * G * rho * rs**3
    phi = (-G * Md / np.sqrt(disk_squared) - G * Mb / (r + ab)
           - halo_scale * np.log1p(r / rs) / r)
    dphi_dR = (G * Md * R / disk_squared**1.5 + G * Mb * R / (r * (r + ab)**2)
               + halo_scale * R / r**3 * (np.log1p(r / rs) - r / (r + rs)))
    return phi, dphi_dR


def binned_weighted_median(x, y, values, weights, x_edges, y_edges):
    """Mass-weighted medians in occupied cells; include the final bin edges."""
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    ix = np.searchsorted(x_edges, x, side="right") - 1
    iy = np.searchsorted(y_edges, y, side="right") - 1
    ix[x == x_edges[-1]] = nx - 1
    iy[y == y_edges[-1]] = ny - 1
    if np.any((ix < 0) | (ix >= nx) | (iy < 0) | (iy >= ny)):
        raise ValueError("Launch conditions fall outside the median heatmap")
    positive = weights > 0
    cells = (ix * ny + iy)[positive]
    values, weights = values[positive], weights[positive]
    result = np.full(nx * ny, np.nan)
    if not cells.size:
        return result.reshape(nx, ny)
    order = np.lexsort((values, cells))
    cells, values, weights = cells[order], values[order], weights[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(cells)) + 1, len(cells)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        cumulative = np.cumsum(weights[start:stop])
        half = cumulative[-1] / 2
        index = min(int(np.searchsorted(cumulative, half)), stop - start - 1)
        median = values[start + index]
        if index + 1 < stop - start and np.isclose(cumulative[index], half, rtol=1e-12, atol=0):
            median = (median + values[start + index + 1]) / 2
        result[cells[start]] = median
    return result.reshape(nx, ny)


def plot_physical_diagnostics(axes, parcels, parameters, radius_edges):
    """Derive four physical panels from saved launches and first returns."""
    angular_ax, median_ax, energy_ax, area_ax = axes
    fields = ("R0_kpc", "z0_kpc", "vR0_kms", "vz0_kms", "vphi0_kms",
              "j_z_kpc2_per_Myr", "v_k_kms", "theta_rad", "weight")
    values = {key: np.array([float(row[key]) for row in parcels]) for key in fields}
    if any(not np.all(np.isfinite(array)) for array in values.values()):
        raise ValueError("Physical diagnostics require finite launch properties")
    statuses = np.array([row["status"] for row in parcels])
    returned = statuses == "returned"
    weights = values["weight"] / values["weight"].sum()
    returned_weights = weights[returned]
    returned_fraction = returned_weights.sum()
    R = np.array([float(row["R_land_kpc"]) for row in parcels if row["status"] == "returned"])
    conversion = float(parameters["kms_to_kpc_per_Myr"])

    # Extend the radial display bins to include returns outside the kernel grid.
    width = radius_edges[-1] - radius_edges[-2]
    extra = max(0, math.ceil((float(R.max()) - radius_edges[-1]) / width)) if R.size else 0
    radial_edges = np.r_[radius_edges, radius_edges[-1] + width * np.arange(1, extra + 1)]
    if R.size:
        _, gradient = potential_and_radial_gradient(R, 0.0, parameters)
        if np.any(R <= 0) or np.any(gradient <= 0) or not np.all(np.isfinite(gradient)):
            raise ValueError("Landing radii must have finite, positive circular speeds")
        eta = values["j_z_kpc2_per_Myr"][returned] / (R * np.sqrt(R * gradient))
        if not np.all(np.isfinite(eta)):
            raise ValueError("Nonfinite angular-momentum ratios")
        eta_edges = np.linspace(min(0.0, float(eta.min())), max(1.1, float(eta.max())), 101)
        mass, _, _ = np.histogram2d(R, eta, bins=(radial_edges, eta_edges), weights=returned_weights)
        if not np.isclose(mass.sum(), returned_fraction, rtol=1e-10, atol=1e-12):
            raise ValueError("Angular-momentum heatmap lost returned mass")
        positive = mass[mass > 0]
        color_map = plt.get_cmap("magma").copy()
        color_map.set_bad("white")
        lo, hi = (float(positive.min()), float(positive.max())) if positive.size else (1e-6, 1.0)
        mesh = angular_ax.pcolormesh(
            radial_edges, eta_edges, np.ma.masked_equal(mass.T, 0), cmap=color_map,
            norm=LogNorm(vmin=lo / 10 if lo == hi else lo, vmax=hi), shading="flat", rasterized=True,
        )
        square_colorbar(mesh, angular_ax, label=r"$\Delta f_{\rm ret}$")
        if returned_fraction > 0:
            print(f"Returned mass with eta_j < 1: {returned_weights[eta < 1].sum() / returned_fraction:.6f}")

        speed_edges = np.arange(0, max(25.0, np.ceil(values["v_k_kms"].max() / 25) * 25) + 25, 25)
        theta_edges = np.linspace(0, np.pi / 2, 31)
        median = binned_weighted_median(
            values["v_k_kms"][returned], values["theta_rad"][returned],
            R - values["R0_kpc"][returned], returned_weights, speed_edges, theta_edges,
        )
        finite = median[np.isfinite(median)]
        limit = max(0.1, float(np.abs(finite).max())) if finite.size else 1.0
        color_map = plt.get_cmap("RdBu_r").copy()
        color_map.set_bad("white")
        mesh = median_ax.pcolormesh(
            speed_edges, np.rad2deg(theta_edges), np.ma.masked_invalid(median.T), cmap=color_map,
            norm=SymLogNorm(linthresh=0.1, vmin=-limit, vmax=limit, base=10), shading="flat", rasterized=True,
        )
        square_colorbar(mesh, median_ax, label=r"Median $\Delta R$ [kpc]")
    else:
        for ax in (angular_ax, median_ax):
            ax.text(0.5, 0.5, "No returning parcels", ha="center", transform=ax.transAxes)
    angular_ax.axhline(1, color="black", linestyle="--", linewidth=1.2)
    angular_ax.set(xlabel=r"$R_{\rm land}$ [kpc]",
                   ylabel=r"$j_z/R_{\rm land}v_c(R_{\rm land})$",
                   xlim=(0, 25), ylim=(-3, 3))
    median_ax.set_xlabel(r"$v_k$ [km s$^{-1}$]")
    median_ax.set_ylabel(r"$\theta$ [${}^{\circ}$]", math_fontfamily="cm")
    median_ax.set(ylim=(0, 90), yticks=np.arange(0, 91, 15))

    # Include the full launch velocity, including disk rotation, in specific energy.
    phi, _ = potential_and_radial_gradient(values["R0_kpc"], values["z0_kpc"], parameters)
    energy = phi / conversion**2 + 0.5 * sum(
        values[key]**2 for key in ("vR0_kms", "vz0_kms", "vphi0_kms")
    )
    if not np.all(np.isfinite(energy)):
        raise ValueError("Nonfinite launch energies")
    lo, hi = min(float(energy.min()), 0.0), max(float(energy.max()), 0.0)
    padding = max(1.0, 0.03 * (hi - lo))
    energy_edges = np.linspace(lo - padding, hi + padding, 101)
    positive_densities = []
    energy_fraction = 0.0
    energy_handles = []
    # Empty histogram bins already break the steps, so encode status with color
    # and markers instead of dash patterns that disappear in sparse tails.
    for status, label, color, marker in (("returned", "Returned", "black", None),
                                          ("timed_out", "Timed out", "red", "o"),
                                          ("integration_failed", "Integration failed", "#CC79A7", "x")):
        selected = statuses == status
        if not np.any(selected):
            continue
        mass, _ = np.histogram(energy[selected], bins=energy_edges, weights=weights[selected])
        if not np.isclose(mass.sum(), weights[selected].sum(), rtol=1e-10, atol=1e-12):
            raise ValueError("Energy histogram lost launched mass")
        energy_fraction += mass.sum()
        density = mass / np.diff(energy_edges)
        positive_densities.extend(density[density > 0])
        energy_ax.stairs(np.where(density > 0, density, np.nan), energy_edges,
                         baseline=None, color=color, linestyle="-", linewidth=1.2)
        if marker is not None:
            centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
            occupied = density > 0
            energy_ax.plot(centers[occupied], density[occupied], linestyle="none",
                           marker=marker, color=color, markersize=3, zorder=3)
        energy_handles.append(Line2D([], [], color=color, linestyle="-", linewidth=1.2,
                                     marker=marker, markersize=3, label=label))
    if not np.isclose(energy_fraction, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("Energy histograms do not account for all launch statuses")
    energy_ax.set(yscale="log", ylim=(min(positive_densities) / 2, max(positive_densities) * 2),
                  xlabel=r"$E_0$ [km$^2$ s$^{-2}$]",
                  ylabel=r"$\mathrm{d}f/\mathrm{d}E_0$ [s$^2$ km$^{-2}$]")
    energy_handles.append(energy_ax.axvline(0, color="black", linestyle=":", linewidth=1.2, label=r"$E_0=0$"))
    energy_ax.legend(handles=energy_handles)
    timed_out = statuses == "timed_out"
    print(f"Timed out: {np.sum(timed_out & (energy < 0))} with E0 < 0; "
          f"{np.sum(timed_out & (energy >= 0))} with E0 >= 0")

    radial_mass, _ = np.histogram(R, bins=radial_edges, weights=returned_weights)
    annular_area = np.pi * np.diff(radial_edges**2)
    area_density = radial_mass / annular_area
    if not np.isclose(area_density @ annular_area, returned_fraction, rtol=1e-10, atol=1e-12):
        raise ValueError("Surface-deposition profile lost returned mass")
    plot_log_histogram(area_ax, area_density, radial_edges)
    area_ax.set(xlabel=r"$R_{\rm land}$ [kpc]",
                ylabel=r"$\mathrm{d}f_{\rm ret}/\mathrm{d}A$ [kpc$^{-2}$]",
                xlim=(radial_edges[0], radial_edges[-1]))
    print(f"Surface-deposition profile integrates to {area_density @ annular_area:.6f} of launched mass.")


def plot_launches(parcels, run_directory, output_directory):
    """Compare all launches (including nonreturns) with the specified PDFs."""
    if "v_k_kms" not in parcels[0]:
        return  # Older fixed-grid runs only have kernel diagnostics.
    with (run_directory / "parameters.csv").open(newline="") as stream:
        parameters = {row["parameter"]: row["value"] for row in csv.DictReader(stream)}
    values = {
        name: np.array([float(row[name]) for row in parcels])
        for name in ("R0_kpc", "v_k_kms", "theta_rad", "psi_rad")
    }
    if any(not np.all(np.isfinite(array)) for array in values.values()):
        raise ValueError("Launch draws must be finite")
    radius, speed, theta, psi = values.values()
    if (np.any(radius <= 0) or np.any(speed < 0) or
            np.any((theta < 0) | (theta > np.pi / 2)) or
            np.any((psi < 0) | (psi >= 2 * np.pi))):
        raise ValueError("Launch draws fall outside the configured PDF support")
    if len(parcels) != int(parameters["number_of_parcels"]):
        raise ValueError("Parcel count does not match parameters.csv")

    ring = float(parameters["R_ring_kpc"])
    sigma_R = float(parameters["sigma_R_kpc"])
    h_v = float(parameters["h_v_kms"])
    sigma_theta = float(parameters["sigma_theta_rad"])
    weights = np.array([float(row["weight"]) for row in parcels])
    figure, axes = diagnostic_page(2, 2)
    labels = ("Launch radius [kpc]", "Kick speed [km/s]",
              r"Polar angle $\theta$ [${}^{\circ}$]", r"Kick azimuth $\psi$ [${}^{\circ}$]")
    plot_values = (radius, speed, np.rad2deg(theta), np.rad2deg(psi))
    for ax, array, label in zip(axes.flat, plot_values, labels):
        ax.hist(array, bins=50, weights=weights, density=True,
                color="#2166ac", alpha=0.45, label="Sampled launches")
        ax.set(xlabel=label, ylabel="Probability density", ylim=(0, None))
    axes[1, 0].xaxis.label.set_math_fontfamily("cm")
    for ax in axes[1]:
        ax.set_ylabel(r"Probability density [$({}^{\circ})^{-1}$]")

    ax = axes[0, 0]
    if sigma_R == 0:
        if not np.all(radius == ring):
            raise ValueError("Fixed-ring draws do not match R_ring_kpc")
        ax.axvline(ring, color="black", label="Fixed ring")
    else:
        lo, hi = (float(parameters[name]) for name in
                  ("radius_cdf_min_kpc", "radius_cdf_max_kpc"))
        x = np.linspace(lo, hi, 2001)
        a, b = (lo - ring) / sigma_R, (hi - ring) / sigma_R
        normalization = (
            ring * sigma_R * math.sqrt(math.pi / 2) *
            (math.erf(b / math.sqrt(2)) - math.erf(a / math.sqrt(2))) +
            sigma_R**2 * (math.exp(-a*a / 2) - math.exp(-b*b / 2))
        )
        pdf = x * np.exp(-0.5 * ((x - ring) / sigma_R)**2) / normalization
        ax.plot(x, pdf, color="black", label="Area-weighted Gaussian")

    x = np.linspace(0, max(speed.max(), 5 * h_v), 1001)
    pdf = np.sqrt(2 / np.pi) * x**2 / h_v**3 * np.exp(-0.5 * (x / h_v)**2)
    axes[0, 1].plot(x, pdf, color="black", label="Maxwell PDF")

    ax = axes[1, 0]
    if sigma_theta == 0:
        if not np.all(theta == 0):
            raise ValueError("Vertical-only draws have nonzero polar angles")
        ax.axvline(0, color="black", label="Vertical kicks")
    else:
        x = np.linspace(0, min(np.pi / 2, 10 * sigma_theta), 4001)
        pdf = np.sin(x) * np.exp(-0.5 * (x / sigma_theta)**2)
        pdf /= np.sum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(x))
        ax.plot(np.rad2deg(x), pdf * np.pi / 180, color="black", label="Gaussian per solid angle")
    axes[1, 1].plot([0, 360], [1 / 360] * 2,
                    color="black", label="Uniform azimuth")
    for ax in axes.flat:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02))
    save_diagnostics(figure, output_directory, "launch",
                     ("radius", "kick_speed", "polar_angle", "azimuth"))
    print(f"Mean kick speed: {np.average(speed, weights=weights):.3f} km/s; "
          f"Maxwell expectation: {math.sqrt(8 / math.pi) * h_v:.3f}")
    print(f"RMS kick speed: {np.sqrt(np.average(speed**2, weights=weights)):.3f} km/s; "
          f"Maxwell expectation: {math.sqrt(3) * h_v:.3f}")
    print(f"Mean sin(psi): {np.average(np.sin(psi), weights=weights):.5f}; "
          f"mean cos(psi): {np.average(np.cos(psi), weights=weights):.5f}; expectations: 0")


def plot_kernel(run_directory, output_directory, full_grid=False):
    with (run_directory / "orbits.csv").open(newline="") as stream:
        parcels = list(csv.DictReader(stream))
    if not parcels:
        raise ValueError("No launch records in orbits.csv")
    returned = [row for row in parcels if row["status"] == "returned"]
    weights = np.array([float(row["weight"]) for row in parcels])
    total_weight = float(weights.sum())
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or total_weight <= 0:
        raise ValueError("Launch weights must be finite and nonnegative with positive total")

    cells = np.atleast_1d(np.genfromtxt(
        run_directory / "kernel.csv", delimiter=",", names=True,
    ))
    radius_edges = np.unique(np.concatenate((cells["R_lo_kpc"], cells["R_hi_kpc"])))
    delay_edges = np.unique(np.concatenate((cells["tau_lo_Myr"], cells["tau_hi_Myr"])))
    radius_indices = np.searchsorted(radius_edges, cells["R_lo_kpc"])
    delay_indices = np.searchsorted(delay_edges, cells["tau_lo_Myr"])
    density = np.full((len(radius_edges) - 1, len(delay_edges) - 1), np.nan)
    density[radius_indices, delay_indices] = cells["K_per_kpc_per_Myr"]
    if not np.all(np.isfinite(density)) or np.any(density < 0) or total_weight <= 0:
        raise ValueError("Expected a complete nonnegative kernel grid and positive launched weight")

    # Check both the density convention and the actual parcel-to-cell accounting.
    cell_fractions = cells["K_per_kpc_per_Myr"] * (
        cells["R_hi_kpc"] - cells["R_lo_kpc"]) * (
        cells["tau_hi_Myr"] - cells["tau_lo_Myr"])
    if not np.allclose(cell_fractions, cells["mass_fraction"], rtol=1e-10, atol=1e-12):
        raise ValueError("Kernel density does not integrate to its stored cell fractions")
    returned_radii = np.array([float(row["R_land_kpc"]) for row in returned])
    returned_delays = np.array([float(row["tau_Myr"]) for row in returned])
    if (not np.all(np.isfinite(returned_radii)) or not np.all(np.isfinite(returned_delays)) or
            np.any(returned_radii < 0) or np.any(returned_delays <= 0)):
        raise ValueError("Invalid returned radius or flight time")
    measured, _, _ = np.histogram2d(
        returned_radii, returned_delays, bins=(radius_edges, delay_edges),
        weights=[float(row["weight"]) / total_weight for row in returned],
    )
    expected = density * np.diff(radius_edges)[:, None] * np.diff(delay_edges)[None, :]
    if not np.allclose(measured, expected, rtol=1e-10, atol=1e-12):
        raise ValueError("Kernel cell fractions disagree with the returned parcels")

    # Integrate over the other coordinate; retain the original launch normalization.
    radius_density = density @ np.diff(delay_edges)
    delay_density = np.diff(radius_edges) @ density
    binned_fraction = float(radius_density @ np.diff(radius_edges))
    returned_weights = np.array([float(row["weight"]) for row in returned]) / total_weight
    returned_fraction = float(returned_weights.sum())
    extra_panels = "v_k_kms" in parcels[0]

    smplotlib.set_style(usetex=False, fontsize=9, figsize=SINGLE_COLUMN_FIGURE_SIZE, dpi=144)
    plt.rcParams.update({
        "axes.grid": False, "axes.labelsize": 9, "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.formatter.limits": (-3, 4), "axes.formatter.use_mathtext": True,
        "savefig.bbox": None, "pdf.fonttype": 42,
    })
    if extra_panels:
        figure, axes = diagnostic_page(3, 4)
        velocity_ax, kernel_ax, angular_ax, median_ax = axes[0]
        radius_ax, delay_ax, displacement_ax, area_ax = axes[1]
        cumulative_radius_ax, cumulative_delay_ax, fraction_ax, energy_ax = axes[2]
        physical_axes = (angular_ax, median_ax, energy_ax, area_ax)
        panel_names = (
            "launch_velocity", "density", "angular_momentum", "median_displacement",
            "radius_distribution", "delay_distribution", "displacement", "surface_deposition",
            "cumulative_radius", "cumulative_delay", "return_fraction", "launch_energy",
        )
    else:
        figure, axes = diagnostic_page(2, 2)
        velocity_ax, kernel_ax, radius_ax, delay_ax = axes.flat
        panel_names = ("launch_velocity", "density", "radius_distribution", "delay_distribution")

    ax = velocity_ax
    if returned:
        launch_vz = np.array([float(row["vz0_kms"]) for row in returned])
        if not np.all(np.isfinite(launch_vz)):
            raise ValueError("Returned parcels must have finite launch velocities")
        # Use the kernel's bin sizes, extending the display to include every
        # returned parcel (the mass kernel itself retains its original grid).
        velocity_edges = []
        for edges, values in ((radius_edges, returned_radii), (delay_edges, returned_delays)):
            width = edges[-1] - edges[-2]
            extra = max(0, math.ceil((float(values.max()) - edges[-1]) / width))
            velocity_edges.append(np.r_[edges, edges[-1] + width * np.arange(1, extra + 1)])
        bin_mass, _, _ = np.histogram2d(
            returned_radii, returned_delays, bins=velocity_edges, weights=returned_weights,
        )
        bin_momentum, _, _ = np.histogram2d(
            returned_radii, returned_delays, bins=velocity_edges,
            weights=returned_weights * launch_vz,
        )
        mean_vz = np.divide(bin_momentum, bin_mass, out=np.full_like(bin_mass, np.nan),
                            where=bin_mass > 0)
        if not np.isclose(bin_mass.sum(), returned_fraction, atol=1e-12, rtol=1e-10):
            raise ValueError("Velocity heatmap lost returned mass")
        velocity_map = plt.get_cmap("viridis").copy()
        velocity_map.set_bad("white")
        velocity_mesh = ax.pcolormesh(
            *velocity_edges, np.ma.masked_invalid(mean_vz.T), cmap=velocity_map,
            shading="flat", vmin=float(launch_vz.min()), vmax=float(launch_vz.max()), rasterized=True,
        )
        square_colorbar(velocity_mesh, ax, label=r"$\langle v_{z,0}\rangle_M$ [km s$^{-1}$]")
    else:
        ax.text(0.5, 0.5, "No returning parcels", ha="center", transform=ax.transAxes)
    ax.set(xlabel=r"$R_{\rm land}$ [kpc]", ylabel=r"$\tau$ [Myr]")

    ax = kernel_ax
    color_map = plt.get_cmap("magma").copy()
    color_map.set_bad("white")
    positive_density = density[density > 0]
    vmin = float(positive_density.min()) if positive_density.size else 1e-6
    vmax = float(positive_density.max()) if positive_density.size else 1.0
    if vmin == vmax:
        vmin *= 0.1
    mesh = ax.pcolormesh(
        radius_edges, delay_edges, np.ma.masked_equal(density.T, 0.0),
        cmap=color_map, shading="flat", norm=LogNorm(vmin=vmin, vmax=vmax), rasterized=True,
    )
    square_colorbar(mesh, ax, label=r"$K(R_{\rm land},\tau)$ [kpc$^{-1}$ Myr$^{-1}$]")
    ax.set(xlabel=r"$R_{\rm land}$ [kpc]", ylabel=r"$\tau$ [Myr]")

    ax.add_patch(Rectangle((0.39, 0.40), 0.59, 0.58, transform=ax.transAxes,
                           facecolor="white", edgecolor="none", zorder=5))
    zoom_ax = ax.inset_axes([0.52, 0.52, 0.43, 0.43], zorder=6)
    zoom_ax.pcolormesh(
        radius_edges, delay_edges, np.ma.masked_equal(density.T, 0.0),
        cmap=color_map, norm=mesh.norm, shading="flat", rasterized=True,
    )
    zoom_ax.set(xlim=(0, 10), ylim=(0, 200), box_aspect=1,
                xticks=(0, 5, 10), yticks=(0, 100, 200))
    zoom_ax.tick_params(which="major", labelsize=6, length=2, pad=2)
    zoom_ax.tick_params(which="minor", length=1)
    plot_log_histogram(radius_ax, radius_density, radius_edges)
    radius_ax.set(
        xlabel=r"$R_{\rm land}$ [kpc]",
        ylabel=r"$\mathrm{d}f_{\rm ret}/\mathrm{d}R_{\rm land}$ [kpc$^{-1}$]",
    )
    plot_log_histogram(delay_ax, delay_density, delay_edges)
    delay_ax.set(
        xlabel=r"$\tau$ [Myr]",
        ylabel=r"$\mathrm{d}f_{\rm ret}/\mathrm{d}\tau$ [Myr$^{-1}$]",
    )

    radius_limit, delay_limit = radius_edges[-1], delay_edges[-1]
    occupied_radius, occupied_delay = np.nonzero(density)
    if not full_grid and occupied_radius.size:
        # Show the populated region with one extra bin, without rebinning or smoothing.
        radius_limit = radius_edges[min(int(occupied_radius.max()) + 2, len(radius_edges) - 1)]
        delay_limit = delay_edges[min(int(occupied_delay.max()) + 2, len(delay_edges) - 1)]
    kernel_ax.set(xlim=(radius_edges[0], radius_limit), ylim=(delay_edges[0], delay_limit))
    radius_ax.set_xlim(radius_edges[0], radius_limit)
    delay_ax.set_xlim(delay_edges[0], delay_limit)

    if extra_panels:
        with (run_directory / "parameters.csv").open(newline="") as stream:
            parameters = {row["parameter"]: row["value"] for row in csv.DictReader(stream)}
        t_stop = float(parameters["t_stop_Myr"])

        # Displacement and cumulative panels include ALL returns, even those
        # outside the kernel grid. Their integrals retain all-launch normalization.
        displacement = returned_radii - np.array([float(row["R0_kpc"]) for row in returned])
        ax = displacement_ax
        if returned:
            displacement_mass, displacement_edges = np.histogram(
                displacement, bins=120, weights=returned_weights,
            )
            if not np.isclose(displacement_mass.sum(), returned_fraction, atol=1e-12, rtol=1e-10):
                raise ValueError("Displacement histogram lost returned mass")
            plot_log_histogram(ax, displacement_mass / np.diff(displacement_edges),
                               displacement_edges)
            inward = returned_weights[displacement < 0].sum()
            outward = returned_weights[displacement > 0].sum()
            ax.text(0.97, 0.95,
                    f"Inward: {inward:.2%}\nOutward: {outward:.2%}\nof launched mass",
                    ha="right", va="top", transform=ax.transAxes, fontsize=8)
        else:
            ax.set(yscale="log", ylim=(1e-6, 1))
            ax.text(0.5, 0.5, "No returning parcels", ha="center", transform=ax.transAxes)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
        ax.set(xlabel=r"$\Delta R=R_{\rm land}-R_0$ [kpc]",
               ylabel=r"$\mathrm{d}f_{\rm ret}/\mathrm{d}\Delta R$ [kpc$^{-1}$]")

        speeds = np.array([float(row["v_k_kms"]) for row in parcels])
        if not np.all(np.isfinite(speeds)) or np.any(speeds < 0):
            raise ValueError("Kick speeds must be finite and nonnegative")
        speed_edges = np.arange(0, max(25.0, np.ceil(speeds.max() / 25) * 25) + 25, 25)
        launch_mass, _ = np.histogram(speeds, bins=speed_edges, weights=weights / total_weight)
        returned_speeds = np.array([float(row["v_k_kms"]) for row in returned])
        return_mass, _ = np.histogram(returned_speeds, bins=speed_edges, weights=returned_weights)
        fraction = np.divide(return_mass, launch_mass, out=np.full_like(launch_mass, np.nan),
                             where=launch_mass > 0)
        if np.any(fraction[np.isfinite(fraction)] > 1 + 1e-10):
            raise ValueError("Return fraction exceeds one in a kick-speed bin")
        ax = fraction_ax
        ax.stairs(fraction, speed_edges, baseline=None, color="black", linewidth=1.2)
        ax.set(xlabel=r"$v_k$ [km s$^{-1}$]", ylabel=r"$f_{\rm ret}$",
               xlim=(0, speed_edges[-1]), ylim=(0, 1.05))

        for ax, values, label, cumulative_label, limit in (
            (cumulative_radius_ax, returned_radii, r"$R_{\rm land}$ [kpc]", r"$f_{\rm ret}(<R_{\rm land})$",
             max(radius_edges[-1], float(returned_radii.max())) if returned else radius_edges[-1]),
            (cumulative_delay_ax, returned_delays, r"$\tau$ [Myr]", r"$f_{\rm ret}(<\tau)$", t_stop),
        ):
            cumulative_mass, cumulative_edges = np.histogram(
                values, bins=200, range=(0, limit), weights=returned_weights,
            )
            cumulative = np.r_[0.0, np.cumsum(cumulative_mass)]
            if not np.isclose(cumulative[-1], returned_fraction, atol=1e-12, rtol=1e-10):
                raise ValueError("Cumulative distribution lost returned mass")
            ax.step(cumulative_edges, cumulative, where="post", color="black", linewidth=1.2)
            ax.axhline(returned_fraction, color="black", linestyle="--", linewidth=1.2,
                       label=f"All returns: {returned_fraction:.4%}")
            ax.set(xlabel=label, ylabel=cumulative_label,
                   xlim=(0, limit), ylim=(0, 1.05))
            ax.legend(loc="lower right")

        plot_physical_diagnostics(physical_axes, parcels, parameters, radius_edges)

    output_directory.mkdir(parents=True, exist_ok=True)
    save_diagnostics(figure, output_directory, "kernel", panel_names)
    print(f"Both marginal distributions integrate to {binned_fraction:.6f} of launched mass.")
    print(f"Returned outside kernel grid: {max(0.0, returned_fraction - binned_fraction):.6f}")
    print(f"Integration failures: {sum(row['status'] == 'integration_failed' for row in parcels)}")
    plot_launches(parcels, run_directory, output_directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="directory containing orbits.csv and kernel.csv")
    parser.add_argument("output_directory", type=Path, nargs="?", help="defaults to RUN_DIRECTORY/plots")
    parser.add_argument("--full-grid", action="store_true", help="show all radius and delay bins, including empty tails")
    args = parser.parse_args()
    plot_kernel(args.run_directory, args.output_directory or args.run_directory / "plots", args.full_grid)


if __name__ == "__main__":
    main()
