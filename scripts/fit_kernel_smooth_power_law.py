#!/usr/bin/env python3
"""Fit the returned-wind surface profile with a smooth broken power law.

Run: conda run -n malca python scripts/fit_kernel_smooth_power_law.py
The fit uses equal weights in log surface density at 0.1-kpc bin centers,
outside the central turnover (1--15 kpc by default). No orbits are rerun.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts" / "plot"))
from axis_labels import quantity_unit_label


def log_profile(radius, parameters):
    """Parameters: ln(A), inner slope, slope increase, ln(Rb), ln(alpha)."""
    log_a, inner, increase, log_break, log_alpha = parameters
    log_x = np.log(radius) - log_break
    alpha = np.exp(log_alpha)
    return (log_a - inner * log_x - increase / alpha * (np.logaddexp(0, alpha * log_x) - np.log(2)))


def read_profile(source):
    cells = np.genfromtxt(source, delimiter=",", names=True)
    edges = np.unique(np.concatenate((cells["R_lo_kpc"], cells["R_hi_kpc"])))
    index = np.searchsorted(edges, cells["R_lo_kpc"])
    delay_width = cells["tau_hi_Myr"] - cells["tau_lo_Myr"]
    surface = np.bincount(index, weights=cells["K_per_kpc2_per_Myr"] * delay_width, minlength=len(edges) - 1)
    area = np.pi * np.diff(edges**2)
    return edges, (edges[:-1] + edges[1:]) / 2, surface, area


def fit_profile(radius, surface, lower, upper):
    selected = (radius >= lower) & (radius <= upper) & (surface > 0)
    r, log_y = radius[selected], np.log(surface[selected])
    if r.size <= 5:
        raise ValueError("Need more than five positive bins in the fitted interval")
    bounds = ([-50, 0, 0, np.log(lower), np.log(0.05)], [10, 15, 20, np.log(upper), np.log(100)])
    solutions = []
    for break_radius in (2.5, 4.0, 6.0):
        break_radius = np.clip(break_radius, lower * 1.01, upper / 1.01)
        log_a = np.interp(np.log(break_radius), np.log(r), log_y)
        for alpha in (0.3, 1.0, 3.0, 10.0):
            for inner in (0.5, 2.5):
                initial = [log_a, inner, 7 - inner, np.log(break_radius), np.log(alpha)]
                result = least_squares(lambda p: log_profile(r, p) - log_y,
                                       initial, bounds=bounds, max_nfev=3000,
                                       ftol=1e-11, xtol=1e-11, gtol=1e-11)
                if result.success:
                    solutions.append(result)
    if not solutions:
        raise RuntimeError("No fit converged")
    best = min(solutions, key=lambda result: result.cost)
    return selected, best, solutions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", nargs="?", type=Path, default=ROOT / "output/kernel/sampled_10000000_r0p1")
    parser.add_argument("--r-min", type=float, default=1.0)
    parser.add_argument("--r-max", type=float, default=15.0)
    args = parser.parse_args()
    if not 0 < args.r_min < args.r_max:
        parser.error("Require 0 < r-min < r-max")
    edges, radius, surface, area = read_profile(args.run_directory / "kernel.csv")
    selected, best, solutions = fit_profile(radius, surface, args.r_min, args.r_max)
    fit_radius = radius[selected]
    fitted = np.exp(log_profile(fit_radius, best.x))
    residual = np.log10(surface[selected] / fitted)
    log_a, inner, increase, log_break, log_alpha = best.x
    single_slope, single_intercept = np.polyfit(np.log(fit_radius), np.log(surface[selected]), 1)
    single_residual = (np.log(surface[selected]) - single_intercept - single_slope * np.log(fit_radius)) / np.log(10)
    integral = quad(lambda r: 2 * np.pi * r * np.exp(log_profile(r, best.x)), args.r_min, args.r_max, epsabs=1e-10)[0]
    near_best = [s for s in solutions if s.cost <= best.cost * (1 + 1e-6) + 1e-12]
    summary = {
        "source": str((args.run_directory / "kernel.csv").resolve()),
        "component": "returned initial nuclear mass; excludes condensed CGM mass",
        "profile": "q(R) = integral K_wind(R,tau) dtau; Sigma_dot_wind = Mdot_launch * q",
        "formula": "A*x**(-lambda_in)*((1+x**alpha)/2)**(-(lambda_out-lambda_in)/alpha); x=R/R_b",
        "fit_interval_kpc": [args.r_min, args.r_max],
        "objective": "equal radial-bin weights in natural-log surface density, evaluated at bin centers",
        "number_of_bins": int(selected.sum()),
        "A_kpc_minus2": float(np.exp(log_a)),
        "lambda_in": float(inner), "lambda_out": float(inner + increase),
        "R_break_kpc": float(np.exp(log_break)), "alpha": float(np.exp(log_alpha)),
        "rms_residual_dex": float(np.sqrt(np.mean(residual**2))),
        "max_absolute_residual_dex": float(np.max(np.abs(residual))),
        "single_power_law_lambda": float(-single_slope),
        "single_power_law_rms_dex": float(np.sqrt(np.mean(single_residual**2))),
        "measured_mass_fraction_in_fit_interval": float(surface[selected] @ area[selected]),
        "fitted_mass_fraction_in_fit_interval": float(integral),
        "successful_starts": len(solutions), "starts_at_best_minimum": len(near_best),
        "bound_active": best.active_mask.tolist(),
        "note": "Shape fit, without a mass-normalization constraint or statistical parameter errors; not extrapolated into the central turnover or inserted into the steady solver.",
    }
    output = args.run_directory / "fits" / "smooth_broken_power_law"
    output.mkdir(parents=True, exist_ok=True)
    (output / "fit.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "profile.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["R_kpc", "q_kernel_kpc-2", "q_fit_kpc-2",
                         "df_ret_dR_kernel_kpc-1", "df_ret_dR_fit_kpc-1", "log10_kernel_over_fit"])
        for r, measured, model, delta in zip(fit_radius, surface[selected], fitted, residual):
            writer.writerow([r, measured, model, 2*np.pi*r*measured, 2*np.pi*r*model, delta])

    plt.rcParams.update({"font.size": 10, "axes.spines.top": True, "axes.spines.right": True, "mathtext.fontset": "cm"})
    figure = plt.figure(figsize=(10, 6.4), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(2.1, 1))
    radial_ax = figure.add_subplot(grid[0, 0])
    surface_ax = figure.add_subplot(grid[0, 1])
    residual_ax = figure.add_subplot(grid[1, :])
    dense_radius = np.geomspace(args.r_min, args.r_max, 1000)
    dense_surface = np.exp(log_profile(dense_radius, best.x))
    for ax, data, model, ylabel in (
        (radial_ax, surface * area / np.diff(edges), 2*np.pi*dense_radius*dense_surface,
         r"$\mathrm{d}f_{\rm ret}/\mathrm{d}R$ [kpc$^{-1}$]"),
        (surface_ax, surface, dense_surface, r"$q(R)$ [kpc$^{-2}$]"),
    ):
        ax.stairs(np.where(data > 0, data, np.nan), edges, baseline=None,
                  color="black", linewidth=1, label="Saved kernel")
        ax.plot(dense_radius, model, color="#AA4499", linewidth=2, label="Smooth broken power law")
        ax.axvline(summary["R_break_kpc"], color="#777777", linewidth=0.8, linestyle=":", label=r"$R_b$")
        ax.set_yscale("log")
        ax.set(xlabel=r"$R$ [kpc]", ylabel=quantity_unit_label(ylabel))
        ax.tick_params(which="both", direction="in", top=True, right=True)
    radial_ax.set(xlim=(0, 20), ylim=(1e-6, 1))
    radial_ax.legend(loc="upper right", frameon=False, fontsize=9)
    surface_ax.set_xscale("log")
    surface_ax.set(xlim=(0.3, 20), ylim=(1e-8, 0.2))
    surface_ax.set_title(
        rf"$\lambda_{{\rm in}}={inner:.2f},\quad\lambda_{{\rm out}}={inner+increase:.2f}$"
        + "\n" + rf"$R_b={np.exp(log_break):.2f}$ kpc, $\alpha={np.exp(log_alpha):.2f}$")
    residual_ax.plot(fit_radius, residual, color="black", linewidth=1)
    residual_ax.axhline(0, color="#AA4499", linewidth=1)
    residual_ax.set(xlim=(args.r_min, args.r_max), xlabel=r"$R$ [kpc]",
                    ylabel=r"$\log_{10}(q_{\rm kernel}/q_{\rm fit})$")
    residual_ax.text(0.02, 0.93, f"Fit range: {args.r_min:g}–{args.r_max:g} kpc; RMS = {summary['rms_residual_dex']:.3f} dex",
                     transform=residual_ax.transAxes, va="top", fontsize=9,
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    residual_ax.tick_params(which="both", direction="in", top=True, right=True)

    figure.savefig(output / "fit.pdf", format="pdf")
    plt.close(figure)
    print(json.dumps(summary, indent=2))
    print(f"Wrote fit.json, profile.csv, and fit.pdf in {output}")


if __name__ == "__main__":
    main()
