"""Display-only weighted abundance spline and its exact logarithmic gradient.

Fit the measured 12+log10(O/H), using the adopted uncertainties (including
the existing intrinsic-scatter term). GCV selects the cubic spline's smoothing.
This comparison never replaces the saved observed profiles used by a model.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_smoothing_spline
from matplotlib.ticker import MaxNLocator

from observed_disk_comparison import FIT_STYLE


@dataclass(frozen=True)
class MetallicitySpline:
    radius: np.ndarray
    logarithmic_abundance: BSpline
    measurement_count: int
    distinct_radius_count: int

    @property
    def abundance_solar(self):
        return 10 ** (self.logarithmic_abundance(self.radius) - 8.69)

    @property
    def gradient_dex_kpc(self):
        # log10(Z/Z_sun) = 12+log10(O/H) - 8.69.
        return self.logarithmic_abundance.derivative()(self.radius)


def fit_metallicity_spline(rows):
    """Combine coincident radii, then fit with inverse-variance weights."""
    radius = np.asarray([float(row["R_kpc"]) for row in rows])
    abundance = np.asarray([float(row["OH12"]) for row in rows])
    sigma = np.asarray([float(row["sigma_OH12_dex"]) for row in rows])
    if not (np.all(np.isfinite(radius)) and np.all(np.isfinite(abundance))
            and np.all(np.isfinite(sigma) & (sigma > 0))):
        raise ValueError("Abundance spline requires finite data and positive uncertainties")
    # Catalog radii differing only through floating-point roundoff describe
    # the same radius. Preserve their total inverse-variance weight.
    unique, inverse = np.unique(np.round(radius, 10), return_inverse=True)
    weight = sigma ** -2
    total_weight = np.bincount(inverse, weights=weight)
    mean = np.bincount(inverse, weights=weight * abundance) / total_weight
    spline = make_smoothing_spline(unique, mean, w=total_weight, lam=None)
    return MetallicitySpline(np.linspace(unique[0], unique[-1], 750), spline, len(rows), len(unique))


def plot_metallicity_spline(axes, comparison, label):
    """Use one spline for both Z/Z_sun and dlog10(Z)/dR, without extrapolation."""
    style = {**FIT_STYLE, "linewidth": 1.2, "zorder": 8}
    abundance, = axes[0].plot(comparison.radius, comparison.abundance_solar, label=label, **style)
    gradient, = axes[1].plot(comparison.radius, comparison.gradient_dex_kpc, label=label, **style)
    return abundance, gradient


def include_spline_gradient(axis, comparison):
    """Include the negative derivative while keeping the view capped at zero."""
    gradient = comparison.gradient_dex_kpc
    padding = .05 * abs(min(gradient.min(), 0))
    lower, _ = axis.get_ylim()
    step = .05
    axis.set_ylim(min(lower, np.floor((gradient.min() - padding) / step) * step), 0)
    axis.yaxis.set_major_locator(MaxNLocator(5))
