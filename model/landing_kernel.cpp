#include "landing_kernel.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace {

void validate_edges(const DoubleVec& edges) {
    if (edges.size() < 2) {
        throw std::invalid_argument("Kernel bins need at least two edges");
    }
    for (std::size_t i = 0; i < edges.size(); ++i) {
        if (!std::isfinite(edges[i]) || edges[i] < 0.0 ||
            (i > 0 && edges[i] <= edges[i - 1])) {
            throw std::invalid_argument("Kernel edges must be finite, nonnegative and strictly increasing");
        }
    }
}

std::size_t bin_index(double value, const DoubleVec& edges) {
    const std::size_t count = edges.size() - 1;
    if (value < edges.front() || value > edges.back()) {
        return count; // outside the grid
    }
    if (value == edges.back()) {
        return count - 1;
    }
    return static_cast<std::size_t>(std::upper_bound(edges.begin(), edges.end(), value) - edges.begin() - 1);
}

}

DoubleVec build_kernel(const std::vector<LandingPoint>& landings,
                       const DoubleVec& R_bins, const DoubleVec& tau_bins) {
    validate_edges(R_bins);
    validate_edges(tau_bins);
    const std::size_t n_R = R_bins.size() - 1;
    const std::size_t n_tau = tau_bins.size() - 1;
    DoubleVec kernel(n_R * n_tau, 0.0);

    double total_weight = 0.0;
    for (const LandingPoint& landing : landings) {
        if (!std::isfinite(landing.weight) || landing.weight < 0.0) {
            throw std::invalid_argument("Launch weights must be finite and nonnegative");
        }
        total_weight += landing.weight;
        if (!landing.returned) {
            continue;
        }
        if (!std::isfinite(landing.R_land) || landing.R_land < 0.0 ||
            !std::isfinite(landing.tau) || landing.tau < 0.0) {
            throw std::invalid_argument("Returned parcels need finite, nonnegative radius and delay");
        }
        const std::size_t i = bin_index(landing.R_land, R_bins);
        const std::size_t j = bin_index(landing.tau, tau_bins);
        if (i < n_R && j < n_tau) {
            kernel[i * n_tau + j] += landing.weight;
        }
    }
    if (!std::isfinite(total_weight) || total_weight <= 0.0) {
        throw std::invalid_argument("Total launched weight must be finite and positive");
    }

    for (std::size_t i = 0; i < n_R; ++i) {
        for (std::size_t j = 0; j < n_tau; ++j) {
            kernel[i * n_tau + j] /= total_weight;
            kernel[i * n_tau + j] /= (R_bins[i + 1] - R_bins[i]) * (tau_bins[j + 1] - tau_bins[j]);
        }
    }
    return kernel;
}

DoubleVec sigmadot_land_kernel(const DoubleVec& kernel,
                              const DoubleVec& R_bins, const DoubleVec& tau_bins,
                              double t_Myr, const std::function<double(double)>& Mdot_launch) {
    validate_edges(R_bins);
    validate_edges(tau_bins);
    const std::size_t n_R = R_bins.size() - 1;
    const std::size_t n_tau = tau_bins.size() - 1;
    if (kernel.size() / n_tau != n_R || kernel.size() % n_tau != 0) {
        throw std::invalid_argument("Kernel dimensions must match the radial and delay bins");
    }
    if (!std::isfinite(t_Myr) || !Mdot_launch) {
        throw std::invalid_argument("Landing time must be finite and launch history must be callable");
    }
    DoubleVec launch_rates(n_tau);
    for (std::size_t j = 0; j < n_tau; ++j) {
        const double tau_mid = tau_bins[j] + 0.5 * (tau_bins[j + 1] - tau_bins[j]);
        const double launch_time = t_Myr - tau_mid;
        if (!std::isfinite(launch_time)) {
            throw std::invalid_argument("Retarded launch time must be finite");
        }
        launch_rates[j] = Mdot_launch(launch_time);
        if (!std::isfinite(launch_rates[j]) || launch_rates[j] < 0.0) {
            throw std::invalid_argument("Launch history must return finite nonnegative mass rates");
        }
    }

    DoubleVec landing_rate(n_R, 0.0);
    for (std::size_t i = 0; i < n_R; ++i) {
        const double dR = R_bins[i + 1] - R_bins[i];
        // Factored difference of squares also works for the central bin R_lo = 0.
        const double area = std::numbers::pi * dR * (R_bins[i + 1] + R_bins[i]);
        if (!std::isfinite(area) || area <= 0.0) {
            throw std::invalid_argument("Receiving annulus area must be finite and positive");
        }
        double annulus_rate = 0.0;
        for (std::size_t j = 0; j < n_tau; ++j) {
            const double density = kernel[i * n_tau + j];
            if (!std::isfinite(density) || density < 0.0) {
                throw std::invalid_argument("Kernel density must be finite and nonnegative");
            }
            annulus_rate += density * dR * (tau_bins[j + 1] - tau_bins[j]) * launch_rates[j];
        }
        landing_rate[i] = annulus_rate / area;
        if (!std::isfinite(annulus_rate) || !std::isfinite(landing_rate[i])) {
            throw std::overflow_error("Kernel landing surface rate overflowed");
        }
    }
    return landing_rate;
}

DoubleVec sigmadot_land_kernel(const DoubleVec& kernel,
                              const DoubleVec& R_bins, const DoubleVec& tau_bins,
                              double Mdot_launch) {
    return sigmadot_land_kernel(kernel, R_bins, tau_bins, 0.0,
                               [=](double) { return Mdot_launch; });
}
