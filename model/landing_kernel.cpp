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

void build_kernel(const std::vector<LandingPoint>& landings,
                  const DoubleVec& R_bins, const DoubleVec& tau_bins,
                  DoubleVec& K_wind, DoubleVec& K_mass, DoubleVec& K_j) {
    validate_edges(R_bins);
    validate_edges(tau_bins);
    const std::size_t n_R = R_bins.size() - 1;
    const std::size_t n_tau = tau_bins.size() - 1;
    K_wind.assign(n_R * n_tau, 0.0);
    K_mass.assign(n_R * n_tau, 0.0);
    K_j.assign(n_R * n_tau, 0.0);

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
            !std::isfinite(landing.tau) || landing.tau < 0.0 ||
            !std::isfinite(landing.mass_ratio) ||
            !std::isfinite(landing.j_land)) {
            throw std::invalid_argument("Returned parcels need finite, nonnegative radius and delay");
        }
        const std::size_t i = bin_index(landing.R_land, R_bins);
        const std::size_t j = bin_index(landing.tau, tau_bins);
        if (i < n_R && j < n_tau) {
            const std::size_t cell = i * n_tau + j;
            K_wind[cell] += landing.weight;
            K_mass[cell] += landing.weight * landing.mass_ratio;
            K_j[cell] += landing.weight * landing.mass_ratio * landing.j_land;
        }
    }
    if (!std::isfinite(total_weight) || total_weight <= 0.0) {
        throw std::invalid_argument("Total launched weight must be finite and positive");
    }

    for (std::size_t i = 0; i < n_R; ++i) {
        for (std::size_t j = 0; j < n_tau; ++j) {
            const double area = std::numbers::pi * (R_bins[i+1]-R_bins[i]) * (R_bins[i+1]+R_bins[i]);
            const double norm = total_weight * area * (tau_bins[j+1]-tau_bins[j]);
            K_wind[i*n_tau+j] /= norm;
            K_mass[i*n_tau+j] /= norm;
            K_j[i*n_tau+j] /= norm;
            if (!std::isfinite(K_wind[i*n_tau+j]) || !std::isfinite(K_mass[i*n_tau+j]) ||
                !std::isfinite(K_j[i*n_tau+j])) {
                throw std::invalid_argument("Kernel moments must be finite");
            }
        }
    }
}

DoubleVec convolve_kernel(const DoubleVec& kernel,
                              const DoubleVec& R_bins, const DoubleVec& tau_bins,
                              double t_Myr, const std::function<double(double)>& Mdot_launch) {
    const std::size_t n_R = R_bins.size() - 1;
    const std::size_t n_tau = tau_bins.size() - 1;
    if (!std::isfinite(t_Myr)) {
        throw std::invalid_argument("Landing time must be finite");
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
        double source = 0.0;
        for (std::size_t j = 0; j < n_tau; ++j) {
            const double density = kernel[i * n_tau + j];
            source += density * (tau_bins[j + 1] - tau_bins[j]) * launch_rates[j];
        }
        landing_rate[i] = source;
        if (!std::isfinite(source)) {
            throw std::overflow_error("Kernel landing surface rate overflowed");
        }
    }
    return landing_rate;
}
