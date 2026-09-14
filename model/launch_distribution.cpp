#include "launch_distribution.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

LaunchDistribution make_launch_distribution(const LaunchDistributionParameters& p) {
    if (!std::isfinite(p.R_ring_kpc) || !std::isfinite(p.sigma_R_kpc) ||
        !std::isfinite(p.h_v_kms) || !std::isfinite(p.sigma_theta_rad) ||
        p.R_ring_kpc < 0.0 || p.sigma_R_kpc < 0.0 || p.h_v_kms <= 0.0 ||
        p.sigma_theta_rad < 0.0 || (p.R_ring_kpc == 0.0 && p.sigma_R_kpc == 0.0)) {
        throw std::invalid_argument("Invalid launch distribution parameters");
    }
    LaunchDistribution distribution{p, {}, {}};
    if (p.sigma_R_kpc == 0.0) return distribution;

    constexpr int intervals = 4096;
    const double lower = std::max(0.0, p.R_ring_kpc - 8.0 * p.sigma_R_kpc);
    const double upper = p.R_ring_kpc + 8.0 * p.sigma_R_kpc;
    if (!std::isfinite(upper) || upper <= lower) {
        throw std::invalid_argument("Radial distribution has an unresolvable width");
    }
    double previous_pdf = 0.0;
    double integral = 0.0;
    for (int i = 0; i <= intervals; ++i) {
        const double radius = lower + (upper - lower) * i / intervals;
        const double x = (radius - p.R_ring_kpc) / p.sigma_R_kpc;
        const double pdf = radius * std::exp(-0.5 * x * x);
        if (i > 0) {
            if (radius <= distribution.radius_grid.back()) {
                throw std::invalid_argument("Radial CDF grid is not resolved in floating point");
            }
            integral += 0.5 * (previous_pdf + pdf) * (radius - distribution.radius_grid.back());
        }
        distribution.radius_grid.push_back(radius);
        distribution.radius_cdf.push_back(integral);
        previous_pdf = pdf;
    }
    if (!std::isfinite(integral) || integral <= 0.0) {
        throw std::invalid_argument("Radial distribution cannot be normalized");
    }
    for (double& value : distribution.radius_cdf) value /= integral;
    return distribution;
}

LaunchDraw sample_launch(const LaunchDistribution& distribution, std::mt19937_64& rng) {
    const auto& p = distribution.parameters;
    constexpr double pi = std::numbers::pi;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    // Open interval avoids log(0) and a zero-radius endpoint draw.
    const auto unit = [&]() {
        double value;
        do { value = uniform(rng); } while (value == 0.0);
        return value;
    };
    double radius = p.R_ring_kpc;
    if (p.sigma_R_kpc > 0.0) {
        const double u = unit();
        const auto& cdf = distribution.radius_cdf;
        const auto& grid = distribution.radius_grid;
        const std::size_t right = std::upper_bound(cdf.begin(), cdf.end(), u) - cdf.begin();
        const double fraction = (u - cdf[right - 1]) / (cdf[right] - cdf[right - 1]);
        radius = grid[right - 1] + fraction * (grid[right] - grid[right - 1]);
    }

    std::normal_distribution<double> normal(0.0, 1.0);
    const double g1 = normal(rng), g2 = normal(rng), g3 = normal(rng);
    const double speed = p.h_v_kms * std::hypot(g1, g2, g3);

    double theta = 0.0;
    if (p.sigma_theta_rad > 0.0) {
        while (true) {
            if (p.sigma_theta_rad < 1.0) {
                // Rayleigh proposal: theta exp(-theta^2/(2 sigma^2)).
                theta = p.sigma_theta_rad * std::sqrt(-2.0 * std::log(unit()));
                if (theta >= pi / 2.0) continue;
                if (unit() <= (theta == 0.0 ? 1.0 : std::sin(theta) / theta)) break;
            } else {
                // Uniform solid-angle proposal is efficient for broad cones.
                theta = std::acos(unit());
                const double x = theta / p.sigma_theta_rad;
                if (unit() <= std::exp(-0.5 * x * x)) break;
            }
        }
    }
    return {radius, speed, theta, 2.0 * pi * uniform(rng)};
}
