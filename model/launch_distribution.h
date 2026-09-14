#pragma once

#include <random>
#include <vector>

struct LaunchDistributionParameters {
    double R_ring_kpc;
    double sigma_R_kpc;     // zero selects a fixed ring
    double h_v_kms;         // one-component Gaussian dispersion of Maxwell speed
    double sigma_theta_rad; // Gaussian per solid angle; zero selects vertical kicks
};

struct LaunchDraw {
    double R0_kpc;
    double v_k_kms;
    double theta_rad; // from +z, in [0, pi/2]
    double psi_rad;   // kick azimuth, in [0, 2*pi)
};

struct LaunchDistribution {
    LaunchDistributionParameters parameters;
    std::vector<double> radius_grid;
    std::vector<double> radius_cdf;
};

// p(R) proportional to R exp[-(R-R_ring)^2/(2 sigma_R^2)], R > 0.
// Tabulate once over R >= 0 and within eight sigma of the ring center.
LaunchDistribution make_launch_distribution(const LaunchDistributionParameters& parameters);
LaunchDraw sample_launch(const LaunchDistribution& distribution, std::mt19937_64& rng);
