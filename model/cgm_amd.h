#pragma once

#include "orbits.h"

// The potential and all dynamical quantities use kpc, Myr and Msun.
// M_CGM only normalizes density: gas self-gravity is not included.
struct CGMAMDParameters {
    OrbitParameters potential{};
    double R200=200.0, Rcore=0.1, M200=1.2e12, M_CGM=1.e10;
    double T_K=2.e6, mu_mol=0.6, lambda_spin=0.035, fj=1.0;
    double jmax_over_jDM=4.0;
    int vertical_intervals=256;
};

struct CGMAMDSolution {
    CGMAMDParameters parameters;
    RK4Solution trajectory; // ascending ln(R/R200); state {C, ln(rho0 R200^3/M_CGM)}
    double jmax, xi, cs2, core_factor;
};

CGMAMDSolution solve_cgm_amd(const CGMAMDParameters& parameters,
    double h_low=-30.0,double h_high=8.0,
    double step_logR=0.02,double atol=1.e-10,double rtol=1.e-8);
double cgm_angular_momentum(const CGMAMDSolution& solution,double R); // kpc^2/Myr
double cgm_angular_momentum_derivative(const CGMAMDSolution& solution,double R);
double cgm_azimuthal_velocity(const CGMAMDSolution& solution,double R); // kpc/Myr
double cgm_beta(const CGMAMDSolution& solution,double R); // relative to potential vc
double cgm_rho_over_mass(const CGMAMDSolution& solution,double R,double z); // kpc^-3
double cgm_density(const CGMAMDSolution& solution,double R,double z); // Msun/kpc^3
double cgm_cumulative_fraction(const CGMAMDSolution& solution,double R);
