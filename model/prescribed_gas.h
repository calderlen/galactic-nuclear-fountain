#pragma once

#include "profiles.h"
#include "rk4.h"

// Separate closure: prescribe gas, landing and an outer inward mass flux;
// use a supplied net stellar sink or the KS law, then derive transport and
// required landing angular momentum.
// Gas: Msun/kpc^2; R: kpc; surface rates: Msun/Gyr/kpc^2;
// integrated rates: Msun/yr; rotation: km/s; j: kpc km/s.
struct PrescribedGasParameters {
    RadialProfile gas;
    RotationCurve rotation;
    double R_in, R_out, Mdot_land, lambda;
    double A_net, N, Mdot_out;
    RadialProfile sigmadot_star{}; // Supplied net sink; otherwise use A_net*gas^N.
};

struct PrescribedGasProfile {
    double gas, star, landing, cumulative_landing;
    double accretion, radial_velocity;
    double circular_velocity, circular_velocity_derivative;
    double disk_angular_momentum, required_landing_angular_momentum;
};

RK4Solution solve_prescribed_gas(const PrescribedGasParameters& parameters,
                               double h0, double atol, double rtol);
PrescribedGasProfile prescribed_gas_profile(double R, const PrescribedGasParameters& parameters,
                                           const RK4Solution& solution);

// Optional chemistry is conditional on an explicitly supplied incident
// composition. It never imposes a reservoir angular-momentum closure.
struct PrescribedGasChemistryParameters {
    double Z_land, yield, Z_inner_boundary, Z_outer_boundary;
    double h0, atol, rtol;
    std::size_t cells;
};
struct PrescribedGasChemistrySolution {
    RK4Solution ode;
    LinearProfile conservative;
    bool used_conservative = false;
    double max_cell_balance_error = 0.0;
};
PrescribedGasChemistrySolution solve_prescribed_gas_chemistry(
    const PrescribedGasParameters& flow, const RK4Solution& dynamics,
    const PrescribedGasChemistryParameters& chemistry);
// Returns Z and dZ/dR in the same units as the shared metallicity equation.
DoubleVec prescribed_gas_chemistry_profile(double R, const PrescribedGasParameters& flow,
    const RK4Solution& dynamics, const PrescribedGasChemistryParameters& chemistry,
    const PrescribedGasChemistrySolution& solution);
