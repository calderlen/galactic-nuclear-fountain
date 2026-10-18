#pragma once

#include "rk4.h"

#include <functional>

struct LandingPoint {
    double R_land; // [kpc]; unused when returned is false
    double tau;    // [Myr]; unused when returned is false
    double weight; // relative launched mass, including nonreturning parcels
    bool returned = true;
    double mass_ratio = 1.0;
    double j_land = 0.0; // [kpc^2/Myr]
};

// Area/delay densities normalized by ALL original launched mass, including nonreturns.
// K_wind is the paper's K [kpc^-2 Myr^-1]; K_mass includes condensed mass;
// K_j includes mass_ratio*j_land and may be signed. All arrays use i*n_tau+j.
// Bins are [left, right), except the final bin includes its right edge.
// Requires caller-validated launch weights with finite positive total and
// returned radii, delays and angular momenta from the orbit integrator.
// Validates bin edges, mass ratios and completed moments; sizes arrays to the grid.
void build_kernel(const std::vector<LandingPoint>& landings,
                  const DoubleVec& R_bins, const DoubleVec& tau_bins,
                  DoubleVec& K_wind, DoubleVec& K_mass, DoubleVec& K_j);

// Convolve an area-normalized kernel with a launch rate in declared mass/time units.
// Source[i] = sum_j K[i,j]*dtau[j]*Mdot_launch(t-tau_mid[j]).
// Output sources carry the mass/time unit supplied by Mdot_launch, per kpc^2;
// canonical disk callers supply Msun/Gyr. K_j additionally carries kpc^2/Myr.
// Delay-bin midpoint quadrature; history must define pre-start launch rates too.
// Requires unchanged bins and kernel from build_kernel(), finite simulation times
// and a configured nonnegative history. Checks calculated rates remain finite.
DoubleVec convolve_kernel(const DoubleVec& kernel,
                          const DoubleVec& R_bins, const DoubleVec& tau_bins,
                          double t_Myr, const std::function<double(double)>& Mdot_launch);
