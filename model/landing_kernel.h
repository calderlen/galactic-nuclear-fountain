#pragma once

#include "rk4.h"

#include <functional>

struct LandingPoint {
    double R_land; // [kpc]; unused when returned is false
    double tau;    // [Myr]; unused when returned is false
    double weight; // relative launched mass, including nonreturning parcels
    bool returned = true;
};

// R_bins and tau_bins are bin edges. K[i*n_tau+j] is a density per
// kpc per Myr, normalized by ALL launched weight. Thus sum(K*dR*dtau)
// is the fraction of launched mass that returned inside the supplied bins.
// Bins are [left, right), except the final bin includes its right edge.
DoubleVec build_kernel(const std::vector<LandingPoint>& landings,
                       const DoubleVec& R_bins, const DoubleVec& tau_bins);

// Landing surface rate at t_Myr, averaged over each annulus [Msun/yr/kpc^2].
// Mdot_launch(time_Myr) returns Msun/yr; kernel uses the build_kernel convention.
// Sigmadot[i](t) = sum_j(K[i,j] * dR[i] * dtau[j] * Mdot_launch(t-tau_mid[j])) / area[i].
// Uses delay-bin midpoints; resolve variations in the launch law with the delay grid.
// Delay widths are in Myr, cancelling K's Myr^-1: no year conversion is needed.
// Retains the in-grid return fraction; no extra return or CGM-mixing factor.
// The history defines rates at all requested times, including negative times;
// supply a history returning zero before its onset if launches have a start time.
DoubleVec sigmadot_land_kernel(const DoubleVec& kernel,
                              const DoubleVec& R_bins, const DoubleVec& tau_bins,
                              double t_Myr, const std::function<double(double)>& Mdot_launch);

// Constant-history convenience overload, evaluated through the same convolution.
DoubleVec sigmadot_land_kernel(const DoubleVec& kernel,
                              const DoubleVec& R_bins, const DoubleVec& tau_bins,
                              double Mdot_launch);
