# pragma once

# include "rk4.h"
# include <limits>

struct OrbitParameters {
    double M_d; // disk mass [M_sun]
    double a_d; // disk scale length [kpc]
    double b_d; // disk scale height [kpc]

    double M_b; // bulge mass [M_sun]
    double a_b; // bulge scale length [kpc]

    double rho_s; // halo scale density [M_sun/kpc^3]
    double r_s; // halo scale radius [kpc]

    double j_z; // angular momentum [kpc^2/Myr]
};

DoubleVec orbit(double t, const DoubleVec& state, const void* params);

enum class OrbitStatus { Returned, TimedOut, IntegrationFailed };

struct OrbitResult {
    OrbitStatus status;
    double R_land = std::numeric_limits<double>::quiet_NaN(); // [kpc], returns only
    double flight_time = std::numeric_limits<double>::quiet_NaN(); // [Myr], returns only
};

// Upward launches only: R > 0, z >= 0, vz > 0. A timeout is not an escape.
OrbitResult integrate_orbit(const DoubleVec& launch_conditions,
                            const OrbitParameters& potential_params,
                            double h_0, double atol, double rtol, double t_stop,
                            double h_max = 1.0);
