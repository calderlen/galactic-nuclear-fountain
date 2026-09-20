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

    double rho_CGM = 0.0; // constant CGM density [M_sun/kpc^3]
    double sigma = 0.0;   // fixed parcel cross-section [kpc^2]
    double C_D = 0.0;
    double epsilon = 0.0;
    double beta = 0.0;    // v_phi,CGM / v_c(R)
};

double circular_velocity(double R, const OrbitParameters& parameters); // [kpc/Myr]
DoubleVec orbit(double t, const DoubleVec& state, const void* params);

enum class OrbitStatus { Returned, TimedOut, IntegrationFailed };

struct OrbitResult {
    OrbitStatus status;
    double R_land = std::numeric_limits<double>::quiet_NaN(); // [kpc], returns only
    double flight_time = std::numeric_limits<double>::quiet_NaN(); // [Myr], returns only
    double j_land = std::numeric_limits<double>::quiet_NaN(); // [kpc^2/Myr]
    double mass_ratio = std::numeric_limits<double>::quiet_NaN(); // m_land/m_initial
};

struct OrbitSample {
    double t; // [Myr]
    DoubleVec state; // [R, z, vR, vz, j_z, m]
};

struct OrbitTrace {
    OrbitResult result;
    std::vector<OrbitSample> samples;
};

// State: [R, z, vR, vz, j_z, m], in kpc, Myr and M_sun.
// Upward launches only: R > 0, z >= 0, vz > 0, m > 0. A timeout is not an escape.
OrbitResult integrate_orbit(const DoubleVec& launch_conditions,
                            const OrbitParameters& potential_params,
                            double h_0, double atol, double rtol, double t_stop,
                            double h_max = 1.0);

// Re-run one launch with the same adaptive integrator, sampling its accepted
// Hermite interpolant at a regular cadence and retaining the exact final state.
OrbitTrace integrate_orbit_trace(const DoubleVec& launch_conditions,
                                 const OrbitParameters& potential_params,
                                 double h_0, double atol, double rtol,
                                 double t_stop, double output_dt,
                                 double h_max = 1.0);
