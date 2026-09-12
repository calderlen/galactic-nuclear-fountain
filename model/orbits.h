# pragma once

# include "rk4.h"

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

DoubleVec integrate_orbit(const DoubleVec& launch_conditions,
                         const OrbitParameters& potential_params,
                         double h_0, double atol, double rtol, double t_stop);