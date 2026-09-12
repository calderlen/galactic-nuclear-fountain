#include "orbits.h"
#include "potential.h"


DoubleVec orbit(double /*t*/, // time -- currently unused b/c potential time-ind
                const DoubleVec& state, // [R, z, vR, vz]
                const void* params // keep as void* so that RK4 can be used with different parameter structs 
                ) {
        
    const OrbitParameters* p = static_cast<const OrbitParameters*>(params);
    // gives disk mass, disk scale length [kpc], disk scale height [kpc], bulge mass [M_sun], bulge scale length [kpc], halo scale density [M_sun/kpc^3], halo scale radius [kpc],angular momentum [kpc^2/Myr]

    double R = state[0];
    double z = state[1];
    double vR = state[2];
    double vz = state[3];

    double dphi_dR = dphi_mn_dR(R, z, p->M_d, p->a_d, p->b_d) + dphi_h_dR(R, z, p->M_b, p->a_b) + dphi_nfw_dR(R, z, p->rho_s, p->r_s);
    double dphi_dz = dphi_mn_dz(R, z, p->M_d, p->a_d, p->b_d) + dphi_h_dz(R, z, p->M_b, p->a_b) + dphi_nfw_dz(R, z, p->rho_s, p->r_s);
    double dvR_dt = -dphi_dR + p->j_z*p->j_z/(R*R*R);
    double dvz_dt = -dphi_dz;

    return {vR, vz, dvR_dt, dvz_dt};
    }


DoubleVec integrate_orbit(const DoubleVec& launch_conditions, //{R_0, z_0, vR_0, vz_0},
                          const OrbitParameters& potential_params, //{M_d, a_d, b_d, M_b, a_b, rho_s, r_s, j_z},
                          double h_0, //initial step size,
                          double atol, //absolute tolerance,
                          double rtol, //relative tolerance,
                          double t_stop) //maximum integration time)
                          {

    DoubleVec y_0 = launch_conditions;    

    RK4Solution solution = integrate_rk4(orbit, &potential_params, 0.0, y_0, h_0, atol, rtol, t_stop);
                            
    // detect first downward disk crossing
    for (std::size_t i = 1; i < solution.x.size(); ++i) {
        double z_prev = solution.y[i - 1][1];
        double z_now  = solution.y[i][1];
        double vz_now = solution.y[i][3];

        if (z_prev > 0.0 && z_now <= 0.0 && vz_now < 0.0) {
            double t_land = solution.x[i];
            double R_land = solution.y[i][0];
            return {R_land, t_land};
            }
        }
        throw std::runtime_error("No downward disk crossing detected within the integration time.");
    }



// An orbital integration wrapper in orbits.cpp. Something like integrate_orbit() that accepts launch conditions, calls the existing RK4 stepper, detects the first downward disk crossing, and returns landing radius and flight time. It also needs configurable tolerances, domain/time limits, outcome statuses, and energy diagnostics. Domain exits and disk returns must be resolved in chronological order.