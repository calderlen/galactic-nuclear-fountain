#include "orbits.h"
#include "potential.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

    // Cylindrical coordinates are singular on the axis. Reject and retry a
    // smaller step if an intermediate RK stage leaves the R > 0 domain.
    if (!std::isfinite(R) || R <= 0.0) {
        throw std::domain_error("Orbit reached the cylindrical axis");
    }

    double dphi_dR = dphi_mn_dR(R, z, p->M_d, p->a_d, p->b_d) + dphi_h_dR(R, z, p->M_b, p->a_b) + dphi_nfw_dR(R, z, p->rho_s, p->r_s);
    double dphi_dz = dphi_mn_dz(R, z, p->M_d, p->a_d, p->b_d) + dphi_h_dz(R, z, p->M_b, p->a_b) + dphi_nfw_dz(R, z, p->rho_s, p->r_s);
    double dvR_dt = -dphi_dR + p->j_z*p->j_z/(R*R*R);
    double dvz_dt = -dphi_dz;

    return {vR, vz, dvR_dt, dvz_dt};
    }


OrbitResult integrate_orbit(const DoubleVec& launch_conditions,
                            const OrbitParameters& potential_params,
                            double h_0, double atol, double rtol, double t_stop,
                            double h_max) {
    const auto finite = [](const DoubleVec& values) {
        return std::all_of(values.begin(), values.end(),
                           [](double value) { return std::isfinite(value); });
    };
    const auto& p = potential_params;
    if (launch_conditions.size() != 4 || !finite(launch_conditions) ||
        launch_conditions[0] <= 0.0 || launch_conditions[1] < 0.0 ||
        launch_conditions[3] <= 0.0 ||
        !finite({h_0, atol, rtol, t_stop, h_max, p.M_d, p.a_d, p.b_d,
                 p.M_b, p.a_b, p.rho_s, p.r_s, p.j_z}) ||
        h_0 <= 0.0 || atol <= 0.0 || rtol < 0.0 || t_stop <= 0.0 || h_max <= 0.0 ||
        p.M_d < 0.0 || p.a_d < 0.0 || p.b_d <= 0.0 || p.M_b < 0.0 ||
        p.a_b < 0.0 || p.rho_s < 0.0 || p.r_s <= 0.0) {
        throw std::invalid_argument("Invalid upward launch, potential, or orbit tolerances");
    }

    DoubleVec y = launch_conditions;
    double t = 0.0;
    double h = std::min(h_0, h_max);
    constexpr int max_attempts = 100000;
    for (int attempt = 0; attempt < max_attempts && t < t_stop; ++attempt) {
        // Resolve motion near the axis as well as imposing an absolute step cap.
        const double speed = std::hypot(std::hypot(y[2], y[3]), p.j_z / y[0]);
        h = std::min({h, h_max, t_stop - t, 0.1 * y[0] / speed});
        if (!std::isfinite(h) || h <= 0.0 || t + h == t) {
            return {OrbitStatus::IntegrationFailed};
        }

        RK4Step step;
        try {
            step = rk4_step_doubling(orbit, &p, t, y, h);
        } catch (const std::domain_error&) {
            h *= 0.5;
            continue;
        }
        if (!finite(step.y) || !finite(step.error) || step.y[0] <= 0.0) {
            h *= 0.5;
            continue;
        }
        double error_squared = 0.0;
        for (std::size_t i = 0; i < y.size(); ++i) {
            const double scale = atol + rtol * std::max(std::abs(y[i]), std::abs(step.y[i]));
            const double ratio = step.error[i] / scale;
            error_squared += ratio * ratio;
        }
        const double error = std::sqrt(error_squared / y.size());
        if (error <= 1.0) {
            const double next_t = h == t_stop - t ? t_stop : t + h;
            if (y[1] > 0.0 && step.y[1] <= 0.0) {
                // Use the existing Hermite interpolator on this accepted step.
                const RK4Solution bracket{{t, next_t}, {y, step.y},
                    {orbit(t, y, &p), orbit(next_t, step.y, &p)}};
                double left = t;
                double right = next_t;
                for (int iteration = 0; iteration < 60; ++iteration) {
                    const double middle = left + (right - left) / 2.0;
                    if (middle == left || middle == right) break;
                    if (evaluate(bracket, middle)[1] > 0.0) left = middle;
                    else right = middle;
                }
                const double landing_time = left + (right - left) / 2.0;
                const DoubleVec landing = evaluate(bracket, landing_time);
                if (!finite(landing) || landing[0] <= 0.0 || landing[3] >= 0.0) {
                    return {OrbitStatus::IntegrationFailed};
                }
                return {OrbitStatus::Returned, landing[0], landing_time};
            }
            t = next_t;
            y = step.y;
        }
        h *= error == 0.0 ? 5.0 : std::clamp(0.9 * std::pow(error, -0.2), 0.2, 5.0);
    }
    return {t >= t_stop ? OrbitStatus::TimedOut : OrbitStatus::IntegrationFailed};
}
