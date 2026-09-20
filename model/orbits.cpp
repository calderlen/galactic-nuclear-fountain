#include "orbits.h"
#include "potential.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

double circular_velocity(double R, const OrbitParameters& p) {
    const double v_squared = R * (
        dphi_mn_dR(R, 0.0, p.M_d, p.a_d, p.b_d) +
        dphi_h_dR(R, 0.0, p.M_b, p.a_b) +
        dphi_nfw_dR(R, 0.0, p.rho_s, p.r_s));
    if (!std::isfinite(v_squared) || v_squared < 0.0) {
        throw std::invalid_argument("Potential gives an invalid circular velocity");
    }
    return std::sqrt(v_squared);
}

DoubleVec orbit(double /*t*/, // fixed potential and CGM
                const DoubleVec& state, // [R, z, vR, vz, j_z, m]
                const void* params // keep as void* so that RK4 can be used with different parameter structs 
                ) {
        
    const OrbitParameters* p = static_cast<const OrbitParameters*>(params);

    double R = state[0];
    double z = state[1];
    double vR = state[2];
    double vz = state[3];
    double j_z = state[4];
    double mass = state[5];

    // Cylindrical coordinates are singular on the axis. Reject and retry a
    // smaller step if an intermediate RK stage leaves the R > 0 domain.
    if (!std::isfinite(R) || R <= 0.0 || !std::isfinite(mass)) {
        throw std::domain_error("Orbit reached an invalid cylindrical radius or parcel mass");
    }

    double dphi_dR = dphi_mn_dR(R, z, p->M_d, p->a_d, p->b_d) + dphi_h_dR(R, z, p->M_b, p->a_b) + dphi_nfw_dR(R, z, p->rho_s, p->r_s);
    double dphi_dz = dphi_mn_dz(R, z, p->M_d, p->a_d, p->b_d) + dphi_h_dz(R, z, p->M_b, p->a_b) + dphi_nfw_dz(R, z, p->rho_s, p->r_s);
    const double v_c = circular_velocity(R, *p);
    const double u = std::hypot(vR, vz, j_z/R - p->beta*v_c);
    const double swept_mass_rate = p->rho_CGM * p->sigma * u;
    const double Gamma = (p->epsilon + 0.5*p->C_D) * swept_mass_rate / mass;
    double dvR_dt = -dphi_dR + j_z*j_z/(R*R*R) - Gamma*vR;
    double dvz_dt = -dphi_dz - Gamma*vz;

    return {vR, vz, dvR_dt, dvz_dt, Gamma*(p->beta*R*v_c-j_z),
            p->epsilon*swept_mass_rate};
    }


namespace {

void append_terminal_sample(std::vector<OrbitSample>* samples, double t,
                            const DoubleVec& state) {
    if (samples == nullptr) return;
    const double tolerance = 16.0 * std::numeric_limits<double>::epsilon() *
        std::max({std::abs(t), 1.0});
    if (!samples->empty() && std::abs(samples->back().t - t) <= tolerance) {
        samples->back() = {t, state};
    } else {
        samples->push_back({t, state});
    }
}

OrbitResult integrate_orbit_impl(const DoubleVec& launch_conditions,
                                 const OrbitParameters& potential_params,
                                 double h_0, double atol, double rtol,
                                 double t_stop, double h_max,
                                 double output_dt,
                                 std::vector<OrbitSample>* samples) {
    const auto finite = [](const DoubleVec& values) {
        return std::all_of(values.begin(), values.end(),
                           [](double value) { return std::isfinite(value); });
    };
    const auto& p = potential_params;
    if (launch_conditions.size() != 6 || !finite(launch_conditions) ||
        launch_conditions[0] <= 0.0 || launch_conditions[1] < 0.0 ||
        launch_conditions[3] <= 0.0 ||
        !finite({h_0, atol, rtol, t_stop, h_max, p.M_d, p.a_d, p.b_d,
                 p.M_b, p.a_b, p.rho_s, p.r_s, p.rho_CGM, p.sigma, p.C_D, p.epsilon, p.beta}) ||
        h_0 <= 0.0 || atol <= 0.0 || rtol < 0.0 || t_stop <= 0.0 || h_max <= 0.0 ||
        p.M_d < 0.0 || p.a_d < 0.0 || p.b_d <= 0.0 || p.M_b < 0.0 ||
        p.a_b < 0.0 || p.rho_s < 0.0 || p.r_s <= 0.0 ||
        (samples != nullptr && !std::isfinite(output_dt))) {
        throw std::invalid_argument("Invalid upward launch, potential, or orbit tolerances");
    }

    DoubleVec y = launch_conditions;
    double t = 0.0;
    double h = std::min(h_0, h_max);
    std::size_t sample_index = 1;
    if (samples != nullptr) {
        samples->clear();
        samples->push_back({t, y});
    }
    const auto append_regular_samples = [&](const RK4Solution& bracket,
                                            double segment_end) {
        if (samples == nullptr) return;
        const double tolerance = 16.0 * std::numeric_limits<double>::epsilon() *
            std::max({std::abs(segment_end), 1.0});
        while (true) {
            const double sample_time = output_dt * static_cast<double>(sample_index);
            if (!std::isfinite(sample_time) || sample_time > segment_end + tolerance) break;
            const double query_time = std::min(sample_time, segment_end);
            samples->push_back({query_time, evaluate(bracket, query_time)});
            ++sample_index;
        }
    };
    constexpr int max_attempts = 100000;
    for (int attempt = 0; attempt < max_attempts && t < t_stop; ++attempt) {
        // Resolve motion near the axis as well as imposing an absolute step cap.
        const double speed = std::hypot(y[2], y[3], y[4] / y[0]);
        h = std::min({h, h_max, t_stop - t, 0.1 * y[0] / speed});
        if (!std::isfinite(h) || h <= 0.0 || t + h == t) {
            append_terminal_sample(samples, t, y);
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
            const bool returned = y[1] > 0.0 && step.y[1] <= 0.0;
            RK4Solution bracket;
            if (returned || samples != nullptr) {
                bracket = {{t, next_t}, {y, step.y},
                           {orbit(t, y, &p), orbit(next_t, step.y, &p)}};
            }
            if (returned) {
                // Locate the first downward crossing on this accepted step.
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
                // Preserve the exact constant-mass limit through Hermite interpolation.
                const double mass_ratio=p.epsilon*p.rho_CGM*p.sigma==0.0 ? 1.0 : landing[5]/launch_conditions[5];
                if (!finite(landing) || landing[0] <= 0.0 || landing[3] >= 0.0) {
                    append_terminal_sample(samples, t, y);
                    return {OrbitStatus::IntegrationFailed};
                }
                append_regular_samples(bracket, landing_time);
                append_terminal_sample(samples, landing_time, landing);
                return {OrbitStatus::Returned, landing[0], landing_time, landing[4],
                        mass_ratio};
            }
            append_regular_samples(bracket, next_t);
            t = next_t;
            y = step.y;
        }
        h *= error == 0.0 ? 5.0 : std::clamp(0.9 * std::pow(error, -0.2), 0.2, 5.0);
    }
    append_terminal_sample(samples, t, y);
    return {t >= t_stop ? OrbitStatus::TimedOut : OrbitStatus::IntegrationFailed};
}

}

OrbitResult integrate_orbit(const DoubleVec& launch_conditions,
                            const OrbitParameters& potential_params,
                            double h_0, double atol, double rtol, double t_stop,
                            double h_max) {
    return integrate_orbit_impl(launch_conditions, potential_params, h_0, atol,
                                rtol, t_stop, h_max, 0.0, nullptr);
}

OrbitTrace integrate_orbit_trace(const DoubleVec& launch_conditions,
                                 const OrbitParameters& potential_params,
                                 double h_0, double atol, double rtol,
                                 double t_stop, double output_dt,
                                 double h_max) {
    OrbitTrace trace;
    trace.result = integrate_orbit_impl(launch_conditions, potential_params,
                                        h_0, atol, rtol, t_stop, h_max,
                                        output_dt, &trace.samples);
    return trace;
}
