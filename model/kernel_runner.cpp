#include "landing_kernel.h"
#include "launch_distribution.h"
#include "models.h"
#include "csv.h"
#include "orbits.h"
#include "physics.h"
#include "potential.h"

#include <algorithm>
#include <cmath>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace {

std::uint64_t unsigned_argument(const std::string& text) {
    std::uint64_t value;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument("Expected an unsigned integer, got: " + text);
    }
    return value;
}

double finite_argument(const std::string& text) {
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("Expected a finite number, got: " + text);
    }
    return value;
}

struct BurstHistory {
    double Mdot_0, Mdot_b, t_b, sigma_b, eta;
    double period = 0.0;

    double operator()(double t_Myr) const {
        if (period == 0.0) {
            return mdot_launch(mdot_nuc_burst(t_Myr, Mdot_0, Mdot_b, t_b, sigma_b), eta);
        }
        // Sum the neighboring bursts, including overlaps and the pre-zero history.
        const double phase = std::remainder(t_Myr - t_b, period);
        const int neighbors = static_cast<int>(std::ceil(8.0 * sigma_b / period));
        double rate = Mdot_0;
        for (int k = -neighbors; k <= neighbors; ++k) {
            const double offset = (phase - k * period) / sigma_b;
            rate += Mdot_b * std::exp(-0.5 * offset * offset);
        }
        return mdot_launch(rate, eta);
    }
};

DoubleVec read_initial_disk(const std::string& filename, MassContinuityParameters& p) {
    namespace csv = galactic_nuclear_fountain::csv;
    const auto table = csv::read(filename);
    const std::size_t n = p.R_bins.size() - 1;
    if (table.rows.size() != n) {
        throw std::invalid_argument("Initial disk must have one row per kernel annulus");
    }
    if (std::find(table.header.begin(), table.header.end(), "Z") == table.header.end()) {
        throw std::invalid_argument("Initial disk CSV needs a Z column containing the initial metal mass fraction in each annulus");
    }
    DoubleVec state(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& row = table.rows[i];
        const auto number = [&](const char* key) { return finite_argument(csv::value(row, key)); };
        if (number("R_lo_kpc") != p.R_bins[i] || number("R_hi_kpc") != p.R_bins[i + 1]) {
            throw std::invalid_argument("Initial disk rows must match the ordered kernel annulus edges");
        }
        const double gas = number("Sigma_g_Msun_kpc2");
        const double star = number("Sigmadot_star_Msun_yr_kpc2");
        const double Z = number("Z");
        if (gas < 0.0 || star < 0.0) {
            throw std::invalid_argument("Initial gas density and star-formation sink must be nonnegative");
        }
        if (Z < 0.0 || Z > 1.0) throw std::invalid_argument("Initial Z must be a metal mass fraction between zero and one");
        state[i] = gas;
        state[n + i] = gas * Z;
        p.sigmadot_star.push_back(star);
    }
    return state;
}

const char* status_name(OrbitStatus status) {
    switch (status) {
        case OrbitStatus::Returned: return "returned";
        case OrbitStatus::TimedOut: return "timed_out";
        case OrbitStatus::IntegrationFailed: return "integration_failed";
    }
    throw std::logic_error("Unknown orbit status");
}

double circular_velocity(double R, const OrbitParameters& p) {
    const double v_squared = R * (
        dphi_mn_dR(R, 0.0, p.M_d, p.a_d, p.b_d) +
        dphi_h_dR(R, 0.0, p.M_b, p.a_b) +
        dphi_nfw_dR(R, 0.0, p.rho_s, p.r_s));
    if (!std::isfinite(v_squared) || v_squared < 0.0) {
        throw std::invalid_argument("Potential gives an invalid circular velocity");
    }
    return std::sqrt(v_squared); // [kpc/Myr]
}

double disk_circular_velocity(double R, const void* parameters) {
    return circular_velocity(R, *static_cast<const OrbitParameters*>(parameters));
}

double disk_circular_velocity_derivative(double R, const void* parameters) {
    // The disk uses the same fixed potential as the ballistic orbits.
    const double h = 1e-4 * R;
    return (disk_circular_velocity(R + h, parameters) - disk_circular_velocity(R - h, parameters)) / (2.0 * h);
}

}

int main(int argc, char* argv[]) {
    std::string period_argument, tau_bins_argument = "100";
    while (argc >= 3) {
        const std::string option = argv[argc - 2];
        if (option == "--period") period_argument = argv[argc - 1];
        else if (option == "--tau-bins") tau_bins_argument = argv[argc - 1];
        else break;
        argc -= 2;
    }
    const bool chemistry_options = argc >= 5 && std::string(argv[argc - 4]) == "--chemistry";
    const int chemistry_argument = chemistry_options ? argc - 3 : 0;
    if (chemistry_options) argc -= 4;
    const bool evolve_mass = argc >= 5 && std::string(argv[argc - 4]) == "--evolve";
    const std::string disk_file = evolve_mass ? argv[argc - 3] : "";
    const std::string mu_argument = evolve_mass ? argv[argc - 2] : "0";
    const std::string beta_argument = evolve_mass ? argv[argc - 1] : "0";
    if (evolve_mass) argc -= 4;
    const bool burst_mode = argc > 4 && std::string(argv[4]) == "--burst";
    const bool valid_arguments = (burst_mode ? argc == 13 : argc <= 5) && (!evolve_mass || burst_mode)
                                 && (!chemistry_options || evolve_mass) && (period_argument.empty() || burst_mode);
    if (!valid_arguments || (argc > 1 && std::string(argv[1]) == "--help")) {
        std::cout << "Usage: " << argv[0] << " [output-directory [number-of-parcels [seed [Mdot-launch-Msun-yr]]]]\n"
                  << "   or: " << argv[0] << " output-directory number-of-parcels seed --burst Mdot_0 Mdot_b t_b sigma_b eta t_min t_max dt [--evolve disk.csv mu beta [--chemistry Z_nucl Z_CGM yield_y]] [--period T] [--tau-bins N]\n"
                  << "Defaults: output/kernel/sampled, 10000 parcels, seed 42, launch rate 1 Msun/yr.\n"
                  << "Burst: Mdot_launch(t) = eta * mdot_nuc_burst(t, Mdot_0, Mdot_b, t_b, sigma_b).\n"
                  << "Nuclear rates are Msun/yr, all times are Myr, and eta is dimensionless.\n"
                  << "The Gaussian plus baseline extends to negative times; no onset cutoff is imposed.\n"
                  << "--period repeats bursts every T Myr, with t_b identifying a reference peak, not the first burst.\n"
                  << "--tau-bins sets the number of uniform flight-delay bins over 0--1000 Myr (default 100).\n"
                  << "Append --period and --tau-bins after any evolution/chemistry options. No stellar-feedback delay is applied.\n"
                  << "Writes landing_sources.csv at t_min + n*dt <= t_max using delay-bin midpoints.\n"
                  << "--evolve advances gas and metal surface densities together, using j_land_mixing at every RK4 stage.\n"
                  << "mu is the CGM/nuclear mass ratio; beta is j_CGM/j_disk. Landing CSVs then include (1+mu) times nuclear returns.\n"
                  << "The disk uses the orbit potential's rotation curve and R_nucl = R_ring.\n"
                  << "Disk CSV columns: R_lo_kpc,R_hi_kpc,Sigma_g_Msun_kpc2,Sigmadot_star_Msun_yr_kpc2,Z\n"
                  << "Use one row per kernel annulus (0--30 kpc in 0.5 kpc bins). Input velocities are not used.\n"
                  << "gas_evolution.csv includes v_R_outer_kpc_yr (positive outward), Sigma_Z_Msun_kpc2, M_Z_Msun and Z.\n"
                  << "It also saves instantaneous RK4 gas/metal derivatives, SFR and the rotation curve for reconstruction.\n"
                  << "Metals use Z_land=(Z_nucl+mu*Z_CGM)/(1+mu), stellar removal at local Z, and enrichment yield_y*Sigmadot_star.\n"
                  << "Chemistry defaults: Z_nucl=0.02, Z_CGM=0.003, yield_y=0.015; --chemistry overrides these. Z is a mass fraction.\n"
                  << "Star-formation sinks stay fixed; domain boundaries stay closed.\n"
                  << "dt controls coupled RK4 steps: reduce it if stage or final gas/metal densities become unphysical.\n"
                  << "Physical and integration parameters are editable examples in model/kernel_runner.cpp.\n";
        return valid_arguments ? 0 : 1;
    }

    try {
        // 1. Example galaxy parameters; replace these for the intended galaxy.
        // Internal units throughout the orbit integration: kpc, Myr, M_sun.
        OrbitParameters potential_params{
            .M_d = 6.0e10, // disk mass [M_sun]
            .a_d = 3.0,    // disk scale length [kpc]
            .b_d = 0.3,    // disk scale height [kpc]
            .M_b = 1.0e10, // bulge mass [M_sun]
            .a_b = 0.7,    // bulge scale length [kpc]
            .rho_s = 8.0e6,// halo scale density [M_sun/kpc^3]
            .r_s = 20.0,   // halo scale radius [kpc]
            .j_z = 0.0    // set separately for each sampled parcel below
        };
        // Illustrative launch parameters, not a fit to a particular galaxy.
        const LaunchDistributionParameters launch_parameters{
            .R_ring_kpc = 1.0,
            .sigma_R_kpc = 0.1,
            .h_v_kms = 150.0,
            .sigma_theta_rad = 15.0 * std::numbers::pi / 180.0
        };
        const LaunchDistribution distribution = make_launch_distribution(launch_parameters);
        const std::uint64_t number_of_parcels = argc > 2 ? unsigned_argument(argv[2]) : 10000;
        const std::uint64_t seed = argc > 3 ? unsigned_argument(argv[3]) : 42;
        BurstHistory launch_history{!burst_mode && argc > 4 ? finite_argument(argv[4]) : 1.0,
                                    0.0, 0.0, 1.0, 1.0};
        double t_min = 0.0, t_max = 0.0, dt = 1.0;
        if (burst_mode) {
            launch_history = {finite_argument(argv[5]), finite_argument(argv[6]),
                              finite_argument(argv[7]), finite_argument(argv[8]), finite_argument(argv[9])};
            t_min = finite_argument(argv[10]);
            t_max = finite_argument(argv[11]);
            dt = finite_argument(argv[12]);
        }
        if (!period_argument.empty()) {
            launch_history.period = finite_argument(period_argument);
            if (launch_history.period <= 0.0 ||
                !std::isfinite(8.0 * launch_history.sigma_b / launch_history.period) ||
                std::ceil(8.0 * launch_history.sigma_b / launch_history.period) >= std::numeric_limits<int>::max()) {
                throw std::invalid_argument("Burst period must be positive and the number of overlapping bursts representable");
            }
        }
        if (launch_history.Mdot_0 < 0.0 || launch_history.Mdot_b < 0.0 ||
            launch_history.sigma_b <= 0.0 || launch_history.eta < 0.0 ||
            !std::isfinite(launch_history(launch_history.t_b))) {
            throw std::invalid_argument("Burst rates and eta must be nonnegative, sigma_b positive, and peak launch rate finite");
        }
        if (dt <= 0.0 || t_max < t_min || t_min + dt <= t_min) {
            throw std::invalid_argument("Output times require t_max >= t_min and a positive representable dt");
        }
        // Include an endpoint one rounding unit below an integer number of steps
        // (for example, 0.3 / 0.1), then clamp the final time to the requested bound.
        const double intervals = std::floor(std::nextafter((t_max - t_min) / dt,
                                                          std::numeric_limits<double>::infinity()));
        if (!std::isfinite(intervals) || intervals >= static_cast<double>(std::numeric_limits<std::size_t>::max())) {
            throw std::invalid_argument("Output time grid is too large");
        }
        const std::size_t time_count = static_cast<std::size_t>(intervals) + 1;
        if (number_of_parcels == 0 || number_of_parcels > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("Number of parcels must be positive and fit in size_t");
        }
        std::mt19937_64 rng(seed);
        const double z0 = 0.0; // launch upwards from the disk midplane [kpc]

        const double h_0 = 0.01;   // initial time step [Myr]
        const double h_max = 1.0; // maximum time step [Myr]
        const double atol = 1e-9;
        const double rtol = 1e-7;
        const double t_stop = 1000.0; // no return after this time is unresolved [Myr]

        // Shared annulus/delay edges for kernel construction and disk evolution.
        DoubleVec R_bins, tau_bins;
        const std::uint64_t delay_bins = unsigned_argument(tau_bins_argument);
        if (delay_bins == 0 || delay_bins >= tau_bins.max_size()) {
            throw std::invalid_argument("Number of delay bins must be positive and fit in a vector");
        }
        for (int i = 0; i <= 60; ++i) R_bins.push_back(0.5 * i);
        for (std::uint64_t j = 0; j <= delay_bins; ++j) tau_bins.push_back(t_stop * j / delay_bins);
        MassContinuityParameters mass_parameters;
        DoubleVec disk_state;
        if (evolve_mass) {
            mass_parameters.R_bins = R_bins;
            mass_parameters.tau_bins = tau_bins;
            mass_parameters.Mdot_launch = launch_history;
            mass_parameters.rot_curve = {{disk_circular_velocity, &potential_params},
                                         {disk_circular_velocity_derivative, &potential_params}};
            mass_parameters.R_nucl = launch_parameters.R_ring_kpc;
            mass_parameters.mu = finite_argument(mu_argument);
            mass_parameters.beta = finite_argument(beta_argument);
            if (mass_parameters.mu < 0.0) throw std::invalid_argument("CGM/nuclear mass ratio mu must be nonnegative");
            if (chemistry_options) {
                mass_parameters.Z_nucl = finite_argument(argv[chemistry_argument]);
                mass_parameters.Z_CGM = finite_argument(argv[chemistry_argument + 1]);
                mass_parameters.yield = finite_argument(argv[chemistry_argument + 2]);
            }
            if (mass_parameters.Z_nucl < 0.0 || mass_parameters.Z_nucl > 1.0 ||
                mass_parameters.Z_CGM < 0.0 || mass_parameters.Z_CGM > 1.0 || mass_parameters.yield < 0.0) {
                throw std::invalid_argument("Chemistry needs 0 <= Z_nucl,Z_CGM <= 1 and a nonnegative yield_y");
            }
            disk_state = read_initial_disk(disk_file, mass_parameters);
        }

        // 2. Independent PDF draws represent equal amounts of launched mass.
        constexpr double kms_to_kpc_per_Myr = 1.022712165e-3;
        const double weight = 1.0 / static_cast<double>(number_of_parcels);
        std::vector<LandingPoint> landings;
        landings.reserve(static_cast<std::size_t>(number_of_parcels));
        double returned_weight = 0.0;
        std::uint64_t timed_out = 0, failed = 0;

        const std::filesystem::path output_dir = argc > 1 ? argv[1] : "output/kernel/sampled";
        std::filesystem::create_directories(output_dir);
        std::ofstream orbit_csv;
        orbit_csv.exceptions(std::ios::failbit | std::ios::badbit);
        orbit_csv.open(output_dir / "orbits.csv");
        orbit_csv << std::setprecision(17)
                  << "parcel_id,R0_kpc,z0_kpc,v_k_kms,theta_rad,psi_rad,vR0_kms,vz0_kms,vphi0_kms,j_z_kpc2_per_Myr,weight,status,R_land_kpc,tau_Myr\n";

        for (std::uint64_t n = 0; n < number_of_parcels; ++n) {
            const LaunchDraw draw = sample_launch(distribution, rng);
            const double vR_kms = draw.v_k_kms * std::sin(draw.theta_rad) * std::cos(draw.psi_rad);
            const double vz_kms = draw.v_k_kms * std::cos(draw.theta_rad);
            const double vphi0 = circular_velocity(draw.R0_kpc, potential_params) +
                draw.v_k_kms * kms_to_kpc_per_Myr * std::sin(draw.theta_rad) * std::sin(draw.psi_rad);
            OrbitParameters parcel_params = potential_params;
            parcel_params.j_z = draw.R0_kpc * vphi0;
            const DoubleVec launch_conditions{
                draw.R0_kpc, z0, vR_kms * kms_to_kpc_per_Myr, vz_kms * kms_to_kpc_per_Myr
            };
            const double missing = std::numeric_limits<double>::quiet_NaN();
            LandingPoint landing{missing, missing, weight, false};
            const OrbitResult result = integrate_orbit(
                launch_conditions, parcel_params, h_0, atol, rtol, t_stop, h_max);
            if (result.status == OrbitStatus::Returned) {
                landing = {result.R_land, result.flight_time, weight, true};
                returned_weight += weight;
            }
            timed_out += result.status == OrbitStatus::TimedOut;
            failed += result.status == OrbitStatus::IntegrationFailed;
            landings.push_back(landing);
            orbit_csv << n << ',' << draw.R0_kpc << ',' << z0 << ',' << draw.v_k_kms << ','
                      << draw.theta_rad << ',' << draw.psi_rad << ',' << vR_kms << ',' << vz_kms << ','
                      << vphi0 / kms_to_kpc_per_Myr << ',' << parcel_params.j_z << ','
                      << weight << ',' << status_name(result.status) << ',';
            if (landing.returned) {
                orbit_csv << landing.R_land << ',' << landing.tau << '\n';
            } else {
                orbit_csv << ",\n";
            }
        }
        orbit_csv.close();

        // 3. Build K(R, tau). These arrays contain edges, not bin centers.
        const DoubleVec kernel = build_kernel(landings, R_bins, tau_bins);
        if (evolve_mass) mass_parameters.kernel = kernel;

        std::ofstream parameters_csv;
        parameters_csv.exceptions(std::ios::failbit | std::ios::badbit);
        parameters_csv.open(output_dir / "parameters.csv");
        parameters_csv << std::setprecision(17) << "parameter,value\n";
        const auto save = [&](const char* name, const auto& value) {
            parameters_csv << name << ',' << value << '\n';
        };
        save("number_of_parcels", number_of_parcels);
        save("seed", seed);
        save("launch_history", launch_history.period > 0.0 ? "baseline_plus_periodic_gaussians" :
                               (burst_mode ? "baseline_plus_gaussian" : "constant"));
        if (!burst_mode) save("Mdot_launch_Msun_yr", launch_history.Mdot_0);
        save("Mdot_0_Msun_yr", launch_history.Mdot_0);
        save("Mdot_b_Msun_yr", launch_history.Mdot_b);
        save("t_b_Myr", launch_history.t_b);
        save("sigma_b_Myr", launch_history.sigma_b);
        save("eta", launch_history.eta);
        save("burst_period_Myr", launch_history.period);
        save("stellar_feedback_delay", "none_instantaneous_eta_times_nuclear_rate");
        if (launch_history.period > 0.0) {
            save("mean_Mdot_launch_Msun_yr", launch_history.eta * (launch_history.Mdot_0 +
                 std::sqrt(2.0 * std::numbers::pi) * launch_history.Mdot_b * launch_history.sigma_b / launch_history.period));
        }
        save("launch_history_before_zero", "same_analytic_law");
        save("landing_t_min_Myr", t_min);
        save("landing_t_max_requested_Myr", t_max);
        save("landing_dt_Myr", dt);
        save("landing_time_count", time_count);
        save("landing_delay_quadrature", "midpoint");
        save("mass_evolution", evolve_mass ? "rk4_mixing" : "disabled");
        save("metal_evolution", evolve_mass ? "same_rk4_state_conservative_transport" : "disabled");
        save("kernel_mass_origin", "nuclear_launch");
        save("landing_mass_origin", evolve_mass ? "nuclear_plus_cgm" : "nuclear_return");
        if (evolve_mass) {
            save("initial_disk_csv", galactic_nuclear_fountain::csv::escape(disk_file));
            save("mass_boundary_fluxes", "zero_at_both_edges");
            save("inner_mass_flux_Msun_yr", mass_parameters.inner_mass_flux);
            save("outer_mass_flux_Msun_yr", mass_parameters.outer_mass_flux);
            save("Z_inner_inflow", mass_parameters.Z_inner_inflow);
            save("Z_outer_inflow", mass_parameters.Z_outer_inflow);
            save("mass_velocity", "j_land_mixing_each_rk4_stage");
            save("mass_sfr", "fixed_input_profile");
            save("mass_rotation_curve", "orbit_potential");
            save("mixing_R_nucl_kpc", mass_parameters.R_nucl);
            save("j_nucl_kpc_kms", mass_parameters.R_nucl * circular_velocity(mass_parameters.R_nucl, potential_params) / kms_to_kpc_per_Myr);
            save("mu", mass_parameters.mu);
            save("beta", mass_parameters.beta);
            save("Z_nucl", mass_parameters.Z_nucl);
            save("Z_CGM", mass_parameters.Z_CGM);
            save("yield_y", mass_parameters.yield);
            save("Z_land", z_land_mixing(mass_parameters.Z_nucl, mass_parameters.Z_CGM, mass_parameters.mu));
        }
        save("rng", "mt19937_64");
        save("R_ring_kpc", launch_parameters.R_ring_kpc);
        save("sigma_R_kpc", launch_parameters.sigma_R_kpc);
        save("h_v_kms", launch_parameters.h_v_kms);
        save("sigma_theta_rad", launch_parameters.sigma_theta_rad);
        save("theta_pdf", "sin(theta)*exp(-theta^2/(2*sigma_theta_rad^2))");
        save("theta_max_rad", std::numbers::pi / 2.0);
        save("radius_cdf_points", distribution.radius_grid.size());
        save("radius_cdf_min_kpc", distribution.radius_grid.empty() ? launch_parameters.R_ring_kpc : distribution.radius_grid.front());
        save("radius_cdf_max_kpc", distribution.radius_grid.empty() ? launch_parameters.R_ring_kpc : distribution.radius_grid.back());
        save("M_d_Msun", potential_params.M_d);
        save("a_d_kpc", potential_params.a_d);
        save("b_d_kpc", potential_params.b_d);
        save("M_b_Msun", potential_params.M_b);
        save("a_b_kpc", potential_params.a_b);
        save("rho_s_Msun_kpc3", potential_params.rho_s);
        save("r_s_kpc", potential_params.r_s);
        save("z0_kpc", z0);
        save("h_0_Myr", h_0);
        save("h_max_Myr", h_max);
        save("atol", atol);
        save("rtol", rtol);
        save("t_stop_Myr", t_stop);
        save("kms_to_kpc_per_Myr", kms_to_kpc_per_Myr);
        // Exact radial and delay bin edges are also written in kernel.csv.
        save("kernel_R_min_kpc", R_bins.front());
        save("kernel_R_max_kpc", R_bins.back());
        save("kernel_R_bins", R_bins.size() - 1);
        save("kernel_tau_min_Myr", tau_bins.front());
        save("kernel_tau_max_Myr", tau_bins.back());
        save("kernel_tau_bins", tau_bins.size() - 1);
        parameters_csv.close();

        // 4. Write the density and the launched-mass fraction in each cell.
        // K is per dR dtau; an annular surface source additionally needs area.
        std::ofstream kernel_csv;
        kernel_csv.exceptions(std::ios::failbit | std::ios::badbit);
        kernel_csv.open(output_dir / "kernel.csv");
        kernel_csv << std::setprecision(17)
                   << "R_lo_kpc,R_hi_kpc,tau_lo_Myr,tau_hi_Myr,K_per_kpc_per_Myr,mass_fraction\n";
        const std::size_t n_tau = tau_bins.size() - 1;
        double binned_fraction = 0.0;
        for (std::size_t i = 0; i + 1 < R_bins.size(); ++i) {
            for (std::size_t j = 0; j < n_tau; ++j) {
                const double density = kernel[i * n_tau + j];
                const double fraction = density * (R_bins[i + 1] - R_bins[i]) * (tau_bins[j + 1] - tau_bins[j]);
                binned_fraction += fraction;
                kernel_csv << R_bins[i] << ',' << R_bins[i + 1] << ','
                           << tau_bins[j] << ',' << tau_bins[j + 1] << ','
                           << density << ',' << fraction << '\n';
            }
        }
        kernel_csv.close();

        // 5. Convolve the launch history with flight delays at each output time.
        std::ofstream landing_csv;
        landing_csv.exceptions(std::ios::failbit | std::ios::badbit);
        landing_csv.open(output_dir / "landing_sources.csv");
        landing_csv << std::setprecision(17)
                    << "t_Myr,R_lo_kpc,R_hi_kpc,area_kpc2,Sigmadot_land_Msun_yr_kpc2,Mdot_land_Msun_yr\n";
        std::ofstream history_csv;
        history_csv.exceptions(std::ios::failbit | std::ios::badbit);
        history_csv.open(output_dir / "landing_history.csv");
        history_csv << std::setprecision(17) << "t_Myr,Mdot_launch_Msun_yr,Mdot_land_Msun_yr\n";
        std::ofstream gas_csv;
        if (evolve_mass) {
            gas_csv.exceptions(std::ios::failbit | std::ios::badbit);
            gas_csv.open(output_dir / "gas_evolution.csv");
            gas_csv << std::setprecision(17) << "t_Myr,R_lo_kpc,R_hi_kpc,Sigma_g_Msun_kpc2,M_g_Msun,v_R_outer_kpc_yr,Sigma_Z_Msun_kpc2,M_Z_Msun,Z,dSigma_g_dt_Msun_kpc2_Myr,dSigma_Z_dt_Msun_kpc2_Myr,Sigmadot_star_Msun_yr_kpc2,v_c_kms,dv_c_dR_kms_kpc,v_c_outer_kms,dv_c_dR_outer_kms_kpc\n";
        }
        double total_landing_rate = 0.0;
        double previous_time = t_min;
        for (std::size_t n = 0; n < time_count; ++n) {
            const double t = std::min(t_max, t_min + static_cast<double>(n) * dt);
            if (evolve_mass && n > 0) {
                disk_state = advance_mass_continuity(previous_time, t - previous_time, disk_state, mass_parameters);
            }
            previous_time = t;
            const DoubleVec landing_rate = evolve_mass ? mass_continuity_landing_rate(t, mass_parameters) :
                sigmadot_land_kernel(kernel, R_bins, tau_bins, t, launch_history);
            const DoubleVec gas_density = evolve_mass ? DoubleVec(disk_state.begin(), disk_state.begin() + landing_rate.size()) : DoubleVec{};
            const DoubleVec velocities = evolve_mass ? mass_continuity_velocity(gas_density, landing_rate, mass_parameters) : DoubleVec{};
            const DoubleVec derivative = evolve_mass ? mass_continuity_rhs(t, disk_state, &mass_parameters) : DoubleVec{};
            total_landing_rate = 0.0;
            for (std::size_t i = 0; i < landing_rate.size(); ++i) {
                const double area = std::numbers::pi * (R_bins[i + 1] - R_bins[i]) * (R_bins[i + 1] + R_bins[i]);
                const double annulus_rate = landing_rate[i] * area;
                total_landing_rate += annulus_rate;
                landing_csv << t << ',' << R_bins[i] << ',' << R_bins[i + 1] << ',' << area << ','
                            << landing_rate[i] << ',' << annulus_rate << '\n';
                if (evolve_mass) {
                    const double R = 0.5 * (R_bins[i] + R_bins[i + 1]);
                    const double metals = disk_state[landing_rate.size() + i];
                    const double Z = gas_density[i] > 0.0 ? metals / gas_density[i] : std::numeric_limits<double>::quiet_NaN();
                    gas_csv << t << ',' << R_bins[i] << ',' << R_bins[i + 1] << ','
                            << gas_density[i] << ',' << gas_density[i] * area << ','
                            << (i < velocities.size() ? velocities[i] : 0.0) << ','
                            << metals << ',' << metals * area << ',' << Z << ','
                            << derivative[i] << ',' << derivative[landing_rate.size() + i] << ','
                            << mass_parameters.sigmadot_star[i] << ','
                            << circular_velocity(R, potential_params) / kms_to_kpc_per_Myr << ','
                            << disk_circular_velocity_derivative(R, &potential_params) / kms_to_kpc_per_Myr << ','
                            << circular_velocity(R_bins[i+1], potential_params) / kms_to_kpc_per_Myr << ','
                            << disk_circular_velocity_derivative(R_bins[i+1], &potential_params) / kms_to_kpc_per_Myr << '\n';
                }
            }
            history_csv << t << ',' << launch_history(t) << ',' << total_landing_rate << '\n';
        }
        landing_csv.close();
        history_csv.close();
        if (evolve_mass) gas_csv.close();
        double total_weight = 0.0;
        double outside_weight = 0.0;
        for (const LandingPoint& landing : landings) {
            total_weight += landing.weight;
            if (landing.returned &&
                (landing.R_land < R_bins.front() || landing.R_land > R_bins.back() ||
                 landing.tau < tau_bins.front() || landing.tau > tau_bins.back())) {
                outside_weight += landing.weight;
            }
        }
        std::cout << "Integrated " << landings.size() << " sampled launches (seed " << seed << ").\n"
                  << "Returned by " << t_stop << " Myr: " << returned_weight / total_weight << '\n'
                  << "Timed out: " << timed_out << '\n'
                  << "Integration failures: " << failed << '\n'
                  << "Returned inside kernel grid: " << binned_fraction << '\n'
                  << "Returned outside kernel grid: " << outside_weight / total_weight << '\n'
                  << "Launch history: " << (launch_history.period > 0.0 ? "baseline plus periodic Gaussian bursts" :
                                            (burst_mode ? "baseline plus Gaussian burst" : "constant")) << '\n'
                  << "Landing profiles written: " << time_count << '\n'
                  << "Landing mass rate at last output time [Msun/yr]: " << total_landing_rate << '\n'
                  << "Wrote " << output_dir / "orbits.csv" << ", " << output_dir / "kernel.csv"
                  << ", " << output_dir / "landing_sources.csv" << " and " << output_dir / "landing_history.csv" << '\n';
        if (evolve_mass) std::cout << "Evolved disk gas and metals and wrote " << output_dir / "gas_evolution.csv" << '\n';
        if (failed > 0) {
            std::cerr << "Kernel contains unresolved numerical failures; inspect orbits.csv.\n";
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << "Kernel runner: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
