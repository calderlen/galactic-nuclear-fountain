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

#include <zlib.h>

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
    double Mdot_0, Mdot_b, t_b, sigma_b, eta_nuc;
    double period = 0.0;

    double operator()(double t_Myr) const {
        if (period == 0.0) {
            return mdot_launch(mdot_nuc_burst(t_Myr, Mdot_0, Mdot_b, t_b, sigma_b), eta_nuc);
        }
        // Sum the neighboring bursts, including overlaps and the pre-zero history.
        const double phase = std::remainder(t_Myr - t_b, period);
        const int neighbors = static_cast<int>(std::ceil(8.0 * sigma_b / period));
        double rate = Mdot_0;
        for (int k = -neighbors; k <= neighbors; ++k) {
            const double offset = (phase - k * period) / sigma_b;
            rate += Mdot_b * std::exp(-0.5 * offset * offset);
        }
        return mdot_launch(rate, eta_nuc);
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
        const double Z = number("Z");
        if (gas < 0.0) {
            throw std::invalid_argument("Initial gas density must be nonnegative");
        }
        if (Z < 0.0 || Z > 1.0) throw std::invalid_argument("Initial Z must be a metal mass fraction between zero and one");
        state[i] = gas;
        state[n + i] = gas * Z;
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

double disk_circular_velocity(double R, const void* parameters) {
    return circular_velocity(R, *static_cast<const OrbitParameters*>(parameters));
}

double disk_circular_velocity_derivative(double R, const void* parameters) {
    // The disk uses the same fixed potential as the parcel orbits.
    const double h = 1e-4 * R;
    return (disk_circular_velocity(R + h, parameters) - disk_circular_velocity(R - h, parameters)) / (2.0 * h);
}

}

int main(int argc, char* argv[]) {
    if (argc > 4 || (argc > 1 && std::string(argv[1]) == "--help")) {
        std::cout << "Usage: " << argv[0] << " [output-directory [config.csv [initial-disk.csv]]]\n"
                  << "Default config: config/kernel.csv. Supplying a disk evolves gas/metals with a KS law.\n"
                  << "Disk columns: R_lo_kpc,R_hi_kpc,Sigma_g_Msun_kpc2,Z. Annuli must match config.\n"
                  << "All physical parameters, launch history, time grids and boundaries are in the config.\n"
                  << "The CGM density and parcel cross-section are constant; no evaporation is included.\n";
        return argc > 4 ? 1 : 0;
    }
    try {
        namespace csv = galactic_nuclear_fountain::csv;
        const std::string config_file = argc > 2 ? argv[2] : "config/kernel.csv";
        csv::Row config;
        for (const auto& row : csv::read(config_file).rows) config[csv::value(row,"name")] = csv::value(row,"value");
        const auto setting = [&](const char* key) { return finite_argument(csv::value(config,key)); };
        const auto count = [&](const char* key) { return unsigned_argument(csv::value(config,key)); };
        const bool evolve_mass = argc == 4;
        const std::string disk_file = evolve_mass ? argv[3] : "";
        OrbitParameters potential_params{
            .M_d=setting("M_d_Msun"), .a_d=setting("a_d_kpc"), .b_d=setting("b_d_kpc"),
            .M_b=setting("M_b_Msun"), .a_b=setting("a_b_kpc"),
            .rho_s=setting("rho_s_Msun_kpc3"), .r_s=setting("r_s_kpc"),
            .rho_CGM=setting("rho_CGM_Msun_kpc3"), .sigma=setting("parcel_sigma_kpc2"),
            .C_D=setting("C_D"), .epsilon=setting("epsilon"), .beta=setting("beta")
        };
        const double m0 = setting("parcel_m0_Msun");
        const LaunchDistributionParameters launch_parameters{
            setting("R_ring_kpc"), setting("sigma_R_kpc"), setting("h_v_kms"),
            setting("sigma_theta_deg") * std::numbers::pi / 180.0
        };
        const LaunchDistribution distribution = make_launch_distribution(launch_parameters);
        const std::uint64_t number_of_parcels = count("number_of_parcels"), seed = count("seed");
        BurstHistory launch_history{setting("Mdot_0_Msun_yr"), setting("Mdot_b_Msun_yr"),
            setting("t_b_Myr"), setting("sigma_b_Myr"), setting("eta_nuc"), setting("burst_period_Myr")};
        const bool burst_mode = launch_history.Mdot_b > 0.0;
        const double t_min=setting("t_min_Myr"), t_max=setting("t_max_Myr"), dt=setting("dt_Myr");
        if (launch_history.Mdot_0 < 0 || launch_history.Mdot_b < 0 || launch_history.eta_nuc < 0 ||
            launch_history.sigma_b <= 0 || launch_history.period < 0 ||
            (launch_history.period > 0 && std::ceil(8*launch_history.sigma_b/launch_history.period) >= std::numeric_limits<int>::max()))
            throw std::invalid_argument("Invalid launch history in config");
        if (dt <= 0 || t_max < t_min || t_min+dt <= t_min || number_of_parcels == 0)
            throw std::invalid_argument("Require positive dt and parcel count and t_max >= t_min");
        const double intervals=std::floor(std::nextafter((t_max-t_min)/dt, std::numeric_limits<double>::infinity()));
        if (!std::isfinite(intervals) || intervals >= static_cast<double>(std::numeric_limits<std::size_t>::max()))
            throw std::invalid_argument("Output time grid is too large");
        const std::size_t time_count=static_cast<std::size_t>(intervals)+1;
        std::mt19937_64 rng(seed);
        const double z0=0.0;
        const double h_0=setting("h_0_Myr"), h_max=setting("h_max_Myr"),
                     atol=setting("atol"), rtol=setting("rtol"), t_stop=setting("t_stop_Myr");
        DoubleVec R_bins, tau_bins;
        const auto radial_bins=count("R_bins"), delay_bins=count("tau_bins");
        const double R_min=setting("R_min_kpc"), R_max=setting("R_max_kpc");
        for (std::uint64_t i=0; i<=radial_bins; ++i) R_bins.push_back(R_min+(R_max-R_min)*i/radial_bins);
        for (std::uint64_t j=0; j<=delay_bins; ++j) tau_bins.push_back(t_stop*j/delay_bins);
        MassContinuityParameters mass_parameters;
        mass_parameters.R_bins=R_bins;
        mass_parameters.tau_bins=tau_bins;
        mass_parameters.Mdot_launch=launch_history;
        mass_parameters.rot_curve={{disk_circular_velocity,&potential_params}, {disk_circular_velocity_derivative,&potential_params}};
        const double return_fraction=setting("return_fraction"), A_SFR=setting("KS_A_SFR");
        mass_parameters.A=(1-return_fraction)*A_SFR;
        mass_parameters.N=setting("KS_N");
        mass_parameters.Z_nucl=setting("Z_nucl");
        mass_parameters.Z_CGM=setting("Z_CGM");
        mass_parameters.yield=setting("yield_y");
        mass_parameters.inner_mass_flux=setting("inner_mass_flux_Msun_yr");
        mass_parameters.outer_mass_flux=setting("outer_mass_flux_Msun_yr");
        mass_parameters.Z_inner_inflow=setting("Z_inner_inflow");
        mass_parameters.Z_outer_inflow=setting("Z_outer_inflow");
        DoubleVec disk_state;
        if (evolve_mass) disk_state=read_initial_disk(disk_file,mass_parameters);

        // 2. Independent PDF draws represent equal amounts of launched mass.
        constexpr double kms_to_kpc_per_Myr = 1.022712165e-3;
        const double weight = 1.0 / static_cast<double>(number_of_parcels);
        std::vector<LandingPoint> landings;
        landings.reserve(static_cast<std::size_t>(number_of_parcels));
        double returned_weight = 0.0;
        std::uint64_t timed_out = 0, failed = 0;

        const std::filesystem::path output_dir = argc > 1 ? argv[1] : "output/kernel/sampled";
        std::filesystem::create_directories(output_dir);
        if (std::filesystem::absolute(config_file).lexically_normal()!=std::filesystem::absolute(output_dir/"config.csv").lexically_normal())
            std::filesystem::copy_file(config_file,output_dir/"config.csv",std::filesystem::copy_options::overwrite_existing);
        const std::filesystem::path orbit_path=output_dir/"orbits.csv.gz";
        gzFile orbit_csv=gzopen(orbit_path.string().c_str(),"wb");
        if (orbit_csv==nullptr || gzbuffer(orbit_csv,1U<<20)!=0)
            throw std::runtime_error("Cannot open "+orbit_path.string());
        if (gzputs(orbit_csv,"parcel_id,R0_kpc,z0_kpc,v_k_kms,theta_rad,psi_rad,vR0_kms,vz0_kms,vphi0_kms,j_z0_kpc2_per_Myr,weight,status,R_land_kpc,tau_Myr,j_land_kpc2_per_Myr,mass_ratio\n")<0)
            throw std::runtime_error("Cannot write "+orbit_path.string());

        for (std::uint64_t n = 0; n < number_of_parcels; ++n) {
            const LaunchDraw draw = sample_launch(distribution, rng);
            const double vR_kms = draw.v_k_kms * std::sin(draw.theta_rad) * std::cos(draw.psi_rad);
            const double vz_kms = draw.v_k_kms * std::cos(draw.theta_rad);
            const double vphi0 = circular_velocity(draw.R0_kpc, potential_params) +
                draw.v_k_kms * kms_to_kpc_per_Myr * std::sin(draw.theta_rad) * std::sin(draw.psi_rad);
            const double j0 = draw.R0_kpc * vphi0;
            const DoubleVec launch_conditions{
                draw.R0_kpc, z0, vR_kms * kms_to_kpc_per_Myr, vz_kms * kms_to_kpc_per_Myr, j0, m0
            };
            const double missing = std::numeric_limits<double>::quiet_NaN();
            LandingPoint landing{missing, missing, weight, false};
            const OrbitResult result = integrate_orbit(
                launch_conditions, potential_params, h_0, atol, rtol, t_stop, h_max);
            if (result.status == OrbitStatus::Returned) {
                landing = {result.R_land, result.flight_time, weight, true, result.mass_ratio, result.j_land};
                returned_weight += weight;
            }
            timed_out += result.status == OrbitStatus::TimedOut;
            failed += result.status == OrbitStatus::IntegrationFailed;
            landings.push_back(landing);
            if (gzprintf(orbit_csv,"%llu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%s,",
                         static_cast<unsigned long long>(n),draw.R0_kpc,z0,draw.v_k_kms,
                         draw.theta_rad,draw.psi_rad,vR_kms,vz_kms,vphi0/kms_to_kpc_per_Myr,j0,
                         weight,status_name(result.status))<=0)
                throw std::runtime_error("Cannot write "+orbit_path.string());
            if (landing.returned) {
                if (gzprintf(orbit_csv,"%.17g,%.17g,%.17g,%.17g\n",landing.R_land,landing.tau,
                             landing.j_land,landing.mass_ratio)<=0)
                    throw std::runtime_error("Cannot write "+orbit_path.string());
            } else {
                if (gzputs(orbit_csv,",,,\n")<0)
                    throw std::runtime_error("Cannot write "+orbit_path.string());
            }
        }
        if (gzclose(orbit_csv)!=Z_OK) throw std::runtime_error("Cannot close "+orbit_path.string());

        // 3. Build K(R, tau). These arrays contain edges, not bin centers.
        build_kernel(landings, R_bins, tau_bins, mass_parameters.K_wind, mass_parameters.K_mass, mass_parameters.K_j);
        if (evolve_mass) {
            for (std::size_t edge = 1; edge + 1 < R_bins.size(); ++edge) {
                const double R = R_bins[edge];
                const double dj_dR = dj_disk_dR(R, rotation_velocity(mass_parameters.rot_curve, R),
                                               rotation_velocity_derivative(mass_parameters.rot_curve, R));
                if (!std::isfinite(dj_dR) || dj_dR <= 0.0)
                    throw std::invalid_argument("Disk transport requires positive dj/dR");
            }
        }

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
        if (!burst_mode) save("Mdot_launch_Msun_yr", launch_history(0.0));
        save("Mdot_0_Msun_yr", launch_history.Mdot_0);
        save("Mdot_b_Msun_yr", launch_history.Mdot_b);
        save("t_b_Myr", launch_history.t_b);
        save("sigma_b_Myr", launch_history.sigma_b);
        save("eta_nuc", launch_history.eta_nuc);
        save("burst_period_Myr", launch_history.period);
        save("stellar_feedback_delay", "none_instantaneous_eta_nuc_times_nuclear_rate");
        if (launch_history.period > 0.0) {
            save("mean_Mdot_launch_Msun_yr", launch_history.eta_nuc * (launch_history.Mdot_0 +
                 std::sqrt(2.0 * std::numbers::pi) * launch_history.Mdot_b * launch_history.sigma_b / launch_history.period));
        }
        save("launch_history_before_zero", "same_analytic_law");
        save("landing_t_min_Myr", t_min);
        save("landing_t_max_requested_Myr", t_max);
        save("landing_dt_Myr", dt);
        save("landing_time_count", time_count);
        save("landing_delay_quadrature", "midpoint");
        save("mass_evolution", evolve_mass ? "rk4_kernel_moments" : "disabled");
        save("metal_evolution", evolve_mass ? "same_rk4_state_conservative_transport" : "disabled");
        save("kernel_mass_origin", "nuclear_launch");
        save("landing_mass_origin", "nuclear_plus_cgm");
        if (evolve_mass) {
            save("initial_disk_csv", galactic_nuclear_fountain::csv::escape(disk_file));
            save("mass_boundary_fluxes", "configured_outward_positive");
            save("inner_mass_flux_Msun_yr", mass_parameters.inner_mass_flux);
            save("outer_mass_flux_Msun_yr", mass_parameters.outer_mass_flux);
            save("Z_inner_inflow", mass_parameters.Z_inner_inflow);
            save("Z_outer_inflow", mass_parameters.Z_outer_inflow);
            save("mass_velocity", "kernel_angular_momentum_each_rk4_stage");
            save("mass_sfr", "KS_from_stage_gas_density");
            save("mass_rotation_curve", "orbit_potential");
            save("R_nucl_kpc", launch_parameters.R_ring_kpc);
            save("j_nucl_kpc_kms", launch_parameters.R_ring_kpc * circular_velocity(launch_parameters.R_ring_kpc, potential_params) / kms_to_kpc_per_Myr);
        }
        save("Z_nucl", mass_parameters.Z_nucl);
        save("Z_CGM", mass_parameters.Z_CGM);
        save("yield_y", mass_parameters.yield);
        save("stellar_sink", "net_locked_mass");
        save("effective_accretion_definition", "Eulerian_dSigma_g_dt_plus_net_star_sink");
        save("config_csv", csv::escape(std::filesystem::absolute(config_file).string()));
        save("CGM_density_law", "constant");
        save("parcel_cross_section_law", "constant");
        save("rho_CGM_Msun_kpc3", potential_params.rho_CGM);
        save("parcel_sigma_kpc2", potential_params.sigma);
        save("parcel_m0_Msun", m0);
        save("C_D", potential_params.C_D);
        save("epsilon", potential_params.epsilon);
        save("beta", potential_params.beta);
        save("return_fraction", return_fraction);
        save("KS_A_SFR", A_SFR);
        save("KS_A_net", mass_parameters.A);
        save("KS_N", mass_parameters.N);
        save("kernel_normalization", "initial_launch_mass_per_annular_area_per_delay");
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

        // 4. Write the density and the launched-mass fraction in each cell.
        // All three moments are per annular area per delay; K_wind is the paper K.
        std::ofstream kernel_csv;
        kernel_csv.exceptions(std::ios::failbit | std::ios::badbit);
        kernel_csv.open(output_dir / "kernel.csv");
        kernel_csv << std::setprecision(17)
                   << "R_lo_kpc,R_hi_kpc,tau_lo_Myr,tau_hi_Myr,K_per_kpc2_per_Myr,K_mass_per_kpc2_per_Myr,K_j_per_Myr2,mass_fraction,total_mass_fraction\n";
        const std::size_t n_tau = tau_bins.size() - 1;
        double binned_fraction = 0.0, total_fraction = 0.0;
        for (std::size_t i = 0; i + 1 < R_bins.size(); ++i) {
            for (std::size_t j = 0; j < n_tau; ++j) {
                const std::size_t cell = i*n_tau+j;
                const double density = mass_parameters.K_wind[cell];
                const double measure = std::numbers::pi*(R_bins[i+1]*R_bins[i+1]-R_bins[i]*R_bins[i])*(tau_bins[j+1]-tau_bins[j]);
                const double fraction = density*measure;
                const double mass_fraction = mass_parameters.K_mass[cell]*measure;
                total_fraction += mass_fraction;
                binned_fraction += fraction;
                kernel_csv << R_bins[i] << ',' << R_bins[i + 1] << ','
                           << tau_bins[j] << ',' << tau_bins[j + 1] << ','
                           << density << ',' << mass_parameters.K_mass[cell] << ',' << mass_parameters.K_j[cell] << ','
                           << fraction << ',' << mass_fraction << '\n';
            }
        }
        kernel_csv.close();
        save("f_ret", binned_fraction);
        save("f_new", total_fraction-binned_fraction);
        save("f_tot", total_fraction);
        save("CGM_landing_mass_fraction", total_fraction > 0 ? (total_fraction-binned_fraction)/total_fraction : std::numeric_limits<double>::quiet_NaN());
        parameters_csv.close();

        // 5. Convolve the launch history with flight delays at each output time.
        std::ofstream landing_csv;
        landing_csv.exceptions(std::ios::failbit | std::ios::badbit);
        landing_csv.open(output_dir / "landing_sources.csv");
        landing_csv << std::setprecision(17)
                    << "t_Myr,R_lo_kpc,R_hi_kpc,area_kpc2,Sigmadot_land_Msun_yr_kpc2,Mdot_land_Msun_yr,Sigmadot_wind_Msun_yr_kpc2,Sigmadot_CGM_Msun_yr_kpc2,Jdot_land_Msun_yr_Myr,Zdot_land_Msun_yr_kpc2\n";
        std::ofstream history_csv;
        history_csv.exceptions(std::ios::failbit | std::ios::badbit);
        history_csv.open(output_dir / "landing_history.csv");
        history_csv << std::setprecision(17) << "t_Myr,Mdot_launch_Msun_yr,Mdot_land_Msun_yr,Mdot_wind_Msun_yr,Mdot_CGM_Msun_yr\n";
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
            DoubleVec wind_rate, landing_rate, angular_rate;
            mass_continuity_landing_sources(t, mass_parameters, wind_rate, landing_rate, angular_rate);
            const DoubleVec gas_density = evolve_mass ? DoubleVec(disk_state.begin(), disk_state.begin() + landing_rate.size()) : DoubleVec{};
            const DoubleVec velocities = evolve_mass ? mass_continuity_velocity(gas_density, landing_rate, angular_rate, mass_parameters) : DoubleVec{};
            const DoubleVec derivative = evolve_mass ? mass_continuity_rhs(t, disk_state, &mass_parameters) : DoubleVec{};
            const DoubleVec star = evolve_mass ? mass_continuity_star(gas_density, mass_parameters) : DoubleVec{};
            total_landing_rate = 0.0;
            double total_wind_rate = 0.0;
            for (std::size_t i = 0; i < landing_rate.size(); ++i) {
                const double area = std::numbers::pi * (R_bins[i + 1] - R_bins[i]) * (R_bins[i + 1] + R_bins[i]);
                const double annulus_rate = landing_rate[i] * area;
                total_landing_rate += annulus_rate;
                total_wind_rate += wind_rate[i]*area;
                const double cgm_rate = landing_rate[i]-wind_rate[i];
                const double metal_source = mass_parameters.Z_nucl*wind_rate[i] + mass_parameters.Z_CGM*cgm_rate;
                landing_csv << t << ',' << R_bins[i] << ',' << R_bins[i + 1] << ',' << area << ','
                            << landing_rate[i] << ',' << annulus_rate << ',' << wind_rate[i] << ',' << cgm_rate << ','
                            << angular_rate[i] << ',' << metal_source << '\n';
                if (evolve_mass) {
                    const double R = 0.5 * (R_bins[i] + R_bins[i + 1]);
                    const double metals = disk_state[landing_rate.size() + i];
                    const double Z = gas_density[i] > 0.0 ? metals / gas_density[i] : std::numeric_limits<double>::quiet_NaN();
                    const double outer_velocity=i<velocities.size() ? velocities[i] :
                        (mass_parameters.outer_mass_flux==0 ? 0.0 :
                         mass_parameters.outer_mass_flux>0 && gas_density[i]>0 ?
                         mass_parameters.outer_mass_flux/(2*std::numbers::pi*R_bins[i+1]*gas_density[i]) :
                         std::numeric_limits<double>::quiet_NaN());
                    gas_csv << t << ',' << R_bins[i] << ',' << R_bins[i + 1] << ','
                            << gas_density[i] << ',' << gas_density[i] * area << ','
                            << outer_velocity << ','
                            << metals << ',' << metals * area << ',' << Z << ','
                            << derivative[i] << ',' << derivative[landing_rate.size() + i] << ','
                            << star[i] << ','
                            << circular_velocity(R, potential_params) / kms_to_kpc_per_Myr << ','
                            << disk_circular_velocity_derivative(R, &potential_params) / kms_to_kpc_per_Myr << ','
                            << circular_velocity(R_bins[i+1], potential_params) / kms_to_kpc_per_Myr << ','
                            << disk_circular_velocity_derivative(R_bins[i+1], &potential_params) / kms_to_kpc_per_Myr << '\n';
                }
            }
            history_csv << t << ',' << launch_history(t) << ',' << total_landing_rate << ',' << total_wind_rate << ',' << total_landing_rate-total_wind_rate << '\n';
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
                  << "Wrote " << output_dir / "orbits.csv.gz" << ", " << output_dir / "kernel.csv"
                  << ", " << output_dir / "landing_sources.csv" << " and " << output_dir / "landing_history.csv" << '\n';
        if (evolve_mass) std::cout << "Evolved disk gas and metals and wrote " << output_dir / "gas_evolution.csv" << '\n';
        if (failed > 0) {
            std::cerr << "Kernel contains unresolved numerical failures; inspect orbits.csv.gz.\n";
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << "Kernel runner: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
