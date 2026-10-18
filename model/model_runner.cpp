#include "models.h"
#include "physics.h"
#include "units.h"
#include "profiles.h"
#include "csv.h"
#include "cgm_io.h"
#include "prescribed_gas_io.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr double pi=3.14159265358979323846;
constexpr double kpc_per_year_to_km_per_second=9.7779222168e8;
constexpr const char* primary_observational_rotation="SPARC_spline";
using CsvRow=galactic_nuclear_fountain::csv::Row;
using CsvTable=galactic_nuclear_fountain::csv::Table;
using galactic_nuclear_fountain::csv::escape;
using galactic_nuclear_fountain::csv::number;
using galactic_nuclear_fountain::csv::read;
using galactic_nuclear_fountain::csv::value;
struct AnalyticProfileParameters {double R_g; double sigma_g0; double A; double N;};
struct LandingProfileParameters {
    double Mdot_land, R_in, R_out, lambda;
    std::string kind="power_law";
    SmoothBrokenPowerLawParameters smooth{};
    double normalization=std::numeric_limits<double>::quiet_NaN();
    double fit_R_min=1.0, fit_R_max=15.0;
};
struct RotationInput {
    std::string source;
    std::string kind;
    FlatRotationParameters flat;
    LinearProfile tabulated_velocity;
    LinearProfile tabulated_derivative;
    double chi2;
    double dof;
    double reduced_chi2;
};
struct EmpiricalInverseVelocityParameters {const RK4Solution* solution; const EmpiricalInverseModelParameters* parameters;};

struct ProfileRow {
    std::string model;
    std::string source;
    double R;
    double gas;
    double star;
    double landing;
    double cumulative_landing;
    double radial_velocity;
    double radial_velocity_kms;
    double radial_mass_flux;
    double accretion;
    double inflow_time;
    double depletion_time;
    double circular_velocity;
    double circular_velocity_derivative;
    double disk_angular_momentum;
    double cgm_angular_momentum;
    double landing_angular_momentum;
    double angular_momentum_difference;
    double landing_to_disk;
    double landing_to_nuclear;
    double mu_angular_momentum;
    double metallicity;
    double equilibrium_metallicity;
    double metallicity_derivative;
    double required_landing_metallicity;
    double mu_metallicity;
    // Extra columns for evolving snapshots; legacy steady rows write NaN.
    DoubleVec time_columns{};
};

struct ModelResult {
    std::string source;
    std::vector<ProfileRow> profiles;
    double total_landing;
    double initial_radial_velocity;
    double initial_landing_rate;
    double rotation_chi2;
    double rotation_dof;
    double rotation_reduced_chi2;
};

double surface_rate_number(const CsvRow& row,const std::string& column){
    if (row.contains(column)) return number(row,column);
    std::string legacy=column;
    const auto marker=legacy.find("_Msun_Gyr_");
    if (marker==std::string::npos) return number(row,column);
    if (column.starts_with("dSigma_")) {
        legacy.replace(marker,std::string::npos,"_Msun_kpc2_Myr");
        return number(row,legacy)*units::myr_per_gyr;
    }
    legacy.replace(marker,10,"_Msun_yr_");
    return units::per_gyr(number(row,legacy));
}

std::unordered_map<std::string,std::string> read_parameters(const std::filesystem::path& path){
    const CsvTable table=read(path);
    std::unordered_map<std::string,std::string> parameters;
    for (const CsvRow& row: table.rows) {
        parameters[value(row, row.contains("name") ? "name" : "parameter")]=value(row,"value");
    }
    if (parameters.contains("surface_rate_time_unit") && parameters.at("surface_rate_time_unit")!="Gyr")
        throw std::invalid_argument("Surface-rate and KS units must use Gyr");
    return parameters;
}

std::string parameter_string(const std::unordered_map<std::string,std::string>& parameters,const std::string& name,const std::string& fallback){
    const auto match=parameters.find(name);
    return match==parameters.end() || match->second.empty() ? fallback : match->second;
}

double parameter_double(const std::unordered_map<std::string,std::string>& parameters,const std::string& name,double fallback){
    const auto match=parameters.find(name);
    return match==parameters.end() || match->second.empty() ? fallback : std::stod(match->second);
}

double parameter_double(const std::unordered_map<std::string,std::string>& parameters,const std::string& name){
    return std::stod(parameters.at(name));
}
std::vector<double> radial_grid(double start,double stop,std::size_t count){
    std::vector<double> grid(count);
    for (std::size_t index=0; index<count; ++index) {
        grid[index]=start+(stop-start)*static_cast<double>(index)/static_cast<double>(count-1);
    }
    return grid;
}

void write_power_law_profiles(const std::filesystem::path& input,const std::filesystem::path& output){
    const auto settings=read_parameters(input/"parameters.csv");
    const double R_nucl=parameter_double(settings,"R_nucl_kpc");
    const double A_net=parameter_double(settings,"KS_A_net",std::numeric_limits<double>::quiet_NaN());
    const double A=std::isnan(A_net) ?
        (1.0-parameter_double(settings,"return_fraction"))*parameter_double(settings,"KS_A_SFR") : A_net;
    const PowerLawFountainParameters p{
        R_nucl,parameter_double(settings,"R_in_kpc",R_nucl),parameter_double(settings,"R_out_kpc"),
        parameter_double(settings,"Mdot_land_Msun_yr"),A,parameter_double(settings,"KS_N"),
        parameter_double(settings,"lambda",2.0),parameter_double(settings,"Z_land"),
        parameter_double(settings,"yield_y"),parameter_double(settings,"Z_outer_boundary")};
    for (double value: {p.R_nucl,p.R_in,p.R_out,p.Mdot_land,p.A,p.N,p.lambda,p.Z_land,p.yield,p.Z_out}) {
        if (!std::isfinite(value)) throw std::invalid_argument("Power-law parameters must be finite");
    }
    std::vector<double> radii;
    if (std::filesystem::exists(input/"radii.csv")) {
        for (const auto& row: read(input/"radii.csv").rows) radii.push_back(number(row,"R_kpc"));
    } else {
        const double count=parameter_double(settings,"n_points",1000.0);
        if (!std::isfinite(count))
            throw std::invalid_argument("n_points must be finite");
        radii=radial_grid(p.R_in,p.R_out,static_cast<std::size_t>(count));
    }
    std::filesystem::create_directories(output);
    std::ofstream stream(output/"profiles.csv");
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << std::setprecision(17)
        << "R_kpc,Sigmadot_land_Msun_Gyr_kpc2,Sigmadot_star_Msun_Gyr_kpc2,Sigma_g_Msun_kpc2,v_R_kpc_yr,v_R_kms,Mdot_acc_Msun_yr,cumulative_landing_Msun_yr,Z,dZ_dR_per_kpc,j_land_over_j_disk\n";
    const PowerLawDiskParameters disk_parameters{p.R_nucl,p.R_in,p.R_out,p.Mdot_land,p.A,p.N,p.lambda};
    for (double R: radii) {
        const auto row=power_law_fountain_profile(R,p);
        stream << R << ',' << row.landing << ',' << row.star << ',' << row.gas << ','
            << row.radial_velocity << ',' << row.radial_velocity*kpc_per_year_to_km_per_second << ','
            << row.accretion << ',' << row.cumulative_landing << ',' << row.metallicity << ','
            << row.metallicity_derivative << ','
            << 1.0-power_law_transport_length(R,disk_parameters)/R << '\n';
    }
    std::ofstream resolved(output/"parameters.csv");
    resolved.exceptions(std::ios::failbit | std::ios::badbit);
    resolved << std::setprecision(17) << "name,value\n"
        << "R_nucl_kpc," << p.R_nucl << "\nR_in_kpc," << p.R_in << "\nR_out_kpc," << p.R_out
        << "\nMdot_land_Msun_yr," << p.Mdot_land << "\nKS_A_net," << p.A << "\nKS_N," << p.N
        << "\nlambda," << p.lambda << "\nZ_land," << p.Z_land << "\nyield_y," << p.yield
        << "\nZ_outer_boundary," << p.Z_out << '\n';
    std::cout << "Wrote power-law profiles to " << output << '\n';
}

double analytic_gas(double R,const void* raw_parameters){
    const AnalyticProfileParameters& parameters=*static_cast<const AnalyticProfileParameters*>(raw_parameters);
    return sigma_g(R,parameters.R_g,parameters.sigma_g0);
}

double analytic_gas_derivative(double R,const void* raw_parameters){
    const AnalyticProfileParameters& parameters=*static_cast<const AnalyticProfileParameters*>(raw_parameters);
    return -analytic_gas(R,raw_parameters)/parameters.R_g;
}

double analytic_star(double R,const void* raw_parameters){
    const AnalyticProfileParameters& parameters=*static_cast<const AnalyticProfileParameters*>(raw_parameters);
    return sigma_star_ks(R,parameters.R_g,parameters.sigma_g0,parameters.A,parameters.N);
}

double prescribed_landing(double R,const void* raw_parameters){
    const LandingProfileParameters& parameters=*static_cast<const LandingProfileParameters*>(raw_parameters);
    if (parameters.kind=="smooth_broken_power_law")
        return parameters.Mdot_land/parameters.normalization*smooth_landing_shape(R,parameters.smooth);
    return sigmadot_land(R,parameters.Mdot_land,parameters.R_in,parameters.R_out,parameters.lambda);
}

double prescribed_cumulative_landing(double R,const LandingProfileParameters& parameters){
    if (R<=parameters.R_in) return 0.0;
    if (R>=parameters.R_out) return parameters.Mdot_land;
    if (parameters.kind=="smooth_broken_power_law")
        return parameters.Mdot_land/parameters.normalization*
            smooth_landing_integral(parameters.R_in,R,parameters.smooth);
    return cumulative_mdot_land(R,parameters.Mdot_land,parameters.R_in,parameters.R_out,parameters.lambda);
}

void write_landing_profile(const std::filesystem::path& path,const LandingProfileParameters& p){
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << std::setprecision(17) << "name,value\nlanding_profile," << p.kind
        << "\nMdot_land_Msun_yr," << p.Mdot_land << "\nR_in_kpc," << p.R_in
        << "\nR_out_kpc," << p.R_out << '\n';
    if (p.kind=="power_law") {
        stream << "lambda," << p.lambda << '\n';
    } else {
        stream << "landing_lambda_in," << p.smooth.lambda_in
            << "\nlanding_lambda_out," << p.smooth.lambda_out
            << "\nlanding_R_break_kpc," << p.smooth.R_break
            << "\nlanding_alpha," << p.smooth.alpha
            << "\nshape_integral_kpc2," << p.normalization
            << "\namplitude_at_break_Msun_Gyr_kpc2," << p.Mdot_land/p.normalization
            << "\nlanding_fit_R_min_kpc," << p.fit_R_min
            << "\nlanding_fit_R_max_kpc," << p.fit_R_max << '\n';
    }
}

void write_landing_prescription(const std::filesystem::path& path,
                               const LandingProfileParameters& p,double R_nucl,
                               const std::vector<ProfileRow>& rows){
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << std::setprecision(17)
        << "R_kpc,Sigmadot_land_Msun_Gyr_kpc2,cumulative_landing_Msun_yr,lambda_effective,"
        << "fixed_j_flat_stellar_sink_ratio,fixed_j_flat_stellar_sink_Msun_Gyr_kpc2\n";
    for (const auto& row: rows) {
        const double slope=p.kind=="smooth_broken_power_law" ?
            smooth_landing_log_slope(row.R,p.smooth) : p.lambda;
        // This diagnoses the unmixed, flat-rotation j_land=j_nucl closure;
        // it is separate from the supplied stellar sink in an inverse run.
        const double ratio=power_law_stellar_sink_ratio(row.R,R_nucl,slope);
        stream << row.R << ',' << row.landing << ',' << row.cumulative_landing
            << ',' << slope << ',' << ratio << ',' << row.landing*ratio << '\n';
    }
}

double empirical_inverse_velocity(double R,const void* raw_parameters){
    const EmpiricalInverseVelocityParameters& parameters=*static_cast<const EmpiricalInverseVelocityParameters*>(raw_parameters);
    return empirical_inverse_radial_velocity(R,*parameters.solution,*parameters.parameters);
}

RotationCurve make_rotation_curve(const RotationInput& input){
    if (input.kind=="flat") {
        return {{flat_rotation_velocity,&input.flat},{flat_rotation_velocity_derivative,&input.flat}};
    }
    return {{linear_profile_value,&input.tabulated_velocity},{linear_profile_value,&input.tabulated_derivative}};
}

void validate_tabulated_profile(const LinearProfile& profile,const std::string& description){
    for (std::size_t index=0; index<profile.radius.size(); ++index) {
        if (!std::isfinite(profile.radius[index]) || !std::isfinite(profile.value[index])) {
            throw std::runtime_error(description+" contains non-finite values");
        }
    }
}

void load_tabulated_rotations(const std::filesystem::path& path,std::vector<RotationInput>& rotations){
    std::unordered_map<std::string,RotationInput*> tabulated;
    for (RotationInput& rotation: rotations) {
        if (rotation.kind=="tabulated") {
            tabulated[rotation.source]=&rotation;
        }
    }
    if (tabulated.empty()) {
        return;
    }
    const CsvTable table=read(path);
    for (const CsvRow& row: table.rows) {
        const std::string source=value(row,"source");
        const auto match=tabulated.find(source);
        if (match==tabulated.end()) {
            continue;
        }
        RotationInput& rotation=*match->second;
        const double radius=number(row,"R_kpc");
        rotation.tabulated_velocity.radius.push_back(radius);
        rotation.tabulated_velocity.value.push_back(number(row,"V_kms"));
        rotation.tabulated_derivative.radius.push_back(radius);
        rotation.tabulated_derivative.value.push_back(number(row,"dV_dR_kms_kpc"));
    }
    for (const auto& entry: tabulated) {
        validate_tabulated_profile(entry.second->tabulated_velocity,entry.first+" velocity profile");
        validate_tabulated_profile(entry.second->tabulated_derivative,entry.first+" derivative profile");
    }
}

std::vector<RotationInput> read_rotations(const std::filesystem::path& path,const std::filesystem::path& profiles_path){
    const CsvTable table=read(path);
    std::vector<RotationInput> rotations;
    for (const CsvRow& row: table.rows) {
        RotationInput input{};
        input.source=value(row,"source");
        input.kind=value(row,"kind");
        if (input.kind=="flat") {
            input.flat.velocity=number(row,"Vflat_kms");
        }
        input.chi2=number(row,"chi2");
        input.dof=number(row,"dof");
        input.reduced_chi2=number(row,"reduced_chi2");
        rotations.push_back(input);
    }
    load_tabulated_rotations(profiles_path,rotations);
    return rotations;
}

const RotationInput& primary_rotation(const std::vector<RotationInput>& rotations,const std::string& profile_type){
    if (profile_type=="tabulated") {
        const auto match=std::find_if(rotations.begin(),rotations.end(),[](const RotationInput& rotation){
            return rotation.source==primary_observational_rotation;
        });
        if (match!=rotations.end()) return *match;
    }
    return rotations.front();
}

void load_tabulated_star_profile(const std::filesystem::path& path,LinearProfile& star){
    const CsvTable table=read(path);
    for (const CsvRow& row: table.rows) {
        const double radius=number(row,"R_kpc");
        star.radius.push_back(radius);
        star.value.push_back(surface_rate_number(row,"Sigmadot_star_Msun_Gyr_kpc2"));
    }
    validate_tabulated_profile(star,"Leroy star-formation profile");
}

void load_observed_metallicity(const std::filesystem::path& path,LinearProfile& metallicity,LinearProfile& derivative){
    for (const CsvRow& row: read(path).rows) {
        const double R=number(row,"R_kpc");
        metallicity.radius.push_back(R);
        metallicity.value.push_back(number(row,"Z"));
        derivative.radius.push_back(R);
        derivative.value.push_back(number(row,"dZ_dR_per_kpc"));
    }
    validate_tabulated_profile(metallicity,"Observed metallicity profile");
    validate_tabulated_profile(derivative,"Observed metallicity derivative");
}

void load_tabulated_gas_profiles(
    const std::filesystem::path& path,
    LinearProfile& gas,
    LinearProfile& gas_derivative
){
    const CsvTable table=read(path);
    for (const CsvRow& row: table.rows) {
        const double radius=number(row,"R_kpc");
        gas.radius.push_back(radius);
        gas.value.push_back(number(row,"Sigma_g_Msun_kpc2"));
        gas_derivative.radius.push_back(radius);
        gas_derivative.value.push_back(number(row,"dSigma_g_dR_Msun_kpc3"));
    }
    validate_tabulated_profile(gas,"Leroy gas spline");
    validate_tabulated_profile(gas_derivative,"Leroy gas spline derivative");
}

void write_profiles(const std::filesystem::path& path,const std::vector<ModelResult>& results){
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << "model,source,R_kpc,Sigma_g_Msun_kpc2,Sigmadot_star_Msun_Gyr_kpc2,Sigmadot_land_Msun_Gyr_kpc2,cumulative_landing_Msun_yr,v_R_kpc_yr,v_R_kms,F_R_Msun_yr,Mdot_acc_Msun_yr,t_inflow_Gyr,t_depletion_Gyr,v_c_kms,dv_c_dR_kms_kpc,j_disk_kpc_kms,j_CGM_kpc_kms,j_land_kpc_kms,delta_j_kpc_kms,j_land_over_j_disk,j_land_over_j_nucl,mu_j,Z,Z_eq,dZ_dR_per_kpc,Z_land_required,mu_Z,t_Myr,R_lo_kpc,R_hi_kpc,dSigma_g_dt_Msun_Gyr_kpc2,dSigma_Z_dt_Msun_Gyr_kpc2,dZ_dt_per_Myr,mass_flux_inner_Msun_yr,mass_flux_outer_Msun_yr,metal_flux_inner_Msun_yr,metal_flux_outer_Msun_yr,v_R_outer_kpc_yr,j_land_supplied_kpc_kms,Z_land_supplied,j_nucl_kpc_kms,R_j_kpc,Sigmadot_wind_Msun_Gyr_kpc2,Sigmadot_CGM_Msun_Gyr_kpc2,Jdot_land_Msun_Gyr_Myr,Zdot_land_Msun_Gyr_kpc2,mu_effective,ell_kpc,Sigmadot_eff_Eulerian_Msun_Gyr_kpc2\n";
    stream << std::setprecision(17);
    for (const ModelResult& result: results) {
        for (const ProfileRow& row: result.profiles) {
            stream << escape(row.model) << ',' << escape(row.source) << ',' << row.R << ',' << row.gas << ',' << row.star << ',' << row.landing << ',' << row.cumulative_landing << ',' << row.radial_velocity << ',' << row.radial_velocity_kms << ',' << row.radial_mass_flux << ',' << row.accretion << ',' << row.inflow_time << ',' << row.depletion_time << ',' << row.circular_velocity << ',' << row.circular_velocity_derivative << ',' << row.disk_angular_momentum << ',' << row.cgm_angular_momentum << ',' << row.landing_angular_momentum << ',' << row.angular_momentum_difference << ',' << row.landing_to_disk << ',' << row.landing_to_nuclear << ',' << row.mu_angular_momentum << ',' << row.metallicity << ',' << row.equilibrium_metallicity << ',' << row.metallicity_derivative << ',' << row.required_landing_metallicity << ',' << row.mu_metallicity;
            for (std::size_t i=0; i<22; ++i) {
                double value=row.time_columns.empty() ? std::numeric_limits<double>::quiet_NaN() : row.time_columns.at(i);
                if (i==3 || i==4) value*=units::myr_per_gyr;
                stream << ',' << value;
            }
            stream << '\n';
        }
    }
}

bool landing_nonnegative(const ModelResult& result){
    double minimum_landing=std::numeric_limits<double>::infinity();
    double maximum_absolute_landing=0.0;
    for (const ProfileRow& row: result.profiles) {
        minimum_landing=std::min(minimum_landing,row.landing);
        maximum_absolute_landing=std::max(maximum_absolute_landing,std::abs(row.landing));
    }
    const double tolerance=1e-10*std::max(maximum_absolute_landing,units::per_gyr(1.0));
    return minimum_landing>=-tolerance;
}

void write_summary(const std::filesystem::path& path,const std::string& model,const std::string& profile_type,const std::string& galaxy,const std::vector<ModelResult>& results,double R_nucl,double R_out,double Mdot_land,double Mdot_out,double mu,double beta,double Z_nucl,double Z_CGM,double yield,double Z_boundary,double R_Z_boundary,bool metallicity_solved,double t_Myr=std::numeric_limits<double>::quiet_NaN(),double R_min=std::numeric_limits<double>::quiet_NaN()){
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << "model,profile_type,galaxy,source,R_nucl_kpc,R_out_kpc,Mdot_land_Msun_yr,Mdot_out_Msun_yr,mu,beta,Z_nucl,Z_CGM,yield_y,Z_outer_boundary,R_Z_boundary_kpc,metallicity_solved,total_landing_Msun_yr,initial_radial_velocity_kpc_yr,initial_landing_rate_Msun_Gyr_kpc2,chi2,dof,reduced_chi2,landing_nonnegative,t_Myr,R_min_kpc\n";
    stream << std::setprecision(17);
    for (const ModelResult& result: results) {
        stream << escape(model) << ',' << escape(profile_type) << ',' << escape(galaxy) << ',' << escape(result.source) << ',' << R_nucl << ',' << R_out << ',' << Mdot_land << ',' << (model=="forward" ? result.profiles.back().accretion : Mdot_out) << ',' << mu << ',' << beta << ',' << Z_nucl << ',' << Z_CGM << ',' << yield << ',' << Z_boundary << ',' << R_Z_boundary << ',' << (metallicity_solved ? 1 : 0) << ',' << result.total_landing << ',' << result.initial_radial_velocity << ',' << result.initial_landing_rate << ',' << result.rotation_chi2 << ',' << result.rotation_dof << ',' << result.rotation_reduced_chi2 << ',' << (landing_nonnegative(result) ? "true" : "false") << ',' << t_Myr << ',' << R_min << '\n';
    }
}

void copy_if_present(const std::filesystem::path& source,const std::filesystem::path& destination){
    if (std::filesystem::exists(source)) {
        std::filesystem::copy_file(source,destination,std::filesystem::copy_options::overwrite_existing);
    }
}

// Reuse the usual profile writer for snapshots, retaining the finite-volume
// time derivatives and face transport instead of invoking the steady solver.
void write_evolving_snapshots(const std::string& model,const std::filesystem::path& input,
                             const std::filesystem::path& output,DoubleVec times){
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const auto parameters=read_parameters(input/"parameters.csv");
    const auto cgm_solution=load_cgm(parameters);
    const auto finite=[](const CsvRow& row,const char* key){
        const double result=number(row,key);
        if (!std::isfinite(result)) throw std::runtime_error(std::string("Snapshot needs finite ")+key+"; regenerate older kernel outputs with the current config and initial disk");
        return result;
    };
    const CsvTable gas_table=read(input/"gas_evolution.csv");
    const CsvTable landing_table=read(input/"landing_sources.csv");
    std::map<double,std::vector<const CsvRow*>> gas_times,landing_times;
    for (const auto& row: gas_table.rows) gas_times[finite(row,"t_Myr")].push_back(&row);
    for (const auto& row: landing_table.rows) landing_times[finite(row,"t_Myr")].push_back(&row);
    if (times.empty()) times.push_back(gas_times.rbegin()->first);
    // Resolve only saved times; never silently treat an interpolated state as a saved snapshot.
    for (double& t: times) {
        const auto match=std::find_if(gas_times.begin(),gas_times.end(),[&](const auto& entry){
            return std::abs(entry.first-t)<=1e-10*std::max(1.0,std::abs(t));
        });
        t=match->first;
    }
    std::sort(times.begin(),times.end());
    times.erase(std::unique(times.begin(),times.end()),times.end());
    std::vector<std::string> directories;
    for (const double t: times) {
        auto rows=gas_times.at(t);
        auto landing_rows=landing_times.at(t);
        const auto by_radius=[&](const CsvRow* a,const CsvRow* b){return finite(*a,"R_lo_kpc")<finite(*b,"R_lo_kpc");};
        std::sort(rows.begin(),rows.end(),by_radius);
        std::sort(landing_rows.begin(),landing_rows.end(),by_radius);
        const std::size_t n=rows.size();
        MassContinuityParameters p;
        const double R_nucl=parameter_double(parameters,"R_nucl_kpc");
        const double beta=cgm_solution ? nan : parameter_double(parameters,"beta");
        const double velocity_conversion=parameter_double(parameters,"kms_to_kpc_per_Myr");
        p.Z_nucl=parameter_double(parameters,"Z_nucl");
        p.Z_CGM=parameter_double(parameters,"Z_CGM");
        p.yield=parameter_double(parameters,"yield_y");
        p.inner_mass_flux=parameter_double(parameters,"inner_mass_flux_Msun_yr");
        p.outer_mass_flux=parameter_double(parameters,"outer_mass_flux_Msun_yr");
        p.Z_inner_inflow=parameter_double(parameters,"Z_inner_inflow");
        p.Z_outer_inflow=parameter_double(parameters,"Z_outer_inflow");
        const double j_nucl=parameter_double(parameters,"j_nucl_kpc_kms");

        DoubleVec wind(n),angular(n),metals(n);
        DoubleVec state(2*n),derivative(2*n),landing(n),radius(n),area(n),Z(n),vc(n),dvc(n),edge_velocity(n),edge_vc(n),edge_dvc(n);
        for (std::size_t i=0; i<n; ++i) {
            const auto& row=*rows[i];
            const double lo=finite(row,"R_lo_kpc"),hi=finite(row,"R_hi_kpc");
            finite(*landing_rows[i],"R_lo_kpc");
            finite(*landing_rows[i],"R_hi_kpc");
            if (i==0) p.R_bins.push_back(lo);
            p.R_bins.push_back(hi);
            radius[i]=0.5*(lo+hi);
            area[i]=pi*(hi-lo)*(hi+lo);
            state[i]=finite(row,"Sigma_g_Msun_kpc2");
            state[n+i]=finite(row,"Sigma_Z_Msun_kpc2");
            if (state[i]<0.0 || state[n+i]<0.0 || state[n+i]>state[i]) {
                throw std::runtime_error("Snapshot densities must satisfy 0 <= Sigma_Z <= Sigma_g");
            }
            derivative[i]=surface_rate_number(row,"dSigma_g_dt_Msun_Gyr_kpc2")/units::myr_per_gyr;
            derivative[n+i]=surface_rate_number(row,"dSigma_Z_dt_Msun_Gyr_kpc2")/units::myr_per_gyr;
            p.sigmadot_star.push_back(surface_rate_number(row,"Sigmadot_star_Msun_Gyr_kpc2"));
            landing[i]=surface_rate_number(*landing_rows[i],"Sigmadot_land_Msun_Gyr_kpc2");
            wind[i]=surface_rate_number(*landing_rows[i],"Sigmadot_wind_Msun_Gyr_kpc2");
            angular[i]=surface_rate_number(*landing_rows[i],"Jdot_land_Msun_Gyr_Myr");
            metals[i]=surface_rate_number(*landing_rows[i],"Zdot_land_Msun_Gyr_kpc2");
            vc[i]=finite(row,"v_c_kms");
            dvc[i]=finite(row,"dv_c_dR_kms_kpc");
            edge_velocity[i]=model=="disk-evolution" && i+1<n ? finite(row,"v_R_outer_kpc_yr") : 0.0;
            edge_vc[i]=finite(row,"v_c_outer_kms");
            edge_dvc[i]=finite(row,"dv_c_dR_outer_kms_kpc");
            if (landing[i]<0.0 || p.sigmadot_star[i]<0.0 || vc[i]<=0.0 || vc[i]+radius[i]*dvc[i]<=0.0) {
                throw std::runtime_error("Snapshot needs nonnegative sources and a positive circular speed and dj_disk/dR");
            }
            Z[i]=state[i]>0.0 ? state[n+i]/state[i] : nan;
        }
        DoubleVec flux(n+1);
        if (model=="reconstruction") {
            flux=reconstruct_mass_flux(derivative,landing,p);
        } else {
            flux.front()=p.inner_mass_flux;
            flux.back()=p.outer_mass_flux;
            for (std::size_t edge=1; edge<n; ++edge) {
                const double velocity=edge_velocity[edge-1];
                flux[edge]=2.0*pi*p.R_bins[edge]*velocity*state[velocity>=0.0 ? edge-1 : edge];
            }
        }
        const DoubleVec metal_flux=disk_metal_flux(state,flux,p);
        const DoubleVec required_Z=reconstruct_landing_metallicity(state,derivative,landing,metal_flux,p);
        const auto divide=[&](double a,double b){return std::isfinite(a) && std::isfinite(b) && b!=0.0 ? a/b : nan;};
        ModelResult result{"orbit_potential",{},0.0,nan,landing.front(),nan,nan,nan};
        double cumulative=0.0;
        for (std::size_t i=0; i<n; ++i) {
            const double R=radius[i],lo=p.R_bins[i],hi=p.R_bins[i+1];
            // Constant annular source/divergence: integrate by area to the midpoint.
            const double inner_area=pi*(R-lo)*(R+lo),fraction=inner_area/area[i];
            const double F=flux[i]+fraction*(flux[i+1]-flux[i]);
            const double velocity=divide(F,2.0*pi*R*state[i]);
            // Angular-momentum closure lives on internal faces, like the solver's
            // transport. R_j_kpc records this staggered sampling explicitly.
            const double disk=j_disk(hi,edge_vc[i]);
            const double cgm=cgm_solution ? cgm_j_kpc_kms(hi,&*cgm_solution) : j_cgm(hi,beta,edge_vc[i]);
            const double weight=i+1<n ? (hi-lo)/(p.R_bins[i+2]-lo) : 0.0;
            const double face_landing=i+1<n ? (1.0-weight)*landing[i]+weight*landing[i+1] : 0.0;
            const double face_angular=i+1<n ? (1.0-weight)*angular[i]+weight*angular[i+1] : 0.0;
            const double supplied_j=divide(face_angular,face_landing*velocity_conversion);
            const double Z_land=divide(metals[i],landing[i]);
            const double required_j=face_landing>0.0 ? disk+units::per_gyr(flux[i+1])/(2.0*pi*hi*face_landing)*dj_disk_dR(hi,edge_vc[i],edge_dvc[i]) : nan;
            const double incident_j=face_landing>0.0 ? (model=="reconstruction" ? required_j : supplied_j) : nan;
            const std::size_t left=i==0 ? 0 : i-1,right=i+1<n ? i+1 : i;
            const double dZ_dR=divide(Z[right]-Z[left],radius[right]-radius[left]);
            const double dZ_dt=divide(derivative[n+i]-Z[i]*derivative[i],state[i]);
            double outer_velocity=nan;
            if (i+1<n) outer_velocity=divide(flux[i+1],2.0*pi*hi*state[flux[i+1]>=0.0 ? i : i+1]);
            else if (flux.back()==0.0) outer_velocity=0.0;
            else if (flux.back()>0.0) outer_velocity=divide(flux.back(),2.0*pi*hi*state[i]);
            const double cumulative_at_R=cumulative+units::per_year(inner_area*landing[i]);
            cumulative+=units::per_year(area[i]*landing[i]);
            result.profiles.push_back({model,result.source,R,state[i],p.sigmadot_star[i],landing[i],cumulative_at_R,
                velocity,velocity*kpc_per_year_to_km_per_second,F/(2.0*pi),-F,
                divide(R,std::abs(velocity))/1e9,divide(state[i],p.sigmadot_star[i]),vc[i],dvc[i],disk,cgm,
                incident_j,incident_j-disk,divide(incident_j,disk),divide(incident_j,j_nucl),
                nan,Z[i],landing[i]>0.0 ? Z_land+p.yield*p.sigmadot_star[i]/landing[i] : nan,
                dZ_dR,required_Z[i],divide(required_Z[i]-p.Z_nucl,p.Z_CGM-required_Z[i]),
                {t,lo,hi,derivative[i],derivative[n+i],dZ_dt,flux[i],flux[i+1],metal_flux[i],metal_flux[i+1],outer_velocity,supplied_j,Z_land,j_nucl,hi,
                 wind[i],landing[i]-wind[i],angular[i],metals[i],divide(landing[i]-wind[i],wind[i]),
                 divide(supplied_j-disk,dj_disk_dR(hi,edge_vc[i],edge_dvc[i])),derivative[i]*units::myr_per_gyr+p.sigmadot_star[i]}});
        }
        result.total_landing=cumulative;
        result.initial_radial_velocity=result.profiles.front().radial_velocity;
        char buffer[64];
        const auto converted=std::to_chars(buffer,buffer+sizeof(buffer),t);
        const std::string directory="t_"+std::string(buffer,converted.ptr)+"_Myr";
        const auto destination=output/directory;
        std::filesystem::create_directories(destination);
        const std::vector<ModelResult> results{std::move(result)};
        write_profiles(destination/"profiles.csv",results);
        write_summary(destination/"summary.csv",model,"evolving","illustrative",results,R_nucl,p.R_bins.back(),
            cumulative,-p.outer_mass_flux,nan,beta,p.Z_nucl,p.Z_CGM,p.yield,nan,nan,true,t,p.R_bins.front());
        std::ofstream rotations(destination/"rotation_curves.csv");
        rotations.exceptions(std::ios::failbit | std::ios::badbit);
        rotations << "source,kind,Vflat_kms,lflat_kpc,chi2,dof,reduced_chi2\norbit_potential,tabulated,nan,nan,nan,nan,nan\n";
        copy_if_present(input/"parameters.csv",destination/"parameters.csv");
        if (cgm_solution) write_cgm_profile(destination,*cgm_solution);
        std::ofstream provenance(destination/"snapshot.csv");
        provenance.exceptions(std::ios::failbit | std::ios::badbit);
        provenance << std::setprecision(17) << "parameter,value\ninput_directory," << escape(std::filesystem::absolute(input).string())
            << "\nt_Myr," << t << "\nmodel," << model
            << "\ntime_derivatives,instantaneous_saved_rhs\nlanding_mass_origin,nuclear_plus_cgm\ncumulative_from_R_kpc," << p.R_bins.front()
            << "\nradial_sampling,annulus_midpoints_with_area_integrated_flux\nangular_sampling,R_j_kpc_internal_faces\nmetal_transport,shared_upwind_faces\n"
            << "inner_mass_flux_inferred_Msun_yr," << flux.front() << '\n';
        directories.push_back(directory);
    }
    std::ofstream manifest(output/"snapshots.csv");
    manifest.exceptions(std::ios::failbit | std::ios::badbit);
    manifest << std::setprecision(17) << "t_Myr,directory\n";
    for (std::size_t i=0; i<times.size(); ++i) manifest << times[i] << ',' << directories[i] << '\n';
    std::cout << "Wrote " << times.size() << " " << model << " snapshots to " << output << '\n';
}

double power_law_disk_gas(double R,const void* raw){
    return power_law_disk_profile(R,*static_cast<const PowerLawDiskParameters*>(raw)).gas;
}
double power_law_disk_star(double R,const void* raw){
    return power_law_disk_profile(R,*static_cast<const PowerLawDiskParameters*>(raw)).star;
}
double power_law_disk_landing(double R,const void* raw){
    return power_law_disk_profile(R,*static_cast<const PowerLawDiskParameters*>(raw)).landing;
}
double power_law_disk_velocity(double R,const void* raw){
    return power_law_disk_profile(R,*static_cast<const PowerLawDiskParameters*>(raw)).radial_velocity;
}

void write_mixed_power_law_profiles(const std::filesystem::path& input,const std::filesystem::path& output){
    const auto settings=read_parameters(input/"parameters.csv");
    if (parameter_string(settings,"closure","")!="j_mix" ||
        parameter_string(settings,"landing_profile","")!="power_law" ||
        parameter_string(settings,"cgm_model","")!="constant_beta" ||
        parameter_string(settings,"rotation_model","")!="flat")
        throw std::invalid_argument("Require j_mix closure, power-law landing, constant-beta CGM and flat rotation");
    for (const char* key: {"R_g_kpc","M_g_Msun","Mdot_out_Msun_yr","profile_type","metallicity_mode"}) {
        if (settings.contains(key))
            throw std::invalid_argument(std::string(key)+" is not an input to the mixed power-law disk");
    }
    const double mu=parameter_double(settings,"mu");
    const double beta=parameter_double(settings,"beta");
    const double eta=parameter_double(settings,"eta");
    const double nuclear_rate=parameter_double(settings,"Mdot_nuc_Msun_yr");
    const double returned=parameter_double(settings,"return_fraction");
    if (!(std::isfinite(eta) && eta>0.0 && std::isfinite(nuclear_rate) && nuclear_rate>0.0 &&
          std::isfinite(returned) && returned>=0.0 && returned<1.0))
        throw std::invalid_argument("Require positive eta/source rate and 0 <= return_fraction < 1");
    const PowerLawDiskParameters p{
        parameter_double(settings,"R_nucl_kpc"),parameter_double(settings,"R_in_kpc"),
        parameter_double(settings,"R_out_kpc"),eta*(1.0+mu)*nuclear_rate,
        (1.0-returned)*parameter_double(settings,"KS_A_SFR"),parameter_double(settings,"KS_N"),
        parameter_double(settings,"lambda"),mu,beta};
    validate_power_law_disk(p);
    if (power_law_transport_length(p.R_in,p)<=0.0 || power_law_transport_length(p.R_out,p)<=0.0)
        throw std::invalid_argument("The mixed metallicity solve requires inward transport throughout the annulus");
    const double vc=parameter_double(settings,"v_c_nucl_kms");
    const double Z_nucl=parameter_double(settings,"Z_nucl");
    const double Z_CGM=parameter_double(settings,"Z_CGM");
    const double yield=parameter_double(settings,"yield_y");
    const double Z_out=parameter_double(settings,"Z_outer_boundary");
    const double count=parameter_double(settings,"n_points");
    const double h0=parameter_double(settings,"h_0_kpc");
    const double rtol=parameter_double(settings,"rtol");
    const double atol=parameter_double(settings,"metallicity_atol");
    for (double value: {vc,Z_nucl,Z_CGM,yield,Z_out,count,h0,rtol,atol}) {
        if (!std::isfinite(value)) throw std::invalid_argument("Mixed disk parameters must be finite");
    }
    if (vc<=0.0 || Z_nucl<0.0 || Z_nucl>1.0 || Z_CGM<0.0 || Z_CGM>1.0 ||
        yield<0.0 || Z_out<0.0 || Z_out>1.0 || count<2.0 || count>1000000.0 ||
        std::floor(count)!=count || h0<=0.0 || rtol<=0.0 || atol<=0.0)
        throw std::invalid_argument("Invalid rotation, chemistry, grid or integration controls");
    const double Z_land=z_land_mixing(Z_nucl,Z_CGM,mu);
    const EmpiricalMetallicityModelParameters chemistry{
        {power_law_disk_gas,&p},{power_law_disk_velocity,&p},
        {power_law_disk_landing,&p},{power_law_disk_star,&p},Z_land,yield};
    const auto solution=integrate_rk4(empirical_metallicity_rhs,&chemistry,p.R_out,{Z_out},
                                    h0,atol,rtol,p.R_in,h0);
    const auto radii=radial_grid(p.R_in,p.R_out,static_cast<std::size_t>(count));
    std::filesystem::create_directories(output);
    std::ofstream stream(output/"profiles.csv");
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << std::setprecision(17)
        << "model,source,R_kpc,Sigmadot_land_Msun_Gyr_kpc2,Sigmadot_star_Msun_Gyr_kpc2,"
        << "Sigma_g_Msun_kpc2,v_R_kpc_yr,v_R_kms,Mdot_acc_Msun_yr,cumulative_landing_Msun_yr,"
        << "v_c_kms,j_disk_kpc_kms,j_CGM_kpc_kms,j_land_kpc_kms,Z_land,Z,dZ_dR_per_kpc\n";
    for (double R: radii) {
        const auto disk=power_law_disk_profile(R,p);
        const double Z=evaluate(solution,R)[0];
        if (!std::isfinite(Z) || Z<0.0 || Z>1.0)
            throw std::runtime_error("Mixed disk metallicity must be finite and between zero and one");
        const double gradient=empirical_metallicity_rhs(R,{Z},&chemistry)[0];
        const double incident_j=j_land_mixing(R,p.R_nucl,mu,beta,vc,vc);
        stream << "mixed-power-law,analytic_flat," << R << ',' << disk.landing << ',' << disk.star << ','
            << disk.gas << ',' << disk.radial_velocity << ',' << disk.radial_velocity*kpc_per_year_to_km_per_second
            << ',' << disk.accretion << ',' << disk.cumulative_landing << ',' << vc << ',' << j_disk(R,vc)
            << ',' << j_cgm(R,beta,vc) << ',' << incident_j << ',' << Z_land << ',' << Z << ',' << gradient << '\n';
    }
    copy_if_present(input/"parameters.csv",output/"parameters.csv");
    const auto inner=power_law_disk_profile(p.R_in,p),outer=power_law_disk_profile(p.R_out,p);
    std::ofstream summary(output/"summary.csv");
    summary.exceptions(std::ios::failbit | std::ios::badbit);
    summary << std::setprecision(17) << "quantity,value\nclosure,j_mix\ngas_profile,derived_from_net_KS\n"
        << "Mdot_land_Msun_yr," << p.Mdot_land << "\nMdot_acc_inner_Msun_yr," << inner.accretion
        << "\nMdot_acc_outer_Msun_yr," << outer.accretion << "\nZ_land," << Z_land << '\n';
    std::ofstream method(output/"metallicity_solver.csv");
    method.exceptions(std::ios::failbit | std::ios::badbit);
    method << "method,boundary\nRK4_mixed_power_law,Z_outer_boundary\n";
    std::cout << "Wrote mixed power-law profiles to " << output << '\n';
}

}

int main(int argc,char** argv){
    try {
        const auto cgm_config=cgm_config_argument(argc,argv);
        if (argc<4) throw std::invalid_argument("Supply a model command, input directory and output directory");
        std::string snapshot_mode=argv[1];
        // Accept earlier commands, but write only the current names to outputs.
        if (snapshot_mode=="forward-time") snapshot_mode="disk-evolution";
        if (snapshot_mode=="inverse-time") snapshot_mode="reconstruction";
        if (snapshot_mode=="disk-evolution" || snapshot_mode=="reconstruction") {
            DoubleVec times;
            for (int i=4; i<argc; ++i) {
                std::size_t used=0;
                const double t=std::stod(argv[i],&used);
                if (!std::isfinite(t) || used!=std::string(argv[i]).size()) throw std::runtime_error("Snapshot times must be finite Myr values");
                times.push_back(t);
            }
            write_evolving_snapshots(snapshot_mode,argv[2],argv[3],times);
            return 0;
        }
        if (snapshot_mode=="power-law") {
            write_power_law_profiles(argv[2],argv[3]);
            return 0;
        }
        if (snapshot_mode=="mixed-power-law") {
            if (argc!=4 || !cgm_config.empty())
                throw std::invalid_argument("mixed-power-law uses its explicit constant-beta parameters only");
            write_mixed_power_law_profiles(argv[2],argv[3]);
            return 0;
        }
        if (snapshot_mode=="prescribed-gas") {
            if (argc!=4 || !cgm_config.empty())
                throw std::invalid_argument("prescribed-gas takes its own explicit gas/landing/rotation parameters");
            write_prescribed_gas_model(argv[2],argv[3]);
            return 0;
        }

        const std::string model=argv[1];
        const std::filesystem::path input_directory=argv[2];
        const std::filesystem::path output_directory=argv[3];
        const auto parameters=read_parameters(input_directory/"parameters.csv");
        const auto cgm_solution=load_cgm(parameters,cgm_config);
        const RadialProfile cgm_j_profile=cgm_solution ? RadialProfile{cgm_j_kpc_kms,&*cgm_solution} : RadialProfile{};
        const std::string metallicity_mode=parameter_string(parameters,"metallicity_mode","predicted");
        const bool observed_chemistry=metallicity_mode=="observed";
        LinearProfile observed_metallicity,observed_metallicity_derivative;
        if (observed_chemistry)
            load_observed_metallicity(input_directory/"metallicity_profiles.csv",observed_metallicity,observed_metallicity_derivative);
        const std::string profile_type=parameter_string(parameters,"profile_type","tabulated");
        const std::string galaxy=parameter_string(parameters,"galaxy",profile_type=="analytic" ? "analytic" : "unknown");
        const double R_nucl=parameter_double(parameters,"R_nucl_kpc");
        const double R_in=parameter_double(parameters,"R_in_kpc");
        const double R_out=parameter_double(parameters,"R_out_kpc");
        const double lambda=parameter_double(parameters,"lambda",2.0);
        const std::string landing_kind=parameter_string(parameters,"landing_profile","power_law");
        const bool analytic_power_law=profile_type=="analytic" && landing_kind=="power_law";
        const double mu=parameter_double(parameters,"mu",1.0);
        const double beta=cgm_solution ? std::numeric_limits<double>::quiet_NaN() : parameter_double(parameters,"beta",0.75);
        const double eta=parameter_double(parameters,"eta");
        const double Mdot_nuc=parameter_double(parameters,"Mdot_nuc_Msun_yr");
        const double Mdot_land=eta*(1.0+mu)*Mdot_nuc;
        const double return_fraction=parameter_double(parameters,"return_fraction");
        const double Z_nucl=parameter_double(parameters,"Z_nucl",0.02);
        const double Z_CGM=parameter_double(parameters,"Z_CGM",0.003);
        const double yield=parameter_double(parameters,"yield_y",0.015);
        const double Z_boundary=parameter_double(parameters,"Z_outer_boundary",0.006);
        const double h_0=parameter_double(parameters,"h_0_kpc",0.01);
        const double rtol=parameter_double(parameters,"rtol",1e-9);
        const double dynamics_atol=parameter_double(parameters,"dynamics_atol",1e-18);
        const double metallicity_atol=parameter_double(parameters,"metallicity_atol",1e-12);
        const std::size_t point_count=static_cast<std::size_t>(parameter_double(parameters,"n_points",300.0));
        const std::vector<double> grid=radial_grid(R_in,R_out,point_count);
        const std::vector<RotationInput> rotations=read_rotations(
            input_directory/"rotation_curves.csv",
            input_directory/"rotation_profiles.csv"
        );
        const auto launch_velocity=[&](const RotationInput& rotation){
            const double configured=parameter_double(parameters,"v_c_nucl_kms",std::numeric_limits<double>::quiet_NaN());
            if (std::isfinite(configured) && configured>0.0) return configured;
            return rotation_velocity(make_rotation_curve(rotation),R_nucl);
        };

        AnalyticProfileParameters analytic_profiles{};
        LinearProfile tabulated_gas;
        LinearProfile tabulated_gas_derivative;
        LinearProfile tabulated_star;
        RadialProfile gas;
        RadialProfile gas_derivative;
        RadialProfile star;
        if (profile_type=="analytic") {
            const double R_g=parameter_double(parameters,"R_g_kpc");
            const double M_g=parameter_double(parameters,"M_g_Msun");
            const double A=(1.0-return_fraction)*parameter_double(parameters,"KS_A_SFR");
            const double N=parameter_double(parameters,"KS_N");
            analytic_profiles={R_g,sigma_g0(M_g,R_g,R_in,R_out),A,N};
            gas={analytic_gas,&analytic_profiles};
            gas_derivative={analytic_gas_derivative,&analytic_profiles};
            star={analytic_star,&analytic_profiles};
        } else {
            load_tabulated_gas_profiles(
                input_directory/"gas_profiles.csv",
                tabulated_gas,
                tabulated_gas_derivative
            );
            load_tabulated_star_profile(input_directory/"profiles.csv",tabulated_star);
            gas={linear_profile_value,&tabulated_gas};
            gas_derivative={linear_profile_value,&tabulated_gas_derivative};
            star={linear_profile_value,&tabulated_star};
        }

        LandingProfileParameters landing_parameters{Mdot_land,R_in,R_out,lambda};
        landing_parameters.kind=landing_kind;
        if (landing_kind=="smooth_broken_power_law") {
            auto& shape=landing_parameters.smooth;
            shape.lambda_in=parameter_double(parameters,"landing_lambda_in",shape.lambda_in);
            shape.lambda_out=parameter_double(parameters,"landing_lambda_out",shape.lambda_out);
            shape.R_break=parameter_double(parameters,"landing_R_break_kpc",shape.R_break);
            shape.alpha=parameter_double(parameters,"landing_alpha",shape.alpha);
            landing_parameters.fit_R_min=parameter_double(parameters,"landing_fit_R_min_kpc",1.0);
            landing_parameters.fit_R_max=parameter_double(parameters,"landing_fit_R_max_kpc",15.0);
            landing_parameters.normalization=smooth_landing_integral(R_in,R_out,shape);
        }
        const RadialProfile landing_profile{prescribed_landing,&landing_parameters};
        const std::string raw_Mdot_out=parameter_string(parameters,"Mdot_out_Msun_yr","auto");
        double Mdot_out=raw_Mdot_out=="auto" ? std::numeric_limits<double>::quiet_NaN() : std::stod(raw_Mdot_out);

        if (model=="inverse" && std::isnan(Mdot_out)) {
            const auto& rotation=primary_rotation(rotations,profile_type);
            const RotationCurve rot_curve=make_rotation_curve(rotation);
            const ForwardModelParameters forward_parameters{R_nucl,R_out,mu,beta,gas,gas_derivative,star,rot_curve,R_in,launch_velocity(rotation),cgm_j_profile};
            const ForwardModelSolution forward_solution=solve_forward_model(forward_parameters,Mdot_land,h_0,dynamics_atol,rtol);
            Mdot_out=forward_accretion_rate(R_out,forward_parameters,forward_solution);
        }

        std::vector<ModelResult> results;
        const double radial_span=R_out-R_in;
        const double R_Z_stop=mu==0.0 && R_in==R_nucl ? R_in+1e-5*radial_span : R_in;
        const double Z_land=z_land_mixing(Z_nucl,Z_CGM,mu);
        double R_Z_boundary=R_out;
        bool metallicity_solved=true;
        std::string metallicity_method="RK4";
        std::size_t metallicity_cells=0;
        double metallicity_balance_error=std::numeric_limits<double>::quiet_NaN();

        if (model=="forward") {
            for (const RotationInput& rotation: rotations) {
                const RotationCurve rot_curve=make_rotation_curve(rotation);
                const ForwardModelParameters forward_parameters{R_nucl,R_out,mu,beta,gas,gas_derivative,star,rot_curve,R_in,launch_velocity(rotation),cgm_j_profile};
                const ForwardModelSolution forward_solution=solve_forward_model(forward_parameters,Mdot_land,h_0,dynamics_atol,rtol);
                const MetallicityModelParameters metallicity_parameters{&forward_parameters,&forward_solution,Z_nucl,Z_CGM,yield};
                const RK4Solution metallicity_solution=solve_metallicity(metallicity_parameters,R_out,R_Z_stop,Z_boundary,h_0,metallicity_atol,rtol);
                ModelResult result{rotation.source,{},forward_solution.Mdot_land,forward_solution.initial_radial_velocity,forward_solution.initial_landing_rate,rotation.chi2,rotation.dof,rotation.reduced_chi2};
                const double nuclear_angular_momentum=j_disk(R_nucl,launch_velocity(rotation));
                for (const double R: grid) {
                    const double gas_value=profile_value(gas,R);
                    const double star_value=profile_value(star,R);
                    const double radial_velocity=forward_radial_velocity(R,forward_parameters,forward_solution);
                    const double landing=forward_landing_rate(R,forward_parameters,forward_solution);
                    const double circular_velocity=rotation_velocity(rot_curve,R);
                    const double circular_derivative=rotation_velocity_derivative(rot_curve,R);
                    const double disk_angular_momentum=j_disk(R,circular_velocity);
                    const double cgm_angular_momentum=cgm_solution ? profile_value(cgm_j_profile,R) : j_cgm(R,beta,circular_velocity);
                    const double landing_angular_momentum=j_land_from_reservoirs(nuclear_angular_momentum,cgm_angular_momentum,mu);
                    const bool chemistry_defined=R>=R_Z_stop;
                    const double metallicity=chemistry_defined ? evaluate(metallicity_solution,R)[0] : std::numeric_limits<double>::quiet_NaN();
                    const DoubleVec metallicity_state{metallicity};
                    const double metallicity_derivative=chemistry_defined ? metallicity_rhs(R,metallicity_state,&metallicity_parameters)[0] : std::numeric_limits<double>::quiet_NaN();
                    const double required_landing_metallicity=chemistry_defined ? z_land_required(metallicity,metallicity_derivative,gas_value,radial_velocity,landing,star_value,yield) : std::numeric_limits<double>::quiet_NaN();
                    result.profiles.push_back({model,rotation.source,R,gas_value,star_value,landing,forward_cumulative_landing(R,forward_solution),radial_velocity,radial_velocity*kpc_per_year_to_km_per_second,R*gas_value*radial_velocity,forward_accretion_rate(R,forward_parameters,forward_solution),R/std::abs(radial_velocity)/1e9,gas_value/star_value,circular_velocity,circular_derivative,disk_angular_momentum,cgm_angular_momentum,landing_angular_momentum,landing_angular_momentum-disk_angular_momentum,landing_angular_momentum/disk_angular_momentum,landing_angular_momentum/nuclear_angular_momentum,mu_from_mixing(landing_angular_momentum,nuclear_angular_momentum,cgm_angular_momentum),metallicity,Z_land+yield*star_value/landing,metallicity_derivative,required_landing_metallicity,mu_from_mixing(required_landing_metallicity,Z_nucl,Z_CGM)});
                }
                results.push_back(std::move(result));
            }
        } else {
            RK4Solution inverse_solution;
            InverseModelParameters inverse_parameters{};
            EmpiricalInverseModelParameters empirical_parameters{};
            if (analytic_power_law) {
                inverse_parameters={analytic_profiles.R_g,analytic_profiles.sigma_g0,analytic_profiles.A,analytic_profiles.N,Mdot_land,R_in,R_out,Mdot_out,lambda};
                inverse_solution=solve_inverse_model(inverse_parameters,R_in,h_0,dynamics_atol,rtol);
            } else {
                empirical_parameters={gas,star,landing_profile,R_out,Mdot_out};
                inverse_solution=solve_empirical_inverse_model(empirical_parameters,R_in,h_0,dynamics_atol,rtol);
            }
            const double R_Z_start=Mdot_out==0.0 ? R_out-std::min(0.1,0.01*radial_span) : R_out;
            R_Z_boundary=R_Z_start;
            RK4Solution metallicity_solution;
            LinearProfile conservative_metallicity;
            InverseMetallicityModelParameters analytic_metallicity_parameters{};
            EmpiricalInverseVelocityParameters velocity_parameters{&inverse_solution,&empirical_parameters};
            const RadialProfile velocity_profile{empirical_inverse_velocity,&velocity_parameters};
            EmpiricalMetallicityModelParameters empirical_metallicity_parameters{};
            const auto inverse_velocity=[&](double R){
                return analytic_power_law ?
                    inverse_radial_velocity(R,inverse_solution,inverse_parameters) :
                    empirical_inverse_radial_velocity(R,inverse_solution,empirical_parameters);
            };

            bool has_stagnation=false;
            std::vector<double> stagnation_radii;

            constexpr std::size_t scan_points=2001;
            double previous_R=R_Z_stop;
            double previous_v=inverse_velocity(previous_R);

            if (!std::isfinite(previous_v) || previous_v==0.0) {
                has_stagnation=true;
            }

            for (std::size_t i=1; i<scan_points; ++i) {
                const double R=R_Z_stop+(R_Z_start-R_Z_stop)*
                    static_cast<double>(i)/static_cast<double>(scan_points-1);
                const double v=inverse_velocity(R);

                if (!std::isfinite(v) || v==0.0) {
                    has_stagnation=true;
                } else if (
                    std::isfinite(previous_v) &&
                    previous_v!=0.0 &&
                    std::signbit(previous_v)!=std::signbit(v)
                ) {
                    has_stagnation=true;
                    stagnation_radii.push_back(
                        previous_R-previous_v*(R-previous_R)/(v-previous_v)
                    );
                }

                previous_R=R;
                previous_v=v;
            }
            metallicity_solved=!has_stagnation;
            if (has_stagnation && !observed_chemistry) {
                std::cerr << "Inverse v_R stagnates inside the metallicity domain; "
                    << "solving conservative steady metal transport";
                for (const double radius: stagnation_radii) {
                    std::cerr << ' ' << radius << " kpc";
                }
                std::cerr << '\n';
                metallicity_cells=static_cast<std::size_t>(parameter_double(parameters,"metallicity_cells",16384.0));
                const ConservativeMetallicityParameters chemistry{landing_profile,star,Z_land,yield,
                    -Mdot_out,parameter_double(parameters,"Z_inner_boundary",std::numeric_limits<double>::quiet_NaN()),Z_boundary};
                try {
                    const auto solution=solve_conservative_metallicity(chemistry,R_in,R_out,metallicity_cells);
                    conservative_metallicity=solution.metallicity;
                    metallicity_balance_error=solution.max_cell_balance_error;
                    metallicity_solved=true;
                    metallicity_method="conservative_upwind";
                    R_Z_boundary=R_out;
                } catch (const std::exception& error) {
                    metallicity_method="unavailable";
                    std::cerr << "Skipping inverse metallicity: " << error.what() << '\n';
                }
            }

            if (!has_stagnation && !observed_chemistry) {
                if (analytic_power_law) {
                    analytic_metallicity_parameters={&inverse_parameters,&inverse_solution,Z_land,yield};
                    metallicity_solution=solve_inverse_metallicity(analytic_metallicity_parameters,R_Z_start,R_Z_stop,Z_boundary,h_0,metallicity_atol,rtol);
                } else {
                    empirical_metallicity_parameters={gas,velocity_profile,landing_profile,star,Z_land,yield};
                    metallicity_solution=solve_empirical_metallicity(empirical_metallicity_parameters,R_Z_start,R_Z_stop,Z_boundary,h_0,metallicity_atol,rtol);
                }
            }
            if (observed_chemistry) {
                // Infer incident composition from the supplied disk abundance,
                // independent of the trial fountain composition. No chemical ODE
                // or artificial abundance boundary is applied to these inputs.
                if (observed_metallicity.radius.back()<R_in || observed_metallicity.radius.front()>R_out)
                    throw std::runtime_error("Observed metallicity has no overlap with the modeled disk");
                metallicity_solved=true;
                metallicity_method="observed_profile";
                R_Z_boundary=std::numeric_limits<double>::quiet_NaN();
            }
            for (const RotationInput& rotation: rotations) {
                const RotationCurve rot_curve=make_rotation_curve(rotation);
                ModelResult result{rotation.source,{},Mdot_land,std::numeric_limits<double>::quiet_NaN(),profile_value(landing_profile,R_in),rotation.chi2,rotation.dof,rotation.reduced_chi2};
                const double nuclear_angular_momentum=j_disk(R_nucl,launch_velocity(rotation));
                for (const double R: grid) {
                    const double gas_value=profile_value(gas,R);
                    const double star_value=profile_value(star,R);
                    const double landing=profile_value(landing_profile,R);
                    const double radial_velocity=inverse_velocity(R);
                    const double circular_velocity=rotation_velocity(rot_curve,R);
                    const double circular_derivative=rotation_velocity_derivative(rot_curve,R);
                    const double disk_angular_momentum=j_disk(R,circular_velocity);
                    const double cgm_angular_momentum=cgm_solution ? profile_value(cgm_j_profile,R) : j_cgm(R,beta,circular_velocity);
                    const double landing_angular_momentum=j_land_required(R,gas_value,radial_velocity,landing,circular_velocity,circular_derivative);
                    const bool conservative=metallicity_method=="conservative_upwind";
                    const bool chemistry_defined=metallicity_solved && (observed_chemistry ?
                        R>=observed_metallicity.radius.front() && R<=observed_metallicity.radius.back() :
                        conservative || (R>=R_Z_stop && R<=R_Z_start));
                    const double metallicity=!chemistry_defined ? std::numeric_limits<double>::quiet_NaN() :
                        observed_chemistry ? linear_profile_value(R,&observed_metallicity) :
                        conservative ? linear_profile_value(R,&conservative_metallicity) : evaluate(metallicity_solution,R)[0];
                    const DoubleVec metallicity_state{metallicity};
                    double metallicity_derivative=std::numeric_limits<double>::quiet_NaN();
                    if (chemistry_defined && observed_chemistry) {
                        metallicity_derivative=linear_profile_value(R,&observed_metallicity_derivative);
                    } else if (chemistry_defined && conservative) {
                        // Report the steady equation's derivative away from
                        // zero flow. A finite Z need not have a finite slope
                        // at the stagnation point itself.
                        if (radial_velocity!=0.0)
                            metallicity_derivative=metallicity_gradient(metallicity,gas_value,radial_velocity,landing,star_value,Z_land,yield);
                    } else if (chemistry_defined) {
                        metallicity_derivative=analytic_power_law ? inverse_metallicity_rhs(R,metallicity_state,&analytic_metallicity_parameters)[0] : empirical_metallicity_rhs(R,metallicity_state,&empirical_metallicity_parameters)[0];
                    }
                    const double required_landing_metallicity=!chemistry_defined ? std::numeric_limits<double>::quiet_NaN() :
                        conservative && radial_velocity==0.0 ? metallicity-yield*star_value/landing :
                        z_land_required(metallicity,metallicity_derivative,gas_value,radial_velocity,landing,star_value,yield);
                    const double cumulative_landing=prescribed_cumulative_landing(R,landing_parameters);
                    result.profiles.push_back({model,rotation.source,R,gas_value,star_value,landing,cumulative_landing,radial_velocity,radial_velocity*kpc_per_year_to_km_per_second,R*gas_value*radial_velocity,mdot_acc_from_velocity(R,gas_value,radial_velocity),R/std::abs(radial_velocity)/1e9,gas_value/star_value,circular_velocity,circular_derivative,disk_angular_momentum,cgm_angular_momentum,landing_angular_momentum,landing_angular_momentum-disk_angular_momentum,landing_angular_momentum/disk_angular_momentum,landing_angular_momentum/nuclear_angular_momentum,mu_from_mixing(landing_angular_momentum,nuclear_angular_momentum,cgm_angular_momentum),metallicity,Z_land+yield*star_value/landing,metallicity_derivative,required_landing_metallicity,mu_from_mixing(required_landing_metallicity,Z_nucl,Z_CGM)});
                }
                results.push_back(std::move(result));
            }
        }

        std::filesystem::create_directories(output_directory);
        if (model=="inverse") {
            write_landing_profile(output_directory/"landing_profile.csv",landing_parameters);
            write_landing_prescription(output_directory/"landing_prescription.csv",landing_parameters,R_nucl,results.front().profiles);
        }
        write_profiles(output_directory/"profiles.csv",results);
        write_summary(output_directory/"summary.csv",model,profile_type,galaxy,results,R_nucl,R_out,Mdot_land,Mdot_out,mu,beta,Z_nucl,Z_CGM,yield,Z_boundary,R_Z_boundary,metallicity_solved,std::numeric_limits<double>::quiet_NaN(),R_in);
        std::ofstream chemistry_info(output_directory/"metallicity_solver.csv");
        chemistry_info.exceptions(std::ios::failbit | std::ios::badbit);
        chemistry_info << std::setprecision(17) << "method,cells,max_cell_metal_balance_error_Msun_yr\n"
                       << metallicity_method << ',' << metallicity_cells << ',' << metallicity_balance_error << '\n';
        copy_if_present(input_directory/"parameters.csv",output_directory/"parameters.csv");
        if (cgm_solution) {
            write_cgm_profile(output_directory,*cgm_solution);
        }
        if (cgm_solution || !cgm_config.empty())
            write_cgm_model_parameters(output_directory/"parameters.csv",parameters,cgm_solution ? &*cgm_solution : nullptr);
        copy_if_present(input_directory/"metallicity_profiles.csv",output_directory/"metallicity_profiles.csv");
        copy_if_present(input_directory/"metallicity_fit.csv",output_directory/"metallicity_fit.csv");
        copy_if_present(input_directory/"metadata.csv",output_directory/"metadata.csv");
        copy_if_present(input_directory/"rotation_curves.csv",output_directory/"rotation_curves.csv");
        copy_if_present(input_directory/"rotation_profiles.csv",output_directory/"rotation_profiles.csv");
        copy_if_present(input_directory/"rotation_spline_diagnostics.csv",output_directory/"rotation_spline_diagnostics.csv");
        copy_if_present(input_directory/"gas_profiles.csv",output_directory/"gas_profiles.csv");
        copy_if_present(input_directory/"gas_spline_diagnostics.csv",output_directory/"gas_spline_diagnostics.csv");
        copy_if_present(input_directory/"sfr_spline_diagnostics.csv",output_directory/"sfr_spline_diagnostics.csv");
        copy_if_present(input_directory/"sfr_crosscalibration.csv",output_directory/"sfr_crosscalibration.csv");
        copy_if_present(input_directory/"sparc_corrected.csv",output_directory/"sparc_corrected.csv");
        copy_if_present(input_directory/"leroy_profiles_used.csv",output_directory/"leroy_profiles_used.csv");
        copy_if_present(input_directory/"bigiel_profiles_used.csv",output_directory/"bigiel_profiles_used.csv");
        std::cout << "Wrote model CSV output to " << output_directory << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "galactic-nuclear-fountain-model: " << error.what() << '\n';
        return 1;
    }
}
