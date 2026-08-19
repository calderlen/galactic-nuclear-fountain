#include "models.h"
#include "physics.h"
#include "profiles.h"
#include "csv.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr double pi=3.14159265358979323846;
constexpr double kpc_per_year_to_km_per_second=9.7779222168e8;
using CsvRow=galacticwind::csv::Row;
using CsvTable=galacticwind::csv::Table;
using galacticwind::csv::escape;
using galacticwind::csv::number;
using galacticwind::csv::read;
using galacticwind::csv::value;
struct AnalyticProfileParameters {double R_g; double sigma_g0; double A; double N;};
struct LandingProfileParameters {double Mdot_land; double R_nucl; double R_out;};
struct RotationInput {
    std::string source;
    std::string kind;
    FlatRotationParameters flat;
    RisingRotationParameters rising;
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

std::unordered_map<std::string,std::string> read_parameters(const std::filesystem::path& path){
    const CsvTable table=read(path);
    std::unordered_map<std::string,std::string> parameters;
    for (const CsvRow& row: table.rows) {
        parameters[value(row,"name")]=value(row,"value");
    }
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
    return sigmadot_land(R,parameters.Mdot_land,parameters.R_nucl,parameters.R_out);
}

double empirical_inverse_velocity(double R,const void* raw_parameters){
    const EmpiricalInverseVelocityParameters& parameters=*static_cast<const EmpiricalInverseVelocityParameters*>(raw_parameters);
    return empirical_inverse_radial_velocity(R,*parameters.solution,*parameters.parameters);
}

RotationCurve make_rotation_curve(const RotationInput& input){
    if (input.kind=="flat") {
        return {{flat_rotation_velocity,&input.flat},{flat_rotation_velocity_derivative,&input.flat}};
    }
    return {{rising_rotation_velocity,&input.rising},{rising_rotation_velocity_derivative,&input.rising}};
}

std::vector<RotationInput> read_rotations(const std::filesystem::path& path){
    const CsvTable table=read(path);
    std::vector<RotationInput> rotations;
    for (const CsvRow& row: table.rows) {
        RotationInput input;
        input.source=value(row,"source");
        input.kind=value(row,"kind");
        input.flat.velocity=number(row,"Vflat_kms");
        input.rising={input.flat.velocity,input.kind=="rising" ? number(row,"lflat_kpc") : 1.0};
        input.chi2=number(row,"chi2");
        input.dof=number(row,"dof");
        input.reduced_chi2=number(row,"reduced_chi2");
        rotations.push_back(input);
    }
    return rotations;
}

void load_tabulated_profiles(const std::filesystem::path& path,LinearProfile& gas,LinearProfile& star){
    const CsvTable table=read(path);
    for (const CsvRow& row: table.rows) {
        const double radius=number(row,"R_kpc");
        gas.radius.push_back(radius);
        star.radius.push_back(radius);
        gas.value.push_back(number(row,"Sigma_g_Msun_kpc2"));
        star.value.push_back(number(row,"Sigmadot_star_Msun_yr_kpc2"));
    }
}

void write_profiles(const std::filesystem::path& path,const std::vector<ModelResult>& results){
    std::ofstream stream(path);
    stream << "model,source,R_kpc,Sigma_g_Msun_kpc2,Sigmadot_star_Msun_yr_kpc2,Sigmadot_land_Msun_yr_kpc2,cumulative_landing_Msun_yr,v_R_kpc_yr,v_R_kms,F_R_Msun_yr,Mdot_acc_Msun_yr,t_inflow_Gyr,t_depletion_Gyr,v_c_kms,dv_c_dR_kms_kpc,j_disk_kpc_kms,j_CGM_kpc_kms,j_land_kpc_kms,delta_j_kpc_kms,j_land_over_j_disk,j_land_over_j_nucl,mu_j,Z,Z_eq,dZ_dR_per_kpc,Z_land_required,mu_Z\n";
    stream << std::setprecision(17);
    for (const ModelResult& result: results) {
        for (const ProfileRow& row: result.profiles) {
            stream << escape(row.model) << ',' << escape(row.source) << ',' << row.R << ',' << row.gas << ',' << row.star << ',' << row.landing << ',' << row.cumulative_landing << ',' << row.radial_velocity << ',' << row.radial_velocity_kms << ',' << row.radial_mass_flux << ',' << row.accretion << ',' << row.inflow_time << ',' << row.depletion_time << ',' << row.circular_velocity << ',' << row.circular_velocity_derivative << ',' << row.disk_angular_momentum << ',' << row.cgm_angular_momentum << ',' << row.landing_angular_momentum << ',' << row.angular_momentum_difference << ',' << row.landing_to_disk << ',' << row.landing_to_nuclear << ',' << row.mu_angular_momentum << ',' << row.metallicity << ',' << row.equilibrium_metallicity << ',' << row.metallicity_derivative << ',' << row.required_landing_metallicity << ',' << row.mu_metallicity << '\n';
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
    const double tolerance=1e-10*std::max(maximum_absolute_landing,1.0);
    return minimum_landing>=-tolerance;
}

void write_summary(const std::filesystem::path& path,const std::string& model,const std::string& profile_type,const std::string& galaxy,const std::vector<ModelResult>& results,double R_nucl,double R_out,double Mdot_land,double Mdot_out,double mu,double beta,double Z_nucl,double Z_CGM,double yield,double Z_boundary,double R_Z_boundary,bool metallicity_solved){
    std::ofstream stream(path);
    stream << "model,profile_type,galaxy,source,R_nucl_kpc,R_out_kpc,Mdot_land_Msun_yr,Mdot_out_Msun_yr,mu,beta,Z_nucl,Z_CGM,yield_y,Z_outer_boundary,R_Z_boundary_kpc,metallicity_solved,total_landing_Msun_yr,initial_radial_velocity_kpc_yr,initial_landing_rate_Msun_yr_kpc2,chi2,dof,reduced_chi2,landing_nonnegative\n";
    stream << std::setprecision(17);
    for (const ModelResult& result: results) {
        stream << escape(model) << ',' << escape(profile_type) << ',' << escape(galaxy) << ',' << escape(result.source) << ',' << R_nucl << ',' << R_out << ',' << Mdot_land << ',' << Mdot_out << ',' << mu << ',' << beta << ',' << Z_nucl << ',' << Z_CGM << ',' << yield << ',' << Z_boundary << ',' << R_Z_boundary << ',' << (metallicity_solved ? 1 : 0) << ',' << result.total_landing << ',' << result.initial_radial_velocity << ',' << result.initial_landing_rate << ',' << result.rotation_chi2 << ',' << result.rotation_dof << ',' << result.rotation_reduced_chi2 << ',' << (landing_nonnegative(result) ? "true" : "false") << '\n';
    }
}

void copy_if_present(const std::filesystem::path& source,const std::filesystem::path& destination){
    if (std::filesystem::exists(source)) {
        std::filesystem::copy_file(source,destination,std::filesystem::copy_options::overwrite_existing);
    }
}

}

int main(int argc,char** argv){
    try {
        if (argc!=4) {
            std::cerr << "Usage: galacticwind_model <forward|inverse> <input-directory> <output-directory>\n";
            return 2;
        }

        const std::string model=argv[1];
        const std::filesystem::path input_directory=argv[2];
        const std::filesystem::path output_directory=argv[3];
        const auto parameters=read_parameters(input_directory/"parameters.csv");
        const std::string profile_type=parameter_string(parameters,"profile_type","tabulated");
        const std::string galaxy=parameter_string(parameters,"galaxy",profile_type=="analytic" ? "analytic" : "unknown");
        const double R_nucl=parameter_double(parameters,"R_nucl_kpc");
        const double R_out=parameter_double(parameters,"R_out_kpc");
        const double Mdot_land=parameter_double(parameters,"Mdot_land_Msun_yr");
        const double mu=parameter_double(parameters,"mu",1.0);
        const double beta=parameter_double(parameters,"beta",0.75);
        const double Z_nucl=parameter_double(parameters,"Z_nucl",0.02);
        const double Z_CGM=parameter_double(parameters,"Z_CGM",0.003);
        const double yield=parameter_double(parameters,"yield_y",0.015);
        const double Z_boundary=parameter_double(parameters,"Z_outer_boundary",0.006);
        const double h_0=parameter_double(parameters,"h_0_kpc",0.01);
        const double rtol=parameter_double(parameters,"rtol",1e-9);
        const double dynamics_atol=parameter_double(parameters,"dynamics_atol",1e-18);
        const double metallicity_atol=parameter_double(parameters,"metallicity_atol",1e-12);
        const std::size_t point_count=static_cast<std::size_t>(parameter_double(parameters,"n_points",300.0));
        const std::vector<double> grid=radial_grid(R_nucl,R_out,point_count);
        const std::vector<RotationInput> rotations=read_rotations(input_directory/"rotation_curves.csv");

        AnalyticProfileParameters analytic_profiles{};
        LinearProfile tabulated_gas;
        LinearProfile tabulated_star;
        RadialProfile gas;
        RadialProfile gas_derivative;
        RadialProfile star;
        if (profile_type=="analytic") {
            const double R_g=parameter_double(parameters,"R_g_kpc");
            const double M_g=parameter_double(parameters,"M_g_Msun");
            const double A=parameter_double(parameters,"A");
            const double N=parameter_double(parameters,"N");
            analytic_profiles={R_g,sigma_g0(M_g,R_g,R_nucl,R_out),A,N};
            gas={analytic_gas,&analytic_profiles};
            gas_derivative={analytic_gas_derivative,&analytic_profiles};
            star={analytic_star,&analytic_profiles};
        } else {
            load_tabulated_profiles(input_directory/"profiles.csv",tabulated_gas,tabulated_star);
            gas={linear_profile_value,&tabulated_gas};
            gas_derivative={linear_profile_derivative,&tabulated_gas};
            star={linear_profile_value,&tabulated_star};
        }

        const LandingProfileParameters landing_parameters{Mdot_land,R_nucl,R_out};
        const RadialProfile landing_profile{prescribed_landing,&landing_parameters};
        const std::string raw_Mdot_out=parameter_string(parameters,"Mdot_out_Msun_yr","auto");
        double Mdot_out=raw_Mdot_out=="auto" ? std::numeric_limits<double>::quiet_NaN() : std::stod(raw_Mdot_out);

        if (model=="inverse" && std::isnan(Mdot_out)) {
            const RotationCurve rot_curve=make_rotation_curve(rotations.front());
            const ForwardModelParameters forward_parameters{R_nucl,R_out,mu,beta,gas,gas_derivative,star,rot_curve};
            const ForwardModelSolution forward_solution=solve_forward_model(forward_parameters,Mdot_land,h_0,dynamics_atol,rtol);
            Mdot_out=-2.0*pi*R_out*profile_value(gas,R_out)*forward_radial_velocity(R_out,forward_parameters,forward_solution);
        }
        if (model=="forward" && std::isnan(Mdot_out)) {
            Mdot_out=0.0;
        }

        std::vector<ModelResult> results;
        const double radial_span=R_out-R_nucl;
        const double R_Z_stop=mu==0.0 ? R_nucl+1e-5*radial_span : R_nucl;
        const double Z_land=z_land_mixing(Z_nucl,Z_CGM,mu);
        double R_Z_boundary=R_out;
        bool metallicity_solved=true;

        if (model=="forward") {
            for (const RotationInput& rotation: rotations) {
                const RotationCurve rot_curve=make_rotation_curve(rotation);
                const ForwardModelParameters forward_parameters{R_nucl,R_out,mu,beta,gas,gas_derivative,star,rot_curve};
                const ForwardModelSolution forward_solution=solve_forward_model(forward_parameters,Mdot_land,h_0,dynamics_atol,rtol);
                const MetallicityModelParameters metallicity_parameters{&forward_parameters,&forward_solution,Z_nucl,Z_CGM,yield};
                const RK4Solution metallicity_solution=solve_metallicity(metallicity_parameters,R_out,R_Z_stop,Z_boundary,h_0,metallicity_atol,rtol);
                ModelResult result{rotation.source,{},forward_solution.Mdot_land,forward_solution.initial_radial_velocity,forward_solution.initial_landing_rate,rotation.chi2,rotation.dof,rotation.reduced_chi2};
                const double nuclear_angular_momentum=j_disk(R_nucl,rotation_velocity(rot_curve,R_nucl));
                for (const double R: grid) {
                    const double gas_value=profile_value(gas,R);
                    const double star_value=profile_value(star,R);
                    const double radial_velocity=forward_radial_velocity(R,forward_parameters,forward_solution);
                    const double landing=forward_landing_rate(R,forward_parameters,forward_solution);
                    const double circular_velocity=rotation_velocity(rot_curve,R);
                    const double circular_derivative=rotation_velocity_derivative(rot_curve,R);
                    const double disk_angular_momentum=j_disk(R,circular_velocity);
                    const double cgm_angular_momentum=j_cgm(R,beta,circular_velocity);
                    const double landing_angular_momentum=j_land_mixing(R,R_nucl,mu,beta,circular_velocity,rotation_velocity(rot_curve,R_nucl));
                    const bool chemistry_defined=R>=R_Z_stop;
                    const double metallicity=chemistry_defined ? evaluate(metallicity_solution,R)[0] : std::numeric_limits<double>::quiet_NaN();
                    const DoubleVec metallicity_state{metallicity};
                    const double metallicity_derivative=chemistry_defined ? metallicity_rhs(R,metallicity_state,&metallicity_parameters)[0] : std::numeric_limits<double>::quiet_NaN();
                    const double required_landing_metallicity=chemistry_defined ? z_land_required(metallicity,metallicity_derivative,gas_value,radial_velocity,landing,star_value,yield) : std::numeric_limits<double>::quiet_NaN();
                    result.profiles.push_back({model,rotation.source,R,gas_value,star_value,landing,forward_cumulative_landing(R,forward_solution),radial_velocity,radial_velocity*kpc_per_year_to_km_per_second,R*gas_value*radial_velocity,mdot_acc_from_velocity(R,gas_value,radial_velocity),R/std::abs(radial_velocity)/1e9,gas_value/star_value/1e9,circular_velocity,circular_derivative,disk_angular_momentum,cgm_angular_momentum,landing_angular_momentum,landing_angular_momentum-disk_angular_momentum,landing_angular_momentum/disk_angular_momentum,landing_angular_momentum/nuclear_angular_momentum,mu_from_mixing(landing_angular_momentum,nuclear_angular_momentum,cgm_angular_momentum),metallicity,Z_land+yield*star_value/landing,metallicity_derivative,required_landing_metallicity,mu_from_mixing(required_landing_metallicity,Z_nucl,Z_CGM)});
                }
                results.push_back(std::move(result));
            }
        } else if (model=="inverse") {
            RK4Solution inverse_solution;
            InverseModelParameters inverse_parameters{};
            EmpiricalInverseModelParameters empirical_parameters{};
            if (profile_type=="analytic") {
                inverse_parameters={analytic_profiles.R_g,analytic_profiles.sigma_g0,analytic_profiles.A,analytic_profiles.N,Mdot_land,R_nucl,R_out,Mdot_out};
                inverse_solution=solve_inverse_model(inverse_parameters,R_nucl,h_0,dynamics_atol,rtol);
            } else {
                empirical_parameters={gas,star,landing_profile,R_out,Mdot_out};
                inverse_solution=solve_empirical_inverse_model(empirical_parameters,R_nucl,h_0,dynamics_atol,rtol);
            }
            const double R_Z_start=Mdot_out==0.0 ? R_out-std::min(0.1,0.01*radial_span) : R_out;
            R_Z_boundary=R_Z_start;
            RK4Solution metallicity_solution;
            InverseMetallicityModelParameters analytic_metallicity_parameters{};
            EmpiricalInverseVelocityParameters velocity_parameters{&inverse_solution,&empirical_parameters};
            const RadialProfile velocity_profile{empirical_inverse_velocity,&velocity_parameters};
            EmpiricalMetallicityModelParameters empirical_metallicity_parameters{};
            const auto inverse_velocity=[&](double R){
                return profile_type=="analytic" ?
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

            if (has_stagnation) {
                std::cerr << "Warning: inverse v_R stagnates inside the "
                    << "metallicity domain; skipping inverse metallicity";
                for (const double radius: stagnation_radii) {
                    std::cerr << ' ' << radius << " kpc";
                }
                std::cerr << '\n';
            }

            if (!has_stagnation) {
                if (profile_type=="analytic") {
                    analytic_metallicity_parameters={&inverse_parameters,&inverse_solution,Z_land,yield};
                    metallicity_solution=solve_inverse_metallicity(analytic_metallicity_parameters,R_Z_start,R_Z_stop,Z_boundary,h_0,metallicity_atol,rtol);
                } else {
                    empirical_metallicity_parameters={gas,velocity_profile,landing_profile,star,Z_land,yield};
                    metallicity_solution=solve_empirical_metallicity(empirical_metallicity_parameters,R_Z_start,R_Z_stop,Z_boundary,h_0,metallicity_atol,rtol);
                }
            }
            for (const RotationInput& rotation: rotations) {
                const RotationCurve rot_curve=make_rotation_curve(rotation);
                ModelResult result{rotation.source,{},Mdot_land,std::numeric_limits<double>::quiet_NaN(),profile_value(landing_profile,R_nucl),rotation.chi2,rotation.dof,rotation.reduced_chi2};
                const double nuclear_angular_momentum=j_disk(R_nucl,rotation_velocity(rot_curve,R_nucl));
                for (const double R: grid) {
                    const double gas_value=profile_value(gas,R);
                    const double star_value=profile_value(star,R);
                    const double landing=profile_value(landing_profile,R);
                    const double radial_velocity=inverse_velocity(R);
                    const double circular_velocity=rotation_velocity(rot_curve,R);
                    const double circular_derivative=rotation_velocity_derivative(rot_curve,R);
                    const double disk_angular_momentum=j_disk(R,circular_velocity);
                    const double cgm_angular_momentum=j_cgm(R,beta,circular_velocity);
                    const double landing_angular_momentum=j_land_required(R,gas_value,radial_velocity,landing,circular_velocity,circular_derivative);
                    const bool chemistry_defined=!has_stagnation && R>=R_Z_stop && R<=R_Z_start;
                    const double metallicity=chemistry_defined ? evaluate(metallicity_solution,R)[0] : std::numeric_limits<double>::quiet_NaN();
                    const DoubleVec metallicity_state{metallicity};
                    const double metallicity_derivative=!chemistry_defined ? std::numeric_limits<double>::quiet_NaN() : profile_type=="analytic" ? inverse_metallicity_rhs(R,metallicity_state,&analytic_metallicity_parameters)[0] : empirical_metallicity_rhs(R,metallicity_state,&empirical_metallicity_parameters)[0];
                    const double required_landing_metallicity=chemistry_defined ? z_land_required(metallicity,metallicity_derivative,gas_value,radial_velocity,landing,star_value,yield) : std::numeric_limits<double>::quiet_NaN();
                    const double cumulative_landing=Mdot_land*std::log(R/R_nucl)/std::log(R_out/R_nucl);
                    result.profiles.push_back({model,rotation.source,R,gas_value,star_value,landing,cumulative_landing,radial_velocity,radial_velocity*kpc_per_year_to_km_per_second,R*gas_value*radial_velocity,mdot_acc_from_velocity(R,gas_value,radial_velocity),R/std::abs(radial_velocity)/1e9,gas_value/star_value/1e9,circular_velocity,circular_derivative,disk_angular_momentum,cgm_angular_momentum,landing_angular_momentum,landing_angular_momentum-disk_angular_momentum,landing_angular_momentum/disk_angular_momentum,landing_angular_momentum/nuclear_angular_momentum,mu_from_mixing(landing_angular_momentum,nuclear_angular_momentum,cgm_angular_momentum),metallicity,Z_land+yield*star_value/landing,metallicity_derivative,required_landing_metallicity,mu_from_mixing(required_landing_metallicity,Z_nucl,Z_CGM)});
                }
                results.push_back(std::move(result));
            }
        } else {
            throw std::runtime_error("Model must be forward or inverse");
        }

        std::filesystem::create_directories(output_directory);
        write_profiles(output_directory/"profiles.csv",results);
        write_summary(output_directory/"summary.csv",model,profile_type,galaxy,results,R_nucl,R_out,Mdot_land,Mdot_out,mu,beta,Z_nucl,Z_CGM,yield,Z_boundary,R_Z_boundary,metallicity_solved);
        copy_if_present(input_directory/"metadata.csv",output_directory/"metadata.csv");
        copy_if_present(input_directory/"rotation_curves.csv",output_directory/"rotation_curves.csv");
        copy_if_present(input_directory/"sparc_corrected.csv",output_directory/"sparc_corrected.csv");
        copy_if_present(input_directory/"leroy_profiles_used.csv",output_directory/"leroy_profiles_used.csv");
        std::cout << "Wrote model CSV output to " << output_directory << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "galacticwind_model: " << error.what() << '\n';
        return 1;
    }
}
