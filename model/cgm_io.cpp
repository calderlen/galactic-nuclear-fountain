#include "cgm_io.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {
constexpr double kms_to_kpc_per_Myr=1.022712165045695e-3;
double number(const galactic_nuclear_fountain::csv::Row& settings,const char* key) {
    const auto it=settings.find(key);
    if (it==settings.end()) throw std::invalid_argument(std::string("CGM config needs ")+key);
    std::size_t used=0;
    const double value=std::stod(it->second,&used);
    if (used!=it->second.size() || !std::isfinite(value))
        throw std::invalid_argument(std::string("CGM config needs a finite number for ")+key);
    return value;
}
std::string formatted(double value) {
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}
}

std::filesystem::path cgm_config_argument(int& argc,char** argv) {
    std::filesystem::path path;
    for (int i=1;i<argc;++i) {
        if (std::string(argv[i])!="--cgm-config") continue;
        if (i+1>=argc || !path.empty())
            throw std::invalid_argument("Supply --cgm-config CONFIG.csv once");
        path=argv[i+1];
        for (int k=i;k+2<argc;++k) argv[k]=argv[k+2];
        argc-=2;
        argv[argc]=nullptr;
        --i;
    }
    return path;
}

std::optional<CGMAMDSolution> load_cgm(const galactic_nuclear_fountain::csv::Row& original,
    const std::filesystem::path& config,const OrbitParameters* orbit_potential) {
    auto settings=original;
    if (!config.empty()) {
        const auto table=galactic_nuclear_fountain::csv::read(config);
        for (const auto& row:table.rows)
            settings[galactic_nuclear_fountain::csv::value(row,table.header.at(0))]=
                galactic_nuclear_fountain::csv::value(row,table.header.at(1));
    }
    const auto mode=settings.find("cgm_model");
    if (mode==settings.end() || mode->second=="constant_beta") return std::nullopt;
    if (mode->second!="exponential_amd")
        throw std::invalid_argument("cgm_model must be constant_beta or exponential_amd");
    CGMAMDParameters p;
    if (orbit_potential) p.potential=*orbit_potential;
    else p.potential={
        .M_d=number(settings,"M_d_Msun"),.a_d=number(settings,"a_d_kpc"),
        .b_d=number(settings,"b_d_kpc"),.M_b=number(settings,"M_b_Msun"),
        .a_b=number(settings,"a_b_kpc"),.rho_s=number(settings,"rho_s_Msun_kpc3"),
        .r_s=number(settings,"r_s_kpc")};
    p.potential.cgm=nullptr;
    p.R200=number(settings,"cgm_R200_kpc");
    p.Rcore=number(settings,"cgm_Rcore_kpc");
    p.M200=number(settings,"cgm_M200_Msun");
    p.M_CGM=number(settings,"cgm_mass_Msun");
    p.T_K=number(settings,"cgm_T_K");
    p.mu_mol=number(settings,"cgm_mu_mol");
    p.lambda_spin=number(settings,"cgm_lambda_spin");
    p.fj=number(settings,"cgm_fj");
    p.jmax_over_jDM=number(settings,"cgm_jmax_over_jDM");
    const double intervals=number(settings,"cgm_vertical_intervals");
    if (intervals<2 || intervals>100000 || std::floor(intervals)!=intervals)
        throw std::invalid_argument("cgm_vertical_intervals must be an even integer in [2,100000]");
    p.vertical_intervals=static_cast<int>(intervals);
    // Numerical controls are fixed here so saved physical inputs reproduce
    // the same solve in trajectories and time-snapshot reconstruction.
    return solve_cgm_amd(p);
}

galactic_nuclear_fountain::csv::Row cgm_parameter_values(const CGMAMDSolution& s) {
    const auto& p=s.parameters;
    const auto& v=p.potential;
    return {{"cgm_model","exponential_amd"},
        {"cgm_R200_kpc",formatted(p.R200)},{"cgm_Rcore_kpc",formatted(p.Rcore)},
        {"cgm_M200_Msun",formatted(p.M200)},{"cgm_mass_Msun",formatted(p.M_CGM)},
        {"cgm_T_K",formatted(p.T_K)},{"cgm_mu_mol",formatted(p.mu_mol)},
        {"cgm_lambda_spin",formatted(p.lambda_spin)},{"cgm_fj",formatted(p.fj)},
        {"cgm_jmax_over_jDM",formatted(p.jmax_over_jDM)},
        {"cgm_vertical_intervals",formatted(p.vertical_intervals)},
        {"M_d_Msun",formatted(v.M_d)},{"a_d_kpc",formatted(v.a_d)},
        {"b_d_kpc",formatted(v.b_d)},{"M_b_Msun",formatted(v.M_b)},
        {"a_b_kpc",formatted(v.a_b)},{"rho_s_Msun_kpc3",formatted(v.rho_s)},
        {"r_s_kpc",formatted(v.r_s)}};
}

void write_cgm_model_parameters(const std::filesystem::path& path,
    galactic_nuclear_fountain::csv::Row settings,const CGMAMDSolution* solution) {
    if (solution) {
        for (const auto& [key,value]:cgm_parameter_values(*solution)) settings[key]=value;
    } else settings["cgm_model"]="constant_beta";
    std::vector<std::pair<std::string,std::string>> sorted(settings.begin(),settings.end());
    std::sort(sorted.begin(),sorted.end());
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit|std::ios::badbit);
    stream << "name,value\n";
    for (const auto& [key,value]:sorted)
        stream << galactic_nuclear_fountain::csv::escape(key) << ','
               << galactic_nuclear_fountain::csv::escape(value) << '\n';
}

double cgm_j_kpc_kms(double R,const void* solution) {
    return cgm_angular_momentum(*static_cast<const CGMAMDSolution*>(solution),R)/kms_to_kpc_per_Myr;
}

void write_cgm_profile(const std::filesystem::path& directory,const CGMAMDSolution& s) {
    std::filesystem::create_directories(directory);
    std::ofstream stream(directory/"cgm_profile.csv");
    stream.exceptions(std::ios::failbit|std::ios::badbit);
    stream << std::setprecision(17)
        << "R_kpc,C,rho0_Msun_kpc3,j_CGM_kpc2_Myr,j_CGM_kpc_kms,vphi_CGM_kms,beta_CGM\n";
    for (std::size_t i=0;i<s.trajectory.x.size();++i) {
        const double R=i==0 ? s.parameters.Rcore : i+1==s.trajectory.x.size() ?
            s.parameters.R200 : s.parameters.R200*std::exp(s.trajectory.x[i]);
        stream << R << ',' << s.trajectory.y[i][0] << ',' << cgm_density(s,R,0.0)
            << ',' << cgm_angular_momentum(s,R) << ',' << cgm_j_kpc_kms(R,&s)
            << ',' << cgm_azimuthal_velocity(s,R)/kms_to_kpc_per_Myr
            << ',' << cgm_beta(s,R) << '\n';
    }
    write_cgm_model_parameters(directory/"cgm_parameters.csv",{},&s);
    std::ofstream diagnostics(directory/"cgm_solver.csv");
    diagnostics.exceptions(std::ios::failbit|std::ios::badbit);
    diagnostics << std::setprecision(17)
        << "xi,jmax_kpc2_Myr,cs2_kpc2_Myr2,inner_mass_residual,mass_Msun,vertical_cutoff,core_rotation\n"
        << s.xi << ',' << s.jmax << ',' << s.cs2 << ','
        << s.trajectory.y.front()[0]-s.core_factor*std::exp(s.trajectory.y.front()[1])
        << ',' << s.parameters.M_CGM << ",spherical_R200,solid_body\n";
}
