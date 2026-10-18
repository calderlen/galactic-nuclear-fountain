#include "csv.h"
#include "comparison.h"
#include "models.h"
#include "physics.h"
#include "potential.h"
#include "units.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numbers>
#include <stdexcept>

namespace csv=galactic_nuclear_fountain::csv;
namespace abundance=galactic_nuclear_fountain;
using Path=std::filesystem::path;

namespace {
constexpr double kpc_per_year_to_kms=3.0856775814913673e16/31557600.0;

std::ofstream output(const Path& path,const char* header){
    std::ofstream stream(path);
    stream.exceptions(std::ios::failbit|std::ios::badbit);
    stream << std::setprecision(17) << header << '\n';
    return stream;
}

std::map<std::string,double> parameters(const Path& input){
    std::map<std::string,double> result;
    for (const auto& row: csv::read(input/"parameters.csv").rows)
        result[csv::value(row,"name")]=csv::number(row,"value");
    return result;
}

void potential(const Path& input,const Path& destination){
    const auto p=parameters(input);
    std::ifstream points(input/"points.csv");
    if (!points) throw std::runtime_error("Could not read potential points");
    auto stream=output(destination/"values.csv","Phi_kpc2_Myr2,dPhi_dR_kpc_Myr2");
    std::string line;
    std::getline(points,line);
    // Stream large parcel catalogs rather than retaining a second table in memory.
    while (std::getline(points,line)) {
        const auto comma=line.find(',');
        const double R=std::stod(line.substr(0,comma)),z=std::stod(line.substr(comma+1));
        stream << phi_mn(R,z,p.at("M_d_Msun"),p.at("a_d_kpc"),p.at("b_d_kpc"))
                    +phi_h(R,z,p.at("M_b_Msun"),p.at("a_b_kpc"))
                    +phi_nfw(R,z,p.at("rho_s_Msun_kpc3"),p.at("r_s_kpc")) << ','
               << dphi_mn_dR(R,z,p.at("M_d_Msun"),p.at("a_d_kpc"),p.at("b_d_kpc"))
                    +dphi_h_dR(R,z,p.at("M_b_Msun"),p.at("a_b_kpc"))
                    +dphi_nfw_dR(R,z,p.at("rho_s_Msun_kpc3"),p.at("r_s_kpc")) << '\n';
    }
}

double one(double,const void*){return 1.0;}
double inward_transport(double R,const void* raw){
    return units::per_year(-power_law_transport_length(R,*static_cast<const PowerLawDiskParameters*>(raw)));
}

DoubleVec chemistry(const DoubleVec& radii,const DoubleVec& star,const DoubleVec& landing,
                    const std::map<std::string,double>& p){
    if (radii.empty()) throw std::invalid_argument("Metallicity needs at least one radius");
    LinearProfile ratio{radii,{}};
    for (std::size_t i=0;i<radii.size();++i) ratio.value.push_back(star[i]/landing[i]);
    if (radii.size()==1)
        return {radii.front()==p.at("R_nucl_kpc") ? p.at("Z_land")+p.at("yield_y")*ratio.value.front() : p.at("Z_outer_boundary")};
    const PowerLawDiskParameters disk{p.at("R_nucl_kpc"),radii.front(),radii.back(),1,1,1,2};
    // Preserve the original piecewise-linear star/landing enrichment profile,
    // flat rotation, and outer Z boundary. Gas and landing cancel from this ODE.
    const EmpiricalMetallicityModelParameters model{
        {one,nullptr},{inward_transport,&disk},{one,nullptr},{linear_profile_value,&ratio},
        p.at("Z_land"),p.at("yield_y")};
    const double stop=radii.front()==disk.R_nucl ? radii[1] : radii.front();
    const auto solution=integrate_rk4(empirical_metallicity_rhs,&model,radii.back(),
        {p.at("Z_outer_boundary")},0.001,1e-13,1e-11,stop,0.001);
    DoubleVec result;
    for (double R: radii)
        result.push_back(R==disk.R_nucl ? p.at("Z_land")+p.at("yield_y")*ratio.value.front() : evaluate(solution,R)[0]);
    return result;
}

void metallicity(const Path& input,const Path& destination){
    const auto p=parameters(input);
    DoubleVec radii,star,landing;
    for (const auto& row: csv::read(input/"points.csv").rows) {
        radii.push_back(csv::number(row,"R_kpc"));
        star.push_back(csv::number(row,"Sigmadot_star_Msun_Gyr_kpc2"));
        landing.push_back(csv::number(row,"Sigmadot_land_Msun_Gyr_kpc2"));
    }
    auto stream=output(destination/"values.csv","Z");
    for (double Z: chemistry(radii,star,landing,p)) stream << Z << '\n';
}

void unmixed_observed(const Path& input,const Path& destination){
    const auto p=parameters(input);
    LinearProfile gas,star;
    for (const auto& row: csv::read(input/"points.csv").rows) {
        const double R=csv::number(row,"R_kpc");
        gas.radius.push_back(R);star.radius.push_back(R);
        gas.value.push_back(csv::number(row,"Sigma_g_Msun_kpc2"));
        star.value.push_back(csv::number(row,"Sigmadot_star_Msun_Gyr_kpc2"));
    }
    const auto& radii=gas.radius;
    // The flat circular-speed amplitude cancels; retain the regular zero-flow
    // launch boundary rather than normalizing this solution to a supplied rate.
    const FlatRotationParameters flat{1};
    const ForwardModelParameters model{p.at("R_nucl_kpc"),radii.back(),0,0,
        {linear_profile_value,&gas},{},{linear_profile_value,&star},
        {{flat_rotation_velocity,&flat},{flat_rotation_velocity_derivative,&flat}},radii.front(),1};
    const auto solution=solve_forward_nonmixing(model,0.001,1e-13,1e-11);
    DoubleVec prescribed,required,velocity;
    for (double R: radii) {
        prescribed.push_back(sigmadot_land(R,p.at("Mdot_land_Msun_yr"),
            p.at("R_in_kpc"),p.at("R_out_kpc"),p.at("lambda")));
        required.push_back(forward_landing_rate(R,model,solution));
        velocity.push_back(forward_radial_velocity(R,model,solution));
        if (!std::isfinite(required.back()) || required.back()<=0)
            throw std::runtime_error("The inferred landing profile must be finite and positive");
    }
    // Keep the legacy diagnostic's sampled trapezoidal total, including its
    // use in the matched R^-2 comparison, rather than changing its quadrature.
    double total=0;
    for (std::size_t i=1;i<radii.size();++i)
        total+=std::numbers::pi*(radii[i]-radii[i-1])*(radii[i]*required[i]+radii[i-1]*required[i-1]);
    total=units::per_year(total);
    const auto Z_prescribed=chemistry(radii,star.value,prescribed,p);
    const auto Z_required=chemistry(radii,star.value,required,p);
    auto stream=output(destination/"values.csv",
        "Sigmadot_w_prescribed_Msun_Gyr_kpc2,v_R_prescribed_kpc_yr,v_R_prescribed_kms,"
        "residual_Msun_Gyr_kpc2,external_source_required_Msun_Gyr_kpc2,"
        "Sigmadot_w_required_Msun_Gyr_kpc2,v_R_required_kpc_yr,v_R_required_kms,"
        "Sigmadot_w_Rm2_same_total_Msun_Gyr_kpc2,Z_prescribed,Z_required");
    const PowerLawDiskParameters disk{p.at("R_nucl_kpc"),p.at("R_in_kpc"),p.at("R_out_kpc"),1,1,1,p.at("lambda")};
    for (std::size_t i=0;i<radii.size();++i) {
        const double R=radii[i];
        const double v=units::per_year(-prescribed[i]*power_law_transport_length(R,disk)/gas.value[i]);
        const double residual=prescribed[i]*power_law_stellar_sink_ratio(R,disk.R_nucl,disk.lambda)-star.value[i];
        stream << prescribed[i] << ',' << v << ',' << v*kpc_per_year_to_kms << ','
            << residual << ',' << -residual << ',' << required[i] << ',' << velocity[i] << ','
            << velocity[i]*kpc_per_year_to_kms << ','
            << sigmadot_land(R,total,disk.R_in,disk.R_out,2) << ',' << Z_prescribed[i] << ',' << Z_required[i] << '\n';
    }
    auto summary=output(destination/"summary.csv","Mdot_land_required_Msun_yr");
    summary << total << '\n';
}

std::vector<abundance::Observation> observations(const Path& input){
    std::vector<abundance::Observation> result;
    for (const auto& row: csv::read(input/"points.csv").rows) {
        const auto optional=[&](const char* key)->std::optional<double>{
            const auto& text=csv::value(row,key);
            return text.empty() ? std::nullopt : std::optional<double>(std::stod(text));
        };
        result.push_back({static_cast<int>(csv::number(row,"source_seq")),"",
            optional("offRA_arcsec"),optional("offDE_arcsec"),csv::number(row,"R_kpc"),
            csv::number(row,"OH12"),csv::number(row,"e_OH12")});
    }
    return result;
}

void collapse(const Path& input,const Path& destination){
    const double scatter=parameters(input).at("abundance_scatter_dex");
    auto stream=output(destination/"values.csv","source_seq,offRA_arcsec,offDE_arcsec,R_kpc,OH12,e_OH12,source_sequences,n_catalog_rows,sigma_OH12_dex,Z,e_Z_low,e_Z_high");
    for (const auto& row: abundance::collapse_repeated_regions(observations(input))) {
        stream << row.source_sequence << ',';
        if (row.offset_ra_arcsec) stream << *row.offset_ra_arcsec;
        stream << ',';
        if (row.offset_de_arcsec) stream << *row.offset_de_arcsec;
        stream << ',' << row.R_kpc << ',' << row.OH12 << ',' << row.e_OH12 << ',';
        for (std::size_t i=0;i<row.source_sequences.size();++i)
            stream << (i ? ";" : "") << row.source_sequences[i];
        const double Z=abundance::metallicity_from_oxygen_abundance(row.OH12);
        stream << ',' << row.source_sequences.size() << ',' << std::hypot(row.e_OH12,scatter) << ',' << Z
            << ',' << Z*(1-std::pow(10.0,-row.e_OH12)) << ',' << Z*(std::pow(10.0,row.e_OH12)-1) << '\n';
    }
}

void fit_abundance(const Path& input,const Path& destination){
    const auto p=parameters(input);
    DoubleVec radius,oxygen,sigma;
    double weight=0,pivot=0;
    for (const auto& row: csv::read(input/"points.csv").rows) {
        radius.push_back(csv::number(row,"R_kpc"));oxygen.push_back(csv::number(row,"OH12"));
        sigma.push_back(std::hypot(csv::number(row,"e_OH12"),p.at("abundance_scatter_dex")));
        const double w=1/(sigma.back()*sigma.back());weight+=w;pivot+=w*radius.back();
    }
    pivot/=weight;
    DoubleVec centered=radius;
    for (double& R: centered) R-=pivot;
    const auto fit=abundance::weighted_line_fit(centered,oxygen,sigma);
    if (!fit.slope) throw std::runtime_error("Abundance fit needs two distinct radii");
    double chi2=0;
    for (std::size_t i=0;i<radius.size();++i)
        chi2+=std::pow((oxygen[i]-*fit.intercept-*fit.slope*centered[i])/sigma[i],2);
    const auto bounds=std::minmax_element(radius.begin(),radius.end());
    auto stream=output(destination/"fit.csv","R_min_kpc,R_max_kpc,R_pivot_kpc,OH12_at_pivot,e_OH12_at_pivot,OH12_slope_dex_per_kpc,e_slope_dex_per_kpc,fit_chi2,fit_dof");
    stream << *bounds.first << ',' << *bounds.second << ',' << pivot << ',' << *fit.intercept << ','
        << *fit.e_intercept << ',' << *fit.slope << ',' << *fit.e_slope << ',' << chi2 << ',' << radius.size()-2 << '\n';
    auto profile=output(destination/"profiles.csv","R_kpc,Z,dZ_dR_per_kpc,OH12_fit,e_OH12_fit_dex,e_Z_low,e_Z_high");
    const int count=static_cast<int>(p.at("sample_count"));
    for (int i=0;i<count;++i) {
        const double R=*bounds.first+(*bounds.second-*bounds.first)*i/(count-1);
        const double OH12=*fit.intercept+*fit.slope*(R-pivot);
        const double Z=abundance::metallicity_from_oxygen_abundance(OH12);
        const double error=std::hypot(*fit.e_intercept,(R-pivot)*(*fit.e_slope));
        profile << R << ',' << Z << ',' << std::log(10.0)*(*fit.slope)*Z << ',' << OH12 << ',' << error << ','
            << Z*(1-std::pow(10.0,-error)) << ',' << Z*(std::pow(10.0,error)-1) << '\n';
    }
}

void compare_abundance(const Path& input,const Path& destination){
    abundance::ModelCurve curve{"","","","",{}, {}};
    for (const auto& row: csv::read(input/"curve.csv").rows) {
        curve.radius_kpc.push_back(csv::number(row,"R_kpc"));curve.metallicity.push_back(csv::number(row,"Z"));
    }
    const auto fit=abundance::comparison_statistics("",abundance::Calibration::KK04,
        observations(input),curve,{0,0,false,""});
    auto stream=output(destination/"fit.csv","n_catalog_usable,n_unique_catalog_regions,n_model_overlap,fit_status,weighted_mean_residual_dex,chi2_fixed_model,reduced_chi2_fixed_model,chi2_after_offset,reduced_chi2_after_offset");
    stream << fit.n_catalog_usable << ',' << fit.n_unique_catalog_regions << ',' << fit.n_model_overlap << ',' << fit.fit_status;
    for (const auto& value: {fit.weighted_mean_residual_dex,fit.chi2_fixed_model,fit.reduced_chi2_fixed_model,
                           fit.chi2_after_offset,fit.reduced_chi2_after_offset}) {
        stream << ',';if (value) stream << *value;
    }
    stream << '\n';
}
}

int main(int argc,char** argv){
    try {
        if (argc!=4) throw std::invalid_argument("Supply calculation, input directory and output directory");
        const std::string command=argv[1];
        const Path input=argv[2],destination=argv[3];
        std::filesystem::create_directories(destination);
        if (command=="potential") potential(input,destination);
        else if (command=="unmixed-observed") unmixed_observed(input,destination);
        else if (command=="metallicity") metallicity(input,destination);
        else if (command=="collapse-abundances") collapse(input,destination);
        else if (command=="fit-abundance") fit_abundance(input,destination);
        else if (command=="compare-abundance") compare_abundance(input,destination);
        else throw std::invalid_argument("Unknown calculation: "+command);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';return 1;
    }
}
