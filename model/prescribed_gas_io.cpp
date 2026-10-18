#include "prescribed_gas_io.h"
#include "prescribed_gas.h"
#include "physics.h"
#include "csv.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace {
namespace csv=galactic_nuclear_fountain::csv;
using Settings=std::map<std::string,std::string>;
double number(const Settings& s,const std::string& key){
    std::size_t used=0;
    const auto& text=s.at(key);
    const double value=std::stod(text,&used);
    if (used!=text.size() || !std::isfinite(value))
        throw std::invalid_argument("Expected finite numeric parameter: "+key);
    return value;
}
void require_support(const LinearProfile& profile,double lo,double hi){
    if (profile.radius.size()<2 || profile.radius.front()>lo || profile.radius.back()<hi)
        throw std::invalid_argument("Tabulated input does not cover the model annulus; extrapolation is disabled");
}
LinearProfile table_profile(const std::filesystem::path& path,const std::string& column,
                            double lo,double hi,bool positive,const std::string& source=""){
    LinearProfile result;
    for (const auto& row: csv::read(path).rows) {
        if (!source.empty() && csv::value(row,"source")!=source) continue;
        const double R=csv::number(row,"R_kpc"),value=csv::number(row,column);
        if (!std::isfinite(R) || R<=0.0 || !std::isfinite(value) || (positive && value<=0.0) ||
            (!result.radius.empty() && R<=result.radius.back()))
            throw std::invalid_argument("Tabulated profiles need finite values and strictly increasing positive radii");
        result.radius.push_back(R);
        result.value.push_back(value);
    }
    require_support(result,lo,hi);
    return result;
}
struct ExponentialGas {double R_g,normalization;};
double exponential_gas(double R,const void* raw){
    const auto& p=*static_cast<const ExponentialGas*>(raw);
    return sigma_g(R,p.R_g,p.normalization);
}
std::filesystem::path input_file(const Settings& s,const std::string& key,const std::filesystem::path& input){
    const std::filesystem::path path=s.at(key);
    return path.is_absolute() ? path : input/path;
}
}

void write_prescribed_gas_model(const std::filesystem::path& input,const std::filesystem::path& output){
    const std::set<std::string> allowed{
        "closure","gas_profile","gas_file","R_g_kpc","M_g_Msun","landing_profile",
        "R_in_kpc","R_out_kpc","Mdot_land_Msun_yr","lambda","KS_A_SFR","KS_N",
        "return_fraction","Mdot_out_Msun_yr","rotation_model","v_c_kms","rotation_file",
        "rotation_source","n_points","h_0_kpc","rtol","dynamics_atol",
        "supply_model","Mdot_nuc_Msun_yr","eta","mu","metallicity_mode",
        "Z_land","Z_nucl","Z_CGM","yield_y","Z_inner_boundary","Z_outer_boundary",
        "metallicity_atol","metallicity_cells","surface_rate_time_unit",
        "stellar_sink","sfr_file","sfr_parameters_file",
        "angular_momentum_comparison","R_nucl_kpc","v_c_nucl_kms","beta"};
    Settings s;
    for (const auto& row: csv::read(input/"parameters.csv").rows) {
        const auto key=csv::value(row,"name");
        if (!allowed.contains(key)) throw std::invalid_argument("Unknown prescribed-gas parameter: "+key);
        if (!s.emplace(key,csv::value(row,"value")).second)
            throw std::invalid_argument("Duplicate prescribed-gas parameter: "+key);
    }
    if (s.contains("surface_rate_time_unit") && s.at("surface_rate_time_unit")!="Gyr")
        throw std::invalid_argument("Surface-rate and KS units must use Gyr");
    if (s.at("closure")!="required_j" || s.at("landing_profile")!="power_law")
        throw std::invalid_argument("prescribed-gas requires required_j closure and power-law landing");
    const double lo=number(s,"R_in_kpc"),hi=number(s,"R_out_kpc");
    const double returned=number(s,"return_fraction");
    const double count=number(s,"n_points");
    if (!(lo>0.0 && hi>lo && returned>=0.0 && returned<1.0 &&
          count>=2.0 && count<=1000000.0 && std::floor(count)==count))
        throw std::invalid_argument("Invalid annulus, return fraction or sampling");
    ExponentialGas analytic{};
    LinearProfile tabulated_gas,velocity,derivative;
    RadialProfile gas{};
    if (s.at("gas_profile")=="exponential") {
        if (s.contains("gas_file")) throw std::invalid_argument("Exponential gas uses R_g and M_g, not gas_file");
        const double R_g=number(s,"R_g_kpc"),M_g=number(s,"M_g_Msun");
        if (R_g<=0.0 || M_g<=0.0) throw std::invalid_argument("Gas scale length and annular mass must be positive");
        analytic={R_g,sigma_g0(M_g,R_g,lo,hi)};
        gas={exponential_gas,&analytic};
    } else if (s.at("gas_profile")=="tabulated") {
        if (s.contains("R_g_kpc") || s.contains("M_g_Msun"))
            throw std::invalid_argument("Tabulated gas is independent of exponential-disk parameters");
        tabulated_gas=table_profile(input_file(s,"gas_file",input),"Sigma_g_Msun_kpc2",lo,hi,true);
        gas={linear_profile_value,&tabulated_gas};
    } else throw std::invalid_argument("gas_profile must be exponential or tabulated");
    const std::string sink=s.contains("stellar_sink") ? s.at("stellar_sink") : "ks";
    LinearProfile tabulated_star;
    RadialProfile star{};
    double A_net=0.0,N=0.0,input_returned=returned,sfr_scale=1.0;
    if (sink=="observed_sfr") {
        Settings exported;
        for (const auto& row: csv::read(input_file(s,"sfr_parameters_file",input)).rows)
            exported.emplace(csv::value(row,"name"),csv::value(row,"value"));
        if (exported.at("stellar_sink")!="net_locked_mass")
            throw std::invalid_argument("Observed SFR spline must declare its saved net_locked_mass convention");
        input_returned=number(exported,"return_fraction");
        if (!(input_returned>=0.0 && input_returned<1.0))
            throw std::invalid_argument("Saved SFR return fraction must lie in [0,1)");
        // The exporter already applied its return fraction. Rescale the net
        // spline only if this run selects a different fraction; never apply it twice.
        sfr_scale=(1.0-returned)/(1.0-input_returned);
        tabulated_star=table_profile(input_file(s,"sfr_file",input),"Sigmadot_star_Msun_Gyr_kpc2",lo,hi,true);
        for (double& value: tabulated_star.value) value*=sfr_scale;
        star={linear_profile_value,&tabulated_star};
    } else if (sink=="ks") {
        A_net=(1.0-returned)*number(s,"KS_A_SFR");
        N=number(s,"KS_N");
    } else throw std::invalid_argument("stellar_sink must be ks or observed_sfr");
    FlatRotationParameters flat{};
    RotationCurve rotation{};
    if (s.at("rotation_model")=="flat") {
        if (s.contains("rotation_file") || s.contains("rotation_source"))
            throw std::invalid_argument("Flat rotation uses v_c_kms only");
        flat.velocity=number(s,"v_c_kms");
        if (flat.velocity<=0.0) throw std::invalid_argument("Flat rotation speed must be positive");
        rotation={{flat_rotation_velocity,&flat},{flat_rotation_velocity_derivative,&flat}};
    } else if (s.at("rotation_model")=="tabulated") {
        if (s.contains("v_c_kms")) throw std::invalid_argument("Tabulated rotation uses its supplied curve");
        const auto path=input_file(s,"rotation_file",input);
        const auto source=s.at("rotation_source");
        if (source.empty()) throw std::invalid_argument("Select an explicit rotation_source");
        velocity=table_profile(path,"V_kms",lo,hi,true,source);
        derivative=table_profile(path,"dV_dR_kms_kpc",lo,hi,false,source);
        for (std::size_t i=0; i<velocity.radius.size(); ++i) {
            if (dj_disk_dR(velocity.radius[i],velocity.value[i],derivative.value[i])<=0.0)
                throw std::invalid_argument("Tabulated rotation has nonpositive d(R*v_c)/dR");
        }
        rotation={{linear_profile_value,&velocity},{linear_profile_value,&derivative}};
    } else throw std::invalid_argument("rotation_model must be flat or tabulated");
    const bool reservoir_supply=s.contains("supply_model") && s.at("supply_model")=="nuclear_plus_cgm";
    if (s.contains("supply_model") && !reservoir_supply && s.at("supply_model")!="direct")
        throw std::invalid_argument("supply_model must be direct or nuclear_plus_cgm");
    double total_landing;
    if (reservoir_supply) {
        if (s.contains("Mdot_land_Msun_yr"))
            throw std::invalid_argument("Reservoir supply derives Mdot_land; do not also prescribe it");
        const double nuclear=number(s,"Mdot_nuc_Msun_yr"),eta=number(s,"eta"),mu=number(s,"mu");
        if (!(nuclear>0.0 && eta>0.0 && mu>=0.0))
            throw std::invalid_argument("Reservoir supply requires positive source/eta and nonnegative mu");
        total_landing=mdot_land_mixing(mdot_launch(nuclear,eta),mu);
    } else {
        for (const auto& key: {"Mdot_nuc_Msun_yr","eta","mu","Z_nucl","Z_CGM"})
            if (s.contains(key)) throw std::invalid_argument("Reservoir inputs require explicit nuclear_plus_cgm supply_model");
        total_landing=number(s,"Mdot_land_Msun_yr");
    }
    const std::string comparison=s.contains("angular_momentum_comparison") ?
        s.at("angular_momentum_comparison") : "";
    if (comparison.empty()) {
        for (const auto& key: {"R_nucl_kpc","v_c_nucl_kms","beta"})
            if (s.contains(key))
                throw std::invalid_argument("Angular comparison inputs require angular_momentum_comparison");
    }
    double launch_radius=0.0,launch_velocity=0.0,comparison_mu=0.0,comparison_beta=0.0;
    if (!comparison.empty()) {
        launch_radius=number(s,"R_nucl_kpc");
        launch_velocity=number(s,"v_c_nucl_kms");
        if (launch_radius<=0.0 || launch_velocity<=0.0)
            throw std::invalid_argument("Angular comparison requires positive launch radius and velocity");
        if (comparison=="mixture") {
            comparison_mu=number(s,"mu");
            comparison_beta=number(s,"beta");
        } else if (comparison!="nuclear")
            throw std::invalid_argument("Angular comparison must be nuclear or mixture");
    }
    const PrescribedGasParameters p{gas,rotation,lo,hi,total_landing,number(s,"lambda"),
        A_net,N,number(s,"Mdot_out_Msun_yr"),star};
    const auto solution=solve_prescribed_gas(p,number(s,"h_0_kpc"),number(s,"dynamics_atol"),number(s,"rtol"));
    const bool chemistry_enabled=s.contains("metallicity_mode") && s.at("metallicity_mode")=="prescribed_incident";
    if (s.contains("metallicity_mode") && !chemistry_enabled)
        throw std::invalid_argument("metallicity_mode must be prescribed_incident when supplied");
    PrescribedGasChemistryParameters chemistry{};
    PrescribedGasChemistrySolution metals;
    if (chemistry_enabled) {
        double incident;
        if (reservoir_supply) {
            if (s.contains("Z_land")) throw std::invalid_argument("Reservoir chemistry derives Z_land from Z_nucl/Z_CGM");
            const double zn=number(s,"Z_nucl"),zc=number(s,"Z_CGM");
            if (zn<0.0 || zn>1.0 || zc<0.0 || zc>1.0)
                throw std::invalid_argument("Reservoir abundances must be between zero and one");
            incident=z_land_mixing(zn,zc,number(s,"mu"));
        } else incident=number(s,"Z_land");
        const double cells=number(s,"metallicity_cells");
        if (cells<2.0 || cells>1000000.0 || std::floor(cells)!=cells)
            throw std::invalid_argument("metallicity_cells must be an integer from 2 to 1000000");
        chemistry={incident,number(s,"yield_y"),number(s,"Z_inner_boundary"),number(s,"Z_outer_boundary"),
            number(s,"h_0_kpc"),number(s,"metallicity_atol"),number(s,"rtol"),static_cast<std::size_t>(cells)};
        metals=solve_prescribed_gas_chemistry(p,solution,chemistry);
    } else {
        for (const auto& key: {"Z_land","Z_nucl","Z_CGM","yield_y","Z_inner_boundary","Z_outer_boundary",
                              "metallicity_atol","metallicity_cells"})
            if (s.contains(key)) throw std::invalid_argument("Chemistry inputs require explicit prescribed_incident metallicity_mode");
    }
    std::filesystem::create_directories(output);
    std::ofstream stream(output/"profiles.csv");
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << std::setprecision(17)
        << "model,R_kpc,Sigma_g_Msun_kpc2,Sigmadot_star_Msun_Gyr_kpc2,Sigmadot_land_Msun_Gyr_kpc2,"
        << "cumulative_landing_Msun_yr,Mdot_acc_Msun_yr,v_R_kpc_yr,v_R_kms,v_c_kms,dv_c_dR_kms_kpc,"
        << "j_disk_kpc_kms,j_land_required_kpc_kms,j_land_required_over_j_disk";
    if (chemistry_enabled) stream << ",Z_land,Z,dZ_dR_per_kpc";
    if (!comparison.empty())
        stream << ",j_nucl_kpc_kms,j_land_comparison_kpc_kms,j_land_required_minus_comparison_kpc_kms";
    stream << '\n';
    std::size_t negative=0,outward=0;
    const std::size_t points=static_cast<std::size_t>(count);
    for (std::size_t i=0; i<points; ++i) {
        const double R=i+1==points ? hi : lo+(hi-lo)*static_cast<double>(i)/static_cast<double>(points-1);
        const auto row=prescribed_gas_profile(R,p,solution);
        negative+=row.required_landing_angular_momentum<0.0;
        outward+=row.radial_velocity>0.0;
        stream << "prescribed-gas," << R << ',' << row.gas << ',' << row.star << ',' << row.landing
            << ',' << row.cumulative_landing << ',' << row.accretion << ',' << row.radial_velocity
            << ',' << row.radial_velocity*9.7779222168e8 << ',' << row.circular_velocity
            << ',' << row.circular_velocity_derivative << ',' << row.disk_angular_momentum
            << ',' << row.required_landing_angular_momentum
            << ',' << row.required_landing_angular_momentum/row.disk_angular_momentum;
        if (chemistry_enabled) {
            const auto metal=prescribed_gas_chemistry_profile(R,p,solution,chemistry,metals);
            stream << ',' << chemistry.Z_land << ',' << metal[0] << ',' << metal[1];
        }
        if (!comparison.empty()) {
            const double assumed=j_land_mixing(R,launch_radius,comparison_mu,comparison_beta,
                                               row.circular_velocity,launch_velocity);
            stream << ',' << launch_radius*launch_velocity << ',' << assumed
                << ',' << row.required_landing_angular_momentum-assumed;
        }
        stream << '\n';
    }
    std::ofstream summary(output/"summary.csv");
    summary.exceptions(std::ios::failbit | std::ios::badbit);
    summary << std::setprecision(17) << "quantity,value\nclosure,required_j\nstellar_sink,"
        << (sink=="observed_sfr" ? "net_observed_sfr_spline" : "net_KS_from_supplied_gas") << '\n'
        << "gas_profile," << s.at("gas_profile") << "\nrotation_model," << s.at("rotation_model")
        << "\nMdot_land_Msun_yr," << p.Mdot_land << "\nMdot_out_Msun_yr," << p.Mdot_out
        << "\nnegative_required_j_samples," << negative << "\noutward_flow_samples," << outward << '\n';
    summary << "supply_model," << (reservoir_supply ? "nuclear_plus_cgm" : "direct") << '\n';
    summary << "return_fraction," << returned << '\n';
    if (!comparison.empty())
        summary << "angular_momentum_comparison," << comparison << "\nj_nucl_kpc_kms,"
            << launch_radius*launch_velocity << '\n';
    if (sink=="observed_sfr")
        summary << "sfr_input_return_fraction," << input_returned << "\nsfr_net_rescale_factor," << sfr_scale << '\n';
    if (chemistry_enabled) {
        summary << "Z_land," << chemistry.Z_land << "\nmetallicity_solver,"
            << (metals.used_conservative ? "conservative_upwind" : "RK4")
            << "\nmetallicity_max_cell_balance_error," << metals.max_cell_balance_error << '\n';
    }
    std::filesystem::copy_file(input/"parameters.csv",output/"parameters.csv",std::filesystem::copy_options::overwrite_existing);
}
