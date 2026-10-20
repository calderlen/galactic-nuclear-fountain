#include "prescribed_gas.h"
#include "models.h"
#include "physics.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {
double gas_value(double R,const void* raw){
    const auto& p=*static_cast<const PrescribedGasParameters*>(raw);
    const double gas=profile_value(p.gas,R);
    if (!std::isfinite(gas) || gas<=0.0)
        throw std::invalid_argument("Prescribed gas density must be finite and positive");
    return gas;
}
double star_value(double R,const void* raw){
    const auto& p=*static_cast<const PrescribedGasParameters*>(raw);
    const double star=p.sigmadot_star.value ? profile_value(p.sigmadot_star,R) :
        stellar_sink_from_gas(gas_value(R,raw),p.A_net,p.N);
    if (!std::isfinite(star) || star<=0.0)
        throw std::invalid_argument("Net stellar sink must be finite and positive");
    return star;
}
double landing_value(double R,const void* raw){
    const auto& p=*static_cast<const PrescribedGasParameters*>(raw);
    return sigmadot_land(R,p.Mdot_land,p.R_in,p.R_out,p.lambda);
}
struct FlowContext {const PrescribedGasParameters* flow; const RK4Solution* solution;};
double velocity_value(double R,const void* raw){
    const auto& context=*static_cast<const FlowContext*>(raw);
    return prescribed_gas_profile(R,*context.flow,*context.solution).radial_velocity;
}
}

RK4Solution solve_prescribed_gas(const PrescribedGasParameters& p,double h0,double atol,double rtol){
    for (double value: {p.R_in,p.R_out,p.Mdot_land,p.lambda,p.A_net,p.N,p.Mdot_out,h0,atol,rtol}) {
        if (!std::isfinite(value)) throw std::invalid_argument("Prescribed-gas parameters must be finite");
    }
    if (!(p.R_in>0.0 && p.R_out>p.R_in && p.Mdot_land>0.0 &&
          (p.sigmadot_star.value || (p.A_net>0.0 && p.N>0.0)) &&
          h0>0.0 && atol>0.0 && rtol>0.0) ||
        !p.gas.value || !p.rotation.velocity.value || !p.rotation.derivative.value)
        throw std::invalid_argument("Invalid prescribed-gas domain, rates, profiles or solver controls");
    const EmpiricalInverseModelParameters flow{
        {gas_value,&p},{star_value,&p},{landing_value,&p},p.R_out,p.Mdot_out};
    // Share the existing mass-conservation RHS. Cap dense-output spacing at h0.
    return integrate_rk4(empirical_inverse_model_rhs,&flow,p.R_out,{p.Mdot_out},
                         h0,atol,rtol,p.R_in,h0);
}

PrescribedGasProfile prescribed_gas_profile(double R,const PrescribedGasParameters& p,const RK4Solution& solution){
    // Match the tabulated profiles' endpoint-roundoff tolerance. Two RK4
    // half steps can land one ulp beyond the same closed annulus endpoint.
    const double tolerance=16.0*std::numeric_limits<double>::epsilon()*std::max({std::abs(p.R_in),std::abs(p.R_out),1.0});
    if (!std::isfinite(R) || R<p.R_in-tolerance || R>p.R_out+tolerance)
        throw std::out_of_range("Requested radius is outside the prescribed-gas model");
    R=std::clamp(R,p.R_in,p.R_out);
    constexpr double pi=3.14159265358979323846;
    const double gas=gas_value(R,&p),star=star_value(R,&p),landing=landing_value(R,&p);
    const double accretion=evaluate(solution,R)[0];
    const double velocity=-accretion/(2.0*pi*R*gas);
    const double vc=rotation_velocity(p.rotation,R),dvc=rotation_velocity_derivative(p.rotation,R);
    if (!std::isfinite(vc) || vc<=0.0 || !std::isfinite(dvc) || dj_disk_dR(R,vc,dvc)<=0.0)
        throw std::invalid_argument("Rotation requires positive velocity and d(R*v_c)/dR");
    const double required=j_land_required(R,gas,velocity,landing,vc,dvc);
    if (!std::isfinite(required) || !std::isfinite(velocity))
        throw std::runtime_error("Nonfinite prescribed-gas transport or required angular momentum");
    return {gas,star,landing,cumulative_mdot_land(R,p.Mdot_land,p.R_in,p.R_out,p.lambda),
        accretion,velocity,vc,dvc,j_disk(R,vc),required};
}

PrescribedGasChemistrySolution solve_prescribed_gas_chemistry(
    const PrescribedGasParameters& p,const RK4Solution& dynamics,
    const PrescribedGasChemistryParameters& c){
    for (double value: {c.Z_land,c.yield,c.Z_inner_boundary,c.Z_outer_boundary,c.h0,c.atol,c.rtol})
        if (!std::isfinite(value)) throw std::invalid_argument("Chemistry inputs must be finite");
    if (!(c.Z_land>=0.0 && c.Z_land<=1.0 && c.yield>=0.0 && c.Z_inner_boundary>=0.0 &&
          c.Z_inner_boundary<=1.0 && c.Z_outer_boundary>=0.0 && c.Z_outer_boundary<=1.0 &&
          c.h0>0.0 && c.atol>0.0 && c.rtol>0.0 && c.cells>=2))
        throw std::invalid_argument("Invalid prescribed incident chemistry or solver controls");
    bool inward=true,outward=true;
    // Inspect the capped dynamics grid and interval midpoints before using an
    // advective ODE. Mixed-sign/zero transport uses conservative metal balance.
    for (std::size_t i=0; i<dynamics.x.size(); ++i) {
        const double mass=dynamics.y[i][0];
        inward=inward && mass>0.0;
        outward=outward && mass<0.0;
        if (i) {
            const double midpoint=evaluate(dynamics,(dynamics.x[i]+dynamics.x[i-1])/2.0)[0];
            inward=inward && midpoint>0.0;
            outward=outward && midpoint<0.0;
        }
    }
    PrescribedGasChemistrySolution result;
    if (inward || outward) {
        const FlowContext context{&p,&dynamics};
        const EmpiricalMetallicityModelParameters chemistry{
            {gas_value,&p},{velocity_value,&context},{landing_value,&p},{star_value,&p},c.Z_land,c.yield};
        const double start=inward ? p.R_out : p.R_in;
        const double stop=inward ? p.R_in : p.R_out;
        const double boundary=inward ? c.Z_outer_boundary : c.Z_inner_boundary;
        result.ode=integrate_rk4(empirical_metallicity_rhs,&chemistry,start,{boundary},
                                 c.h0,c.atol,c.rtol,stop,c.h0);
    } else {
        const ConservativeMetallicityParameters chemistry{{landing_value,&p},{star_value,&p},
            c.Z_land,c.yield,-p.Mdot_out,c.Z_inner_boundary,c.Z_outer_boundary};
        const auto solution=solve_conservative_metallicity(chemistry,p.R_in,p.R_out,c.cells);
        result.conservative=solution.metallicity;
        result.max_cell_balance_error=solution.max_cell_balance_error;
        result.used_conservative=true;
    }
    return result;
}

DoubleVec prescribed_gas_chemistry_profile(double R,const PrescribedGasParameters& p,
    const RK4Solution& dynamics,const PrescribedGasChemistryParameters& c,
    const PrescribedGasChemistrySolution& solution){
    double Z,gradient;
    if (solution.used_conservative) {
        const auto& table=solution.conservative;
        Z=linear_profile_value(R,&table);
        auto hi=std::upper_bound(table.radius.begin(),table.radius.end(),R);
        std::size_t index=static_cast<std::size_t>(hi-table.radius.begin());
        index=std::clamp(index,std::size_t{1},table.radius.size()-1);
        gradient=(table.value[index]-table.value[index-1])/(table.radius[index]-table.radius[index-1]);
    } else {
        Z=evaluate(solution.ode,R)[0];
        const auto row=prescribed_gas_profile(R,p,dynamics);
        gradient=metallicity_gradient(Z,row.gas,row.radial_velocity,row.landing,row.star,c.Z_land,c.yield);
    }
    if (!std::isfinite(Z) || Z<=0.0 || Z>1.0 || !std::isfinite(gradient))
        throw std::runtime_error("Prescribed-gas chemistry produced invalid metallicity");
    return {Z,gradient};
}
