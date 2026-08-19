#include "models.h"

#include "physics.h"

namespace {

constexpr double pi=3.14159265358979323846;

struct AngularMomentumValues {double disk; double disk_derivative; double landing;};

AngularMomentumValues angular_momentum_values(double R,const ForwardModelParameters& parameters){
    const double v_c=rotation_velocity(parameters.rot_curve,R);
    const double dv_c_dR=rotation_velocity_derivative(parameters.rot_curve,R);
    const double v_c_nucl=rotation_velocity(parameters.rot_curve,parameters.R_nucl);
    const double disk=j_disk(R,v_c);
    const double disk_derivative=dj_disk_dR(R,v_c,dv_c_dR);
    const double landing=j_land_mixing(R,parameters.R_nucl,parameters.mu,parameters.beta,v_c,v_c_nucl);
    return {disk,disk_derivative,landing};
}

double initial_velocity_from_landing_rate(double sigmadot_land_initial,const ForwardModelParameters& parameters){
    const AngularMomentumValues angular=angular_momentum_values(parameters.R_nucl,parameters);
    const double gap=angular_momentum_gap(angular.landing,angular.disk);
    const double gas=profile_value(parameters.sigma_g,parameters.R_nucl);
    return sigmadot_land_initial*gap/(gas*angular.disk_derivative);
}

}

DoubleVec inverse_model_rhs(double R,const DoubleVec& state,const void* raw_parameters){
    (void)state;
    const InverseModelParameters& parameters=*static_cast<const InverseModelParameters*>(raw_parameters);
    const double landing=sigmadot_land(R,parameters.Mdot_land,parameters.R_nucl,parameters.R_out);
    const double star=sigma_star_ks(R,parameters.R_g,parameters.sigma_g0,parameters.A,parameters.N);
    return {R*(landing-star)};
}

DoubleVec empirical_inverse_model_rhs(double R,const DoubleVec& state,const void* raw_parameters){
    (void)state;
    const EmpiricalInverseModelParameters& parameters=*static_cast<const EmpiricalInverseModelParameters*>(raw_parameters);
    return {R*(profile_value(parameters.sigmadot_land,R)-profile_value(parameters.sigmadot_star,R))};
}

RK4Solution solve_inverse_model(const InverseModelParameters& parameters,double R_stop,double h_0,double atol,double rtol){
    return integrate_rk4(inverse_model_rhs,&parameters,parameters.R_out,{0.0},h_0,atol,rtol,R_stop);
}

RK4Solution solve_empirical_inverse_model(const EmpiricalInverseModelParameters& parameters,double R_stop,double h_0,double atol,double rtol){
    return integrate_rk4(empirical_inverse_model_rhs,&parameters,parameters.R_out,{0.0},h_0,atol,rtol,R_stop);
}

double inverse_radial_velocity(double R,const RK4Solution& solution,const InverseModelParameters& parameters){
    const double integral=evaluate(solution,R)[0];
    return (integral-parameters.Mdot_out/(2.0*pi))/(R*sigma_g(R,parameters.R_g,parameters.sigma_g0));
}

double empirical_inverse_radial_velocity(double R,const RK4Solution& solution,const EmpiricalInverseModelParameters& parameters){
    const double integral=evaluate(solution,R)[0];
    return (integral-parameters.Mdot_out/(2.0*pi))/(R*profile_value(parameters.sigma_g,R));
}

double inverse_landing_angular_momentum(double R,const RK4Solution& solution,const InverseModelParameters& parameters,const RotationCurve& rot_curve){
    const double radial_velocity=inverse_radial_velocity(R,solution,parameters);
    return j_land_required(R,sigma_g(R,parameters.R_g,parameters.sigma_g0),radial_velocity,sigmadot_land(R,parameters.Mdot_land,parameters.R_nucl,parameters.R_out),rotation_velocity(rot_curve,R),rotation_velocity_derivative(rot_curve,R));
}

double inverse_landing_angular_momentum_ratio(double R,const RK4Solution& solution,const InverseModelParameters& parameters,const RotationCurve& rot_curve){
    return angular_momentum_ratio(inverse_landing_angular_momentum(R,solution,parameters,rot_curve),j_disk(R,rotation_velocity(rot_curve,R)));
}

double sigmadot_land_mixing_from_velocity(double R,double radial_velocity,const ForwardModelParameters& parameters){
    const AngularMomentumValues angular=angular_momentum_values(R,parameters);
    return sigmadot_land_from_velocity(profile_value(parameters.sigma_g,R),radial_velocity,angular.disk,angular.landing,angular.disk_derivative);
}

DoubleVec forward_mixing_rhs(double R,const DoubleVec& state,const void* raw_parameters){
    const ForwardModelParameters& parameters=*static_cast<const ForwardModelParameters*>(raw_parameters);
    const double radial_velocity=state[0];
    const AngularMomentumValues angular=angular_momentum_values(R,parameters);
    const double gas=profile_value(parameters.sigma_g,R);
    const double gas_derivative=profile_value(parameters.sigma_g_derivative,R);
    const double star=profile_value(parameters.sigmadot_star,R);
    const double landing=sigmadot_land_from_velocity(gas,radial_velocity,angular.disk,angular.landing,angular.disk_derivative);

    return {radial_velocity_gradient(radial_velocity,R,gas,gas_derivative,star,angular.disk,angular.landing,angular.disk_derivative),2.0*pi*R*landing};
}

double radial_velocity_nonmixing(double R,double integral_state,const ForwardModelParameters& parameters){
    if (R==parameters.R_nucl) {
        return 0.0;
    }

    const double v_c=rotation_velocity(parameters.rot_curve,R);
    const double v_c_nucl=rotation_velocity(parameters.rot_curve,parameters.R_nucl);
    const double delta_j=j_disk(R,v_c)-j_disk(parameters.R_nucl,v_c_nucl);
    const double gas=profile_value(parameters.sigma_g,R);
    return -integral_state/(R*gas*delta_j);
}

double sigmadot_land_nonmixing(double R,double radial_velocity,const ForwardModelParameters& parameters){
    if (R==parameters.R_nucl) {
        return profile_value(parameters.sigmadot_star,R)/2.0;
    }

    const double v_c=rotation_velocity(parameters.rot_curve,R);
    const double dv_c_dR=rotation_velocity_derivative(parameters.rot_curve,R);
    const double v_c_nucl=rotation_velocity(parameters.rot_curve,parameters.R_nucl);
    return sigmadot_land_from_velocity(profile_value(parameters.sigma_g,R),radial_velocity,j_disk(R,v_c),j_disk(parameters.R_nucl,v_c_nucl),dj_disk_dR(R,v_c,dv_c_dR));
}

DoubleVec forward_nonmixing_rhs(double R,const DoubleVec& state,const void* raw_parameters){
    const ForwardModelParameters& parameters=*static_cast<const ForwardModelParameters*>(raw_parameters);
    const double v_c=rotation_velocity(parameters.rot_curve,R);
    const double v_c_nucl=rotation_velocity(parameters.rot_curve,parameters.R_nucl);
    const double delta_j=j_disk(R,v_c)-j_disk(parameters.R_nucl,v_c_nucl);
    const double star=profile_value(parameters.sigmadot_star,R);
    const double integral_gradient=R*star*delta_j;
    const double radial_velocity=radial_velocity_nonmixing(R,state[0],parameters);
    const double landing=sigmadot_land_nonmixing(R,radial_velocity,parameters);

    return {integral_gradient,2.0*pi*R*landing};
}

ForwardModelSolution solve_forward_mixing(const ForwardModelParameters& parameters,double target_mdot_land,double h_0,double atol,double rtol){
    const auto solve_for=[&](double initial_landing_rate){
        const double initial_radial_velocity=initial_velocity_from_landing_rate(initial_landing_rate,parameters);
        return integrate_rk4(forward_mixing_rhs,&parameters,parameters.R_nucl,{initial_radial_velocity,0.0},h_0,atol,rtol,parameters.R_out);
    };

    const RK4Solution first=solve_for(0.0);
    const RK4Solution second=solve_for(1.0);
    const double first_total=first.y.back()[1];
    const double second_total=second.y.back()[1];
    const double normalization_slope=second_total-first_total;

    const double normalized_initial_landing=(target_mdot_land-first_total)/normalization_slope;
    const double initial_radial_velocity=initial_velocity_from_landing_rate(normalized_initial_landing,parameters);
    const RK4Solution solution=solve_for(normalized_initial_landing);
    return {solution,solution.y.back()[1],initial_radial_velocity,normalized_initial_landing,false};
}

ForwardModelSolution solve_forward_nonmixing(const ForwardModelParameters& parameters,double h_0,double atol,double rtol){
    const RK4Solution solution=integrate_rk4(forward_nonmixing_rhs,&parameters,parameters.R_nucl,{0.0,0.0},h_0,atol,rtol,parameters.R_out);
    const double initial_landing_rate=profile_value(parameters.sigmadot_star,parameters.R_nucl)/2.0;
    return {solution,solution.y.back()[1],0.0,initial_landing_rate,true};
}

ForwardModelSolution solve_forward_model(const ForwardModelParameters& parameters,double target_mdot_land,double h_0,double atol,double rtol){
    return parameters.mu==0.0 ? solve_forward_nonmixing(parameters,h_0,atol,rtol) : solve_forward_mixing(parameters,target_mdot_land,h_0,atol,rtol);
}

double forward_radial_velocity(double R,const ForwardModelParameters& parameters,const ForwardModelSolution& solution){
    const DoubleVec state=evaluate(solution.solution,R);
    return solution.nonmixing ? radial_velocity_nonmixing(R,state[0],parameters) : state[0];
}

double forward_landing_rate(double R,const ForwardModelParameters& parameters,const ForwardModelSolution& solution){
    const double radial_velocity=forward_radial_velocity(R,parameters,solution);
    return solution.nonmixing ? sigmadot_land_nonmixing(R,radial_velocity,parameters) : sigmadot_land_mixing_from_velocity(R,radial_velocity,parameters);
}

double forward_cumulative_landing(double R,const ForwardModelSolution& solution){
    return evaluate(solution.solution,R)[1];
}

DoubleVec metallicity_rhs(double R,const DoubleVec& state,const void* raw_parameters){
    const MetallicityModelParameters& parameters=*static_cast<const MetallicityModelParameters*>(raw_parameters);

    const ForwardModelParameters& forward=*parameters.forward_parameters;
    const ForwardModelSolution& forward_solution=*parameters.forward_solution;
    const double radial_velocity=forward_radial_velocity(R,forward,forward_solution);
    const double landing=forward_landing_rate(R,forward,forward_solution);
    const double gas=profile_value(forward.sigma_g,R);
    const double star=profile_value(forward.sigmadot_star,R);
    const double Z_land=z_land_mixing(parameters.Z_nucl,parameters.Z_CGM,forward.mu);

    return {metallicity_gradient(state[0],gas,radial_velocity,landing,star,Z_land,parameters.yield)};
}

DoubleVec inverse_metallicity_rhs(double R,const DoubleVec& state,const void* raw_parameters){
    const InverseMetallicityModelParameters& parameters=*static_cast<const InverseMetallicityModelParameters*>(raw_parameters);
    const InverseModelParameters& inverse=*parameters.inverse_parameters;
    const double gas=sigma_g(R,inverse.R_g,inverse.sigma_g0);
    const double star=sigma_star_ks(R,inverse.R_g,inverse.sigma_g0,inverse.A,inverse.N);
    const double landing=sigmadot_land(R,inverse.Mdot_land,inverse.R_nucl,inverse.R_out);
    const double radial_velocity=inverse_radial_velocity(R,*parameters.inverse_solution,inverse);
    return {metallicity_gradient(state[0],gas,radial_velocity,landing,star,parameters.Z_land,parameters.yield)};
}

DoubleVec mixing_metallicity_rhs(double R,const DoubleVec& state,const void* raw_parameters){
    const MixingMetallicityModelParameters& parameters=*static_cast<const MixingMetallicityModelParameters*>(raw_parameters);
    const double Z_land=z_land_mixing(parameters.Z_nucl,parameters.Z_CGM,parameters.mu);
    return {metallicity_gradient(state[0],profile_value(parameters.sigma_g,R),profile_value(parameters.radial_velocity,R),profile_value(parameters.sigmadot_land,R),profile_value(parameters.sigmadot_star,R),Z_land,parameters.yield)};
}

DoubleVec empirical_metallicity_rhs(double R,const DoubleVec& state,const void* raw_parameters){
    const EmpiricalMetallicityModelParameters& parameters=*static_cast<const EmpiricalMetallicityModelParameters*>(raw_parameters);
    return {metallicity_gradient(state[0],profile_value(parameters.sigma_g,R),profile_value(parameters.radial_velocity,R),profile_value(parameters.sigmadot_land,R),profile_value(parameters.sigmadot_star,R),parameters.Z_land,parameters.yield)};
}

RK4Solution solve_metallicity(const MetallicityModelParameters& parameters,double R_start,double R_stop,double Z_initial,double h_0,double atol,double rtol){
    return integrate_rk4(metallicity_rhs,&parameters,R_start,{Z_initial},h_0,atol,rtol,R_stop);
}

RK4Solution solve_inverse_metallicity(const InverseMetallicityModelParameters& parameters,double R_start,double R_stop,double Z_initial,double h_0,double atol,double rtol){
    return integrate_rk4(inverse_metallicity_rhs,&parameters,R_start,{Z_initial},h_0,atol,rtol,R_stop);
}

RK4Solution solve_mixing_metallicity(const MixingMetallicityModelParameters& parameters,double R_start,double R_stop,double Z_initial,double h_0,double atol,double rtol){
    return integrate_rk4(mixing_metallicity_rhs,&parameters,R_start,{Z_initial},h_0,atol,rtol,R_stop);
}

RK4Solution solve_empirical_metallicity(const EmpiricalMetallicityModelParameters& parameters,double R_start,double R_stop,double Z_initial,double h_0,double atol,double rtol){
    return integrate_rk4(empirical_metallicity_rhs,&parameters,R_start,{Z_initial},h_0,atol,rtol,R_stop);
}
