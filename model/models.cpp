#include "models.h"

#include "physics.h"
#include "units.h"
#include "landing_kernel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

constexpr double pi=3.14159265358979323846;

void validate_gas_and_metals(const DoubleVec& state, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(state[i]) || state[i] < 0.0 ||
            !std::isfinite(state[n + i]) || state[n + i] < 0.0 || state[n + i] > state[i]) {
            throw std::invalid_argument("Gas/metal densities must be finite with 0 <= Sigma_Z <= Sigma_g; reduce dt_Myr if a stage or step violates this");
        }
    }
}

struct AngularMomentumValues {double disk; double disk_derivative; double landing;};

AngularMomentumValues angular_momentum_values(double R,const ForwardModelParameters& parameters){
    const double v_c=rotation_velocity(parameters.rot_curve,R);
    const double dv_c_dR=rotation_velocity_derivative(parameters.rot_curve,R);
    const double v_c_nucl=parameters.v_c_nucl;
    const double disk=j_disk(R,v_c);
    const double disk_derivative=dj_disk_dR(R,v_c,dv_c_dR);
    const double cgm=parameters.cgm_angular_momentum.value ?
        profile_value(parameters.cgm_angular_momentum,R) : j_cgm(R,parameters.beta,v_c);
    const double landing=j_land_from_reservoirs(parameters.R_nucl*v_c_nucl,cgm,parameters.mu);
    return {disk,disk_derivative,landing};
}

double initial_accretion_from_landing_rate(double sigmadot_land_initial,const ForwardModelParameters& parameters){
    const AngularMomentumValues angular=angular_momentum_values(parameters.R_in,parameters);
    const double gap=angular_momentum_gap(angular.landing,angular.disk);
    return units::per_year(-2.0*pi*parameters.R_in*sigmadot_land_initial*gap/angular.disk_derivative);
}

double steady_max_step(double R_start,double R_stop,double h_0){
    // Some flat-curve flux solutions have exact RK4 nodes but nonpolynomial
    // profiles between them. Bound cubic-interpolation spacing independently
    // of the adaptive integration error, including for tabulated disk inputs.
    const double span=std::abs(R_stop-R_start);
    return span==0.0 ? std::abs(h_0) : std::min(std::abs(h_0),span/1000.0);
}

}

DoubleVec mass_continuity_star(const DoubleVec& sigma_g, const MassContinuityParameters& p) {
    DoubleVec star(sigma_g.size());
    for (std::size_t i = 0; i < star.size(); ++i) star[i] = p.A * std::pow(sigma_g[i], p.N);
    return star;
}

void mass_continuity_landing_sources(double t_Myr, const MassContinuityParameters& p,
                                    DoubleVec& wind, DoubleVec& mass, DoubleVec& angular) {
    const std::function<double(double)> launch_gyr=[&](double t){return units::per_gyr(p.Mdot_launch(t));};
    wind = convolve_kernel(p.K_wind, p.R_bins, p.tau_bins, t_Myr, launch_gyr);
    mass = convolve_kernel(p.K_mass, p.R_bins, p.tau_bins, t_Myr, launch_gyr);
    angular = convolve_kernel(p.K_j, p.R_bins, p.tau_bins, t_Myr, launch_gyr);
}

DoubleVec mass_continuity_velocity(const DoubleVec& sigma_g, const DoubleVec& landing,
                                  const DoubleVec& angular, const MassContinuityParameters& p) {
    const std::size_t n = sigma_g.size();
    DoubleVec velocity(n-1);
    for (std::size_t edge = 1; edge < n; ++edge) {
        const double R = p.R_bins[edge];
        const double v_c = rotation_velocity(p.rot_curve, R);
        const double dj_dR = dj_disk_dR(R, v_c, rotation_velocity_derivative(p.rot_curve, R));
        const double f = (R-p.R_bins[edge-1]) / (p.R_bins[edge+1]-p.R_bins[edge-1]);
        const double mass_face = (1-f)*landing[edge-1] + f*landing[edge];
        const double angular_face = (1-f)*angular[edge-1] + f*angular[edge];
        const double sigma_v = (angular_face - j_disk(R, v_c)*mass_face) / dj_dR;
        const double donor = sigma_v >= 0.0 ? sigma_g[edge-1] : sigma_g[edge];
        if (sigma_v != 0.0 && donor <= 0.0)
            throw std::invalid_argument("Transport requires gas in the donor cell; reduce dt or revise the initial disk");
        velocity[edge-1] = sigma_v == 0.0 ? 0.0 : units::per_year(sigma_v/donor);
        if (!std::isfinite(velocity[edge-1])) throw std::overflow_error("Disk velocity overflowed");
    }
    return velocity;
}

DoubleVec mass_continuity_rhs(double t_Myr, const DoubleVec& state, const void* raw_parameters) {
    const auto& p = *static_cast<const MassContinuityParameters*>(raw_parameters);
    const std::size_t n = p.R_bins.size() - 1;
    validate_gas_and_metals(state, n);
    const DoubleVec sigma_g(state.begin(), state.begin() + n);
    const DoubleVec star = mass_continuity_star(sigma_g, p);
    DoubleVec Z(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(star[i]) || star[i] < 0.0) {
            throw std::invalid_argument("Star-formation sinks must be finite and nonnegative");
        }
        // An empty cell has no metals to remove or advect. Its reported Z is undefined.
        Z[i] = sigma_g[i] > 0.0 ? state[n + i] / sigma_g[i] : 0.0;
    }
    DoubleVec wind, landing, angular;
    mass_continuity_landing_sources(t_Myr, p, wind, landing, angular);
    const DoubleVec velocities = mass_continuity_velocity(sigma_g, landing, angular, p);

    DoubleVec flux(n + 1);
    flux.front() = p.inner_mass_flux;
    flux.back() = p.outer_mass_flux;
    if ((flux.front() < 0.0 && sigma_g.front() == 0.0) || (flux.back() > 0.0 && sigma_g.back() == 0.0)) {
        throw std::invalid_argument("Boundary outflow requires gas in the donor cell");
    }
    for (std::size_t edge = 1; edge < n; ++edge) {
        const double velocity = velocities[edge - 1];
        const std::size_t donor = velocity >= 0.0 ? edge - 1 : edge;
        flux[edge] = 2.0 * pi * p.R_bins[edge] * velocity * sigma_g[donor];
        if (!std::isfinite(flux[edge])) {
            throw std::overflow_error("Radial mass flux overflowed");
        }
    }
    const DoubleVec metal_flux = disk_metal_flux(state, flux, p);

    DoubleVec derivative(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        const double area = pi * (p.R_bins[i + 1] - p.R_bins[i]) * (p.R_bins[i + 1] + p.R_bins[i]);
        derivative[i] = units::per_myr(landing[i] - star[i]) + 1e6*(flux[i] - flux[i + 1])/area;
        const double metals = p.Z_nucl*wind[i] + p.Z_CGM*(landing[i]-wind[i]);
        derivative[n + i] = units::per_myr(metals + (p.yield - Z[i]) * star[i])
                                 + 1e6*(metal_flux[i] - metal_flux[i + 1])/area;
        if (!std::isfinite(derivative[i]) || !std::isfinite(derivative[n + i])) {
            throw std::overflow_error("Gas or metal surface-density derivative overflowed");
        }
    }
    return derivative;
}

DoubleVec advance_mass_continuity(double t_Myr, double dt_Myr, const DoubleVec& state,
                                 const MassContinuityParameters& p) {
    const double dt_yr = dt_Myr * 1e6;
    if (!std::isfinite(dt_yr) || dt_yr <= 0.0 || !std::isfinite(t_Myr) ||
        !std::isfinite(t_Myr + dt_Myr) || t_Myr + dt_Myr <= t_Myr) {
        throw std::invalid_argument("Mass continuity needs a finite forward time step");
    }
    const DoubleVec next = rk4_step(mass_continuity_rhs, &p, t_Myr, state, dt_Myr);
    validate_gas_and_metals(next, p.R_bins.size() - 1);
    return next;
}

DoubleVec reconstruct_mass_flux(const DoubleVec& derivative, const DoubleVec& landing,
                                const MassContinuityParameters& p) {
    const std::size_t n = landing.size();
    if (n == 0 || derivative.size() != 2*n || p.R_bins.size() != n+1 || p.sigmadot_star.size() != n) {
        throw std::invalid_argument("Flow reconstruction requires matching annuli, sources and gas/metal derivatives");
    }
    DoubleVec flux(n+1);
    flux.back() = p.outer_mass_flux;
    double roundoff = 8.0*std::numeric_limits<double>::epsilon()*std::abs(flux.back());
    for (std::size_t i = n; i-- > 0;) {
        const double area = pi*(p.R_bins[i+1]-p.R_bins[i])*(p.R_bins[i+1]+p.R_bins[i]);
        flux[i] = flux[i+1] + area*(derivative[i]/1e6 + units::per_year(p.sigmadot_star[i]-landing[i]));
        if (!(area > 0.0) || !std::isfinite(flux[i])) {
            throw std::invalid_argument("Flow reconstruction needs finite derivatives/sources and increasing radius edges");
        }
        // Subtracting balanced, double-precision sources leaves roundoff, not
        // resolved transport. Do not turn that residue into a finite R/|v_R|.
        roundoff += 8.0*std::numeric_limits<double>::epsilon()*area*
                    (std::abs(derivative[i]/1e6)+units::per_year(std::abs(landing[i])+std::abs(p.sigmadot_star[i])));
        if (std::abs(flux[i]) <= roundoff) flux[i] = 0.0;
    }
    return flux;
}

DoubleVec disk_metal_flux(const DoubleVec& state, const DoubleVec& flux,
                         const MassContinuityParameters& p) {
    const std::size_t n = p.R_bins.size()-1;
    const auto Z = [&](std::size_t i) { return state[i] > 0.0 ? state[n+i]/state[i] : 0.0; };
    DoubleVec metals(n+1);
    metals.front() = flux.front()*(flux.front() >= 0.0 ? p.Z_inner_inflow : Z(0));
    metals.back() = flux.back()*(flux.back() <= 0.0 ? p.Z_outer_inflow : Z(n-1));
    for (std::size_t edge = 1; edge < n; ++edge) {
        metals[edge] = flux[edge]*Z(flux[edge] >= 0.0 ? edge-1 : edge);
    }
    return metals;
}

DoubleVec reconstruct_landing_metallicity(const DoubleVec& state, const DoubleVec& derivative,
                                         const DoubleVec& landing, const DoubleVec& metal_flux,
                                         const MassContinuityParameters& p) {
    const std::size_t n = landing.size();
    if (derivative.size() != 2*n || metal_flux.size() != n+1 || p.R_bins.size() != n+1 || p.sigmadot_star.size() != n) {
        throw std::invalid_argument("Metal reconstruction requires matching annuli, derivatives and face fluxes");
    }
    DoubleVec required(n, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < n; ++i) {
        if (landing[i] <= 0.0) continue;
        const double area = pi*(p.R_bins[i+1]-p.R_bins[i])*(p.R_bins[i+1]+p.R_bins[i]);
        const double Z = state[i] > 0.0 ? state[n+i]/state[i] : 0.0;
        required[i] = (derivative[n+i]*units::myr_per_gyr + units::per_gyr((metal_flux[i+1]-metal_flux[i])/area)
                       + (Z-p.yield)*p.sigmadot_star[i])/landing[i];
    }
    return required;
}

DoubleVec inverse_model_rhs(double R,const DoubleVec& state,const void* raw_parameters){
    (void)state;
    const InverseModelParameters& parameters=*static_cast<const InverseModelParameters*>(raw_parameters);
    const double landing=sigmadot_land(R,parameters.Mdot_land,parameters.R_in,parameters.R_out,parameters.lambda);
    const double star=sigma_star_ks(R,parameters.R_g,parameters.sigma_g0,parameters.A,parameters.N);
    return {units::per_year(2.0*pi*R*(star-landing))};
}

DoubleVec empirical_inverse_model_rhs(double R,const DoubleVec& state,const void* raw_parameters){
    (void)state;
    const EmpiricalInverseModelParameters& parameters=*static_cast<const EmpiricalInverseModelParameters*>(raw_parameters);
    return {units::per_year(2.0*pi*R*(profile_value(parameters.sigmadot_star,R)-profile_value(parameters.sigmadot_land,R)))};
}

RK4Solution solve_inverse_model(const InverseModelParameters& parameters,double R_stop,double h_0,double atol,double rtol){
    return integrate_rk4(inverse_model_rhs,&parameters,parameters.R_out,{parameters.Mdot_out},h_0,atol,rtol,R_stop,
        steady_max_step(parameters.R_out,R_stop,h_0));
}

RK4Solution solve_empirical_inverse_model(const EmpiricalInverseModelParameters& parameters,double R_stop,double h_0,double atol,double rtol){
    return integrate_rk4(empirical_inverse_model_rhs,&parameters,parameters.R_out,{parameters.Mdot_out},h_0,atol,rtol,R_stop,
        steady_max_step(parameters.R_out,R_stop,h_0));
}

double inverse_radial_velocity(double R,const RK4Solution& solution,const InverseModelParameters& parameters){
    const double accretion=evaluate(solution,R)[0];
    return -accretion/(2.0*pi*R*sigma_g(R,parameters.R_g,parameters.sigma_g0));
}

double empirical_inverse_radial_velocity(double R,const RK4Solution& solution,const EmpiricalInverseModelParameters& parameters){
    const double accretion=evaluate(solution,R)[0];
    return -accretion/(2.0*pi*R*profile_value(parameters.sigma_g,R));
}

double inverse_landing_angular_momentum(double R,const RK4Solution& solution,const InverseModelParameters& parameters,const RotationCurve& rot_curve){
    const double radial_velocity=inverse_radial_velocity(R,solution,parameters);
    return j_land_required(R,sigma_g(R,parameters.R_g,parameters.sigma_g0),radial_velocity,sigmadot_land(R,parameters.Mdot_land,parameters.R_in,parameters.R_out,parameters.lambda),rotation_velocity(rot_curve,R),rotation_velocity_derivative(rot_curve,R));
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
    const double accretion=state[0];
    const AngularMomentumValues angular=angular_momentum_values(R,parameters);
    const double star=profile_value(parameters.sigmadot_star,R);
    const double transport=accretion*angular.disk_derivative/(angular.landing-angular.disk);

    // Mdot_acc = -2*pi*R*Sigma_g*v_R.  Angular-momentum balance gives
    // 2*pi*R*Sigmadot_land = -Mdot_acc*j'/(j_land-j), while continuity
    // gives Mdot_acc' = 2*pi*R*(Sigmadot_star-Sigmadot_land).
    return {transport+units::per_year(2.0*pi*R*star),-transport};
}

double radial_velocity_nonmixing(double R,double integral_state,const ForwardModelParameters& parameters){
    if (R==parameters.R_nucl) {
        return 0.0;
    }

    const double v_c=rotation_velocity(parameters.rot_curve,R);
    const double v_c_nucl=parameters.v_c_nucl;
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
    const double v_c_nucl=parameters.v_c_nucl;
    return sigmadot_land_from_velocity(profile_value(parameters.sigma_g,R),radial_velocity,j_disk(R,v_c),j_disk(parameters.R_nucl,v_c_nucl),dj_disk_dR(R,v_c,dv_c_dR));
}

DoubleVec forward_nonmixing_rhs(double R,const DoubleVec& state,const void* raw_parameters){
    const ForwardModelParameters& parameters=*static_cast<const ForwardModelParameters*>(raw_parameters);
    const double v_c=rotation_velocity(parameters.rot_curve,R);
    const double v_c_nucl=parameters.v_c_nucl;
    const double delta_j=j_disk(R,v_c)-j_disk(parameters.R_nucl,v_c_nucl);
    const double star=profile_value(parameters.sigmadot_star,R);
    const double integral_gradient=units::per_year(R*star*delta_j);
    const double radial_velocity=radial_velocity_nonmixing(R,state[0],parameters);
    const double landing=sigmadot_land_nonmixing(R,radial_velocity,parameters);

    return {integral_gradient,units::per_year(2.0*pi*R*landing)};
}

ForwardModelSolution solve_forward_mixing(const ForwardModelParameters& parameters,double target_mdot_land,double h_0,double atol,double rtol){
    const auto solve_for=[&](double initial_landing_rate){
        const double initial_accretion=initial_accretion_from_landing_rate(initial_landing_rate,parameters);
        return integrate_rk4(forward_mixing_rhs,&parameters,parameters.R_in,{initial_accretion,0.0},h_0,atol,rtol,parameters.R_out,
            steady_max_step(parameters.R_in,parameters.R_out,h_0));
    };

    const RK4Solution first=solve_for(0.0);
    const RK4Solution second=solve_for(units::per_gyr(1.0));
    const double first_total=first.y.back()[1];
    const double second_total=second.y.back()[1];
    const double normalization_slope=(second_total-first_total)/units::per_gyr(1.0);

    if (!std::isfinite(normalization_slope) || normalization_slope==0.0)
        throw std::runtime_error("Forward boundary cannot be normalized to the nuclear launch rate");
    const double normalized_initial_landing=(target_mdot_land-first_total)/normalization_slope;
    const double initial_accretion=initial_accretion_from_landing_rate(normalized_initial_landing,parameters);
    const double initial_radial_velocity=-initial_accretion/
        (2.0*pi*parameters.R_in*profile_value(parameters.sigma_g,parameters.R_in));
    const RK4Solution solution=solve_for(normalized_initial_landing);
    return {solution,solution.y.back()[1],initial_radial_velocity,normalized_initial_landing,false};
}

ForwardModelSolution solve_forward_nonmixing(const ForwardModelParameters& parameters,double h_0,double atol,double rtol){
    const RK4Solution solution=integrate_rk4(forward_nonmixing_rhs,&parameters,parameters.R_nucl,{0.0,0.0},h_0,atol,rtol,parameters.R_out,
        steady_max_step(parameters.R_nucl,parameters.R_out,h_0));
    const double initial_landing_rate=profile_value(parameters.sigmadot_star,parameters.R_nucl)/2.0;
    return {solution,solution.y.back()[1],0.0,initial_landing_rate,true};
}

ForwardModelSolution solve_forward_model(const ForwardModelParameters& parameters,double target_mdot_land,double h_0,double atol,double rtol){
    const auto angular=angular_momentum_values(parameters.R_in,parameters);
    if (parameters.cgm_angular_momentum.value && parameters.mu>0.0) {
        const double initial_gap=angular.landing-angular.disk;
        for (int i=0;i<=1000;++i) {
            const double R=parameters.R_in+(parameters.R_out-parameters.R_in)*i/1000.0;
            const auto local=angular_momentum_values(R,parameters);
            const double gap=local.landing-local.disk;
            if (!std::isfinite(gap) || gap==0.0 || std::signbit(gap)!=std::signbit(initial_gap))
                throw std::runtime_error("Local CGM mixing makes j_land-j_disk vanish or cross zero near R="+
                    std::to_string(R)+" kpc; the forward closure is singular on this disk domain");
        }
    }
    if (parameters.mu==0.0 && parameters.R_in==parameters.R_nucl && angular.landing==angular.disk) {
        const auto solution=solve_forward_nonmixing(parameters,h_0,atol,rtol);
        if (std::abs(solution.Mdot_land-target_mdot_land)>1e-6*std::max(target_mdot_land,1e-12))
            throw std::runtime_error("The regular mu=0 solution at R_in=R_nucl fixes its own total landing rate; choose R_in>R_nucl or a compatible nuclear rate");
        return solution;
    }
    if (angular.landing==angular.disk)
        throw std::runtime_error("Zero angular-momentum gap at R_in makes the forward boundary singular");
    return solve_forward_mixing(parameters,target_mdot_land,h_0,atol,rtol);
}

double forward_accretion_rate(double R,const ForwardModelParameters& parameters,const ForwardModelSolution& solution){
    const DoubleVec state=evaluate(solution.solution,R);
    if (!solution.nonmixing) return state[0];
    if (R==parameters.R_nucl) return 0.0;
    const double disk=j_disk(R,rotation_velocity(parameters.rot_curve,R));
    const double nuclear=j_disk(parameters.R_nucl,parameters.v_c_nucl);
    return 2.0*pi*state[0]/(disk-nuclear);
}

double forward_radial_velocity(double R,const ForwardModelParameters& parameters,const ForwardModelSolution& solution){
    return -forward_accretion_rate(R,parameters,solution)/
        (2.0*pi*R*profile_value(parameters.sigma_g,R));
}

double forward_landing_rate(double R,const ForwardModelParameters& parameters,const ForwardModelSolution& solution){
    if (solution.nonmixing && R==parameters.R_nucl)
        return profile_value(parameters.sigmadot_star,R)/2.0;
    const AngularMomentumValues angular=angular_momentum_values(R,parameters);
    const double landing_angular=solution.nonmixing ?
        j_disk(parameters.R_nucl,parameters.v_c_nucl) : angular.landing;
    return units::per_gyr(-forward_accretion_rate(R,parameters,solution)*angular.disk_derivative/
        (2.0*pi*R*(landing_angular-angular.disk)));
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
    const double landing=sigmadot_land(R,inverse.Mdot_land,inverse.R_in,inverse.R_out,inverse.lambda);
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

ConservativeMetallicitySolution solve_conservative_metallicity(
    const ConservativeMetallicityParameters& p,double R_in,double R_out,std::size_t cells){
    if (!(R_in>0.0 && R_out>R_in) || cells<2)
        throw std::invalid_argument("Steady metal conservation needs positive increasing radii and at least two cells");
    const double width=(R_out-R_in)/static_cast<double>(cells);
    DoubleVec edges(cells+1),landing(cells),star(cells),flux(cells+1);
    // Two-point Gauss quadrature integrates R times a linear input exactly
    // within each input segment; mesh refinement resolves its knots.
    for (std::size_t i=0; i<=cells; ++i) edges[i]=R_in+width*static_cast<double>(i);
    edges.back()=R_out;
    for (std::size_t i=0; i<cells; ++i) {
        const double midpoint=(edges[i]+edges[i+1])/2.0;
        for (const double sign: {-1.0,1.0}) {
            const double R=midpoint+sign*width/(2.0*std::sqrt(3.0));
            const double land=profile_value(p.sigmadot_land,R);
            const double sink=profile_value(p.sigmadot_star,R);
            if (!std::isfinite(land) || !std::isfinite(sink) || land<0.0 || sink<0.0)
                throw std::runtime_error("Steady metal conservation requires finite nonnegative landing and stellar sinks");
            landing[i]+=units::per_year(pi*width*R*land);
            star[i]+=units::per_year(pi*width*R*sink);
        }
    }
    flux.back()=p.outer_mass_flux;
    double flux_scale=std::abs(flux.back());
    for (std::size_t i=cells; i-- > 0;) {
        flux[i]=flux[i+1]-landing[i]+star[i];
        flux_scale+=landing[i]+star[i];
    }
    double Z_inner=p.Z_inner_boundary;
    // The regular zero-flux limit also supplies boundary Z when the mass
    // closure differs from zero only at the input/ODE numerical precision.
    const bool closed_inner=std::abs(flux.front())<=1e-6*flux_scale;
    if (closed_inner) {
        const double land=profile_value(p.sigmadot_land,R_in);
        Z_inner=land>0.0 ? p.Z_land+p.yield*profile_value(p.sigmadot_star,R_in)/land : p.Z_land;
    }
    if (flux.front()>0.0 && !std::isfinite(Z_inner))
        throw std::runtime_error("An inner inflow needs Z_inner_boundary; the outer abundance alone does not determine both flow branches");
    if (flux.back()<0.0 && !std::isfinite(p.Z_outer_boundary))
        throw std::runtime_error("An outer inflow needs Z_outer_boundary");

    DoubleVec diagonal(cells),lower(cells),upper(cells),rhs(cells),Z(cells);
    for (std::size_t i=0; i<cells; ++i) {
        diagonal[i]=std::max(flux[i+1],0.0)-std::min(flux[i],0.0)+star[i];
        if (i>0) lower[i]=-std::max(flux[i],0.0);
        if (i+1<cells) upper[i]=std::min(flux[i+1],0.0);
        rhs[i]=p.Z_land*landing[i]+p.yield*star[i];
    }
    if (flux.front()>0.0) rhs.front()+=flux.front()*Z_inner;
    if (flux.back()<0.0) rhs.back()-=flux.back()*p.Z_outer_boundary;
    for (std::size_t i=0; i<cells; ++i) {
        if (i>0) {
            const double factor=lower[i]/diagonal[i-1];
            diagonal[i]-=factor*upper[i-1];
            rhs[i]-=factor*rhs[i-1];
        }
        if (!std::isfinite(diagonal[i]) || diagonal[i]<=0.0)
            throw std::runtime_error("Steady metallicity is unconstrained in a source-free, zero-flux cell");
    }
    for (std::size_t i=cells; i-- > 0;) {
        Z[i]=(rhs[i]-(i+1<cells ? upper[i]*Z[i+1] : 0.0))/diagonal[i];
        if (!std::isfinite(Z[i]) || Z[i]<0.0 || Z[i]>1.0)
            throw std::runtime_error("Steady metal conservation produced an unphysical metal mass fraction");
    }
    DoubleVec metal_flux(cells+1);
    metal_flux.front()=flux.front()*(flux.front()>0.0 ? Z_inner : Z.front());
    metal_flux.back()=flux.back()*(flux.back()<0.0 ? p.Z_outer_boundary : Z.back());
    for (std::size_t i=1; i<cells; ++i)
        metal_flux[i]=flux[i]*(flux[i]>0.0 ? Z[i-1] : Z[i]);
    double balance_error=0.0;
    for (std::size_t i=0; i<cells; ++i)
        balance_error=std::max(balance_error,std::abs(metal_flux[i+1]-metal_flux[i]
            +star[i]*Z[i]-p.Z_land*landing[i]-p.yield*star[i]));
    LinearProfile profile;
    profile.radius.push_back(R_in);
    profile.value.push_back(closed_inner || flux.front()>0.0 ? Z_inner : Z.front());
    for (std::size_t i=0; i<cells; ++i) {
        profile.radius.push_back((edges[i]+edges[i+1])/2.0);
        profile.value.push_back(Z[i]);
    }
    profile.radius.push_back(R_out);
    profile.value.push_back(flux.back()<0.0 ? p.Z_outer_boundary : Z.back());
    return {std::move(profile),flux.front(),balance_error};
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
