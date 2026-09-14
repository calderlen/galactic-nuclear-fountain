#pragma once

#include "profiles.h"
#include "rk4.h"

#include <functional>

struct MassContinuityParameters {
    DoubleVec R_bins;   // annulus edges [kpc]
    DoubleVec tau_bins; // delay edges [Myr]
    DoubleVec kernel;   // build_kernel density [kpc^-1 Myr^-1]
    std::function<double(double)> Mdot_launch; // time [Myr] -> nuclear launch rate [Msun/yr]
    RotationCurve rot_curve{}; // fixed circular rotation; velocity/derivative in consistent units
    double R_nucl = 0.0; // [kpc], nuclear angular momentum is R_nucl*v_c(R_nucl)
    double mu = 0.0;     // CGM/nuclear mass ratio; total deposition is (1+mu)*kernel deposition
    double beta = 0.0;   // j_CGM/j_disk, as in j_land_mixing()
    double Z_nucl = 0.02; // nuclear and CGM metal mass fractions, as in z_land_mixing()
    double Z_CGM = 0.003;
    double yield = 0.015; // newly produced metals per unit star-formation sink mass
    DoubleVec sigmadot_star;              // [Msun/yr/kpc^2], length n
    // Signed fluxes through the two domain boundaries, positive toward increasing R.
    // Zero gives closed boundaries; negative inner flux removes gas into the nucleus.
    double inner_mass_flux = 0.0; // [Msun/yr]
    double outer_mass_flux = 0.0; // [Msun/yr]
    double Z_inner_inflow = 0.0; // incoming boundary gas metallicity; outflow uses local stage Z
    double Z_outer_inflow = 0.0;
};

// Advance one state containing n gas densities followed by n metal densities:
// state = [Sigma_g[0..n-1], Sigma_Z[0..n-1]], Sigma_Z = Sigma_g*Z [Msun/kpc^2].
// dSigma_i/dt = landing_i - star_i + (F_i - F_{i+1}) / area_i.
// dSigma_Z_i/dt = Z_land*landing_i + (yield-Z_i)*star_i + (F_Z_i-F_Z_{i+1})/area_i.
// Uses the existing rk4_step(), with shared upwind fluxes F = 2*pi*R*Sigma_upwind*v_R.
// Metal flux is F_Z = F*Z_upwind, using that same stage's gas and metal densities.
// Landing, mixing velocity, gas/metal transport and stellar terms are recomputed each stage.
// v_R comes from j_land_mixing() and the current landing source/gas density, assuming
// fixed j_disk(R), no additional torques, and star formation removing local disk j.
// Sigma_g cancels in Sigma_g*v_R; the resulting flux is source-driven, not prescribed advection.
// Rejects negative stage/final densities, Sigma_Z > Sigma_g and transport through empty donors;
// reduce dt_Myr instead of clipping mass. Time steps must resolve the launch history.
DoubleVec advance_mass_continuity(double t_Myr, double dt_Myr, const DoubleVec& state,
                                 const MassContinuityParameters& parameters);
// Total nuclear+CGM landing source [Msun/yr/kpc^2].
DoubleVec mass_continuity_landing_rate(double t_Myr, const MassContinuityParameters& parameters);
// INTERNAL edge velocities [kpc/yr], positive outward. Interpolate the total landing
// source from annulus midpoints to each edge and use the upwind stage gas density.
DoubleVec mass_continuity_velocity(const DoubleVec& sigma_g, const DoubleVec& landing,
                                  const MassContinuityParameters& parameters);
// RK4 derivative in Msun/kpc^2/Myr; parameters points to MassContinuityParameters.
DoubleVec mass_continuity_rhs(double t_Myr, const DoubleVec& state, const void* parameters);

// Reconstruct snapshot flow by integrating inward from outer_mass_flux.
// derivative has the same ordering and Myr units as mass_continuity_rhs; its gas
// time derivative is retained, so a snapshot is not assumed to be in steady state.
DoubleVec reconstruct_mass_flux(const DoubleVec& derivative, const DoubleVec& landing,
                                const MassContinuityParameters& parameters);
// The same upwind metal transport is used by evolution and reconstruction.
DoubleVec disk_metal_flux(const DoubleVec& state, const DoubleVec& mass_flux,
                         const MassContinuityParameters& parameters);
// Required incident composition from dSigma_Z/dt + div(F_Z) = Z_land*landing
// + (yield-Z)*star. Undefined where no mass lands; returns NaN there.
DoubleVec reconstruct_landing_metallicity(const DoubleVec& state, const DoubleVec& derivative,
                                         const DoubleVec& landing, const DoubleVec& metal_flux,
                                         const MassContinuityParameters& parameters);

struct InverseModelParameters {
    double R_g;
    double sigma_g0;
    double A;
    double N;
    double Mdot_land;
    double R_nucl;
    double R_out;
    double Mdot_out;
};

struct EmpiricalInverseModelParameters {
    RadialProfile sigma_g;
    RadialProfile sigmadot_star;
    RadialProfile sigmadot_land;
    double R_out;
    double Mdot_out;
};

DoubleVec inverse_model_rhs(double R,const DoubleVec& state,const void* parameters);
DoubleVec empirical_inverse_model_rhs(double R,const DoubleVec& state,const void* parameters);
RK4Solution solve_inverse_model(const InverseModelParameters& parameters,double R_stop,double h_0,double atol,double rtol);
RK4Solution solve_empirical_inverse_model(const EmpiricalInverseModelParameters& parameters,double R_stop,double h_0,double atol,double rtol);
// radial velocity of disk gas induced by mass and angular #momnetum deposition of incident fountain gas
double inverse_radial_velocity(double R,const RK4Solution& solution,const InverseModelParameters& parameters);
// Radial velocity from observed gas and SFR profiles, in kpc/yr.
double empirical_inverse_radial_velocity(double R,const RK4Solution& solution,const EmpiricalInverseModelParameters& parameters);
// Required landing angular momentum for an arbitrary rotation curve.
double inverse_landing_angular_momentum(double R,const RK4Solution& solution,const InverseModelParameters& parameters,const RotationCurve& rot_curve);
// Ratio of incident-fountain to local disk specific angular momentum.
double inverse_landing_angular_momentum_ratio(double R,const RK4Solution& solution,const InverseModelParameters& parameters,const RotationCurve& rot_curve);

struct ForwardModelParameters {
    double R_nucl;
    double R_out;
    double mu;
    double beta;
    RadialProfile sigma_g;
    RadialProfile sigma_g_derivative;
    RadialProfile sigmadot_star;
    RotationCurve rot_curve;
};

struct ForwardModelSolution {
    RK4Solution solution;
    double Mdot_land;
    double initial_radial_velocity;
    double initial_landing_rate;
    bool nonmixing;
};

// Radial-velocity ODE for the mixing forward model.
DoubleVec forward_mixing_rhs(double R,const DoubleVec& state,const void* parameters);
DoubleVec forward_nonmixing_rhs(double R,const DoubleVec& state,const void* parameters);
double sigmadot_land_mixing_from_velocity(double R,double radial_velocity,const ForwardModelParameters& parameters);
// Regular radial velocity for the no-CGM-mixing limit (mu = 0).
double radial_velocity_nonmixing(double R,double integral_state,const ForwardModelParameters& parameters);
// Regular landing profile for the no-CGM-mixing limit (mu = 0).
double sigmadot_land_nonmixing(double R,double radial_velocity,const ForwardModelParameters& parameters);
ForwardModelSolution solve_forward_mixing(const ForwardModelParameters& parameters,double target_mdot_land,double h_0,double atol,double rtol);
ForwardModelSolution solve_forward_nonmixing(const ForwardModelParameters& parameters,double h_0,double atol,double rtol);
// Solve the forward model for arbitrary gas and rotation-curve profiles.
ForwardModelSolution solve_forward_model(const ForwardModelParameters& parameters,double target_mdot_land,double h_0,double atol,double rtol);
double forward_radial_velocity(double R,const ForwardModelParameters& parameters,const ForwardModelSolution& solution);
double forward_landing_rate(double R,const ForwardModelParameters& parameters,const ForwardModelSolution& solution);
double forward_cumulative_landing(double R,const ForwardModelSolution& solution);

struct MetallicityModelParameters {
    const ForwardModelParameters* forward_parameters;
    const ForwardModelSolution* forward_solution;
    double Z_nucl;
    double Z_CGM;
    double yield;
};

struct InverseMetallicityModelParameters {
    const InverseModelParameters* inverse_parameters;
    const RK4Solution* inverse_solution;
    double Z_land;
    double yield;
};

struct MixingMetallicityModelParameters {
    RadialProfile sigma_g;
    RadialProfile radial_velocity;
    RadialProfile sigmadot_land;
    RadialProfile sigmadot_star;
    double Z_nucl;
    double Z_CGM;
    double mu;
    double yield;
};

struct EmpiricalMetallicityModelParameters {
    RadialProfile sigma_g;
    RadialProfile radial_velocity;
    RadialProfile sigmadot_land;
    RadialProfile sigmadot_star;
    double Z_land;
    double yield;
};

DoubleVec metallicity_rhs(double R,const DoubleVec& state,const void* parameters);
DoubleVec inverse_metallicity_rhs(double R,const DoubleVec& state,const void* parameters);
DoubleVec mixing_metallicity_rhs(double R,const DoubleVec& state,const void* parameters);
DoubleVec empirical_metallicity_rhs(double R,const DoubleVec& state,const void* parameters);
RK4Solution solve_metallicity(const MetallicityModelParameters& parameters,double R_start,double R_stop,double Z_initial,double h_0,double atol,double rtol);
RK4Solution solve_inverse_metallicity(const InverseMetallicityModelParameters& parameters,double R_start,double R_stop,double Z_initial,double h_0,double atol,double rtol);
RK4Solution solve_mixing_metallicity(const MixingMetallicityModelParameters& parameters,double R_start,double R_stop,double Z_initial,double h_0,double atol,double rtol);
RK4Solution solve_empirical_metallicity(const EmpiricalMetallicityModelParameters& parameters,double R_start,double R_stop,double Z_initial,double h_0,double atol,double rtol);
