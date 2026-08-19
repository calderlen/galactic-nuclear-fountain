#pragma once

#include "profiles.h"
#include "rk4.h"

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
