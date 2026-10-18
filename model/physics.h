#pragma once

// Sigma_dot and KS coefficients use Msun/Gyr/kpc^2 with gas in Msun/kpc^2.
// Integrated mass rates use Msun/yr; radial velocities use kpc/yr.

double sigma_g(double R, double R_g, double sigma_g0);
double sigma_g0(double M_g, double R_g, double R_nucl, double R_out);
double stellar_sink_from_gas(double sigma_g, double A_net, double N);
double sigma_star_ks(double R, double R_g, double sigma_g0, double A, double N);

double j_disk(double R,double v_c);
double dj_disk_dR(double R,double v_c,double dv_c_dR);
double j_cgm(double R,double beta,double v_c);
double j_land_from_reservoirs(double j_nucl,double j_CGM,double mu);
double j_land_mixing(double R,double R_nucl,double mu,double beta,double v_c,double v_c_nucl);
double dj_land_dR_mixing(double R,double mu,double beta,double v_c,double dv_c_dR);
double angular_momentum_gap(double j_land,double j_disk);
double angular_momentum_ratio(double j_land,double j_disk);
double j_land_required(double R, double sigma_g, double v_R, double sigmadot_land, double v_c, double dv_c_dR);
double mu_from_mixing(double required,double nuclear,double cgm);

double mdot_land_mixing(double mdot_nucl, double mu);
// Baseline plus Gaussian nuclear burst; use Myr for t, t_b and sigma_b when
// coupling to the landing kernel. Rates are Msun/yr; eta is dimensionless.
double mdot_nuc_burst(double t, double Mdot_0, double Mdot_b, double t_b, double sigma_b);
double mdot_launch(double mdot_nuc_burst, double eta);
// Normalized power-law deposition on R_in <= R <= R_out; lambda=2 is the
// logarithmic case. R_in is independent of the nuclear launch radius.
double sigmadot_land(double R, double Mdot_land, double R_in, double R_out, double lambda=2.0);
double cumulative_mdot_land(double R, double Mdot_land, double R_in, double R_out, double lambda=2.0);
double power_law_stellar_sink_ratio(double R, double R_nucl, double lambda,
                                   double mu=0.0, double beta=0.75);

// Dimensionless shape, equal to one at R_break, fitted to the returned-wind
// surface kernel on 1--15 kpc. Integral units are kpc^2, including annular area.
struct SmoothBrokenPowerLawParameters {
    double lambda_in=2.2763464271480927;
    double lambda_out=7.399360121082115;
    double R_break=3.705196213885244;
    double alpha=4.834302012811455;
};
double smooth_landing_shape(double R, const SmoothBrokenPowerLawParameters& parameters);
double smooth_landing_log_slope(double R, const SmoothBrokenPowerLawParameters& parameters);
double smooth_landing_integral(double R_in, double R_out, const SmoothBrokenPowerLawParameters& parameters);

// Prescribed power-law landing and j_land=(j_nucl+mu*j_CGM)/(1+mu), with
// flat rotation and j_CGM=beta*j_disk. Conservation fixes the stellar sink;
// the net KS law then fixes gas density. The outer mass flux is derived.
struct PowerLawDiskParameters {
    double R_nucl, R_in, R_out, Mdot_land;
    double A, N, lambda;
    double mu=0.0, beta=0.75;
};
struct PowerLawDiskProfile {
    double landing, star, gas, radial_velocity, accretion, cumulative_landing;
};
double power_law_transport_length(double R, const PowerLawDiskParameters& parameters);
void validate_power_law_disk(const PowerLawDiskParameters& parameters);
PowerLawDiskProfile power_law_disk_profile(double R, const PowerLawDiskParameters& parameters);

// The mu=0 limit also has closed-form metallicity.
// Radii: kpc; surface rates: Msun/Gyr/kpc^2; integrated rates: Msun/yr;
// surface densities: Msun/kpc^2; Z and y: mass fractions.
struct PowerLawFountainParameters {
    double R_nucl, R_in, R_out, Mdot_land;
    double A, N, lambda;
    double Z_land, yield, Z_out;
};
struct PowerLawFountainProfile {
    double landing, star, gas, radial_velocity, accretion, cumulative_landing;
    double metallicity, metallicity_derivative;
};
void validate_power_law_fountain(const PowerLawFountainParameters& parameters);
PowerLawFountainProfile power_law_fountain_profile(double R, const PowerLawFountainParameters& parameters);
double radial_velocity_gradient(double radial_velocity,double R,double sigma_g,double dsigma_g_dR,double sigmadot_star,double j_disk,double j_land,double dj_disk_dR);
double sigmadot_land_from_velocity(double sigma_g,double radial_velocity,double j_disk,double j_land,double dj_disk_dR);
double mdot_acc_from_velocity(double R,double sigma_g,double radial_velocity);

double z_land_mixing(double Z_nucl, double Z_CGM, double mu);
double metallicity_gradient(double Z, double sigma_g, double radial_velocity, double sigmadot_land, double sigmadot_star, double Z_land, double y);
double z_land_required(double Z, double dZ_dR, double sigma_g, double v_R, double sigmadot_land, double sigmadot_star, double y);
