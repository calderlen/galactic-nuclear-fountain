#pragma once

double sigma_g(double R, double R_g, double sigma_g0);
double sigma_g0(double M_g, double R_g, double R_nucl, double R_out);
double sigma_star_ks(double R, double R_g, double sigma_g0, double A, double N);

double j_disk(double R,double v_c);
double dj_disk_dR(double R,double v_c,double dv_c_dR);
double j_cgm(double R,double beta,double v_c);
double j_land_mixing(double R,double R_nucl,double mu,double beta,double v_c,double v_c_nucl);
double dj_land_dR_mixing(double R,double mu,double beta,double v_c,double dv_c_dR);
double angular_momentum_gap(double j_land,double j_disk);
double angular_momentum_ratio(double j_land,double j_disk);
double j_land_required(double R, double sigma_g, double v_R, double sigmadot_land, double v_c, double dv_c_dR);
double mu_from_mixing(double required,double nuclear,double cgm);

double mdot_land_mixing(double mdot_nucl, double mu);
double sigmadot_land(double R, double Mdot_land, double R_nucl, double R_out);
double radial_velocity_gradient(double radial_velocity,double R,double sigma_g,double dsigma_g_dR,double sigmadot_star,double j_disk,double j_land,double dj_disk_dR);
double sigmadot_land_from_velocity(double sigma_g,double radial_velocity,double j_disk,double j_land,double dj_disk_dR);
double mdot_acc_from_velocity(double R,double sigma_g,double radial_velocity);

double z_land_mixing(double Z_nucl, double Z_CGM, double mu);
double metallicity_gradient(double Z, double sigma_g, double radial_velocity, double sigmadot_land, double sigmadot_star, double Z_land, double y);
double z_land_required(double Z, double dZ_dR, double sigma_g, double v_R, double sigmadot_land, double sigmadot_star, double y);
