# pragma once

double phi_h(double R, double z, double M_b, double a_b);
double dphi_h_dR(double R, double z, double M_b, double a_b);
double dphi_h_dz(double R, double z, double M_b, double a_b);


double phi_nfw(double R, double z, double rho_s, double r_s);
double dphi_nfw_dR(double R, double z, double rho_s, double r_s);
double dphi_nfw_dz(double R, double z, double rho_s, double r_s);


double phi_mn(double R, double z, double M_d, double a_d, double b_d);
double dphi_mn_dR(double R, double z, double M_d, double a_d, double b_d);
double dphi_mn_dz(double R, double z, double M_d, double a_d, double b_d);

