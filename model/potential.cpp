#include <cmath>

constexpr double pi=3.14159265358979323846;
constexpr double G=4.4985e-12; // [kpc^3 / (M_sun * Myr^2)]

// Hernquist bulge

double phi_h(double R, double z, double M_b, double a_b){
    double r = std::hypot(R, z);
    return -G*M_b/(r+a_b);
}

double dphi_h_dR(double R, double z, double M_b, double a_b){
    double r = std::hypot(R, z);
    return G*M_b*R/(r*std::pow(r+a_b,2));
}

double dphi_h_dz(double R, double z, double M_b, double a_b){
    double r = std::hypot(R, z);
    return G*M_b*z/(r*std::pow(r+a_b,2));
}

// NFW halo

double phi_nfw(double R, double z, double rho_s, double r_s) {
    double r = std::hypot(R, z);
    return -4.0*pi*G*rho_s*r_s*r_s*std::log(1.0+r/r_s)/(r/r_s);
}


double dphi_nfw_dR(double R, double z, double rho_s, double r_s) {
    double r = std::hypot(R, z);
    return 4.0*pi*G*rho_s*std::pow(r_s,3)*R/std::pow(r,3)*(std::log(1.0+r/r_s)-r/(r+r_s));
}


double dphi_nfw_dz(double R, double z, double rho_s, double r_s) {
    double r = std::hypot(R, z);
    return 4.0*pi*G*rho_s*std::pow(r_s,3)*z/std::pow(r,3)*(std::log(1.0+r/r_s) -r/(r+r_s));
}


// Miyamoto-Nagai disk


double phi_mn(double R, double z, double M_d, double a_d, double b_d) {
    double r = std::hypot(R, z);
    return -G*M_d/std::sqrt(R*R + std::pow(a_d + std::sqrt(z*z + b_d*b_d),2));
}

double dphi_mn_dR(double R, double z, double M_d, double a_d, double b_d) {
    return G*M_d*R/std::pow(R*R + std::pow(a_d + std::sqrt(z*z + b_d*b_d),2),1.5);
}

double dphi_mn_dz(double R, double z, double M_d, double a_d, double b_d) {
    return G*M_d*(a_d + std::sqrt(z*z + b_d*b_d))*z/(std::sqrt(z*z + b_d*b_d)*std::pow(R*R + std::pow(a_d + std::sqrt(z*z + b_d*b_d),2),1.5));
}


