#include "physics.h"
#include <cmath>

constexpr double pi=3.14159265358979323846;

// exponential gas surface density profile
double sigma_g(double R, double R_g, double sigma_g0){
    return sigma_g0*std::exp(-R/R_g);
}

// normalization of gas surface density profile (!) this implies that M_g is the gas mass betweeen R_nucl and R_out, not the total gas mass of the disk
double sigma_g0(double M_g, double R_g, double R_nucl, double R_out){
    double F_nucl=(1.0+R_nucl/R_g)*std::exp(-R_nucl/R_g);
    double F_out=(1.0+R_out/R_g)*std::exp(-R_out/R_g);
    return M_g/(2.0*pi*R_g*R_g*(F_nucl-F_out));
}

// Kennicutt-Schmidt SFR surface density
double sigma_star_ks(double R, double R_g, double sigma_g0, double A, double N){
    return A*std::pow(sigma_g(R,R_g,sigma_g0),N);
}

double j_disk(double R,double v_c){
    return R*v_c;
}

double dj_disk_dR(double R,double v_c,double dv_c_dR){
    return v_c+R*dv_c_dR;
}

double j_cgm(double R,double beta,double v_c){
    return beta*R*v_c;
}

// specific angular momentum of incident fountain gas after mixing with CGM
double j_land_mixing(double R,double R_nucl,double mu,double beta,double v_c,double v_c_nucl){
    return (R_nucl*v_c_nucl+mu*beta*R*v_c)/(1.0+mu);
}

double dj_land_dR_mixing(double R,double mu,double beta,double v_c,double dv_c_dR){
    return mu*beta*(v_c+R*dv_c_dR)/(1.0+mu);
}

double angular_momentum_gap(double j_land_value,double j_disk_value){
    return j_land_value-j_disk_value;
}

double angular_momentum_ratio(double j_land_value,double j_disk_value){
    return j_land_value/j_disk_value;
}

// Required landing angular momentum for a general rotation curve.
double j_land_required(double R, double sigma_g, double v_R, double sigmadot_land, double v_c, double dv_c_dR){
    double j_disk=R*v_c;
    double dj_disk_dR=v_c+R*dv_c_dR;
    return j_disk+sigma_g*v_R/sigmadot_land*dj_disk_dR;
}

double mu_from_mixing(double required,double nuclear,double cgm){
    return (required-nuclear)/(cgm-required);
}

// mass flux of nuclear-origin gas after mixing with CGM
double mdot_land_mixing(double mdot_nucl, double mu){
    return mdot_nucl*(1.0+mu);
}

// mass deposition rate per unit area of incident fountain gas
double sigmadot_land(double R, double Mdot_land, double R_nucl, double R_out){
    return Mdot_land/(2.0*pi*R*R*std::log(R_out/R_nucl));
}

// Expanding d(R * Sigma_g * v_R)/dR =
// R * (Sigmadot_land - Sigmadot_star) gives equation (36) of the paper for
// an exponential disk, and the expression below for any Sigma_g(R).
double radial_velocity_gradient(double radial_velocity,double R,double sigma_g_value,double dsigma_g_dR,double sigmadot_star_value,double j_disk_value,double j_land_value,double dj_disk_dR_value){
    return radial_velocity*(
        dj_disk_dR_value/(j_land_value-j_disk_value)-
        1.0/R-
        dsigma_g_dR/sigma_g_value
    )-sigmadot_star_value/sigma_g_value;
}

double sigmadot_land_from_velocity(double sigma_g_value,double radial_velocity,double j_disk_value,double j_land_value,double dj_disk_dR_value){
    // Equation (37): Sigmadot_land = Sigma_g * v_R * j_disk' /
    // (j_land - j_disk).
    return sigma_g_value*radial_velocity*dj_disk_dR_value/
        (j_land_value-j_disk_value);
}

double mdot_acc_from_velocity(double R,double sigma_g_value,double radial_velocity){
    return -2.0*pi*R*sigma_g_value*radial_velocity;
}

double z_land_mixing(double Z_nucl, double Z_CGM, double mu){
    return (Z_nucl+Z_CGM*mu)/(1.0+mu);
}

// Metallicity gradient for empirical disk, star-formation, and flow profiles.
double metallicity_gradient(double Z, double sigma_g, double radial_velocity, double sigmadot_land, double sigmadot_star, double Z_land, double y){
    return (sigmadot_land*(Z_land-Z)+y*sigmadot_star)/(sigma_g*radial_velocity);
}

double z_land_required(double Z, double dZ_dR, double sigma_g, double v_R, double sigmadot_land, double sigmadot_star, double y){
    return Z+(sigma_g*v_R*dZ_dR-y*sigmadot_star)/sigmadot_land;
}
