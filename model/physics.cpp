#include "physics.h"
#include "units.h"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>

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
double stellar_sink_from_gas(double gas, double A, double N){
    return A*std::pow(gas,N);
}

double sigma_star_ks(double R, double R_g, double sigma_g0, double A, double N){
    return stellar_sink_from_gas(sigma_g(R,R_g,sigma_g0),A,N);
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

double j_land_from_reservoirs(double j_nucl,double j_CGM,double mu){
    return (j_nucl+mu*j_CGM)/(1.0+mu);
}

// specific angular momentum of incident fountain gas after mixing with CGM
double j_land_mixing(double R,double R_nucl,double mu,double beta,double v_c,double v_c_nucl){
    return j_land_from_reservoirs(R_nucl*v_c_nucl,j_cgm(R,beta,v_c),mu);
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
    return j_disk+sigma_g*units::per_gyr(v_R)/sigmadot_land*dj_disk_dR;
}

double mu_from_mixing(double required,double nuclear,double cgm){
    return (required-nuclear)/(cgm-required);
}

// mass flux of nuclear-origin gas after mixing with CGM
double mdot_land_mixing(double mdot_nucl, double mu){
    return mdot_nucl*(1.0+mu);
}

double mdot_nuc_burst(double t, double Mdot_0, double Mdot_b, double t_b, double sigma_b){
    return Mdot_0 + Mdot_b*std::exp(-std::pow(t-t_b,2)/(2.0*sigma_b*sigma_b));
}

double mdot_launch(double mdot_nuc_burst, double eta){
    return mdot_nuc_burst*eta;
}

// mass deposition rate per unit area of incident fountain gas
double sigmadot_land(double R, double Mdot_land, double R_in, double R_out, double lambda){
    const double log_span=std::log(R_out/R_in);
    const double x=(2.0-lambda)*log_span;
    // expm1 keeps the general expression accurate arbitrarily close to lambda=2.
    const double correction=x==0.0 ? 1.0 : x/std::expm1(x);
    return units::per_gyr(Mdot_land)/(2.0*pi*R_in*R_in*log_span)*
        std::exp(-lambda*std::log(R/R_in))*correction;
}

double cumulative_mdot_land(double R, double Mdot_land, double R_in, double R_out, double lambda){
    const double exponent=2.0-lambda;
    const double log_radius=std::log(R/R_in);
    const double log_span=std::log(R_out/R_in);
    return Mdot_land*(exponent==0.0 ? log_radius/log_span :
        std::expm1(exponent*log_radius)/std::expm1(exponent*log_span));
}

double power_law_stellar_sink_ratio(double R, double R_nucl, double lambda,
                                   double mu, double beta){
    // D=(j_disk-j_mix)/j_disk' = aR-b. Differentiate Mdot_acc=2*pi*R*landing*D
    // in mass continuity to obtain star/landing below.
    const double a=(1.0+mu*(1.0-beta))/(1.0+mu);
    const double b=R_nucl/(1.0+mu);
    return 1.0+a*(2.0-lambda)+(lambda-1.0)*b/R;
}

namespace {

void validate_smooth_landing(double R, const SmoothBrokenPowerLawParameters& p){
    for (double value: {R,p.lambda_in,p.lambda_out,p.R_break,p.alpha}) {
        if (!std::isfinite(value))
            throw std::invalid_argument("Smooth landing parameters and radius must be finite");
    }
    if (R<=0.0 || p.R_break<=0.0 || p.alpha<=0.0)
        throw std::invalid_argument("Smooth landing radius, break radius and alpha must be positive");
}

double smooth_landing_log_shape(double log_R, const SmoothBrokenPowerLawParameters& p){
    const double log_x=log_R-std::log(p.R_break);
    const double z=p.alpha*log_x;
    // log((1+exp(z))/2)/alpha without overflow or cancellation at the break.
    // The series also preserves the broad-transition limit as alpha tends to zero.
    const double transition=std::abs(z)<1e-6 ?
        log_x*(0.5+z/8.0-z*z*z/192.0) :
        (std::abs(z)<0.5 ? std::log1p(0.5*std::expm1(z))/p.alpha :
            std::max(log_x,0.0)+(std::log1p(std::exp(-std::abs(z)))-std::log(2.0))/p.alpha);
    return -p.lambda_in*log_x-(p.lambda_out-p.lambda_in)*transition;
}

double smooth_annular_integrand(double log_R, const SmoothBrokenPowerLawParameters& p){
    // dR=R*d(log R), so the integrand is 2*pi*R^2 times the shape.
    const double value=std::exp(std::log(2.0*pi)+2.0*log_R+smooth_landing_log_shape(log_R,p));
    if (!std::isfinite(value)) throw std::overflow_error("Smooth landing integral overflowed");
    return value;
}

double smooth_adaptive_simpson(double lo, double hi, double flo, double fmid,
                              double fhi, double whole, double tolerance,
                              int remaining, const SmoothBrokenPowerLawParameters& p){
    const double mid=lo+0.5*(hi-lo);
    const double flmid=smooth_annular_integrand(lo+0.5*(mid-lo),p);
    const double frmid=smooth_annular_integrand(mid+0.5*(hi-mid),p);
    const double left=(mid-lo)/6.0*(flo+4.0*flmid+fmid);
    const double right=(hi-mid)/6.0*(fmid+4.0*frmid+fhi);
    const double change=left+right-whole;
    if (std::abs(change)<=15.0*tolerance) return left+right+change/15.0;
    if (remaining==0 || mid==lo || mid==hi)
        throw std::runtime_error("Smooth landing quadrature did not converge");
    return smooth_adaptive_simpson(lo,mid,flo,flmid,fmid,left,tolerance/2.0,remaining-1,p)+
        smooth_adaptive_simpson(mid,hi,fmid,frmid,fhi,right,tolerance/2.0,remaining-1,p);
}

double smooth_integral_segment(double lo, double hi, const SmoothBrokenPowerLawParameters& p){
    const double flo=smooth_annular_integrand(lo,p);
    const double fmid=smooth_annular_integrand(lo+0.5*(hi-lo),p);
    const double fhi=smooth_annular_integrand(hi,p);
    const double whole=(hi-lo)/6.0*(flo+4.0*fmid+fhi);
    if (!std::isfinite(whole) || whole<=0.0)
        throw std::overflow_error("Smooth landing integral is not finite and positive");
    const double tolerance=std::max(1e-11*whole,100.0*std::numeric_limits<double>::denorm_min());
    return smooth_adaptive_simpson(lo,hi,flo,fmid,fhi,whole,tolerance,24,p);
}

}

double smooth_landing_shape(double R, const SmoothBrokenPowerLawParameters& p){
    validate_smooth_landing(R,p);
    const double shape=std::exp(smooth_landing_log_shape(std::log(R),p));
    if (!std::isfinite(shape) || shape<=0.0)
        throw std::overflow_error("Smooth landing shape is not finite and positive");
    return shape;
}

double smooth_landing_log_slope(double R, const SmoothBrokenPowerLawParameters& p){
    validate_smooth_landing(R,p);
    const double z=p.alpha*(std::log(R)-std::log(p.R_break));
    const double weight=z>=0.0 ? 1.0/(1.0+std::exp(-z)) :
        std::exp(z)/(1.0+std::exp(z));
    return (1.0-weight)*p.lambda_in+weight*p.lambda_out;
}

double smooth_landing_integral(double R_in, double R_out, const SmoothBrokenPowerLawParameters& p){
    validate_smooth_landing(R_in,p);
    validate_smooth_landing(R_out,p);
    if (R_out<R_in) throw std::invalid_argument("Smooth landing integral requires R_out >= R_in");
    if (R_out==R_in) return 0.0;
    const double lo=std::log(R_in),hi=std::log(R_out),br=std::log(p.R_break);
    // Splitting at the break keeps arbitrarily sharp transitions visible to the quadrature.
    const double integral=lo<br && br<hi ?
        smooth_integral_segment(lo,br,p)+smooth_integral_segment(br,hi,p) :
        smooth_integral_segment(lo,hi,p);
    if (!std::isfinite(integral) || integral<=0.0)
        throw std::overflow_error("Smooth landing integral is not finite and positive");
    return integral;
}

double power_law_transport_length(double R, const PowerLawDiskParameters& p){
    return ((1.0+p.mu*(1.0-p.beta))*R-p.R_nucl)/(1.0+p.mu);
}

void validate_power_law_disk(const PowerLawDiskParameters& p){
    for (double value: {p.R_nucl,p.R_in,p.R_out,p.Mdot_land,p.A,p.N,p.lambda,p.mu,p.beta}) {
        if (!std::isfinite(value)) throw std::invalid_argument("Power-law parameters must be finite");
    }
    if (!(0.0<p.R_nucl && p.R_nucl<=p.R_in && p.R_in<p.R_out))
        throw std::invalid_argument("Require 0 < R_nucl <= R_in < R_out");
    if (!(p.Mdot_land>0.0 && p.A>0.0 && p.N>0.0))
        throw std::invalid_argument("Landing rate and net KS parameters must be positive");
    if (p.mu<0.0 || p.beta<0.0 || p.beta>1.0)
        throw std::invalid_argument("Require mu >= 0 and 0 <= beta <= 1");
    // This ratio is monotonic, so its endpoints determine positivity everywhere.
    if (power_law_stellar_sink_ratio(p.R_in,p.R_nucl,p.lambda,p.mu,p.beta)<=0.0 ||
        power_law_stellar_sink_ratio(p.R_out,p.R_nucl,p.lambda,p.mu,p.beta)<=0.0)
        throw std::invalid_argument("Landing slope implies a nonpositive stellar sink in the modeled annulus");
    if (power_law_transport_length(p.R_in,p)<0.0 || power_law_transport_length(p.R_out,p)<0.0)
        throw std::invalid_argument("Mixed landing angular momentum implies outward flow in the modeled annulus");
}

PowerLawDiskProfile power_law_disk_profile(double R, const PowerLawDiskParameters& p){
    if (!std::isfinite(R) || R<p.R_in || R>p.R_out)
        throw std::invalid_argument("Radius must lie between R_in and R_out");
    const double landing=sigmadot_land(R,p.Mdot_land,p.R_in,p.R_out,p.lambda);
    const double star=landing*power_law_stellar_sink_ratio(R,p.R_nucl,p.lambda,p.mu,p.beta);
    const double gas=std::pow(star/p.A,1.0/p.N);
    const double velocity=units::per_year(-landing*power_law_transport_length(R,p)/gas);
    if (!(std::isfinite(gas) && gas>0.0 && std::isfinite(velocity)))
        throw std::runtime_error("Power-law disk has nonpositive or nonfinite gas/flow");
    return {landing,star,gas,velocity,mdot_acc_from_velocity(R,gas,velocity),
        cumulative_mdot_land(R,p.Mdot_land,p.R_in,p.R_out,p.lambda)};
}

void validate_power_law_fountain(const PowerLawFountainParameters& p){
    validate_power_law_disk({p.R_nucl,p.R_in,p.R_out,p.Mdot_land,p.A,p.N,p.lambda});
    for (double value: {p.Z_land,p.yield,p.Z_out}) {
        if (!std::isfinite(value) || value<0.0)
            throw std::invalid_argument("Metallicities and yield must be finite and nonnegative");
    }
}

PowerLawFountainProfile power_law_fountain_profile(double R, const PowerLawFountainParameters& p){
    if (!std::isfinite(R) || R<p.R_in || R>p.R_out)
        throw std::invalid_argument("Radius must lie between R_in and R_out");
    const auto disk=power_law_disk_profile(R,{p.R_nucl,p.R_in,p.R_out,p.Mdot_land,p.A,p.N,p.lambda});
    const double offset=R-p.R_nucl;
    const double boundary_slope=(p.Z_out-p.Z_land-2.0*p.yield)/(p.R_out-p.R_nucl);
    const double coefficient=p.yield*(p.lambda-1.0);
    double Z=p.Z_land+2.0*p.yield+offset*boundary_slope;
    double dZ=boundary_slope;
    if (coefficient!=0.0) {
        if (offset==0.0) {
            // Z is finite at launch, but its derivative diverges except at lambda=1 or y=0.
            dZ=std::copysign(std::numeric_limits<double>::infinity(),-coefficient);
        } else {
            const double log_ratio=std::log(offset/R)-std::log1p(-p.R_nucl/p.R_out);
            Z+=coefficient*offset/p.R_nucl*log_ratio;
            dZ+=coefficient/p.R_nucl*log_ratio+coefficient/R;
        }
    }
    return {disk.landing,disk.star,disk.gas,disk.radial_velocity,disk.accretion,
        disk.cumulative_landing,Z,dZ};
}

// Expanding d(R * Sigma_g * v_R)/dR =
// R * (Sigmadot_land - Sigmadot_star) gives equation (36) of the paper for
// an exponential disk, and the expression below for any Sigma_g(R).
double radial_velocity_gradient(double radial_velocity,double R,double sigma_g_value,double dsigma_g_dR,double sigmadot_star_value,double j_disk_value,double j_land_value,double dj_disk_dR_value){
    return radial_velocity*(
        dj_disk_dR_value/(j_land_value-j_disk_value)-
        1.0/R-
        dsigma_g_dR/sigma_g_value
    )-units::per_year(sigmadot_star_value)/sigma_g_value;
}

double sigmadot_land_from_velocity(double sigma_g_value,double radial_velocity,double j_disk_value,double j_land_value,double dj_disk_dR_value){
    // Equation (37): Sigmadot_land = Sigma_g * v_R * j_disk' /
    // (j_land - j_disk).
    return sigma_g_value*units::per_gyr(radial_velocity)*dj_disk_dR_value/
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
    return (sigmadot_land*(Z_land-Z)+y*sigmadot_star)/(sigma_g*units::per_gyr(radial_velocity));
}

double z_land_required(double Z, double dZ_dR, double sigma_g, double v_R, double sigmadot_land, double sigmadot_star, double y){
    return Z+(sigma_g*units::per_gyr(v_R)*dZ_dR-y*sigmadot_star)/sigmadot_land;
}
