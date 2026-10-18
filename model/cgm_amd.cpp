#include "cgm_amd.h"
#include "potential.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
constexpr double PI = 3.14159265358979323846;
constexpr double G = 4.4985e-12;   // kpc^3 / (Msun Myr^2), same as potential.cpp
constexpr double KB = 1.380649e-16; // erg/K
constexpr double MP = 1.67262192369e-24; // g
constexpr double CM_PER_KPC = 3.085677581491367e21;
constexpr double S_PER_MYR = 3.15576e13;

struct Context {
    const CGMAMDParameters* p;
    double cs2;
    double jmax;
    double xi;
    double core_factor;
};

double potential(double R, double z, const OrbitParameters& p) {
    return phi_mn(R,z,p.M_d,p.a_d,p.b_d)
         + phi_h(R,z,p.M_b,p.a_b)
         + phi_nfw(R,z,p.rho_s,p.r_s);
}

double chi(double R, const Context& c) {
    const auto& p = *c.p;
    // Finite spherical virial-volume cutoff. This is a physical truncation
    // choice; the formal paper expression writes z_max = infinity.
    const double zmax = std::sqrt(std::max(0.0, p.R200*p.R200-R*R));
    if (zmax == 0.0) return 0.0;
    const int n = p.vertical_intervals;
    // Resolve the thin disk and the long halo column on the same grid.
    const double scale=std::min(p.Rcore,p.potential.b_d);
    const double dt=std::asinh(zmax/scale)/n;
    const double phi0 = potential(R,0.0,p.potential);
    double sum = 0.0;
    for (int i = 0; i <= n; ++i) {
        const double z = scale*std::sinh(i*dt);
        const double weight = (i == 0 || i == n) ? 1.0 : (i % 2 == 0 ? 2.0 : 4.0);
        sum += weight * scale*std::cosh(i*dt) * std::exp(-(potential(R,z,p.potential)-phi0)/c.cs2);
    }
    return (sum*dt/3.0) / R;
}

// Caller supplies kappa > 2 for a decreasing exponential.
// Mean of a truncated exponential: jmean/jmax=1/xi-1/(exp(xi)-1).
double find_xi(double kappa) {
    double left = 0.0;
    double right = std::max(128.0,2.0*kappa);
    for (int iter = 0; iter < 100; ++iter) {
        const double m = 0.5*(left+right);
        const double mean_fraction = m<1.e-3 ? 0.5-m/12.0+m*m*m/720.0 :
            1.0/m-1.0/std::expm1(m);
        if (mean_fraction > 1.0/kappa) left=m;
        else right=m;
    }
    return 0.5*(left+right);
}

double j_of_C(double C, const Context& c) {
    // Retain the finite truncation endpoint even when exp(-xi) is too small
    // to survive subtraction from one in double precision.
    if (C==1.0) return c.jmax;
    if (c.xi>1.0 && C>=0.0 && C<1.0)
        return -c.jmax/c.xi*std::log((1.0-C)+C*std::exp(-c.xi));
    const double b = -std::expm1(-c.xi);
    const double argument = -C*b;
    if (!(argument > -1.0))
        throw std::domain_error("Shooting trial left the exponential-AMD domain");
    return -c.jmax/c.xi*std::log1p(argument);
}

// Independent coordinate: x = ln(R/R200).
// y = [C, h], h=ln(rho0 R200^3/M_CGM).
DoubleVec cgm_rhs(double x, const DoubleVec& y, const void* data) {
    const auto& c = *static_cast<const Context*>(data);
    const double R = c.p->R200*std::exp(x);
    const double C=y[0], h=y[1];
    const double j=j_of_C(C,c);
    const double vc=circular_velocity(R,c.p->potential);
    const double dCdx = 4*PI * chi(R,c) * std::pow(R/c.p->R200,3) * std::exp(h);
    const double dhdx = ((j/R)*(j/R)-vc*vc)/c.cs2;
    return {dCdx,dhdx};
}

double calculate_core_factor(const Context& c) {
    // Density in R<Rcore is approximated as rho0(Rcore), as in the paper.
    // Midpoint integration avoids the R=0 coordinate singularity.
    constexpr int n = 512;
    const double dr = c.p->Rcore/n;
    double integral = 0.0;
    for (int i=0; i<n; ++i) {
        const double R=(i+0.5)*dr;
        integral += chi(R,c)*R*R*dr;
    }
    return 4.0*PI*integral/std::pow(c.p->R200,3);
}

struct Trial {
    double residual;
    RK4Solution trajectory;
};

Trial run_shot(const Context& c, double h200,
               double step, double atol, double rtol) {
    // Outer condition C(R200)=1 is imposed exactly; shoot inward.
    auto result = integrate_rk4(cgm_rhs, &c, 0.0, {1.0,h200},
                                step,atol,rtol,std::log(c.p->Rcore/c.p->R200),step);
    const auto& last=result.y.back();
    const double inner_core_mass = c.core_factor * std::exp(last[1]);
    const double residual=last[0]-inner_core_mass;
    if (!std::isfinite(residual)) throw std::runtime_error("Nonfinite CGM shooting residual");
    return {residual, std::move(result)};
}

} // namespace

CGMAMDSolution solve_cgm_amd(const CGMAMDParameters& p,
                              double h_low, double h_high,
                              double step_logR, double atol, double rtol) {
    const double cs_cms = std::sqrt(KB*p.T_K/(p.mu_mol*MP));
    const double cs = cs_cms*S_PER_MYR/CM_PER_KPC;
    const double j_dm = p.lambda_spin*std::sqrt(2.0*G*p.M200*p.R200);
    Context c{&p,cs*cs,p.jmax_over_jDM*j_dm,
              find_xi(p.jmax_over_jDM/p.fj),0.0};
    c.core_factor=calculate_core_factor(c);
    auto left=run_shot(c,h_low,step_logR,atol,rtol);
    auto right=run_shot(c,h_high,step_logR,atol,rtol);
    if ((left.residual>0.0 && right.residual>0.0) ||
        (left.residual<0.0 && right.residual<0.0))
        throw std::invalid_argument("CGM shooting density guesses do not bracket the inner mass condition");

    for (int iter=0; iter<72; ++iter) {
        const double middle=0.5*(h_low+h_high);
        auto shot=run_shot(c,middle,step_logR,atol,rtol);
        // RK4 checks finiteness; C decreases inward from one. Checking the
        // core endpoint is sufficient to keep the entire profile nonnegative.
        if (std::abs(shot.residual)<1.e-9 && shot.trajectory.y.back()[0]>=0.0) {
            std::reverse(shot.trajectory.x.begin(),shot.trajectory.x.end());
            std::reverse(shot.trajectory.y.begin(),shot.trajectory.y.end());
            std::reverse(shot.trajectory.dydx.begin(),shot.trajectory.dydx.end());
            return {p,std::move(shot.trajectory),c.jmax,c.xi,c.cs2,c.core_factor};
        }
        if ((left.residual>0.0)==(shot.residual>0.0)) {
            h_low=middle;
            left=std::move(shot);
        } else {
            h_high=middle;
            right=std::move(shot);
        }
    }
    throw std::runtime_error("CGM shooting did not converge");
}

namespace {
std::array<double,2> state_at(const CGMAMDSolution& s,double R) {
    // Lookups assume finite coordinates in the modeled domain.
    const double x=std::log(std::max(R,s.parameters.Rcore)/s.parameters.R200);
    const auto& t=s.trajectory;
    if (x<=t.x.front()) return {t.y.front()[0],t.y.front()[1]};
    if (x>=t.x.back()) return {t.y.back()[0],t.y.back()[1]};
    // Thousands/millions of orbit stages reuse this solution. Avoid a linear
    // scan through the entire trajectory for every local CGM lookup.
    const auto index=std::upper_bound(t.x.begin(),t.x.end(),x)-t.x.begin()-1;
    const double dx=t.x[index+1]-t.x[index],u=(x-t.x[index])/dx;
    std::array<double,2> y;
    for (int k=0;k<2;++k)
        y[k]=(2*u*u*u-3*u*u+1)*t.y[index][k]+(u*u*u-2*u*u+u)*dx*t.dydx[index][k]
            +(-2*u*u*u+3*u*u)*t.y[index+1][k]+(u*u*u-u*u)*dx*t.dydx[index+1][k];
    return y;
}
}

double cgm_angular_momentum(const CGMAMDSolution& s, double R) {
    const double C=state_at(s,R)[0];
    Context c{&s.parameters,s.cs2,s.jmax,s.xi,s.core_factor};
    const double j=j_of_C(C,c);
    // Frozen-density unresolved core: regular solid-body rotation, continuous
    // at Rcore. The AMD/equilibrium ODEs are enforced only outside the core.
    return R<s.parameters.Rcore ? j*std::pow(R/s.parameters.Rcore,2) : j;
}

double cgm_angular_momentum_derivative(const CGMAMDSolution& s,double R) {
    if (R==0.0 || R==s.parameters.R200) return 0.0;
    if (R<s.parameters.Rcore) return 2.0*cgm_angular_momentum(s,R)/R;
    const auto y=state_at(s,R);
    const Context c{&s.parameters,s.cs2,s.jmax,s.xi,s.core_factor};
    const double b=-std::expm1(-s.xi);
    return s.jmax*b/(s.xi*(1.0-b*y[0]))*
        cgm_rhs(std::log(R/s.parameters.R200),{y[0],y[1]},&c)[0]/R;
}

double cgm_azimuthal_velocity(const CGMAMDSolution& s, double R) {
    return R==0.0 ? 0.0 : cgm_angular_momentum(s,R)/R;
}

double cgm_beta(const CGMAMDSolution& s, double R) {
    if (R==0.0) return 0.0;
    const double vc=circular_velocity(R,s.parameters.potential);
    return vc>0.0 ? cgm_azimuthal_velocity(s,R)/vc : std::numeric_limits<double>::quiet_NaN();
}

double cgm_rho_over_mass(const CGMAMDSolution& s, double R, double z) {
    // Roundoff in a sampled virial-sphere endpoint must not erase its Simpson
    // endpoint weight. The physical boundary is unchanged to float precision.
    const double cutoff=s.parameters.R200;
    if (std::hypot(R,z)>cutoff*(1.0+16.0*std::numeric_limits<double>::epsilon())) return 0.0;
    // Evaluate the axis as its limiting potential, avoiding NFW's 0/0.
    R=std::max(R,1.e-12*s.parameters.Rcore);
    const auto state=state_at(s,R);
    const double phi0=potential(R,0.0,s.parameters.potential);
    const double phiz=potential(R,z,s.parameters.potential);
    return std::exp(state[1]-(phiz-phi0)/s.cs2) / std::pow(s.parameters.R200,3);
}

double cgm_density(const CGMAMDSolution& s,double R,double z) {
    return s.parameters.M_CGM*cgm_rho_over_mass(s,R,z);
}

double cgm_cumulative_fraction(const CGMAMDSolution& s,double R) {
    return state_at(s,R)[0];
}
