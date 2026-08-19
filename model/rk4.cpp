#include "rk4.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

// Classical RK4 has O(h^5) local truncation error; step doubling estimates that error as |y_h2-y|/15.

DoubleVec rk4_step(FunctionDerivative f,const void* parameters,const double x_n,const DoubleVec& y_n,const double h){

    DoubleVec dydx_0 = f(x_n,y_n,parameters);

    DoubleVec k1(y_n.size());
    for (std::size_t i=0; i<y_n.size(); ++i){
        k1[i] = h*dydx_0[i];
    }

    DoubleVec y_temp(y_n.size());
    for (std::size_t i=0; i<y_n.size(); ++i){
        y_temp[i] = y_n[i]+k1[i]/2.0;
    }

    DoubleVec dydx_1 = f(x_n+h/2.0, y_temp, parameters);

    DoubleVec k2(y_n.size());
    for (std::size_t i=0; i<y_n.size(); ++i){
        k2[i] = h*dydx_1[i];
    }

    for (std::size_t i=0; i<y_n.size(); ++i){
        y_temp[i] = y_n[i]+k2[i]/2.0;
    }

    DoubleVec dydx_2 = f(x_n+h/2.0, y_temp, parameters);

    DoubleVec k3(y_n.size());
    for (std::size_t i=0; i<y_n.size(); ++i){
        k3[i] = h*dydx_2[i];
    }

    for (std::size_t i=0; i<y_n.size(); ++i){
        y_temp[i] = y_n[i]+k3[i];
    }

    DoubleVec dydx_3 = f(x_n+h, y_temp, parameters);

    DoubleVec k4(y_n.size());
    for (std::size_t i=0; i<y_n.size(); ++i){
        k4[i] = h * dydx_3[i];
    }

    DoubleVec y_nplus(y_n.size());
    for (std::size_t i=0; i<y_n.size(); ++i){
        y_nplus[i] = y_n[i] + 1.0/6.0*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i]);
    }

    return y_nplus;

}


RK4Step rk4_step_doubling(FunctionDerivative f, const void* parameters, const double x_n, const DoubleVec& y_n, const double h){

    DoubleVec y = rk4_step(f, parameters, x_n, y_n, h);
    DoubleVec y_h1 = rk4_step(f, parameters, x_n, y_n, h/2.0);
    DoubleVec y_h2 = rk4_step(f,parameters,x_n+h/2.0, y_h1, h/2.0);
    DoubleVec error(y_n.size());

    for (std::size_t i=0; i<y_n.size(); ++i){
        error[i] = std::abs(y_h2[i]-y[i])/15.0;
    }

    return {y_h2, error};
}


RK4Solution integrate_rk4(FunctionDerivative f, const void* parameters, const double x_0, const DoubleVec& y_0, const double h_0, const double atol, const double rtol, const double x_stop){

double x=x_0;
DoubleVec y=y_0;

const double direction = x_stop>x_0 ? 1.0 : -1.0;

double h=direction*std::abs(h_0);

const int max_steps = 100000;

RK4Solution solution;
solution.x.push_back(x);
solution.y.push_back(y);
solution.dydx.push_back(f(x,y,parameters));

for (int attempts=0; attempts<max_steps && direction*(x_stop-x)>0.0; ++attempts){
    bool endpoint_step=false;
    if (direction*(x+h-x_stop)>0.0)
    {
        h=x_stop-x;
        endpoint_step=true;
    }

    RK4Step step = rk4_step_doubling(f, parameters, x, y, h);

    if (!std::all_of(step.y.begin(), step.y.end(), [](double value) {return std::isfinite(value);}) ||
        !std::all_of(step.error.begin(), step.error.end(), [](double value) {return std::isfinite(value);})) {
        throw std::runtime_error("The RK4 step returned a nonfinite value");
    }

    double sum_squared = 0.0;


    for (std::size_t i=0; i<y.size(); ++i){
        const double scale = atol + rtol * std::max(std::abs(y[i]), std::abs(step.y[i]));
        const double error_ratio = step.error[i]/scale;

        sum_squared += error_ratio*error_ratio;

    }

    double error = std::sqrt(sum_squared/y.size());

    bool accept = error <= 1.0;

    if (accept) {
        x+=h;
        if (endpoint_step) {
            x=x_stop;
        }
        y = step.y;

        solution.x.push_back(x);
        solution.y.push_back(y);
        solution.dydx.push_back(f(x,y,parameters));
    }

    double factor;

    if (error==0.0) {
        factor=5.0;
    } else {
        factor=0.9*std::pow(error, -1.0/5.0);
        factor=std::clamp(factor, 0.2, 5.0);
    }

    h*=factor;

    if (!std::isfinite(h) || (direction*(x_stop-x)>0.0 && x+h==x)) {
        throw std::runtime_error("The RK4 step size underflowed");
    }
}

if (direction*(x_stop-x)>0.0) {
    throw std::runtime_error("RK4 integration exceeded max_steps");
}

return solution;
}


DoubleVec evaluate(const RK4Solution& solution, const double x_query){

    const double boundary_tolerance=16.0*std::numeric_limits<double>::epsilon()*std::max({std::abs(solution.x.front()),std::abs(solution.x.back()),1.0});

    if (std::abs(x_query-solution.x.front())<=boundary_tolerance) {
        return solution.y.front();
    }

    if (std::abs(x_query-solution.x.back())<=boundary_tolerance) {
        return solution.y.back();
    }

    std::size_t interval=0;
    bool found=false;

    for (std::size_t i=0; i+1<solution.x.size(); ++i) {
        const bool inside =
            (solution.x[i]<=x_query && x_query<=solution.x[i+1]) ||
            (solution.x[i]>=x_query && x_query>=solution.x[i+1]);

        if (inside) {
            interval=i;
            found=true;
            break;
        }
    }

    if (!found) {
        throw std::out_of_range("Interpolation point is outside the RK4 solution");
    }

    const double x_left=solution.x[interval];
    const double x_right=solution.x[interval+1];
    const double dx=x_right-x_left;
    const double t=(x_query-x_left)/dx;

    const double h00=2*t*t*t-3*t*t+1;
    const double h10=t*t*t-2*t*t+t;
    const double h01=-2*t*t*t+3*t*t;
    const double h11=t*t*t-t*t;

    DoubleVec result(solution.y[interval].size());

    for (std::size_t component=0; component<result.size(); ++component) {
        result[component] =
            h00*solution.y[interval][component] +
            h10*dx*solution.dydx[interval][component] +
            h01*solution.y[interval+1][component] +
            h11*dx*solution.dydx[interval+1][component];
    }

    return result;
}
