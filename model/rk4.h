#pragma once

#include <vector>


using DoubleVec = std::vector<double>;
using FunctionDerivative = DoubleVec (*)(double, const DoubleVec&, const void*);
struct RK4Step {DoubleVec y; DoubleVec error;};

struct RK4Solution {DoubleVec x; std::vector<DoubleVec> y; std::vector<DoubleVec> dydx;};

DoubleVec rk4_step(FunctionDerivative f, const void* parameters, const double x_n, const DoubleVec& y_n, const double h);

RK4Step rk4_step_doubling(FunctionDerivative f, const void* parameters, const double x_n, const DoubleVec& y_n, const double h);

RK4Solution integrate_rk4(FunctionDerivative f, const void* parameters, const double x_0, const DoubleVec& y_0, const double h_0, const double atol, const double rtol, const double x_stop);

DoubleVec evaluate(const RK4Solution& solution, const double x_query);
