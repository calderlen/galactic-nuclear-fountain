#include <iostream>
#include <vector>


using DoubleVec = std::vector<double>;
using FunctionDerivative = DoubleVec (*)(double, const DoubleVec&);



// note: for now i think you will need to manually supply the function and function derivatives and the IVP for the function


// 4th-order vector Runge-Kutta method -> O(h^5) error

DoubleVec rk4v(FunctionDerivative f,
              const double x_n,     // single initial x value
              const DoubleVec& y_n, // array of initial y values
              const double h)      // step size
{

    DoubleVec dydx_0 = f(x_n,y_n);

    DoubleVec k1(y_n.size());
    for (int i=0; i<y_n.size(); ++i){
        k1[i] = h*dydx_0[i];
    }

    DoubleVec y_temp(y_n.size());
    for (int i=0; i<y_n.size(); ++i){
        y_temp[i] = y_n[i]+k1[i]/2.0;
    }

    DoubleVec dydx_1 = f(x_n+h/2.0, y_temp);

    DoubleVec k2(y_n.size());
    for (int i=0; i<y_n.size(); ++i){
        k2[i] = h*dydx_1[i];
    }

    for (int i=0; i<y_n.size(); ++i){
        y_temp[i] = y_n[i]+k2[i]/2.0;
    }

    DoubleVec dydx_2 = f(x_n+h/2.0, y_temp);

    DoubleVec k3(y_n.size());
    for (int i=0; i<y_n.size(); ++i){
        k3[i] = h*dydx_2[i];
    }

    for (int i=0; i<y_n.size(); ++i){
        y_temp[i] = y_n[i]+k3[i];
    }

    DoubleVec dydx_3 = f(x_n+h, y_temp);

    DoubleVec k4(y_n.size());
    for (int i=0; i<y_n.size(); ++i){
        k4[i] = h * dydx_3[i];
    }

    DoubleVec y_nplus(y_n.size());
    for (int i=0; i<y_n.size(); ++i){
        y_nplus[i] = y_n[i] + 1.0/6.0*(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i]);
    }

    return y_nplus;

}

int main() {}
