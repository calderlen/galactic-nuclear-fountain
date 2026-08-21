#include <iostream>
#include <cmath>

constexpr double PI=3.14159265358979323846;

double hernquist(double G, double M, double r, double a){
    return -G*M/(r+a);
}

double miyamoto_nagai(double G, double M, double R, double a, double z, double b){
    return -G*M/std::sqrt(R*R+(a+std::sqrt(z*z+b*b))*(a+std::sqrt(z*z+b*b)));
}

double nfw(double G, double rho, double r, double r_s){
    return -4.0*PI*G*rho*r_s*r_s*std::log((1.0+r/r_s)/(r/r_s));
}