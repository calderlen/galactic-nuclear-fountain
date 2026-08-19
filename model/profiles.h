#pragma once

#include <vector>

using RadialFunction = double (*)(double, const void*);

struct RadialProfile {
    RadialFunction value;
    const void* parameters;
};

struct RotationCurve {
    RadialProfile velocity;
    RadialProfile derivative;
};

double profile_value(const RadialProfile& profile, double R);
double rotation_velocity(const RotationCurve& rot_curve, double R);
double rotation_velocity_derivative(const RotationCurve& rot_curve, double R);

struct FlatRotationParameters {
    double velocity;
};

struct RisingRotationParameters {
    double Vflat;
    double lflat;
};

struct LinearProfile {
    std::vector<double> radius;
    std::vector<double> value;
};

double flat_rotation_velocity(double R, const void* parameters);
double flat_rotation_velocity_derivative(double R, const void* parameters);
double rising_rotation_velocity(double R,const void* parameters);
double rising_rotation_velocity_derivative(double R,const void* parameters);
double linear_profile_value(double R,const void* parameters);
double linear_profile_derivative(double R,const void* parameters);
