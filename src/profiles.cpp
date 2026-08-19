#include "profiles.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

double profile_value(const RadialProfile& profile, double R){
    return profile.value(R,profile.parameters);
}

double rotation_velocity(const RotationCurve& rot_curve, double R){
    return profile_value(rot_curve.velocity,R);
}

double rotation_velocity_derivative(const RotationCurve& rot_curve, double R){
    return profile_value(rot_curve.derivative,R);
}

double flat_rotation_velocity(double R, const void* parameters){
    (void)R;
    return static_cast<const FlatRotationParameters*>(parameters)->velocity;
}

double flat_rotation_velocity_derivative(double R, const void* parameters){
    (void)R;
    (void)parameters;
    return 0.0;
}

double rising_rotation_velocity(double R,const void* raw_parameters){
    const RisingRotationParameters& parameters=*static_cast<const RisingRotationParameters*>(raw_parameters);
    return parameters.Vflat*(-std::expm1(-R/parameters.lflat));
}

double rising_rotation_velocity_derivative(double R,const void* raw_parameters){
    const RisingRotationParameters& parameters=*static_cast<const RisingRotationParameters*>(raw_parameters);
    return parameters.Vflat/parameters.lflat*std::exp(-R/parameters.lflat);
}

namespace {

std::size_t linear_profile_interval(const LinearProfile& profile,double R){
    const double tolerance=16.0*std::numeric_limits<double>::epsilon()*std::max({std::abs(profile.radius.front()),std::abs(profile.radius.back()),1.0});
    if (R<profile.radius.front()-tolerance || R>profile.radius.back()+tolerance) {
        throw std::out_of_range("Requested radius is outside the linear profile");
    }
    const double query=std::clamp(R,profile.radius.front(),profile.radius.back());
    if (query==profile.radius.back()) {
        return profile.radius.size()-2;
    }
    return static_cast<std::size_t>(std::upper_bound(profile.radius.begin(),profile.radius.end(),query)-profile.radius.begin()-1);
}

}

double linear_profile_value(double R,const void* raw_parameters){
    const LinearProfile& profile=*static_cast<const LinearProfile*>(raw_parameters);
    const std::size_t interval=linear_profile_interval(profile,R);
    const double query=std::clamp(R,profile.radius.front(),profile.radius.back());
    const double fraction=(query-profile.radius[interval])/(profile.radius[interval+1]-profile.radius[interval]);
    return profile.value[interval]+fraction*(profile.value[interval+1]-profile.value[interval]);
}

double linear_profile_derivative(double R,const void* raw_parameters){
    const LinearProfile& profile=*static_cast<const LinearProfile*>(raw_parameters);
    const std::size_t interval=linear_profile_interval(profile,R);
    return (profile.value[interval+1]-profile.value[interval])/(profile.radius[interval+1]-profile.radius[interval]);
}
