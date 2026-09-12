// These would store parcel results, bin them by landing radius and delay, retain nonreturning mass in the normalization, and implement build_landing_kernel() and convolve_landing_kernel() for mass, angular-momentum, and metal-mass sources.

struct LandingPoint {double R_land;
    double tau;
    double weight;
};

DoubleVec build_kernel(const std::vector<LandingPoint>& landings, const std::vector<double>& R_bins, const std::vector<double>& tau_bins
);