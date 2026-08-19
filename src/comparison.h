#pragma once

#include <optional>
#include <string>
#include <vector>

namespace galacticwind {

constexpr double OH12_sun=8.69;
constexpr double Z_sun=0.0139;
constexpr double abundance_scatter_dex=0.05;

enum class Calibration {KK04,PT05};

struct AbundanceMeasurement {
    bool usable;
    std::optional<double> OH12;
    std::optional<double> e_OH12;
    std::optional<double> Z;
    std::optional<double> e_Z_lo;
    std::optional<double> e_Z_hi;
};

struct CatalogRow {
    std::string galaxy;
    int source_sequence;
    std::string hii_region;
    std::optional<double> offset_ra_arcsec;
    std::optional<double> offset_de_arcsec;
    std::optional<double> R_R25;
    std::optional<double> R_kpc;
    std::optional<double> R_Reff;
    std::optional<double> Reff_kpc;
    AbundanceMeasurement KK04;
    AbundanceMeasurement PT05;
};

struct Observation {
    int source_sequence;
    std::string hii_region;
    std::optional<double> offset_ra_arcsec;
    std::optional<double> offset_de_arcsec;
    double R_R25;
    double R_kpc;
    double R_Reff;
    double OH12;
    double e_OH12;
    double Z;
    double e_Z_lo;
    double e_Z_hi;
};

struct ModelCurve {
    std::string family;
    std::string id;
    std::string label;
    std::string role;
    std::vector<double> radius_kpc;
    std::vector<double> metallicity;
};

struct ComparisonMetadata {
    double Reff_kpc;
    double Z_outer_boundary;
    double R_Z_boundary_kpc;
    bool radial_H2_available;
    std::string H2_treatment;
};

struct ResidualPoint {
    Observation observation;
    double model_log_Z_Zsun;
    double residual_dex;
    double sigma_dex;
};

struct LineFit {
    std::optional<double> intercept;
    std::optional<double> slope;
    std::optional<double> e_intercept;
    std::optional<double> e_slope;
};

struct FitStatistics {
    std::string galaxy;
    std::string calibration;
    std::string model_family;
    std::string model_curve;
    int n_catalog_usable;
    int n_unique_catalog_regions;
    int n_model_overlap;
    int n_unique_model_overlap;
    std::optional<double> R_min_kpc;
    std::optional<double> R_max_kpc;
    std::optional<double> R_min_Reff;
    std::optional<double> R_max_Reff;
    std::optional<double> R_R25_span;
    bool gradient_sample_reliable;
    std::string fit_status;
    std::optional<double> observed_slope_dex_per_kpc;
    std::optional<double> e_observed_slope_dex_per_kpc;
    std::optional<double> observed_slope_dex_per_Reff;
    std::optional<double> e_observed_slope_dex_per_Reff;
    std::optional<double> model_slope_dex_per_kpc;
    std::optional<double> model_slope_dex_per_Reff;
    std::optional<double> residual_offset_at_pivot_dex;
    std::optional<double> e_residual_offset_at_pivot_dex;
    std::optional<double> residual_slope_dex_per_Reff;
    std::optional<double> e_residual_slope_dex_per_Reff;
    std::optional<double> pivot_Reff;
    std::optional<double> weighted_mean_residual_dex;
    std::optional<double> e_weighted_mean_residual_dex;
    std::optional<double> rms_residual_dex;
    std::optional<double> shape_rms_after_offset_dex;
    std::optional<double> chi2_fixed_model;
    std::optional<double> reduced_chi2_fixed_model;
    std::optional<double> chi2_after_offset;
    std::optional<double> reduced_chi2_after_offset;
    double abundance_scatter_added_dex;
    double Reff_adopted_kpc;
    double Z_outer_boundary;
    double R_Z_boundary_kpc;
    bool radial_H2_available;
    std::string H2_treatment;
};

std::string calibration_name(Calibration calibration);
std::vector<Observation> select_observations(
    const std::vector<CatalogRow>& catalog,
    const std::string& galaxy,
    Calibration calibration
);
std::vector<Observation> collapse_repeated_regions(const std::vector<Observation>& observations);
std::vector<ResidualPoint> overlap_residuals(
    const std::vector<Observation>& observations,
    const ModelCurve& curve
);
LineFit weighted_line_fit(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& sigma
);
FitStatistics comparison_statistics(
    const std::string& galaxy,
    Calibration calibration,
    const std::vector<Observation>& observations,
    const ModelCurve& curve,
    const ComparisonMetadata& metadata
);
FitStatistics unavailable_statistics(
    const std::string& galaxy,
    Calibration calibration,
    const std::vector<Observation>& observations,
    const ComparisonMetadata& metadata
);

}
