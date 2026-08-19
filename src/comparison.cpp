#include "comparison.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>

namespace galacticwind {
namespace {

const AbundanceMeasurement& abundance(const CatalogRow& row,Calibration calibration){
    return calibration==Calibration::KK04 ? row.KK04 : row.PT05;
}

double weighted_mean(const std::vector<Observation>& group,double Observation::* member){
    double weighted_sum=0.0;
    double weight_sum=0.0;
    for (const Observation& observation: group) {
        const double weight=1.0/(observation.e_OH12*observation.e_OH12);
        weighted_sum+=weight*(observation.*member);
        weight_sum+=weight;
    }
    return weighted_sum/weight_sum;
}

std::vector<double> interpolate_profile(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& queries
){
    std::vector<double> values;
    values.reserve(queries.size());
    for (const double query: queries) {
        if (query==x.back()) {
            values.push_back(y.back());
            continue;
        }
        const auto right=std::upper_bound(x.begin(),x.end(),query);
        const std::size_t index=static_cast<std::size_t>(right-x.begin()-1);
        const double fraction=(query-x[index])/(x[index+1]-x[index]);
        values.push_back(y[index]+fraction*(y[index+1]-y[index]));
    }
    return values;
}

FitStatistics empty_statistics(
    const std::string& galaxy,
    Calibration calibration,
    const std::string& family,
    const std::string& curve,
    const std::vector<Observation>& observations,
    const std::vector<Observation>& unique_observations,
    const ComparisonMetadata& metadata,
    const std::string& status
){
    return {
        galaxy,calibration_name(calibration),family,curve,
        static_cast<int>(observations.size()),static_cast<int>(unique_observations.size()),0,0,
        std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt,false,status,
        std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt,
        std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt,
        std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt,std::nullopt,
        std::nullopt,std::nullopt,abundance_scatter_dex,metadata.Reff_kpc,
        metadata.Z_outer_boundary,metadata.R_Z_boundary_kpc,
        metadata.radial_H2_available,metadata.H2_treatment
    };
}

std::pair<double,double> minimum_maximum(const std::vector<double>& values){
    const auto limits=std::minmax_element(values.begin(),values.end());
    return {*limits.first,*limits.second};
}

}

std::string calibration_name(Calibration calibration){
    return calibration==Calibration::KK04 ? "KK04" : "PT05";
}

std::vector<Observation> select_observations(
    const std::vector<CatalogRow>& catalog,
    const std::string& galaxy,
    Calibration calibration
){
    std::vector<Observation> observations;
    for (const CatalogRow& row: catalog) {
        const AbundanceMeasurement& selected=abundance(row,calibration);
        if (row.galaxy!=galaxy || !selected.usable || !row.R_R25 || !row.R_kpc ||
            !row.R_Reff || !selected.OH12 || !selected.e_OH12 || !selected.Z ||
            !selected.e_Z_lo || !selected.e_Z_hi) {
            continue;
        }
        observations.push_back({
            row.source_sequence,row.hii_region,row.offset_ra_arcsec,row.offset_de_arcsec,
            *row.R_R25,*row.R_kpc,*row.R_Reff,*selected.OH12,*selected.e_OH12,
            *selected.Z,*selected.e_Z_lo,*selected.e_Z_hi
        });
    }
    std::sort(observations.begin(),observations.end(),[](const Observation& left,const Observation& right){
        return std::tie(left.R_kpc,left.source_sequence)<std::tie(right.R_kpc,right.source_sequence);
    });
    return observations;
}

std::vector<Observation> collapse_repeated_regions(const std::vector<Observation>& observations){
    using PositionKey=std::tuple<bool,double,double,int>;
    std::map<PositionKey,std::vector<Observation>> grouped;
    for (const Observation& observation: observations) {
        const bool has_offset=observation.offset_ra_arcsec.has_value() && observation.offset_de_arcsec.has_value();
        const PositionKey key=has_offset ?
            PositionKey{true,*observation.offset_ra_arcsec,*observation.offset_de_arcsec,0} :
            PositionKey{false,0.0,0.0,observation.source_sequence};
        grouped[key].push_back(observation);
    }

    std::vector<Observation> result;
    result.reserve(grouped.size());
    for (const auto& entry: grouped) {
        const std::vector<Observation>& group=entry.second;
        double weight_sum=0.0;
        double weighted_OH12=0.0;
        for (const Observation& observation: group) {
            const double weight=1.0/(observation.e_OH12*observation.e_OH12);
            weight_sum+=weight;
            weighted_OH12+=weight*observation.OH12;
        }
        const double OH12=weighted_OH12/weight_sum;
        const double e_OH12=std::sqrt(1.0/weight_sum);
        const double Z=Z_sun*std::pow(10.0,OH12-OH12_sun);
        const double Z_low=Z_sun*std::pow(10.0,OH12-e_OH12-OH12_sun);
        const double Z_high=Z_sun*std::pow(10.0,OH12+e_OH12-OH12_sun);
        result.push_back({
            group.front().source_sequence,group.front().hii_region,
            group.front().offset_ra_arcsec,group.front().offset_de_arcsec,
            weighted_mean(group,&Observation::R_R25),weighted_mean(group,&Observation::R_kpc),
            weighted_mean(group,&Observation::R_Reff),
            OH12,e_OH12,Z,Z-Z_low,Z_high-Z
        });
    }
    std::sort(result.begin(),result.end(),[](const Observation& left,const Observation& right){
        return left.R_kpc<right.R_kpc;
    });
    return result;
}

std::vector<ResidualPoint> overlap_residuals(
    const std::vector<Observation>& observations,
    const ModelCurve& curve
){
    std::vector<Observation> overlapping;
    std::vector<double> queries;
    for (const Observation& observation: observations) {
        if (curve.radius_kpc.front()<=observation.R_kpc && observation.R_kpc<=curve.radius_kpc.back()) {
            overlapping.push_back(observation);
            queries.push_back(observation.R_kpc);
        }
    }
    std::vector<double> model_log;
    model_log.reserve(curve.metallicity.size());
    for (const double Z: curve.metallicity) {
        model_log.push_back(std::log10(Z/Z_sun));
    }
    const std::vector<double> interpolated=interpolate_profile(curve.radius_kpc,model_log,queries);
    std::vector<ResidualPoint> residuals;
    residuals.reserve(overlapping.size());
    for (std::size_t index=0; index<overlapping.size(); ++index) {
        const Observation& observation=overlapping[index];
        const double observed_log=observation.OH12-OH12_sun;
        const double sigma=std::sqrt(
            observation.e_OH12*observation.e_OH12+
            abundance_scatter_dex*abundance_scatter_dex
        );
        residuals.push_back({
            observation,interpolated[index],observed_log-interpolated[index],sigma
        });
    }
    return residuals;
}

LineFit weighted_line_fit(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& sigma
){
    if (x.size()<2 || *std::max_element(x.begin(),x.end())<=*std::min_element(x.begin(),x.end())) {
        return {std::nullopt,std::nullopt,std::nullopt,std::nullopt};
    }
    double S=0.0;
    double Sx=0.0;
    double Sy=0.0;
    double Sxx=0.0;
    double Sxy=0.0;
    for (std::size_t index=0; index<x.size(); ++index) {
        const double weight=1.0/(sigma[index]*sigma[index]);
        S+=weight;
        Sx+=weight*x[index];
        Sy+=weight*y[index];
        Sxx+=weight*x[index]*x[index];
        Sxy+=weight*x[index]*y[index];
    }
    const double determinant=S*Sxx-Sx*Sx;
    const double intercept=(Sxx*Sy-Sx*Sxy)/determinant;
    const double slope=(S*Sxy-Sx*Sy)/determinant;
    return {intercept,slope,std::sqrt(Sxx/determinant),std::sqrt(S/determinant)};
}

FitStatistics comparison_statistics(
    const std::string& galaxy,
    Calibration calibration,
    const std::vector<Observation>& observations,
    const ModelCurve& curve,
    const ComparisonMetadata& metadata
){
    const std::vector<Observation> unique_observations=collapse_repeated_regions(observations);
    const std::vector<ResidualPoint> residuals=overlap_residuals(unique_observations,curve);
    if (residuals.empty()) {
        return empty_statistics(
            galaxy,calibration,curve.family,curve.id,observations,unique_observations,
            metadata,"no model-overlap points"
        );
    }

    std::vector<double> R_kpc;
    std::vector<double> R_Reff;
    std::vector<double> R_R25;
    std::vector<double> observed;
    std::vector<double> model;
    std::vector<double> residual_values;
    std::vector<double> sigma;
    for (const ResidualPoint& point: residuals) {
        R_kpc.push_back(point.observation.R_kpc);
        R_Reff.push_back(point.observation.R_Reff);
        R_R25.push_back(point.observation.R_R25);
        observed.push_back(point.observation.OH12-OH12_sun);
        model.push_back(point.model_log_Z_Zsun);
        residual_values.push_back(point.residual_dex);
        sigma.push_back(point.sigma_dex);
    }

    const LineFit observed_kpc=weighted_line_fit(R_kpc,observed,sigma);
    const LineFit observed_Reff=weighted_line_fit(R_Reff,observed,sigma);
    const LineFit model_kpc=weighted_line_fit(R_kpc,model,sigma);
    const LineFit model_Reff=weighted_line_fit(R_Reff,model,sigma);

    double weight_sum=0.0;
    double residual_sum=0.0;
    double pivot_sum=0.0;
    for (std::size_t index=0; index<residuals.size(); ++index) {
        const double weight=1.0/(sigma[index]*sigma[index]);
        weight_sum+=weight;
        residual_sum+=weight*residual_values[index];
        pivot_sum+=weight*R_Reff[index];
    }
    const double offset=residual_sum/weight_sum;
    const double pivot=pivot_sum/weight_sum;
    std::vector<double> pivoted_R;
    pivoted_R.reserve(residuals.size());
    double residual_square_sum=0.0;
    double centered_square_sum=0.0;
    double chi2_fixed=0.0;
    double chi2_offset=0.0;
    for (std::size_t index=0; index<residuals.size(); ++index) {
        const double centered_value=residual_values[index]-offset;
        pivoted_R.push_back(R_Reff[index]-pivot);
        residual_square_sum+=residual_values[index]*residual_values[index];
        centered_square_sum+=centered_value*centered_value;
        chi2_fixed+=residual_values[index]*residual_values[index]/(sigma[index]*sigma[index]);
        chi2_offset+=centered_value*centered_value/(sigma[index]*sigma[index]);
    }
    const LineFit residual_fit=weighted_line_fit(pivoted_R,residual_values,sigma);
    const auto kpc_limits=minimum_maximum(R_kpc);
    const auto Reff_limits=minimum_maximum(R_Reff);
    const auto R25_limits=minimum_maximum(R_R25);
    const double Reff_span=Reff_limits.second-Reff_limits.first;
    const int count=static_cast<int>(residuals.size());
    const bool reliable=count>=8 && Reff_span>=1.0;
    const std::string status=count<2 ? "no gradient fit" :
        count<8 ? "descriptive: fewer than 8 unique regions" :
        Reff_span<1.0 ? "descriptive: radial span below 1 Re" :
        Reff_limits.first>1.0 ? "fit-grade: outer-disk coverage" : "fit-grade";

    return {
        galaxy,calibration_name(calibration),curve.family,curve.id,
        static_cast<int>(observations.size()),static_cast<int>(unique_observations.size()),count,count,
        kpc_limits.first,kpc_limits.second,Reff_limits.first,Reff_limits.second,
        R25_limits.second-R25_limits.first,reliable,status,
        observed_kpc.slope,observed_kpc.e_slope,observed_Reff.slope,observed_Reff.e_slope,
        model_kpc.slope,model_Reff.slope,residual_fit.intercept,residual_fit.e_intercept,
        residual_fit.slope,residual_fit.e_slope,pivot,offset,std::sqrt(1.0/weight_sum),
        std::sqrt(residual_square_sum/static_cast<double>(count)),
        std::sqrt(centered_square_sum/static_cast<double>(count)),chi2_fixed,
        chi2_fixed/static_cast<double>(count),chi2_offset,
        count>1 ? std::optional<double>(chi2_offset/static_cast<double>(count-1)) : std::nullopt,
        abundance_scatter_dex,metadata.Reff_kpc,metadata.Z_outer_boundary,
        metadata.R_Z_boundary_kpc,metadata.radial_H2_available,metadata.H2_treatment
    };
}

FitStatistics unavailable_statistics(
    const std::string& galaxy,
    Calibration calibration,
    const std::vector<Observation>& observations,
    const ComparisonMetadata& metadata
){
    const std::vector<Observation> unique_observations=collapse_repeated_regions(observations);
    return empty_statistics(
        galaxy,calibration,"inverse","inverse_dynamics",observations,unique_observations,
        metadata,"inverse metallicity not solved: radial-flow stagnation"
    );
}

}
