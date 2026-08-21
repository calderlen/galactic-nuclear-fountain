#include "comparison.h"
#include "csv.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using CsvRow=galactic_nuclear_fountain::csv::Row;
using CsvTable=galactic_nuclear_fountain::csv::Table;
using galactic_nuclear_fountain::csv::escape;
using galactic_nuclear_fountain::csv::number;
using galactic_nuclear_fountain::csv::read;
using galactic_nuclear_fountain::csv::value;

struct ModelRun {
    std::vector<galactic_nuclear_fountain::ModelCurve> curves;
    std::vector<galactic_nuclear_fountain::ModelCurve> references;
    galactic_nuclear_fountain::ComparisonMetadata metadata;
};

std::optional<double> optional_double(const std::string& text){
    if (text.empty()) {
        return std::nullopt;
    }
    return std::stod(text);
}

int csv_int(const CsvRow& row,const std::string& name){
    return static_cast<int>(number(row,name));
}

bool csv_bool(const CsvRow& row,const std::string& name){
    const std::string& text=value(row,name);
    return text=="true" || text=="TRUE" || text=="1";
}

galactic_nuclear_fountain::AbundanceMeasurement read_abundance(const CsvRow& row,const std::string& calibration){
    return {
        csv_bool(row,"usable_OH12_"+calibration),
        optional_double(value(row,"OH12_"+calibration+"_dex")),
        optional_double(value(row,"e_OH12_"+calibration+"_dex"))
    };
}

std::vector<galactic_nuclear_fountain::CatalogRow> read_catalog(const std::filesystem::path& path){
    const CsvTable table=read(path);
    std::vector<galactic_nuclear_fountain::CatalogRow> catalog;
    catalog.reserve(table.rows.size());
    for (const CsvRow& row: table.rows) {
        catalog.push_back({
            value(row,"galaxy"),csv_int(row,"source_seq"),value(row,"hii_region"),
            optional_double(value(row,"offRA_arcsec")),optional_double(value(row,"offDE_arcsec")),
            optional_double(value(row,"R_adopted_kpc")),
            read_abundance(row,"KK04"),read_abundance(row,"PT05")
        });
    }
    return catalog;
}

std::vector<const CsvRow*> rows_for_source(const CsvTable& profiles,const std::string& source){
    std::vector<const CsvRow*> rows;
    for (const CsvRow& row: profiles.rows) {
        if (value(row,"source")==source) {
            rows.push_back(&row);
        }
    }
    std::sort(rows.begin(),rows.end(),[](const CsvRow* left,const CsvRow* right){
        return number(*left,"R_kpc")<number(*right,"R_kpc");
    });
    return rows;
}

galactic_nuclear_fountain::ModelCurve make_curve(
    const CsvTable& profiles,
    const std::string& source,
    const std::string& value_column,
    const std::string& family,
    const std::string& id,
    const std::string& label,
    const std::string& role
){
    galactic_nuclear_fountain::ModelCurve curve{family,id,label,role,{},{}};
    for (const CsvRow* row: rows_for_source(profiles,source)) {
        const double radius=number(*row,"R_kpc");
        const std::optional<double> profile_value=optional_double(value(*row,value_column));
        if (profile_value && std::isfinite(*profile_value) && *profile_value>0.0) {
            curve.radius_kpc.push_back(radius);
            curve.metallicity.push_back(*profile_value);
        }
    }
    if (curve.radius_kpc.size()<2) {
        throw std::runtime_error("Missing model curve "+id);
    }
    return curve;
}

ModelRun load_model_run(
    const std::filesystem::path& root,
    const std::string& galaxy,
    const std::string& family
){
    const std::filesystem::path directory=root/galaxy;
    const std::filesystem::path profiles_path=directory/"profiles.csv";
    const std::filesystem::path summary_path=directory/"summary.csv";
    const std::filesystem::path metadata_path=directory/"metadata.csv";
    const CsvTable profiles=read(profiles_path);
    const CsvRow summary=read(summary_path).rows.at(0);
    const CsvRow metadata=read(metadata_path).rows.at(0);
    const double Z_boundary=number(summary,"Z_outer_boundary");
    const double R_Z_boundary=number(summary,"R_Z_boundary_kpc");
    const bool solved=number(summary,"metallicity_solved")!=0.0;

    ModelRun run{
        {},{},
        {Z_boundary,R_Z_boundary,csv_bool(metadata,"radial_H2_available"),
         value(metadata,"H2_treatment")}
    };
    if (family=="forward") {
        run.curves.push_back(make_curve(
            profiles,"SPARC_spline","Z","forward","forward_sparc_spline",
            "forward: SPARC spline","prediction"
        ));
        return run;
    }
    run.references.push_back(make_curve(
        profiles,"SPARC_spline","Z_eq","inverse","inverse_local_equilibrium",
        "local-equilibrium reference","reference"
    ));
    if (solved) {
        run.curves.push_back(make_curve(
            profiles,"SPARC_spline","Z","inverse","inverse_dynamics",
            "inverse-dynamics prediction","prediction"
        ));
    }
    return run;
}

std::string format_double(double value){
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

std::string format_optional(const std::optional<double>& value){
    return value ? format_double(*value) : "";
}

void write_row(std::ofstream& stream,const std::vector<std::string>& values){
    for (std::size_t index=0; index<values.size(); ++index) {
        index>0 && (stream << ',');
        stream << escape(values[index]);
    }
    stream << '\n';
}

std::ofstream output_stream(const std::filesystem::path& path){
    return std::ofstream(path);
}

void write_fit_summary(
    const std::filesystem::path& path,
    const std::vector<galactic_nuclear_fountain::FitStatistics>& statistics
){
    std::ofstream stream=output_stream(path);
    write_row(stream,{
        "galaxy","calibration","model_family","model_curve","n_catalog_usable",
        "n_unique_catalog_regions","n_model_overlap","n_unique_model_overlap","R_min_kpc",
        "R_max_kpc","R_span_kpc","gradient_sample_reliable",
        "fit_status","observed_OH12_slope_dex_per_kpc","e_observed_OH12_slope_dex_per_kpc",
        "model_OH12_slope_dex_per_kpc",
        "OH12_residual_offset_at_pivot_dex","e_OH12_residual_offset_at_pivot_dex",
        "OH12_residual_slope_dex_per_kpc","e_OH12_residual_slope_dex_per_kpc",
        "pivot_kpc","weighted_mean_OH12_residual_dex",
        "e_weighted_mean_OH12_residual_dex","rms_OH12_residual_dex",
        "OH12_shape_rms_after_offset_dex",
        "chi2_fixed_model","reduced_chi2_fixed_model","chi2_after_offset",
        "reduced_chi2_after_offset","abundance_scatter_added_dex",
        "Z_outer_boundary","model_OH12_outer_boundary","R_Z_boundary_kpc",
        "radial_H2_available","H2_treatment"
    });
    for (const galactic_nuclear_fountain::FitStatistics& row: statistics) {
        write_row(stream,{
            row.galaxy,row.calibration,row.model_family,row.model_curve,
            std::to_string(row.n_catalog_usable),std::to_string(row.n_unique_catalog_regions),
            std::to_string(row.n_model_overlap),std::to_string(row.n_unique_model_overlap),
            format_optional(row.R_min_kpc),format_optional(row.R_max_kpc),
            format_optional(row.R_span_kpc),row.gradient_sample_reliable ? "true" : "false",
            row.fit_status,format_optional(row.observed_slope_dex_per_kpc),
            format_optional(row.e_observed_slope_dex_per_kpc),
            format_optional(row.model_slope_dex_per_kpc),
            format_optional(row.residual_offset_at_pivot_dex),
            format_optional(row.e_residual_offset_at_pivot_dex),
            format_optional(row.residual_slope_dex_per_kpc),
            format_optional(row.e_residual_slope_dex_per_kpc),format_optional(row.pivot_kpc),
            format_optional(row.weighted_mean_residual_dex),
            format_optional(row.e_weighted_mean_residual_dex),format_optional(row.rms_residual_dex),
            format_optional(row.shape_rms_after_offset_dex),format_optional(row.chi2_fixed_model),
            format_optional(row.reduced_chi2_fixed_model),format_optional(row.chi2_after_offset),
            format_optional(row.reduced_chi2_after_offset),format_double(row.abundance_scatter_added_dex),
            format_double(row.Z_outer_boundary),
            format_double(galactic_nuclear_fountain::oxygen_abundance_from_metallicity(
                row.Z_outer_boundary
            )),format_double(row.R_Z_boundary_kpc),
            row.radial_H2_available ? "true" : "false",row.H2_treatment
        });
    }
}

void write_observation(
    std::ofstream& stream,
    const std::string& galaxy,
    const std::string& calibration,
    const galactic_nuclear_fountain::Observation& observation
){
    write_row(stream,{
        galaxy,calibration,std::to_string(observation.source_sequence),observation.hii_region,
        format_double(observation.R_kpc),format_double(observation.OH12),
        format_double(observation.e_OH12)
    });
}

void write_curve(
    std::ofstream& stream,
    const std::string& galaxy,
    const galactic_nuclear_fountain::ModelCurve& curve
){
    for (std::size_t index=0; index<curve.radius_kpc.size(); ++index) {
        write_row(stream,{
            galaxy,curve.family,curve.id,curve.role,curve.label,
            format_double(curve.radius_kpc[index]),
            format_double(galactic_nuclear_fountain::oxygen_abundance_from_metallicity(
                curve.metallicity[index]
            ))
        });
    }
}

void write_residual(
    std::ofstream& stream,
    const std::string& galaxy,
    const std::string& calibration,
    const galactic_nuclear_fountain::ModelCurve& curve,
    const galactic_nuclear_fountain::ResidualPoint& point
){
    write_row(stream,{
        galaxy,calibration,curve.family,curve.id,std::to_string(point.observation.source_sequence),
        format_double(point.observation.R_kpc),format_double(point.model_OH12),format_double(point.residual_dex),
        format_double(point.sigma_dex)
    });
}

}

int main(int argc,char** argv){
    try {
        if (argc!=6) {
            std::cerr << "Usage: galactic-nuclear-fountain-compare <galaxy|all> <sings-catalog.csv> <forward-root> <inverse-root> <output-directory>\n";
            return 2;
        }
        const std::string requested=argv[1];
        const std::filesystem::path catalog_path=argv[2];
        const std::filesystem::path forward_root=argv[3];
        const std::filesystem::path inverse_root=argv[4];
        const std::filesystem::path output_directory=argv[5];
        const std::vector<galactic_nuclear_fountain::CatalogRow> catalog=read_catalog(catalog_path);

        std::set<std::string> available;
        for (const galactic_nuclear_fountain::CatalogRow& row: catalog) {
            available.insert(row.galaxy);
        }
        std::vector<std::string> galaxies;
        if (requested=="all" || requested=="ALL") {
            galaxies.assign(available.begin(),available.end());
        } else {
            if (available.count(requested)==0) {
                throw std::runtime_error("Unknown galaxy "+requested);
            }
            galaxies.push_back(requested);
        }

        std::filesystem::create_directories(output_directory);
        std::ofstream observation_stream=output_stream(output_directory/"observations.csv");
        std::ofstream curve_stream=output_stream(output_directory/"model_curves.csv");
        std::ofstream residual_stream=output_stream(output_directory/"residuals.csv");
        write_row(observation_stream,{
            "galaxy","calibration","source_seq","hii_region","R_kpc","OH12","e_OH12"
        });
        write_row(curve_stream,{
            "galaxy","model_family","model_curve","curve_role","label","R_kpc","model_OH12"
        });
        write_row(residual_stream,{
            "galaxy","calibration","model_family","model_curve","source_seq","R_kpc",
            "model_OH12","OH12_residual_dex","sigma_OH12_dex"
        });
        std::vector<galactic_nuclear_fountain::FitStatistics> statistics;
        const std::vector<galactic_nuclear_fountain::Calibration> calibrations={
            galactic_nuclear_fountain::Calibration::KK04,galactic_nuclear_fountain::Calibration::PT05
        };

        for (const std::string& galaxy: galaxies) {
            const ModelRun forward=load_model_run(forward_root,galaxy,"forward");
            const ModelRun inverse=load_model_run(inverse_root,galaxy,"inverse");
            for (const galactic_nuclear_fountain::ModelCurve& curve: forward.curves) {
                write_curve(curve_stream,galaxy,curve);
            }
            for (const galactic_nuclear_fountain::ModelCurve& curve: inverse.curves) {
                write_curve(curve_stream,galaxy,curve);
            }
            for (const galactic_nuclear_fountain::ModelCurve& curve: inverse.references) {
                write_curve(curve_stream,galaxy,curve);
            }

            for (const galactic_nuclear_fountain::Calibration calibration: calibrations) {
                const std::string calibration_text=galactic_nuclear_fountain::calibration_name(calibration);
                const std::vector<galactic_nuclear_fountain::Observation> observations=
                    galactic_nuclear_fountain::select_observations(catalog,galaxy,calibration);
                for (const galactic_nuclear_fountain::Observation& observation: observations) {
                    write_observation(observation_stream,galaxy,calibration_text,observation);
                }
                for (const galactic_nuclear_fountain::ModelCurve& curve: forward.curves) {
                    statistics.push_back(galactic_nuclear_fountain::comparison_statistics(
                        galaxy,calibration,observations,curve,forward.metadata
                    ));
                    for (const galactic_nuclear_fountain::ResidualPoint& residual:
                         galactic_nuclear_fountain::overlap_residuals(observations,curve)) {
                        write_residual(residual_stream,galaxy,calibration_text,curve,residual);
                    }
                }
                if (inverse.curves.empty()) {
                    statistics.push_back(galactic_nuclear_fountain::unavailable_statistics(
                        galaxy,calibration,observations,inverse.metadata
                    ));
                } else {
                    const galactic_nuclear_fountain::ModelCurve& curve=inverse.curves.front();
                    statistics.push_back(galactic_nuclear_fountain::comparison_statistics(
                        galaxy,calibration,observations,curve,inverse.metadata
                    ));
                    for (const galactic_nuclear_fountain::ResidualPoint& residual:
                         galactic_nuclear_fountain::overlap_residuals(observations,curve)) {
                        write_residual(residual_stream,galaxy,calibration_text,curve,residual);
                    }
                }
            }
        }

        std::sort(statistics.begin(),statistics.end(),[](const auto& left,const auto& right){
            return std::tie(left.calibration,left.galaxy,left.model_family,left.model_curve)<
                std::tie(right.calibration,right.galaxy,right.model_family,right.model_curve);
        });
        write_fit_summary(output_directory/"fit_summary.csv",statistics);
        std::cout << "Galaxies compared: " << galaxies.size() << '\n';
        std::cout << "Comparison rows: " << statistics.size() << '\n';
        std::cout << "Wrote comparison CSV output to " << output_directory << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "galactic-nuclear-fountain-compare: " << error.what() << '\n';
        return 1;
    }
}
