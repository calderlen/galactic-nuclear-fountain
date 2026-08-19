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

using CsvRow=galacticwind::csv::Row;
using CsvTable=galacticwind::csv::Table;
using galacticwind::csv::escape;
using galacticwind::csv::number;
using galacticwind::csv::read;
using galacticwind::csv::value;

struct ModelRun {
    std::vector<galacticwind::ModelCurve> curves;
    std::vector<galacticwind::ModelCurve> references;
    galacticwind::ComparisonMetadata metadata;
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

galacticwind::AbundanceMeasurement read_abundance(const CsvRow& row,const std::string& calibration){
    return {
        csv_bool(row,"usable_OH12_"+calibration),
        optional_double(value(row,"OH12_"+calibration+"_dex")),
        optional_double(value(row,"e_OH12_"+calibration+"_dex")),
        optional_double(value(row,"Z_"+calibration+"_scaled_solar")),
        optional_double(value(row,"e_lo_Z_"+calibration+"_scaled_solar")),
        optional_double(value(row,"e_hi_Z_"+calibration+"_scaled_solar"))
    };
}

std::vector<galacticwind::CatalogRow> read_catalog(const std::filesystem::path& path){
    const CsvTable table=read(path);
    std::vector<galacticwind::CatalogRow> catalog;
    catalog.reserve(table.rows.size());
    for (const CsvRow& row: table.rows) {
        catalog.push_back({
            value(row,"galaxy"),csv_int(row,"source_seq"),value(row,"hii_region"),
            optional_double(value(row,"offRA_arcsec")),optional_double(value(row,"offDE_arcsec")),
            optional_double(value(row,"R_R25")),optional_double(value(row,"R_adopted_kpc")),
            optional_double(value(row,"R_Reff_adopted")),
            optional_double(value(row,"Reff_adopted_kpc")),
            read_abundance(row,"KK04"),read_abundance(row,"PT05")
        });
    }
    return catalog;
}

double adopted_reff(const std::vector<galacticwind::CatalogRow>& catalog,const std::string& galaxy){
    std::optional<double> result;
    for (const galacticwind::CatalogRow& row: catalog) {
        if (row.galaxy!=galaxy || !row.Reff_kpc) {
            continue;
        }
        result=*row.Reff_kpc;
        break;
    }
    if (!result) {
        throw std::runtime_error("Missing Reff for "+galaxy);
    }
    return *result;
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

galacticwind::ModelCurve make_curve(
    const CsvTable& profiles,
    const std::string& source,
    const std::string& value_column,
    const std::string& family,
    const std::string& id,
    const std::string& label,
    const std::string& role
){
    galacticwind::ModelCurve curve{family,id,label,role,{},{}};
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
    const std::string& family,
    double Reff_kpc
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
        {Reff_kpc,Z_boundary,R_Z_boundary,csv_bool(metadata,"radial_H2_available"),
         value(metadata,"H2_treatment")}
    };
    if (family=="forward") {
        run.curves.push_back(make_curve(
            profiles,"Leroy_THINGS","Z","forward","forward_leroy",
            "forward: Leroy/THINGS","prediction"
        ));
        run.curves.push_back(make_curve(
            profiles,"SPARC_corrected_fit","Z","forward","forward_sparc_fit",
            "forward: SPARC fit","prediction"
        ));
        return run;
    }
    run.references.push_back(make_curve(
        profiles,"Leroy_THINGS","Z_eq","inverse","inverse_local_equilibrium",
        "local-equilibrium reference","reference"
    ));
    if (solved) {
        run.curves.push_back(make_curve(
            profiles,"Leroy_THINGS","Z","inverse","inverse_dynamics",
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
    const std::vector<galacticwind::FitStatistics>& statistics
){
    std::ofstream stream=output_stream(path);
    write_row(stream,{
        "galaxy","calibration","model_family","model_curve","n_catalog_usable",
        "n_unique_catalog_regions","n_model_overlap","n_unique_model_overlap","R_min_kpc",
        "R_max_kpc","R_min_Reff","R_max_Reff","R_R25_span","gradient_sample_reliable",
        "fit_status","observed_slope_dex_per_kpc","e_observed_slope_dex_per_kpc",
        "observed_slope_dex_per_Reff","e_observed_slope_dex_per_Reff",
        "model_slope_dex_per_kpc","model_slope_dex_per_Reff","residual_offset_at_pivot_dex",
        "e_residual_offset_at_pivot_dex","residual_slope_dex_per_Reff",
        "e_residual_slope_dex_per_Reff","pivot_Reff","weighted_mean_residual_dex",
        "e_weighted_mean_residual_dex","rms_residual_dex","shape_rms_after_offset_dex",
        "chi2_fixed_model","reduced_chi2_fixed_model","chi2_after_offset",
        "reduced_chi2_after_offset","abundance_scatter_added_dex","Reff_adopted_kpc",
        "Z_outer_boundary","R_Z_boundary_kpc","radial_H2_available","H2_treatment"
    });
    for (const galacticwind::FitStatistics& row: statistics) {
        write_row(stream,{
            row.galaxy,row.calibration,row.model_family,row.model_curve,
            std::to_string(row.n_catalog_usable),std::to_string(row.n_unique_catalog_regions),
            std::to_string(row.n_model_overlap),std::to_string(row.n_unique_model_overlap),
            format_optional(row.R_min_kpc),format_optional(row.R_max_kpc),
            format_optional(row.R_min_Reff),format_optional(row.R_max_Reff),
            format_optional(row.R_R25_span),row.gradient_sample_reliable ? "true" : "false",
            row.fit_status,format_optional(row.observed_slope_dex_per_kpc),
            format_optional(row.e_observed_slope_dex_per_kpc),
            format_optional(row.observed_slope_dex_per_Reff),
            format_optional(row.e_observed_slope_dex_per_Reff),
            format_optional(row.model_slope_dex_per_kpc),format_optional(row.model_slope_dex_per_Reff),
            format_optional(row.residual_offset_at_pivot_dex),
            format_optional(row.e_residual_offset_at_pivot_dex),
            format_optional(row.residual_slope_dex_per_Reff),
            format_optional(row.e_residual_slope_dex_per_Reff),format_optional(row.pivot_Reff),
            format_optional(row.weighted_mean_residual_dex),
            format_optional(row.e_weighted_mean_residual_dex),format_optional(row.rms_residual_dex),
            format_optional(row.shape_rms_after_offset_dex),format_optional(row.chi2_fixed_model),
            format_optional(row.reduced_chi2_fixed_model),format_optional(row.chi2_after_offset),
            format_optional(row.reduced_chi2_after_offset),format_double(row.abundance_scatter_added_dex),
            format_double(row.Reff_adopted_kpc),format_double(row.Z_outer_boundary),
            format_double(row.R_Z_boundary_kpc),row.radial_H2_available ? "true" : "false",
            row.H2_treatment
        });
    }
}

void write_observation(
    std::ofstream& stream,
    const std::string& galaxy,
    const std::string& calibration,
    const galacticwind::Observation& observation
){
    write_row(stream,{
        galaxy,calibration,std::to_string(observation.source_sequence),observation.hii_region,
        format_double(observation.R_R25),format_double(observation.R_kpc),
        format_double(observation.R_Reff),format_double(observation.OH12),
        format_double(observation.e_OH12),format_double(observation.Z),
        format_double(observation.e_Z_lo),format_double(observation.e_Z_hi)
    });
}

void write_curve(
    std::ofstream& stream,
    const std::string& galaxy,
    const galacticwind::ModelCurve& curve,
    double Reff_kpc
){
    for (std::size_t index=0; index<curve.radius_kpc.size(); ++index) {
        write_row(stream,{
            galaxy,curve.family,curve.id,curve.role,curve.label,
            format_double(curve.radius_kpc[index]),format_double(curve.radius_kpc[index]/Reff_kpc),
            format_double(curve.metallicity[index])
        });
    }
}

void write_residual(
    std::ofstream& stream,
    const std::string& galaxy,
    const std::string& calibration,
    const galacticwind::ModelCurve& curve,
    const galacticwind::ResidualPoint& point
){
    write_row(stream,{
        galaxy,calibration,curve.family,curve.id,std::to_string(point.observation.source_sequence),
        format_double(point.observation.R_kpc),format_double(point.observation.R_Reff),
        format_double(point.model_log_Z_Zsun),format_double(point.residual_dex),
        format_double(point.sigma_dex)
    });
}

}

int main(int argc,char** argv){
    try {
        if (argc!=6) {
            std::cerr << "Usage: galacticwind_compare <galaxy|all> <sings-catalog.csv> <forward-root> <inverse-root> <output-directory>\n";
            return 2;
        }
        const std::string requested=argv[1];
        const std::filesystem::path catalog_path=argv[2];
        const std::filesystem::path forward_root=argv[3];
        const std::filesystem::path inverse_root=argv[4];
        const std::filesystem::path output_directory=argv[5];
        const std::vector<galacticwind::CatalogRow> catalog=read_catalog(catalog_path);

        std::set<std::string> available;
        for (const galacticwind::CatalogRow& row: catalog) {
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
            "galaxy","calibration","source_seq","hii_region","R_R25","R_kpc","R_Reff",
            "OH12","e_OH12","Z","e_Z_lo","e_Z_hi"
        });
        write_row(curve_stream,{
            "galaxy","model_family","model_curve","curve_role","label","R_kpc","R_Reff","Z"
        });
        write_row(residual_stream,{
            "galaxy","calibration","model_family","model_curve","source_seq","R_kpc","R_Reff",
            "model_log_Z_Zsun","residual_dex","sigma_dex"
        });
        std::vector<galacticwind::FitStatistics> statistics;
        const std::vector<galacticwind::Calibration> calibrations={
            galacticwind::Calibration::KK04,galacticwind::Calibration::PT05
        };

        for (const std::string& galaxy: galaxies) {
            const double Reff=adopted_reff(catalog,galaxy);
            const ModelRun forward=load_model_run(forward_root,galaxy,"forward",Reff);
            const ModelRun inverse=load_model_run(inverse_root,galaxy,"inverse",Reff);
            for (const galacticwind::ModelCurve& curve: forward.curves) {
                write_curve(curve_stream,galaxy,curve,Reff);
            }
            for (const galacticwind::ModelCurve& curve: inverse.curves) {
                write_curve(curve_stream,galaxy,curve,Reff);
            }
            for (const galacticwind::ModelCurve& curve: inverse.references) {
                write_curve(curve_stream,galaxy,curve,Reff);
            }

            for (const galacticwind::Calibration calibration: calibrations) {
                const std::string calibration_text=galacticwind::calibration_name(calibration);
                const std::vector<galacticwind::Observation> observations=
                    galacticwind::select_observations(catalog,galaxy,calibration);
                for (const galacticwind::Observation& observation: observations) {
                    write_observation(observation_stream,galaxy,calibration_text,observation);
                }
                for (const galacticwind::ModelCurve& curve: forward.curves) {
                    statistics.push_back(galacticwind::comparison_statistics(
                        galaxy,calibration,observations,curve,forward.metadata
                    ));
                    for (const galacticwind::ResidualPoint& residual:
                         galacticwind::overlap_residuals(observations,curve)) {
                        write_residual(residual_stream,galaxy,calibration_text,curve,residual);
                    }
                }
                if (inverse.curves.empty()) {
                    statistics.push_back(galacticwind::unavailable_statistics(
                        galaxy,calibration,observations,inverse.metadata
                    ));
                } else {
                    const galacticwind::ModelCurve& curve=inverse.curves.front();
                    statistics.push_back(galacticwind::comparison_statistics(
                        galaxy,calibration,observations,curve,inverse.metadata
                    ));
                    for (const galacticwind::ResidualPoint& residual:
                         galacticwind::overlap_residuals(observations,curve)) {
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
        std::cerr << "galacticwind_compare: " << error.what() << '\n';
        return 1;
    }
}
