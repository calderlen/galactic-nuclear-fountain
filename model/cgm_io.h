#pragma once

#include "cgm_amd.h"
#include "csv.h"
#include <optional>

// Remove the optional flag before the existing positional argument parsers run.
std::filesystem::path cgm_config_argument(int& argc,char** argv);
std::optional<CGMAMDSolution> load_cgm(const galactic_nuclear_fountain::csv::Row& settings,
    const std::filesystem::path& config={},const OrbitParameters* orbit_potential=nullptr);
galactic_nuclear_fountain::csv::Row cgm_parameter_values(const CGMAMDSolution& solution);
void write_cgm_profile(const std::filesystem::path& directory,const CGMAMDSolution& solution);
void write_cgm_model_parameters(const std::filesystem::path& path,
    galactic_nuclear_fountain::csv::Row settings,const CGMAMDSolution* solution=nullptr);
double cgm_j_kpc_kms(double R,const void* solution);
