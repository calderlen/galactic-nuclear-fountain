using CSV
using DataFrames

include(joinpath(@__DIR__, "..", "..", "observations", "observations.jl"))
using .Observations

function catalog_sfr(metadata)
    !ismissing(metadata.SFR_leroy_Msun_yr) && return Float64(metadata.SFR_leroy_Msun_yr)
    !ismissing(metadata.SFR_things_Msun_yr) && return Float64(metadata.SFR_things_Msun_yr)
    error("No catalog SFR is available for $(metadata.galaxy)")
end

"""Return the largest inner radial interval with increasing disk angular momentum."""
function forward_model_bounds(observations; scan_point_count=10_001)
    R_min, R_max = observations.common_bounds
    R_scan = collect(range(R_min, R_max, length=scan_point_count))
    dj_disk_dR = observations.sparc_spline.v.(R_scan) .+
        R_scan .* observations.sparc_spline.dv_dR.(R_scan)
    all(isfinite.(dj_disk_dR)) ||
        error("The SPARC spline produced a non-finite angular-momentum derivative")

    first_nonpositive = findfirst(dj_disk_dR .<= 0)
    first_nonpositive === nothing && return (R_min, R_max, false)
    first_nonpositive > 1 ||
        error("The SPARC spline has non-increasing angular momentum at the inner model boundary")

    # With sub-circular landed gas and inward flow, d(R*v_c)/dR <= 0 would
    # force a negative landing rate. Exclude that outer spline-edge segment.
    return (R_min, R_scan[first_nonpositive - 1], true)
end

galaxy = length(ARGS) >= 1 ? ARGS[1] : "NGC3198"
observations = Observations.load_galaxy_observations(galaxy)
output_directory = length(ARGS) >= 2 ? normpath(ARGS[2]) : joinpath(@__DIR__, "..", "..", "input", "model_inputs", observations.galaxy)
Mdot_land = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : catalog_sfr(observations.metadata)
R_spline_min, R_spline_max, rotation_domain_truncated = forward_model_bounds(observations)
R_nucl = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : max(0.5, R_spline_min)
R_out = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : R_spline_max
mu = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 1.0
beta = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 0.75
Mdot_out = length(ARGS) >= 8 ? string(parse(Float64, ARGS[8])) : "auto"

R_spline_min <= R_nucl < R_out <= R_spline_max ||
    error("Model radii must satisfy $R_spline_min <= R_nucl < R_out <= $R_spline_max kpc")

mkpath(output_directory)

parameter_names = [
    "profile_type",
    "galaxy",
    "R_nucl_kpc",
    "R_out_kpc",
    "Mdot_land_Msun_yr",
    "Mdot_out_Msun_yr",
    "mu",
    "beta",
    "Z_nucl",
    "Z_CGM",
    "yield_y",
    "Z_outer_boundary",
    "n_points",
    "h_0_kpc",
    "rtol",
    "dynamics_atol",
    "metallicity_atol",
]
parameter_values = [
    "tabulated",
    observations.galaxy,
    string(R_nucl),
    string(R_out),
    string(Mdot_land),
    Mdot_out,
    string(mu),
    string(beta),
    "0.02",
    "0.003",
    "0.015",
    "0.006",
    "300",
    "0.01",
    "1e-9",
    "1e-18",
    "1e-12",
]
CSV.write(joinpath(output_directory, "parameters.csv"), DataFrame(name=parameter_names, value=parameter_values))

spline_point_count = 750
R_spline = collect(range(R_spline_min, R_spline_max, length=spline_point_count))

gas_spline_support = observations.gas_spline.support
R_spline_min >= gas_spline_support[1] && R_spline_max <= gas_spline_support[2] ||
    error("The exported gas spline range lies outside its measured support")
Sigma_g_spline = observations.gas_spline.v.(R_spline)
dSigma_g_spline = observations.gas_spline.dv_dR.(R_spline)
all(isfinite.(Sigma_g_spline) .& (Sigma_g_spline .> 0)) ||
    error("The exported gas spline contains non-finite or non-positive values")
all(isfinite.(dSigma_g_spline)) ||
    error("The exported gas spline contains non-finite derivatives")

SFR_spline_support = observations.sfr_spline.support
R_spline_min >= SFR_spline_support[1] && R_spline_max <= SFR_spline_support[2] ||
    error("The exported star-formation spline range lies outside its measured support")
Sigmadot_star_spline = observations.sfr_spline.v.(R_spline)
all(isfinite.(Sigmadot_star_spline) .& (Sigmadot_star_spline .> 0)) ||
    error("The exported star-formation spline contains non-finite or non-positive values")

profile_source = nrow(observations.bigiel) > 0 ?
    "Leroy2008+Bigiel2010_crosscal" : "Leroy2008"

CSV.write(joinpath(output_directory, "profiles.csv"), DataFrame(
    R_kpc=R_spline,
    Sigma_g_Msun_kpc2=Sigma_g_spline,
    Sigmadot_star_Msun_yr_kpc2=Sigmadot_star_spline,
))

CSV.write(joinpath(output_directory, "gas_profiles.csv"), DataFrame(
    source=fill(profile_source, spline_point_count),
    R_kpc=R_spline,
    Sigma_g_Msun_kpc2=Sigma_g_spline,
    dSigma_g_dR_Msun_kpc3=dSigma_g_spline,
))

gas_measurement_diagnostics = DataFrame(
    source=observations.gas_measurements.source,
    row_type=fill("measurement", nrow(observations.gas_measurements)),
    R_kpc=observations.gas_measurements.R_kpc,
    Sigma_g_obs_Msun_kpc2=observations.gas_measurements.value,
    e_Sigma_g_obs_Msun_kpc2=observations.gas_measurements.error,
    Sigma_g_spline_Msun_kpc2=observations.gas_spline.v.(observations.gas_measurements.R_kpc),
    dSigma_g_dR_Msun_kpc3=observations.gas_spline.dv_dR.(observations.gas_measurements.R_kpc),
)
gas_dense_diagnostics = DataFrame(
    source=fill(profile_source, spline_point_count),
    row_type=fill("dense", spline_point_count),
    R_kpc=R_spline,
    Sigma_g_obs_Msun_kpc2=fill(missing, spline_point_count),
    e_Sigma_g_obs_Msun_kpc2=fill(missing, spline_point_count),
    Sigma_g_spline_Msun_kpc2=Sigma_g_spline,
    dSigma_g_dR_Msun_kpc3=dSigma_g_spline,
)
CSV.write(
    joinpath(output_directory, "gas_spline_diagnostics.csv"),
    vcat(gas_measurement_diagnostics, gas_dense_diagnostics),
)

sfr_measurement_diagnostics = DataFrame(
    source=observations.sfr_measurements.source,
    row_type=fill("measurement", nrow(observations.sfr_measurements)),
    R_kpc=observations.sfr_measurements.R_kpc,
    Sigmadot_star_obs_Msun_yr_kpc2=observations.sfr_measurements.value,
    e_Sigmadot_star_obs_Msun_yr_kpc2=observations.sfr_measurements.error,
    Sigmadot_star_spline_Msun_yr_kpc2=observations.sfr_spline.v.(observations.sfr_measurements.R_kpc),
)
sfr_dense_diagnostics = DataFrame(
    source=fill(profile_source, spline_point_count),
    row_type=fill("dense", spline_point_count),
    R_kpc=R_spline,
    Sigmadot_star_obs_Msun_yr_kpc2=fill(missing, spline_point_count),
    e_Sigmadot_star_obs_Msun_yr_kpc2=fill(missing, spline_point_count),
    Sigmadot_star_spline_Msun_yr_kpc2=Sigmadot_star_spline,
)
CSV.write(
    joinpath(output_directory, "sfr_spline_diagnostics.csv"),
    vcat(sfr_measurement_diagnostics, sfr_dense_diagnostics),
)

CSV.write(joinpath(output_directory, "rotation_curves.csv"), DataFrame(
    source=["SPARC_spline"],
    kind=["tabulated"],
    Vflat_kms=[NaN],
    lflat_kpc=[NaN],
    chi2=[NaN],
    dof=[NaN],
    reduced_chi2=[NaN],
))

spline_support = observations.sparc_spline.support
R_spline_min >= spline_support[1] && R_spline_max <= spline_support[2] ||
    error("The exported SPARC spline range lies outside its measured support")
V_spline = observations.sparc_spline.v.(R_spline)
dV_spline = observations.sparc_spline.dv_dR.(R_spline)
all(isfinite.(V_spline)) || error("The exported SPARC spline contains non-finite velocities")
all(isfinite.(dV_spline)) || error("The exported SPARC spline contains non-finite derivatives")
all(V_spline .+ R_spline .* dV_spline .> 0) ||
    error("The exported SPARC spline contains non-increasing disk angular momentum")

CSV.write(joinpath(output_directory, "rotation_profiles.csv"), DataFrame(
    source=fill("SPARC_spline", spline_point_count),
    R_kpc=R_spline,
    V_kms=V_spline,
    dV_dR_kms_kpc=dV_spline,
))

measurement_diagnostics = DataFrame(
    source=fill("SPARC_spline", nrow(observations.sparc)),
    row_type=fill("measurement", nrow(observations.sparc)),
    R_kpc=observations.sparc.R_adopted_kpc,
    Vobs_kms=observations.sparc.Vobs_adopted_kms,
    e_Vobs_kms=observations.sparc.e_Vobs_adopted_kms,
    V_spline_kms=observations.sparc_spline.fitted,
    dV_dR_kms_kpc=observations.sparc_spline.derivative_at_measurements,
)
dense_diagnostics = DataFrame(
    source=fill("SPARC_spline", spline_point_count),
    row_type=fill("dense", spline_point_count),
    R_kpc=R_spline,
    Vobs_kms=fill(missing, spline_point_count),
    e_Vobs_kms=fill(missing, spline_point_count),
    V_spline_kms=V_spline,
    dV_dR_kms_kpc=dV_spline,
)
CSV.write(
    joinpath(output_directory, "rotation_spline_diagnostics.csv"),
    vcat(measurement_diagnostics, dense_diagnostics),
)

geometry = observations.geometry
CSV.write(joinpath(output_directory, "metadata.csv"), DataFrame(
    galaxy=[observations.galaxy],
    D_sparc_Mpc=[geometry.D_sparc_Mpc],
    D_adopted_Mpc=[geometry.D_adopted_Mpc],
    inclination_sparc_deg=[geometry.inclination_sparc_deg],
    inclination_adopted_deg=[geometry.inclination_adopted_deg],
    SPARC_radius_scale=[geometry.distance_scale],
    SPARC_velocity_scale=[geometry.velocity_scale],
    radial_H2_available=[observations.has_radial_H2],
    H2_source=[observations.H2_source],
    H2_treatment=[observations.H2_treatment],
    bigiel2010_available=[nrow(observations.bigiel) > 0],
    bigiel2010_point_count=[nrow(observations.bigiel)],
    SFR_crosscal_factor=[observations.applied_sfr_crosscalibration.factor],
    SFR_crosscal_log_uncertainty=[observations.applied_sfr_crosscalibration.log_uncertainty],
    SFR_crosscal_log_scatter=[observations.applied_sfr_crosscalibration.log_scatter],
    SFR_crosscal_overlap_point_count=[observations.applied_sfr_crosscalibration.point_count],
    SFR_crosscal_global_summary_factor=[observations.sfr_crosscalibration.global_summary_factor],
    SFR_crosscal_galaxy_scatter=[observations.sfr_crosscalibration.galaxy_scatter],
    SFR_crosscal_galaxy_count=[observations.sfr_crosscalibration.galaxy_count],
    SFR_crosscal_point_count=[observations.sfr_crosscalibration.point_count],
    profile_sources=[profile_source],
    common_data_R_max_kpc=[observations.common_bounds[2]],
    model_R_max_kpc=[R_spline_max],
    rotation_domain_truncated=[rotation_domain_truncated],
))

CSV.write(joinpath(output_directory, "sparc_corrected.csv"), observations.sparc)
CSV.write(joinpath(output_directory, "leroy_profiles_used.csv"), observations.radial)
CSV.write(joinpath(output_directory, "bigiel_profiles_used.csv"), observations.bigiel)
CSV.write(
    joinpath(output_directory, "sfr_crosscalibration.csv"),
    observations.sfr_crosscalibration.diagnostics,
)

println("Wrote C++ model input CSVs to: $(output_directory)")
