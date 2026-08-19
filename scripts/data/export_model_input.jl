import GalacticWind
using CSV
using DataFrames

function catalog_sfr(metadata)
    !ismissing(metadata.SFR_leroy_Msun_yr) && return Float64(metadata.SFR_leroy_Msun_yr)
    !ismissing(metadata.SFR_things_Msun_yr) && return Float64(metadata.SFR_things_Msun_yr)
    error("No catalog SFR is available for $(metadata.galaxy)")
end

galaxy = length(ARGS) >= 1 ? ARGS[1] : "NGC3198"
observations = GalacticWind.load_galaxy_observations(galaxy)
output_directory = length(ARGS) >= 2 ? normpath(ARGS[2]) : joinpath(@__DIR__, "..", "..", "input", "model_inputs", observations.galaxy)
Mdot_land = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : catalog_sfr(observations.metadata)
R_nucl = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : max(0.5, observations.common_bounds[1])
R_out = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : observations.common_bounds[2]
mu = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 1.0
beta = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 0.75
Mdot_out = length(ARGS) >= 8 ? string(parse(Float64, ARGS[8])) : "auto"

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

CSV.write(joinpath(output_directory, "profiles.csv"), DataFrame(
    R_kpc=observations.radial.R_kpc,
    Sigma_g_Msun_kpc2=observations.radial.Sigma_g_Msun_kpc2,
    Sigmadot_star_Msun_yr_kpc2=observations.radial.SigmaSFR_Msun_yr_kpc2,
))

CSV.write(joinpath(output_directory, "rotation_curves.csv"), DataFrame(
    source=["Leroy_THINGS", "SPARC_corrected_fit"],
    kind=["rising", "rising"],
    Vflat_kms=[observations.leroy_rotation.Vflat, observations.sparc_rotation.Vflat],
    lflat_kpc=[observations.leroy_rotation.lflat, observations.sparc_rotation.lflat],
    chi2=[NaN, observations.sparc_rotation.chi2],
    dof=[NaN, observations.sparc_rotation.dof],
    reduced_chi2=[NaN, observations.sparc_rotation.reduced_chi2],
))

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
    H2_treatment=[observations.H2_treatment],
))

CSV.write(joinpath(output_directory, "sparc_corrected.csv"), observations.sparc)
CSV.write(joinpath(output_directory, "leroy_profiles_used.csv"), observations.radial)

println("Wrote C++ model input CSVs to: $(output_directory)")
