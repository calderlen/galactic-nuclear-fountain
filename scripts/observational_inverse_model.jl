import GalacticWind
using CairoMakie
using CSV
using DataFrames
using LaTeXStrings
using QuadGK

# Empirical inverse model using Leroy gas/SFR profiles and two kinematic fits:
#   1. the published Leroy/THINGS analytic rotation curve;
#   2. the same analytic form fitted to geometry-corrected SPARC points.
#
# Usage:
#   julia --project=. scripts/observational_inverse_model.jl [galaxy] [Mdot_land] [R_nucl] [R_out]
#
# Defaults: NGC3198, Mdot_land = catalog SFR, R_nucl = max(0.5 kpc,
# first common data radius), and R_out = last common data radius.

const KPC_PER_YEAR_TO_KM_PER_SECOND = 9.7779222168e8

galaxy = length(ARGS) >= 1 ? ARGS[1] : "NGC3198"
observations = GalacticWind.load_galaxy_observations(galaxy)

if !observations.has_radial_H2
    @warn "No radial H2 measurements for $(observations.galaxy); this run uses HI-only gas surface density"
end

function catalog_sfr(metadata)
    !ismissing(metadata.SFR_leroy_Msun_yr) && return Float64(metadata.SFR_leroy_Msun_yr)
    !ismissing(metadata.SFR_things_Msun_yr) && return Float64(metadata.SFR_things_Msun_yr)
    error("No catalog SFR is available to set the default landing rate")
end

Mdot_land = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : catalog_sfr(observations.metadata)
R_nucl = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : max(0.5, observations.common_bounds[1])
R_out = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : observations.common_bounds[2]

Mdot_land > 0 || error("Mdot_land must be positive")
R_nucl >= observations.common_bounds[1] || error("R_nucl is below the common data range")
R_out <= observations.common_bounds[2] || error("R_out is above the common data range")
R_nucl < R_out || error("R_nucl must be smaller than R_out")

Sigmadot_land(R) = GalacticWind.Sigmadot_land(R, Mdot_land, R_nucl, R_out)

R_grid = collect(range(R_nucl, R_out, length=300))
Sigma_g_grid = observations.Sigma_g.(R_grid)
Sigmadot_star_grid = observations.Sigmadot_star.(R_grid)
Sigmadot_land_grid = Sigmadot_land.(R_grid)

v_R_kpc_yr = [
    GalacticWind.v_R_empirical(
        R,
        observations.Sigma_g,
        observations.Sigmadot_star,
        Sigmadot_land,
        R_out,
    )
    for R in R_grid
]
v_R_kms = v_R_kpc_yr .* KPC_PER_YEAR_TO_KM_PER_SECOND

function angular_momentum_results(rotation)
    v_c = rotation.v.(R_grid)
    dv_c_dR = rotation.dv_dR.(R_grid)
    j_disk = R_grid .* v_c
    j_land = GalacticWind.j_land_required.(
        R_grid,
        Sigma_g_grid,
        v_R_kpc_yr,
        Sigmadot_land_grid,
        v_c,
        dv_c_dR,
    )
    j_launch = R_nucl * rotation.v(R_nucl)

    return (
        v_c = v_c,
        dv_c_dR = dv_c_dR,
        j_disk = j_disk,
        j_land = j_land,
        j_launch = j_launch,
        j_ratio = j_land ./ j_launch,
        delta_j = j_land .- j_launch,
    )
end

leroy = angular_momentum_results(observations.leroy_rotation)
sparc = angular_momentum_results(observations.sparc_rotation)

output_dir = joinpath(
    @__DIR__,
    "..",
    "output",
    splitext(basename(@__FILE__))[1],
    observations.galaxy,
)
mkpath(output_dir)

results = DataFrame(
    R_kpc = R_grid,
    Sigma_g_Msun_kpc2 = Sigma_g_grid,
    Sigmadot_star_Msun_yr_kpc2 = Sigmadot_star_grid,
    Sigmadot_land_Msun_yr_kpc2 = Sigmadot_land_grid,
    v_R_kms = v_R_kms,
    v_c_leroy_kms = leroy.v_c,
    v_c_sparc_fit_kms = sparc.v_c,
    j_disk_leroy_kpc_kms = leroy.j_disk,
    j_disk_sparc_fit_kpc_kms = sparc.j_disk,
    j_land_required_leroy_kpc_kms = leroy.j_land,
    j_land_required_sparc_fit_kpc_kms = sparc.j_land,
    j_land_over_j_launch_leroy = leroy.j_ratio,
    j_land_over_j_launch_sparc_fit = sparc.j_ratio,
    delta_j_leroy_kpc_kms = leroy.delta_j,
    delta_j_sparc_fit_kpc_kms = sparc.delta_j,
)
CSV.write(joinpath(output_dir, "results.csv"), results)

geometry = observations.geometry
CSV.write(joinpath(output_dir, "geometry.csv"), DataFrame(
    galaxy = [observations.galaxy],
    D_sparc_Mpc = [geometry.D_sparc_Mpc],
    D_adopted_Mpc = [geometry.D_adopted_Mpc],
    inclination_sparc_deg = [geometry.inclination_sparc_deg],
    inclination_adopted_deg = [geometry.inclination_adopted_deg],
    SPARC_radius_scale = [geometry.distance_scale],
    SPARC_velocity_scale = [geometry.velocity_scale],
    R_nucl_kpc = [R_nucl],
    R_out_kpc = [R_out],
    Mdot_land_Msun_yr = [Mdot_land],
    radial_H2_available = [observations.has_radial_H2],
    H2_treatment = [observations.H2_treatment],
))

CSV.write(joinpath(output_dir, "rotation_fits.csv"), DataFrame(
    source = ["Leroy/THINGS", "SPARC corrected fit"],
    Vflat_kms = [observations.leroy_rotation.Vflat, observations.sparc_rotation.Vflat],
    lflat_kpc = [observations.leroy_rotation.lflat, observations.sparc_rotation.lflat],
    chi2 = [missing, observations.sparc_rotation.chi2],
    dof = [missing, observations.sparc_rotation.dof],
    reduced_chi2 = [missing, observations.sparc_rotation.reduced_chi2],
))

CSV.write(joinpath(output_dir, "sparc_corrected.csv"), observations.sparc)
CSV.write(joinpath(output_dir, "leroy_profiles_used.csv"), observations.radial)

# Rotation-curve comparison
R_rotation_grid = range(
    0.0,
    max(R_out, maximum(observations.sparc.R_adopted_kpc)),
    length=500,
)
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"R\;[\mathrm{kpc}]", ylabel=L"v_c\;[\mathrm{km\,s^{-1}}]")
errorbars!(
    ax,
    observations.sparc.R_adopted_kpc,
    observations.sparc.Vobs_adopted_kms,
    observations.sparc.e_Vobs_adopted_kms,
    color=(:gray40, 0.6),
)
scatter!(ax, observations.sparc.R_adopted_kpc, observations.sparc.Vobs_adopted_kms,
    color=:gray40, markersize=6, label="SPARC corrected data")
lines!(ax, R_rotation_grid, observations.leroy_rotation.v.(R_rotation_grid),
    linewidth=2, label="Leroy/THINGS fit")
lines!(ax, R_rotation_grid, observations.sparc_rotation.v.(R_rotation_grid),
    linewidth=2, linestyle=:dash, label="SPARC corrected fit")
axislegend(ax, position=:rb)
save(joinpath(output_dir, "rotation_curves.pdf"), fig)

# Observed and assumed surface rates
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"R\;[\mathrm{kpc}]",
    ylabel=L"\dot{\Sigma}\;[M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}]", yscale=log10)
lines!(ax, R_grid, Sigmadot_star_grid, linewidth=2, label=L"\dot{\Sigma}_{\star,\mathrm{obs}}")
lines!(ax, R_grid, Sigmadot_land_grid, linewidth=2, label=L"\dot{\Sigma}_{\mathrm{land}}")
axislegend(ax)
save(joinpath(output_dir, "surface_rates.pdf"), fig)

# The inferred radial velocity is independent of the rotation-curve choice.
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"R\;[\mathrm{kpc}]", ylabel=L"v_R\;[\mathrm{km\,s^{-1}}]")
lines!(ax, R_grid, v_R_kms, linewidth=2)
hlines!(ax, [0.0], color=:gray50, linestyle=:dot)
save(joinpath(output_dir, "radial_velocity.pdf"), fig)

# Rotation-curve choice enters the required landing angular momentum.
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"R\;[\mathrm{kpc}]", ylabel=L"j_{\mathrm{land,req}}/j_{\mathrm{launch}}")
lines!(ax, R_grid, leroy.j_ratio, linewidth=2, label="Leroy/THINGS fit")
lines!(ax, R_grid, sparc.j_ratio, linewidth=2, linestyle=:dash, label="SPARC corrected fit")
hlines!(ax, [1.0], color=:gray50, linestyle=:dot)
axislegend(ax)
save(joinpath(output_dir, "angular_momentum_ratio.pdf"), fig)

M_g_annulus = 2pi * quadgk(R -> R * observations.Sigma_g(R), R_nucl, R_out)[1]
SFR_annulus = 2pi * quadgk(R -> R * observations.Sigmadot_star(R), R_nucl, R_out)[1]

println("Galaxy:                  $(observations.galaxy)")
println("Adopted D, i:            $(geometry.D_adopted_Mpc) Mpc, $(geometry.inclination_adopted_deg) deg")
println("SPARC R, V scale:        $(round(geometry.distance_scale, digits=4)), $(round(geometry.velocity_scale, digits=4))")
println("Model radial range:      $R_nucl -- $R_out kpc")
println("Landing rate:            $Mdot_land Msun/yr")
println("Gas mass in annulus:     $(round(M_g_annulus, sigdigits=5)) Msun (includes helium)")
println("SFR in annulus:          $(round(SFR_annulus, sigdigits=5)) Msun/yr")
println("H2 treatment:            $(observations.H2_treatment)")
println("SPARC fit Vflat, lflat:  $(round(observations.sparc_rotation.Vflat, digits=2)) km/s, $(round(observations.sparc_rotation.lflat, digits=3)) kpc")
println("SPARC reduced chi2:      $(round(observations.sparc_rotation.reduced_chi2, digits=2))")
println("Results written to:      $(normpath(output_dir))")
