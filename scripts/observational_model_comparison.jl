import GalacticWind
using CairoMakie
using CSV
using DataFrames
using LaTeXStrings
using Statistics

# Compare completed observational inverse and forward calculations.
#
# Usage:
#   julia --project=. scripts/observational_model_comparison.jl [galaxy]
#
# The two calculations must already have been run with the same adopted
# geometry, radial boundaries, landing-rate normalization, and output grid.

const KPC_PER_YEAR_TO_KM_PER_SECOND = 9.7779222168e8

galaxy = GalacticWind.normalize_galaxy_name(length(ARGS) >= 1 ? ARGS[1] : "NGC3198")
root_dir = normpath(joinpath(@__DIR__, ".."))
inverse_dir = joinpath(root_dir, "output", "observational_inverse_model", galaxy)
forward_dir = joinpath(root_dir, "output", "observational_forward_model", galaxy)

function require_output(path, command)
    isfile(path) && return path
    error("Missing $(normpath(path)). Run this first:\n  $command")
end

inverse_results_path = require_output(
    joinpath(inverse_dir, "results.csv"),
    "julia --project=. scripts/observational_inverse_model.jl $galaxy",
)
inverse_geometry_path = require_output(
    joinpath(inverse_dir, "geometry.csv"),
    "julia --project=. scripts/observational_inverse_model.jl $galaxy",
)
forward_results_path = require_output(
    joinpath(forward_dir, "results.csv"),
    "julia --project=. scripts/observational_forward_model.jl $galaxy",
)
forward_geometry_path = require_output(
    joinpath(forward_dir, "geometry.csv"),
    "julia --project=. scripts/observational_forward_model.jl $galaxy",
)

inverse = CSV.read(inverse_results_path, DataFrame)
forward = CSV.read(forward_results_path, DataFrame)
inverse_geometry = CSV.read(inverse_geometry_path, DataFrame)
forward_geometry = CSV.read(forward_geometry_path, DataFrame)

nrow(inverse_geometry) == 1 || error("Inverse geometry.csv must contain one row")
nrow(forward_geometry) == 1 || error("Forward geometry.csv must contain one row")

function require_same_number(label, inverse_value, forward_value)
    isapprox(inverse_value, forward_value; rtol=1e-10, atol=1e-12) && return
    error("Cannot compare mismatched $label: inverse=$inverse_value, forward=$forward_value")
end

function require_same_vector(label, inverse_values, forward_values)
    length(inverse_values) == length(forward_values) ||
        error("Cannot compare mismatched $label: the output lengths differ")
    all(isapprox.(inverse_values, forward_values; rtol=1e-10, atol=1e-12)) && return
    error("Cannot compare mismatched $label between the inverse and forward outputs")
end

inverse_galaxy = GalacticWind.normalize_galaxy_name(inverse_geometry.galaxy[1])
forward_galaxy = GalacticWind.normalize_galaxy_name(forward_geometry.galaxy[1])
inverse_galaxy == galaxy || error("Inverse output is for $inverse_galaxy, not $galaxy")
forward_galaxy == galaxy || error("Forward output is for $forward_galaxy, not $galaxy")

for column in (
    :D_sparc_Mpc,
    :D_adopted_Mpc,
    :inclination_sparc_deg,
    :inclination_adopted_deg,
    :SPARC_radius_scale,
    :SPARC_velocity_scale,
    :R_nucl_kpc,
    :R_out_kpc,
)
    require_same_number(
        string(column),
        inverse_geometry[1, column],
        forward_geometry[1, column],
    )
end

Mdot_land = inverse_geometry.Mdot_land_Msun_yr[1]
require_same_number(
    "landing-rate normalization",
    Mdot_land,
    forward_geometry.Mdot_land_target_Msun_yr[1],
)
require_same_vector("radial grids", inverse.R_kpc, forward.R_kpc)
require_same_vector(
    "gas surface-density profiles",
    inverse.Sigma_g_Msun_kpc2,
    forward.Sigma_g_Msun_kpc2,
)
require_same_vector(
    "star-formation profiles",
    inverse.Sigmadot_star_Msun_yr_kpc2,
    forward.Sigmadot_star_Msun_yr_kpc2,
)

R = inverse.R_kpc
R_nucl = inverse_geometry.R_nucl_kpc[1]
R_out = inverse_geometry.R_out_kpc[1]
mu = forward_geometry.mu[1]
beta = forward_geometry.beta[1]

# The inverse model assumes Sigma_dot_land proportional to R^-2, so its
# cumulative landing rate has this exact logarithmic form.
cumulative_landing_inverse = Mdot_land .* log.(R ./ R_nucl) ./ log(R_out / R_nucl)

# Both scripts define inward motion with v_R < 0. This converts the inverse
# velocity back to the inward-positive mass flow rate -2*pi*R*Sigma_g*v_R.
Mdot_acc_inverse = -2pi .* R .* inverse.Sigma_g_Msun_kpc2 .*
    (inverse.v_R_kms ./ KPC_PER_YEAR_TO_KM_PER_SECOND)

j_land_over_j_disk_inverse_leroy =
    inverse.j_land_required_leroy_kpc_kms ./ inverse.j_disk_leroy_kpc_kms
j_land_over_j_disk_inverse_sparc =
    inverse.j_land_required_sparc_fit_kpc_kms ./ inverse.j_disk_sparc_fit_kpc_kms

comparison = DataFrame(
    R_kpc = R,
    Sigma_g_Msun_kpc2 = inverse.Sigma_g_Msun_kpc2,
    Sigmadot_star_Msun_yr_kpc2 = inverse.Sigmadot_star_Msun_yr_kpc2,
    Sigmadot_land_inverse_Msun_yr_kpc2 = inverse.Sigmadot_land_Msun_yr_kpc2,
    Sigmadot_land_forward_leroy_Msun_yr_kpc2 =
        forward.Sigmadot_land_leroy_Msun_yr_kpc2,
    Sigmadot_land_forward_sparc_fit_Msun_yr_kpc2 =
        forward.Sigmadot_land_sparc_fit_Msun_yr_kpc2,
    cumulative_landing_inverse_Msun_yr = cumulative_landing_inverse,
    cumulative_landing_forward_leroy_Msun_yr =
        forward.cumulative_landing_leroy_Msun_yr,
    cumulative_landing_forward_sparc_fit_Msun_yr =
        forward.cumulative_landing_sparc_fit_Msun_yr,
    v_R_inverse_kms = inverse.v_R_kms,
    v_R_forward_leroy_kms = forward.v_R_leroy_kms,
    v_R_forward_sparc_fit_kms = forward.v_R_sparc_fit_kms,
    Mdot_acc_inverse_Msun_yr = Mdot_acc_inverse,
    Mdot_acc_forward_leroy_Msun_yr = forward.Mdot_acc_leroy_Msun_yr,
    Mdot_acc_forward_sparc_fit_Msun_yr = forward.Mdot_acc_sparc_fit_Msun_yr,
    j_land_over_j_disk_inverse_leroy = j_land_over_j_disk_inverse_leroy,
    j_land_over_j_disk_inverse_sparc_fit = j_land_over_j_disk_inverse_sparc,
    j_land_over_j_disk_forward_leroy = forward.j_land_over_j_disk_leroy,
    j_land_over_j_disk_forward_sparc_fit = forward.j_land_over_j_disk_sparc_fit,
)

output_dir = joinpath(root_dir, "output", "observational_model_comparison", galaxy)
mkpath(output_dir)
CSV.write(joinpath(output_dir, "comparison.csv"), comparison)
CSV.write(joinpath(output_dir, "parameters.csv"), DataFrame(
    galaxy = [galaxy],
    R_nucl_kpc = [R_nucl],
    R_out_kpc = [R_out],
    Mdot_land_Msun_yr = [Mdot_land],
    mu = [mu],
    beta = [beta],
))

const LEROY_COLOR = :royalblue
const SPARC_COLOR = :darkorange
const INVERSE_STYLE = :dashdot

# Assumed inverse landing profile versus forward-model predictions.
all_rates = vcat(
    inverse.Sigmadot_star_Msun_yr_kpc2,
    inverse.Sigmadot_land_Msun_yr_kpc2,
    forward.Sigmadot_land_leroy_Msun_yr_kpc2,
    forward.Sigmadot_land_sparc_fit_Msun_yr_kpc2,
)
rate_scale = minimum(all_rates) > 0 ? log10 : identity
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"R\;[\mathrm{kpc}]",
    ylabel=L"\dot{\Sigma}\;[M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}]",
    yscale=rate_scale)
lines!(ax, R, inverse.Sigmadot_star_Msun_yr_kpc2, color=:black, linewidth=2,
    label=L"\dot{\Sigma}_{\star,\mathrm{obs}}")
lines!(ax, R, inverse.Sigmadot_land_Msun_yr_kpc2, color=:gray35, linewidth=2,
    linestyle=INVERSE_STYLE, label="inverse: assumed landing")
lines!(ax, R, forward.Sigmadot_land_leroy_Msun_yr_kpc2,
    color=LEROY_COLOR, linewidth=2, label="forward: Leroy/THINGS")
lines!(ax, R, forward.Sigmadot_land_sparc_fit_Msun_yr_kpc2,
    color=SPARC_COLOR, linewidth=2, label="forward: SPARC fit")
axislegend(ax)
save(joinpath(output_dir, "surface_rates.pdf"), fig)

# All three profiles have the same requested total landing normalization.
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"R\;[\mathrm{kpc}]",
    ylabel=L"\dot{M}_{\mathrm{land}}(<R)\;[M_\odot\,\mathrm{yr}^{-1}]")
lines!(ax, R, cumulative_landing_inverse, color=:gray35, linewidth=2,
    linestyle=INVERSE_STYLE, label="inverse: assumed")
lines!(ax, R, forward.cumulative_landing_leroy_Msun_yr,
    color=LEROY_COLOR, linewidth=2, label="forward: Leroy/THINGS")
lines!(ax, R, forward.cumulative_landing_sparc_fit_Msun_yr,
    color=SPARC_COLOR, linewidth=2, label="forward: SPARC fit")
hlines!(ax, [Mdot_land], color=:black, linestyle=:dot, label="total target")
axislegend(ax, position=:rb)
save(joinpath(output_dir, "cumulative_landing_rate.pdf"), fig)

# Inverse-inferred radial flow versus the forward predictions.
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"R\;[\mathrm{kpc}]", ylabel=L"v_R\;[\mathrm{km\,s^{-1}}]")
lines!(ax, R, inverse.v_R_kms, color=:gray35, linewidth=2,
    linestyle=INVERSE_STYLE, label="inverse")
lines!(ax, R, forward.v_R_leroy_kms, color=LEROY_COLOR, linewidth=2,
    label="forward: Leroy/THINGS")
lines!(ax, R, forward.v_R_sparc_fit_kms, color=SPARC_COLOR, linewidth=2,
    label="forward: SPARC fit")
hlines!(ax, [0.0], color=:black, linestyle=:dot)
axislegend(ax)
save(joinpath(output_dir, "radial_velocity.pdf"), fig)

# Corresponding inward-positive radial mass flow rates.
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"R\;[\mathrm{kpc}]",
    ylabel=L"\dot{M}_{\mathrm{acc}}\;[M_\odot\,\mathrm{yr}^{-1}]")
lines!(ax, R, Mdot_acc_inverse, color=:gray35, linewidth=2,
    linestyle=INVERSE_STYLE, label="inverse")
lines!(ax, R, forward.Mdot_acc_leroy_Msun_yr,
    color=LEROY_COLOR, linewidth=2, label="forward: Leroy/THINGS")
lines!(ax, R, forward.Mdot_acc_sparc_fit_Msun_yr,
    color=SPARC_COLOR, linewidth=2, label="forward: SPARC fit")
axislegend(ax)
save(joinpath(output_dir, "accretion_rate.pdf"), fig)

# Inverse-required versus forward-prescribed landing angular momentum. Dividing
# both by the local disk value makes the two model definitions comparable.
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"R\;[\mathrm{kpc}]",
    ylabel=L"j_{\mathrm{land}}/j_{\mathrm{disk}}")
lines!(ax, R, j_land_over_j_disk_inverse_leroy,
    color=LEROY_COLOR, linewidth=2, linestyle=INVERSE_STYLE,
    label="inverse required: Leroy/THINGS")
lines!(ax, R, forward.j_land_over_j_disk_leroy,
    color=LEROY_COLOR, linewidth=2, label="forward prescribed: Leroy/THINGS")
lines!(ax, R, j_land_over_j_disk_inverse_sparc,
    color=SPARC_COLOR, linewidth=2, linestyle=INVERSE_STYLE,
    label="inverse required: SPARC fit")
lines!(ax, R, forward.j_land_over_j_disk_sparc_fit,
    color=SPARC_COLOR, linewidth=2, label="forward prescribed: SPARC fit")
hlines!(ax, [beta], color=:gray45, linestyle=:dot, label=L"\beta")
hlines!(ax, [1.0], color=:black, linestyle=:dash)
axislegend(ax)
save(joinpath(output_dir, "angular_momentum_ratio.pdf"), fig)

velocity_rms_leroy = sqrt(mean((forward.v_R_leroy_kms .- inverse.v_R_kms) .^ 2))
velocity_rms_sparc = sqrt(mean((forward.v_R_sparc_fit_kms .- inverse.v_R_kms) .^ 2))

println("Galaxy:                  $galaxy")
println("Matched radial range:    $R_nucl -- $R_out kpc")
println("Matched landing rate:    $Mdot_land Msun/yr")
println("Forward mu, beta:        $mu, $beta")
println("RMS v_R difference:      $(round(velocity_rms_leroy, digits=3)) km/s (Leroy)")
println("                         $(round(velocity_rms_sparc, digits=3)) km/s (SPARC)")
println("Results written to:      $(normpath(output_dir))")
