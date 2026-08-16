import GalacticWind
import DifferentialEquations as DE
using CairoMakie
using CSV
using DataFrames
using LaTeXStrings
using QuadGK

# Forward mixing model using observed Leroy gas/SFR profiles and both smooth
# rotation-curve choices prepared by load_galaxy_observations.
#
# Usage:
#   julia --project=. scripts/observational_forward_model.jl \
#       [galaxy] [mu] [beta] [Mdot_land] [R_nucl] [R_out]
#
# Defaults: NGC3198, mu=1, beta=0.75, Mdot_land=catalog SFR, and the same
# data-limited radial boundaries used by the observational inverse model.

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

mu = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
beta = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.75
Mdot_land_target = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : catalog_sfr(observations.metadata)
R_nucl = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : max(0.5, observations.common_bounds[1])
R_out = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : observations.common_bounds[2]

mu > 0 || error("mu must be positive")
0 <= beta < 1 || error("beta must satisfy 0 <= beta < 1 for this inflow model")
Mdot_land_target > 0 || error("Mdot_land must be positive")
R_nucl >= observations.common_bounds[1] || error("R_nucl is below the common data range")
R_out <= observations.common_bounds[2] || error("R_out is above the common data range")
R_nucl < R_out || error("R_nucl must be smaller than R_out")

R_grid = collect(range(R_nucl, R_out, length=300))
Sigma_g_grid = observations.Sigma_g.(R_grid)
Sigmadot_star_grid = observations.Sigmadot_star.(R_grid)

function solve_forward(rotation)
    p = (
        Sigmadot_star = observations.Sigmadot_star,
        v_c = rotation.v,
        dv_c_dR = rotation.dv_dR,
        R_nucl = R_nucl,
        mu = mu,
        beta = beta,
    )

    j_disk(R) = GalacticWind.j_disk_empirical(R, rotation.v)
    j_CGM(R) = beta * j_disk(R)
    j_land(R) = GalacticWind.j_land_mixing_empirical(R, R_nucl, mu, beta, rotation.v)
    delta_j(R) = j_land(R) - j_disk(R)

    delta_j_grid = delta_j.(R_grid)
    relative_delta_j = abs.(delta_j_grid) ./ max.(abs.(j_disk.(R_grid)), 1.0)
    minimum(relative_delta_j) > 1e-8 ||
        error("j_land - j_disk is too close to zero in the model domain")

    function solve_for(F0)
        problem = DE.ODEProblem(GalacticWind.radial_mass_flux_ode, F0, (R_nucl, R_out), p)
        return DE.solve(problem, DE.Tsit5(); reltol=1e-9, abstol=1e-12)
    end

    function landing_rate(R, F)
        return GalacticWind.Sigmadot_land_forward_empirical(
            R,
            F,
            R_nucl,
            mu,
            beta,
            rotation.v,
            rotation.dv_dR,
        )
    end

    function total_landing_rate(solution)
        integral = quadgk(R -> R * landing_rate(R, solution(R)), R_nucl, R_out)[1]
        return 2 * pi * integral
    end

    # The ODE and its landing-rate integral are linear in F0. Two trial solves
    # therefore determine the F0 that gives the requested total landing rate.
    F0_a = 0.0
    F0_b = -1.0
    solution_a = solve_for(F0_a)
    solution_b = solve_for(F0_b)
    Mdot_a = total_landing_rate(solution_a)
    Mdot_b = total_landing_rate(solution_b)
    abs(Mdot_b - Mdot_a) > eps(Float64) || error("Could not normalize the landing profile")

    F0 = F0_a + (Mdot_land_target - Mdot_a) * (F0_b - F0_a) / (Mdot_b - Mdot_a)
    solution = solve_for(F0)

    F_grid = solution.(R_grid)
    Sigmadot_land_grid = landing_rate.(R_grid, F_grid)
    v_R_kpc_yr = GalacticWind.v_R_forward_empirical.(
        R_grid,
        F_grid,
        Ref(observations.Sigma_g),
    )
    v_R_kms = v_R_kpc_yr .* KPC_PER_YEAR_TO_KM_PER_SECOND
    Mdot_acc_grid = GalacticWind.Mdot_acc_forward_empirical.(F_grid)
    j_disk_grid = j_disk.(R_grid)
    j_CGM_grid = j_CGM.(R_grid)
    j_land_grid = j_land.(R_grid)

    cumulative_landing_grid = [
        2 * pi * quadgk(r -> r * landing_rate(r, solution(r)), R_nucl, R)[1]
        for R in R_grid
    ]
    Mdot_integrated = cumulative_landing_grid[end]
    normalization_error = abs(Mdot_integrated - Mdot_land_target) / Mdot_land_target
    landing_tolerance = 1e-10 * max(maximum(abs.(Sigmadot_land_grid)), 1.0)
    velocity_tolerance = 1e-10 * max(maximum(abs.(v_R_kms)), 1.0)

    solver_success = DE.successful_retcode(solution_a) &&
        DE.successful_retcode(solution_b) && DE.successful_retcode(solution)
    landing_nonnegative = minimum(Sigmadot_land_grid) >= -landing_tolerance
    inward_flow = maximum(v_R_kms) <= velocity_tolerance
    lower_angular_momentum = maximum(delta_j_grid) < 0
    valid = solver_success && landing_nonnegative && inward_flow &&
        lower_angular_momentum && normalization_error < 1e-6

    return (
        solution = solution,
        F0 = F0,
        F = F_grid,
        Sigmadot_land = Sigmadot_land_grid,
        cumulative_landing = cumulative_landing_grid,
        v_R = v_R_kms,
        Mdot_acc = Mdot_acc_grid,
        v_c = rotation.v.(R_grid),
        j_disk = j_disk_grid,
        j_CGM = j_CGM_grid,
        j_land = j_land_grid,
        delta_j = delta_j_grid,
        j_CGM_ratio = j_CGM_grid ./ j_disk_grid,
        j_land_ratio = j_land_grid ./ j_disk_grid,
        Mdot_integrated = Mdot_integrated,
        normalization_error = normalization_error,
        minimum_relative_delta_j = minimum(relative_delta_j),
        solver_success = solver_success,
        landing_nonnegative = landing_nonnegative,
        inward_flow = inward_flow,
        lower_angular_momentum = lower_angular_momentum,
        valid = valid,
    )
end

leroy = solve_forward(observations.leroy_rotation)
sparc = solve_forward(observations.sparc_rotation)

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
    F_R_leroy_Msun_yr = leroy.F,
    F_R_sparc_fit_Msun_yr = sparc.F,
    Sigmadot_land_leroy_Msun_yr_kpc2 = leroy.Sigmadot_land,
    Sigmadot_land_sparc_fit_Msun_yr_kpc2 = sparc.Sigmadot_land,
    cumulative_landing_leroy_Msun_yr = leroy.cumulative_landing,
    cumulative_landing_sparc_fit_Msun_yr = sparc.cumulative_landing,
    v_R_leroy_kms = leroy.v_R,
    v_R_sparc_fit_kms = sparc.v_R,
    Mdot_acc_leroy_Msun_yr = leroy.Mdot_acc,
    Mdot_acc_sparc_fit_Msun_yr = sparc.Mdot_acc,
    v_c_leroy_kms = leroy.v_c,
    v_c_sparc_fit_kms = sparc.v_c,
    j_disk_leroy_kpc_kms = leroy.j_disk,
    j_disk_sparc_fit_kpc_kms = sparc.j_disk,
    j_CGM_leroy_kpc_kms = leroy.j_CGM,
    j_CGM_sparc_fit_kpc_kms = sparc.j_CGM,
    j_land_leroy_kpc_kms = leroy.j_land,
    j_land_sparc_fit_kpc_kms = sparc.j_land,
    delta_j_leroy_kpc_kms = leroy.delta_j,
    delta_j_sparc_fit_kpc_kms = sparc.delta_j,
    j_land_over_j_disk_leroy = leroy.j_land_ratio,
    j_land_over_j_disk_sparc_fit = sparc.j_land_ratio,
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
    mu = [mu],
    beta = [beta],
    R_nucl_kpc = [R_nucl],
    R_out_kpc = [R_out],
    Mdot_land_target_Msun_yr = [Mdot_land_target],
    radial_H2_available = [observations.has_radial_H2],
    H2_treatment = [observations.H2_treatment],
))

function summary_row(source, result)
    return (
        source = source,
        F_inner_Msun_yr = result.F0,
        F_outer_Msun_yr = result.F[end],
        Mdot_land_target_Msun_yr = Mdot_land_target,
        Mdot_land_integrated_Msun_yr = result.Mdot_integrated,
        normalization_relative_error = result.normalization_error,
        min_Sigmadot_land_Msun_yr_kpc2 = minimum(result.Sigmadot_land),
        max_abs_v_R_kms = maximum(abs.(result.v_R)),
        Mdot_acc_inner_Msun_yr = result.Mdot_acc[1],
        Mdot_acc_outer_Msun_yr = result.Mdot_acc[end],
        minimum_relative_abs_delta_j = result.minimum_relative_delta_j,
        landing_nonnegative = result.landing_nonnegative,
        inward_flow = result.inward_flow,
        lower_angular_momentum = result.lower_angular_momentum,
        solver_success = result.solver_success,
        valid = result.valid,
    )
end

summary = DataFrame([
    summary_row("Leroy/THINGS", leroy),
    summary_row("SPARC corrected fit", sparc),
])
CSV.write(joinpath(output_dir, "summary.csv"), summary)
CSV.write(joinpath(output_dir, "sparc_corrected.csv"), observations.sparc)
CSV.write(joinpath(output_dir, "leroy_profiles_used.csv"), observations.radial)

CSV.write(joinpath(output_dir, "rotation_fits.csv"), DataFrame(
    source = ["Leroy/THINGS", "SPARC corrected fit"],
    Vflat_kms = [observations.leroy_rotation.Vflat, observations.sparc_rotation.Vflat],
    lflat_kpc = [observations.leroy_rotation.lflat, observations.sparc_rotation.lflat],
    chi2 = [missing, observations.sparc_rotation.chi2],
    dof = [missing, observations.sparc_rotation.dof],
    reduced_chi2 = [missing, observations.sparc_rotation.reduced_chi2],
))

# Rotation-curve comparison over the full corrected SPARC range
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

# Predicted landing and observed star-formation surface rates
all_rates = vcat(Sigmadot_star_grid, leroy.Sigmadot_land, sparc.Sigmadot_land)
rate_scale = minimum(all_rates) > 0 ? log10 : identity
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"R\;[\mathrm{kpc}]",
    ylabel=L"\dot{\Sigma}\;[M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}]",
    yscale=rate_scale)
lines!(ax, R_grid, Sigmadot_star_grid, color=:black, linewidth=2,
    label=L"\dot{\Sigma}_{\star,\mathrm{obs}}")
lines!(ax, R_grid, leroy.Sigmadot_land, linewidth=2, label="landing: Leroy/THINGS")
lines!(ax, R_grid, sparc.Sigmadot_land, linewidth=2, linestyle=:dash,
    label="landing: SPARC fit")
axislegend(ax)
save(joinpath(output_dir, "surface_rates.pdf"), fig)

# Cumulative landing-rate normalization
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"R\;[\mathrm{kpc}]",
    ylabel=L"\dot{M}_{\mathrm{land}}(<R)\;[M_\odot\,\mathrm{yr}^{-1}]")
lines!(ax, R_grid, leroy.cumulative_landing, linewidth=2, label="Leroy/THINGS")
lines!(ax, R_grid, sparc.cumulative_landing, linewidth=2, linestyle=:dash,
    label="SPARC fit")
hlines!(ax, [Mdot_land_target], color=:gray50, linestyle=:dot, label="target")
axislegend(ax)
save(joinpath(output_dir, "cumulative_landing_rate.pdf"), fig)

# Predicted radial velocities
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"R\;[\mathrm{kpc}]", ylabel=L"v_R\;[\mathrm{km\,s^{-1}}]")
lines!(ax, R_grid, leroy.v_R, linewidth=2, label="Leroy/THINGS")
lines!(ax, R_grid, sparc.v_R, linewidth=2, linestyle=:dash, label="SPARC fit")
hlines!(ax, [0.0], color=:gray50, linestyle=:dot)
axislegend(ax)
save(joinpath(output_dir, "radial_velocity.pdf"), fig)

# Inward-positive mass accretion rates
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"R\;[\mathrm{kpc}]",
    ylabel=L"\dot{M}_{\mathrm{acc}}\;[M_\odot\,\mathrm{yr}^{-1}]")
lines!(ax, R_grid, leroy.Mdot_acc, linewidth=2, label="Leroy/THINGS")
lines!(ax, R_grid, sparc.Mdot_acc, linewidth=2, linestyle=:dash, label="SPARC fit")
axislegend(ax)
save(joinpath(output_dir, "accretion_rate.pdf"), fig)

# Prescribed angular momentum after fountain/CGM mixing
fig = Figure()
ax = Axis(fig[1, 1], xlabel=L"R\;[\mathrm{kpc}]", ylabel=L"j/j_{\mathrm{disk}}")
lines!(ax, R_grid, leroy.j_land_ratio, linewidth=2, label="landing: Leroy/THINGS")
lines!(ax, R_grid, sparc.j_land_ratio, linewidth=2, linestyle=:dash,
    label="landing: SPARC fit")
hlines!(ax, [beta], color=:gray50, linestyle=:dot, label=L"j_{\mathrm{CGM}}/j_{\mathrm{disk}}=\beta")
hlines!(ax, [1.0], color=:black, linestyle=:dashdot)
axislegend(ax)
save(joinpath(output_dir, "angular_momentum_ratio.pdf"), fig)

println("Galaxy:                  $(observations.galaxy)")
println("Adopted D, i:            $(geometry.D_adopted_Mpc) Mpc, $(geometry.inclination_adopted_deg) deg")
println("SPARC R, V scale:        $(round(geometry.distance_scale, digits=4)), $(round(geometry.velocity_scale, digits=4))")
println("Model radial range:      $R_nucl -- $R_out kpc")
println("mu, beta:                $mu, $beta")
println("Landing-rate target:     $Mdot_land_target Msun/yr")
println("H2 treatment:            $(observations.H2_treatment)")
println("Leroy integrated/valid:  $(round(leroy.Mdot_integrated, sigdigits=6)) Msun/yr / $(leroy.valid)")
println("SPARC integrated/valid:  $(round(sparc.Mdot_integrated, sigdigits=6)) Msun/yr / $(sparc.valid)")
println("Leroy max |v_R|:         $(round(maximum(abs.(leroy.v_R)), digits=3)) km/s")
println("SPARC max |v_R|:         $(round(maximum(abs.(sparc.v_R)), digits=3)) km/s")
println("Results written to:      $(normpath(output_dir))")
