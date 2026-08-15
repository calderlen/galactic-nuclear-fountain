import GalacticWind
import DifferentialEquations as DE
using CairoMakie
using CSV
using DataFrames
using LaTeXStrings
using QuadGK

# model parameters held fixed during the sweeps
R_g = 3.0 #kpc
M_g = 1e9 #Msun between R_nucl and R_out
R_nucl = 0.5 #kpc
R_out = 15.0 #kpc
Mdot_nucl = 1.0 #Msun/yr
f = 0.8
n = 0.7
N = 7//5
A = 2.5e-4 / 1e6^N #KS coefficient for Sigma_g in Msun/kpc^2
Z_nucl = 0.02
Z0 = 0.006
y = 0.015

mu_reference = 1.0
beta_reference = 0.75

mu_values = collect(range(0.25, 2.0, length=8))
beta_values = collect(range(0.4, 0.9, length=11))
Z_CGM_values = [0.0015, 0.003, 0.006]
representative_radii = [2.0, 8.0, 14.0] #kpc

R_grid = range(R_nucl, R_out, length=300)
sigma_g0 = GalacticWind.Sigma_g0(M_g, R_g, R_nucl, R_out)

output_dir = joinpath(@__DIR__, "..", "output", splitext(basename(@__FILE__))[1])
dynamics_dir = joinpath(output_dir, "dynamics")
chemistry_dir = joinpath(output_dir, "chemistry")
mkpath(dynamics_dir)
mkpath(chemistry_dir)


function total_landing_rate(sol)
    integral, error = quadgk(R -> R * sol(R), R_nucl, R_out)
    return 2*pi*integral
end

function solve_landing_profile(mu, beta)
    p = (
        R_g = R_g,
        sigma_g0 = sigma_g0,
        A = A,
        N = N,
        R_nucl = R_nucl,
        mu = mu,
        beta = beta
    )

    solve_for(Sigmadot_land_0) = DE.solve(
        DE.ODEProblem(GalacticWind.Sigmadot_land_ode, Sigmadot_land_0, (R_nucl, R_out), p),
        DE.Tsit5()
    )

    Sigmadot_land_0_a = 0.0
    Sigmadot_land_0_b = 1.0
    sol_a = solve_for(Sigmadot_land_0_a)
    sol_b = solve_for(Sigmadot_land_0_b)
    mdot_a = total_landing_rate(sol_a)
    mdot_b = total_landing_rate(sol_b)
    mdot_target = GalacticWind.Mdot_land_mixing(Mdot_nucl, f, n, mu)

    Sigmadot_land_0 = Sigmadot_land_0_a + (mdot_target - mdot_a) * (Sigmadot_land_0_b - Sigmadot_land_0_a) / (mdot_b - mdot_a)
    sol = solve_for(Sigmadot_land_0)

    return (
        sol = sol,
        Sigmadot_land_0 = Sigmadot_land_0,
        mdot_target = mdot_target,
        mdot_integrated = total_landing_rate(sol),
        solver_success = DE.successful_retcode(sol_a) && DE.successful_retcode(sol_b) && DE.successful_retcode(sol)
    )
end

function solve_metallicity_profile(landing_sol, mu, beta, Z_CGM)
    p_Z = (
        R_g = R_g,
        sigma_g0 = sigma_g0,
        A = A,
        N = N,
        Sigmadot_land = landing_sol,
        R_nucl = R_nucl,
        mu = mu,
        beta = beta,
        Z_nucl = Z_nucl,
        Z_CGM = Z_CGM,
        y = y
    )

    prob = DE.ODEProblem(GalacticWind.metallicity_ode_mixing, Z0, (R_out, R_nucl), p_Z)
    return DE.solve(prob, DE.Tsit5())
end

function linear_slope(x, y)
    x_mean = sum(x) / length(x)
    y_mean = sum(y) / length(y)
    return sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
end


# dynamics sweep over mu and beta
n_mu = length(mu_values)
n_beta = length(beta_values)
Sigmadot_land_0_matrix = fill(NaN, n_mu, n_beta)
max_v_R_matrix = fill(NaN, n_mu, n_beta)
Mdot_acc_inner_matrix = fill(NaN, n_mu, n_beta)
valid_matrix = zeros(n_mu, n_beta)
dynamics_rows = NamedTuple[]
landing_solutions = Dict{Tuple{Float64,Float64},Any}()

for (i, mu) in enumerate(mu_values), (j, beta) in enumerate(beta_values)
    try
        landing = solve_landing_profile(mu, beta)
        landing_solutions[(mu, beta)] = landing.sol

        Sigmadot_land_grid = landing.sol.(R_grid)
        D_grid = R_nucl .+ R_grid .* (mu*(beta-1)-1)
        v_R_grid = GalacticWind.v_R_forward.(R_grid, Sigmadot_land_grid, R_g, sigma_g0, R_nucl, mu, beta)
        v_R_grid .*= 9.7779222168e8 #kpc/yr to km/s

        v_R_representative = [
            GalacticWind.v_R_forward(R, landing.sol(R), R_g, sigma_g0, R_nucl, mu, beta) * 9.7779222168e8
            for R in representative_radii
        ]

        Mdot_acc_inner = GalacticWind.Mdot_acc_forward(R_nucl, landing.sol(R_nucl), R_nucl, mu, beta)
        landing_positive = minimum(Sigmadot_land_grid) > 0
        D_negative = maximum(D_grid) < 0
        normalization_relative_error = abs(landing.mdot_integrated - landing.mdot_target) / landing.mdot_target
        valid = landing.solver_success && landing_positive && D_negative && normalization_relative_error < 1e-5

        Sigmadot_land_0_matrix[i, j] = landing.Sigmadot_land_0
        max_v_R_matrix[i, j] = maximum(abs.(v_R_grid))
        Mdot_acc_inner_matrix[i, j] = Mdot_acc_inner
        valid_matrix[i, j] = valid

        push!(dynamics_rows, (
            mu = mu,
            beta = beta,
            Sigmadot_land_0_msun_yr_kpc2 = landing.Sigmadot_land_0,
            v_R_2kpc_km_s = v_R_representative[1],
            v_R_8kpc_km_s = v_R_representative[2],
            v_R_14kpc_km_s = v_R_representative[3],
            max_abs_v_R_km_s = maximum(abs.(v_R_grid)),
            Mdot_acc_inner_msun_yr = Mdot_acc_inner,
            min_Sigmadot_land_msun_yr_kpc2 = minimum(Sigmadot_land_grid),
            max_D_kpc = maximum(D_grid),
            normalization_relative_error = normalization_relative_error,
            landing_positive = landing_positive,
            D_negative = D_negative,
            solver_success = landing.solver_success,
            valid = valid,
            error = ""
        ))
    catch err
        push!(dynamics_rows, (
            mu = mu,
            beta = beta,
            Sigmadot_land_0_msun_yr_kpc2 = NaN,
            v_R_2kpc_km_s = NaN,
            v_R_8kpc_km_s = NaN,
            v_R_14kpc_km_s = NaN,
            max_abs_v_R_km_s = NaN,
            Mdot_acc_inner_msun_yr = NaN,
            min_Sigmadot_land_msun_yr_kpc2 = NaN,
            max_D_kpc = NaN,
            normalization_relative_error = NaN,
            landing_positive = false,
            D_negative = false,
            solver_success = false,
            valid = false,
            error = sprint(showerror, err)
        ))
    end
end

CSV.write(joinpath(dynamics_dir, "results.csv"), DataFrame(dynamics_rows))


# dynamics heatmaps
fig = Figure(size = (1100, 800))

ax = Axis(fig[1, 1], xlabel = L"\mu", ylabel = L"\beta", title = L"\dot{\Sigma}_{\mathrm{land}}(R_{\mathrm{nucl}})")
hm = heatmap!(ax, mu_values, beta_values, Sigmadot_land_0_matrix)
Colorbar(fig[1, 2], hm, label = L"M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}")

ax = Axis(fig[1, 3], xlabel = L"\mu", ylabel = L"\beta", title = L"\max |v_R|")
hm = heatmap!(ax, mu_values, beta_values, max_v_R_matrix)
Colorbar(fig[1, 4], hm, label = L"\mathrm{km\,s^{-1}}")

ax = Axis(fig[2, 1], xlabel = L"\mu", ylabel = L"\beta", title = L"\dot{M}_{\mathrm{acc}}(R_{\mathrm{nucl}})")
hm = heatmap!(ax, mu_values, beta_values, Mdot_acc_inner_matrix)
Colorbar(fig[2, 2], hm, label = L"M_\odot\,\mathrm{yr}^{-1}")

ax = Axis(fig[2, 3], xlabel = L"\mu", ylabel = L"\beta", title = "physical and numerical validity")
hm = heatmap!(ax, mu_values, beta_values, valid_matrix, colorrange = (0, 1), colormap = [:firebrick, :seagreen])
Colorbar(fig[2, 4], hm, ticks = ([0, 1], ["invalid", "valid"]))

save(joinpath(dynamics_dir, "heatmaps.pdf"), fig)


# representative dynamics profiles at fixed beta
fig = Figure(size = (1000, 400))
ax_land = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"\dot{\Sigma}_{\mathrm{land}}\;[M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}]", yscale = log10)
ax_v_R = Axis(fig[1, 2], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"v_R\;[\mathrm{km\,s^{-1}}]")

for mu in [0.25, 1.0, 2.0]
    sol = landing_solutions[(mu, beta_reference)]
    Sigmadot_land_grid = sol.(R_grid)
    v_R_grid = GalacticWind.v_R_forward.(R_grid, Sigmadot_land_grid, R_g, sigma_g0, R_nucl, mu, beta_reference) .* 9.7779222168e8
    lines!(ax_land, R_grid, Sigmadot_land_grid, label = "μ = $mu")
    lines!(ax_v_R, R_grid, v_R_grid, label = "μ = $mu")
end

axislegend(ax_land)
axislegend(ax_v_R)
save(joinpath(dynamics_dir, "profiles_varying_mu.pdf"), fig)


# representative dynamics profiles at fixed mu
fig = Figure(size = (1000, 400))
ax_land = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"\dot{\Sigma}_{\mathrm{land}}\;[M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}]", yscale = log10)
ax_v_R = Axis(fig[1, 2], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"v_R\;[\mathrm{km\,s^{-1}}]")

for beta in [0.4, 0.75, 0.9]
    sol = landing_solutions[(mu_reference, beta)]
    Sigmadot_land_grid = sol.(R_grid)
    v_R_grid = GalacticWind.v_R_forward.(R_grid, Sigmadot_land_grid, R_g, sigma_g0, R_nucl, mu_reference, beta) .* 9.7779222168e8
    lines!(ax_land, R_grid, Sigmadot_land_grid, label = "β = $beta")
    lines!(ax_v_R, R_grid, v_R_grid, label = "β = $beta")
end

axislegend(ax_land)
axislegend(ax_v_R)
save(joinpath(dynamics_dir, "profiles_varying_beta.pdf"), fig)


# chemistry sweep over Z_CGM at the reference mu and beta
landing_reference = solve_landing_profile(mu_reference, beta_reference)
Sigmadot_land_reference_grid = landing_reference.sol.(R_grid)
Sigmadot_star_grid = GalacticWind.Sigma_star_ks.(R_grid, R_g, sigma_g0, A, N)
gradient_mask = R_grid .>= 5.0
chemistry_rows = NamedTuple[]
chemistry_profiles = Dict{Float64,Any}()

for Z_CGM in Z_CGM_values
    Z_sol = solve_metallicity_profile(landing_reference.sol, mu_reference, beta_reference, Z_CGM)
    Z_grid = Z_sol.(R_grid)
    Z_land = GalacticWind.Z_land_mixing(Z_nucl, Z_CGM, mu_reference)
    Z_eq_grid = Z_land .+ y .* Sigmadot_star_grid ./ Sigmadot_land_reference_grid
    gradient = linear_slope(R_grid[gradient_mask], log10.(Z_grid[gradient_mask]))

    chemistry_profiles[Z_CGM] = (Z = Z_grid, Z_eq = Z_eq_grid)
    push!(chemistry_rows, (
        Z_CGM = Z_CGM,
        Z_land = Z_land,
        metallicity_gradient_5_15_dex_kpc = gradient,
        peak_Z = maximum(Z_grid),
        inner_Z = Z_grid[1],
        outer_Z = Z_grid[end],
        solver_success = DE.successful_retcode(Z_sol)
    ))
end

chemistry_results = DataFrame(chemistry_rows)
CSV.write(joinpath(chemistry_dir, "results.csv"), chemistry_results)


# chemistry profiles
colors = [:dodgerblue, :darkorange, :seagreen]
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"Z")

for (i, Z_CGM) in enumerate(Z_CGM_values)
    profile = chemistry_profiles[Z_CGM]
    lines!(ax, R_grid, profile.Z, color = colors[i], label = "Z, Z_CGM = $Z_CGM")
    lines!(ax, R_grid, profile.Z_eq, color = colors[i], linestyle = :dash, label = "Z_eq, Z_CGM = $Z_CGM")
end

axislegend(ax)
save(joinpath(chemistry_dir, "metallicity_profiles.pdf"), fig)


# chemistry summary
fig = Figure(size = (900, 400))
ax = Axis(fig[1, 1], xlabel = L"Z_{\mathrm{CGM}}", ylabel = L"\mathrm{d}\log_{10}Z/\mathrm{d}R\;[\mathrm{dex\,kpc^{-1}}]")
scatterlines!(ax, chemistry_results.Z_CGM, chemistry_results.metallicity_gradient_5_15_dex_kpc)

ax = Axis(fig[1, 2], xlabel = L"Z_{\mathrm{CGM}}", ylabel = L"\max Z")
scatterlines!(ax, chemistry_results.Z_CGM, chemistry_results.peak_Z)

save(joinpath(chemistry_dir, "summary.pdf"), fig)
