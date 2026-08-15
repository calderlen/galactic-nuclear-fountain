import GalacticWind
import DifferentialEquations as DE
using CairoMakie
using LaTeXStrings
using QuadGK

# shared model parameters
R_g = 3.0 #kpc
M_g = 1e9 #Msun between R_nucl and R_out
R_nucl = 0.5 #kpc
R_out = 15.0 #kpc
Mdot_nucl = 1.0 #Msun/yr
f = 0.8
n = 0.7
mu = 1.0
beta = 0.75
N = 7//5
A = 2.5e-4 / 1e6^N #KS coefficient for Sigma_g in Msun/kpc^2
v_c = 200.0 #km/s
Z_nucl = 0.02
Z_CGM = 0.003
y = 0.015
Z_land = GalacticWind.Z_land_mixing(Z_nucl, Z_CGM, mu)

# shared metallicity boundary condition
Z0 = 0.006
R_Z_start = 14.9 #kpc; avoids the inverse-model v_R = 0 boundary at R_out
R_Z_stop = R_nucl


sigma_g0 = GalacticWind.Sigma_g0(M_g, R_g, R_nucl, R_out)
mdot_land = GalacticWind.Mdot_land_mixing(Mdot_nucl, f, n, mu)

p_landing = (
    R_g = R_g,
    sigma_g0 = sigma_g0,
    A = A,
    N = N,
    R_nucl = R_nucl,
    mu = mu,
    beta = beta
)


function solve_landing(Sigmadot_land_0)
    prob = DE.ODEProblem(GalacticWind.Sigmadot_land_ode, Sigmadot_land_0, (R_nucl, R_out), p_landing)
    return DE.solve(prob, DE.Tsit5())
end

function total_landing_rate(sol)
    integral, error = quadgk(R -> R * sol(R), R_nucl, R_out)
    return 2*pi*integral
end


Sigmadot_land_0_a = 0.0
Sigmadot_land_0_b = 1.0
landing_sol_a = solve_landing(Sigmadot_land_0_a)
landing_sol_b = solve_landing(Sigmadot_land_0_b)
mdot_land_a = total_landing_rate(landing_sol_a)
mdot_land_b = total_landing_rate(landing_sol_b)
Sigmadot_land_0 = Sigmadot_land_0_a + (mdot_land - mdot_land_a) * (Sigmadot_land_0_b - Sigmadot_land_0_a) / (mdot_land_b - mdot_land_a)
landing_sol = solve_landing(Sigmadot_land_0)


p_Z_inverse = (
    R_g = R_g,
    sigma_g0 = sigma_g0,
    R_nucl = R_nucl,
    R_out = R_out,
    A = A,
    N = N,
    mdot_land = mdot_land,
    Z_land = Z_land,
    y = y
)

p_Z_mixing = (
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

function metallicity_ode_inverse(Z, p, R)
    Sigmadot_land = GalacticWind.Sigmadot_land(R, p.mdot_land, p.R_nucl, p.R_out)
    Sigmadot_star = GalacticWind.Sigma_star_ks(R, p.R_g, p.sigma_g0, p.A, p.N)
    v_R = GalacticWind.v_R(R, p.R_g, p.sigma_g0, p.A, p.N, p.mdot_land, p.R_nucl, p.R_out)
    return (Sigmadot_land * (p.Z_land - Z) + p.y * Sigmadot_star) / (GalacticWind.Sigma_g(R, p.R_g, p.sigma_g0) * v_R)
end

Z_prob_inverse = DE.ODEProblem(metallicity_ode_inverse, Z0, (R_Z_start, R_Z_stop), p_Z_inverse)
Z_prob_mixing = DE.ODEProblem(GalacticWind.metallicity_ode_mixing, Z0, (R_Z_start, R_Z_stop), p_Z_mixing)
Z_sol_inverse = DE.solve(Z_prob_inverse, DE.Tsit5())
Z_sol_mixing = DE.solve(Z_prob_mixing, DE.Tsit5())


R_grid = range(R_nucl, R_Z_start, length=200)
Sigma_g_grid = GalacticWind.Sigma_g.(R_grid, R_g, sigma_g0)
Sigmadot_star_grid = GalacticWind.Sigma_star_ks.(R_grid, R_g, sigma_g0, A, N)

Sigmadot_land_inverse_grid = GalacticWind.Sigmadot_land.(R_grid, mdot_land, R_nucl, R_out)
Sigmadot_land_mixing_grid = landing_sol.(R_grid)

Mdot_land_cumulative_inverse_grid = [2*pi*quadgk(r -> r * GalacticWind.Sigmadot_land(r, mdot_land, R_nucl, R_out), R_nucl, R)[1] for R in R_grid]
Mdot_land_cumulative_mixing_grid = [2*pi*quadgk(r -> r * landing_sol(r), R_nucl, R)[1] for R in R_grid]

v_R_inverse_grid = GalacticWind.v_R.(R_grid, R_g, sigma_g0, A, N, mdot_land, R_nucl, R_out)
v_R_mixing_grid = GalacticWind.v_R_forward.(R_grid, Sigmadot_land_mixing_grid, R_g, sigma_g0, R_nucl, mu, beta)

Mdot_acc_inverse_grid = -2*pi .* R_grid .* Sigma_g_grid .* v_R_inverse_grid
Mdot_acc_mixing_grid = GalacticWind.Mdot_acc_forward.(R_grid, Sigmadot_land_mixing_grid, R_nucl, mu, beta)

j_disk_grid = GalacticWind.j_rotcurve_flat.(R_grid, v_c)
j_land_inverse_grid = GalacticWind.j_land.(R_grid, R_g, sigma_g0, A, N, mdot_land, R_nucl, R_out, v_c)
j_land_mixing_grid = GalacticWind.j_land_mixing.(R_grid, R_nucl, mu, beta, v_c)
j_ratio_inverse_grid = j_land_inverse_grid ./ j_disk_grid
j_ratio_mixing_grid = j_land_mixing_grid ./ j_disk_grid

Z_inverse_grid = Z_sol_inverse.(R_grid)
Z_mixing_grid = Z_sol_mixing.(R_grid)
Z_eq_inverse_grid = Z_land .+ y .* Sigmadot_star_grid ./ Sigmadot_land_inverse_grid
Z_eq_mixing_grid = Z_land .+ y .* Sigmadot_star_grid ./ Sigmadot_land_mixing_grid

t_inflow_inverse_grid = R_grid ./ abs.(v_R_inverse_grid) ./ 1e9 #Gyr
t_inflow_mixing_grid = R_grid ./ abs.(v_R_mixing_grid) ./ 1e9 #Gyr
t_depletion_grid = Sigma_g_grid ./ Sigmadot_star_grid ./ 1e9 #Gyr

v_R_inverse_grid .*= 9.7779222168e8 #kpc/yr to km/s
v_R_mixing_grid .*= 9.7779222168e8 #kpc/yr to km/s


output_dir = joinpath(@__DIR__, "..", "output", splitext(basename(@__FILE__))[1])
mkpath(output_dir)

mixing_color = :darkorange
inverse_color = :dodgerblue

# Z and Z_eq vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"Z")
lines!(ax, R_grid, Z_inverse_grid, color = inverse_color, label = L"Z\;\mathrm{(inverse)}")
lines!(ax, R_grid, Z_eq_inverse_grid, color = inverse_color, linestyle = :dash, label = L"Z_{\mathrm{eq}}\;\mathrm{(inverse)}")
lines!(ax, R_grid, Z_mixing_grid, color = mixing_color, label = L"Z\;\mathrm{(mixing)}")
lines!(ax, R_grid, Z_eq_mixing_grid, color = mixing_color, linestyle = :dash, label = L"Z_{\mathrm{eq}}\;\mathrm{(mixing)}")
axislegend(ax)
save(joinpath(output_dir, "metallicity.pdf"), fig)

# normalized metallicity shapes vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"Z(R)/Z(R_{\mathrm{BC}})")
lines!(ax, R_grid, Z_inverse_grid ./ Z_inverse_grid[end], color = inverse_color, label = "inverse")
lines!(ax, R_grid, Z_mixing_grid ./ Z_mixing_grid[end], color = mixing_color, label = "mixing")
axislegend(ax)
save(joinpath(output_dir, "metallicity_normalized.pdf"), fig)

# radial velocity vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"v_R\;[\mathrm{km\,s^{-1}}]")
lines!(ax, R_grid, v_R_inverse_grid, color = inverse_color, label = "inverse")
lines!(ax, R_grid, v_R_mixing_grid, color = mixing_color, label = "mixing")
axislegend(ax)
save(joinpath(output_dir, "radial_velocity.pdf"), fig)

# surface rates vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"\dot{\Sigma}\;[M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}]", yscale = log10)
lines!(ax, R_grid, Sigmadot_star_grid, color = :black, label = L"\dot{\Sigma}_\star")
lines!(ax, R_grid, Sigmadot_land_inverse_grid, color = inverse_color, label = L"\dot{\Sigma}_{\mathrm{land}}\;\mathrm{(inverse)}")
lines!(ax, R_grid, Sigmadot_land_mixing_grid, color = mixing_color, label = L"\dot{\Sigma}_{\mathrm{land}}\;\mathrm{(mixing)}")
axislegend(ax)
save(joinpath(output_dir, "surface_rates.pdf"), fig)

# cumulative landing rate vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"\dot{M}_{\mathrm{land}}(<R)\;[M_\odot\,\mathrm{yr}^{-1}]")
lines!(ax, R_grid, Mdot_land_cumulative_inverse_grid, color = inverse_color, label = "inverse")
lines!(ax, R_grid, Mdot_land_cumulative_mixing_grid, color = mixing_color, label = "mixing")
hlines!(ax, [mdot_land], color = :black, linestyle = :dash, label = "shared target")
axislegend(ax)
save(joinpath(output_dir, "cumulative_landing_rate.pdf"), fig)

# accretion rate vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"\dot{M}_{\mathrm{acc}}\;[M_\odot\,\mathrm{yr}^{-1}]")
lines!(ax, R_grid, Mdot_acc_inverse_grid, color = inverse_color, label = "inverse")
lines!(ax, R_grid, Mdot_acc_mixing_grid, color = mixing_color, label = "mixing")
axislegend(ax)
save(joinpath(output_dir, "accretion_rate.pdf"), fig)

# angular momentum ratio vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"j_{\mathrm{land}}/j_{\mathrm{disk}}")
lines!(ax, R_grid, j_ratio_inverse_grid, color = inverse_color, label = "inverse")
lines!(ax, R_grid, j_ratio_mixing_grid, color = mixing_color, label = "mixing")
hlines!(ax, [1.0], linestyle = :dash, color = :black)
axislegend(ax)
save(joinpath(output_dir, "angular_momentum_ratio.pdf"), fig)

# inflow and depletion timescales vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"t\;[\mathrm{Gyr}]", yscale = log10)
lines!(ax, R_grid, t_inflow_inverse_grid, color = inverse_color, label = L"t_{\mathrm{inflow}}\;\mathrm{(inverse)}")
lines!(ax, R_grid, t_inflow_mixing_grid, color = mixing_color, label = L"t_{\mathrm{inflow}}\;\mathrm{(mixing)}")
lines!(ax, R_grid, t_depletion_grid, color = :black, linestyle = :dash, label = L"t_{\mathrm{depletion}}")
axislegend(ax)
save(joinpath(output_dir, "timescales.pdf"), fig)
