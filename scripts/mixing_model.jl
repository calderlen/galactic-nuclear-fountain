import GalacticWind
import DifferentialEquations as DE
using CairoMakie
using LaTeXStrings
using QuadGK

# given a cgm mixing prescription for j_land, what landing profile and disk response are required?

# model parameters
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

# boundary conditions
R_start = R_nucl
R_stop = R_out
Z0 = 0.006
R_Z_start = R_out
R_Z_stop = R_nucl


sigma_g0 = GalacticWind.Sigma_g0(M_g, R_g, R_nucl, R_out)
mdot_land = GalacticWind.Mdot_land_mixing(Mdot_nucl, f, n, mu)

p = (
    R_g = R_g,
    sigma_g0 = sigma_g0,
    A = A,
    N = N,
    R_nucl = R_nucl,
    mu = mu,
    beta = beta
)


function solve_landing(Sigmadot_land_0)
    prob = DE.ODEProblem(GalacticWind.Sigmadot_land_ode, Sigmadot_land_0, (R_start, R_stop), p)
    return DE.solve(prob, DE.Tsit5())
end

function total_landing_rate(sol)
    integral, error = quadgk(R -> R * sol(R), R_nucl, R_out)
    return 2*pi*integral
end


Sigmadot_land_0_a = 0.0 #Msun/yr/kpc^2
Sigmadot_land_0_b = 1.0 #Msun/yr/kpc^2

landing_sol_a = solve_landing(Sigmadot_land_0_a)
landing_sol_b = solve_landing(Sigmadot_land_0_b)

mdot_land_a = total_landing_rate(landing_sol_a)
mdot_land_b = total_landing_rate(landing_sol_b)

Sigmadot_land_0 = Sigmadot_land_0_a + (mdot_land - mdot_land_a) * (Sigmadot_land_0_b - Sigmadot_land_0_a) / (mdot_land_b - mdot_land_a)
landing_sol = solve_landing(Sigmadot_land_0)

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

Z_prob = DE.ODEProblem(GalacticWind.metallicity_ode_mixing, Z0, (R_Z_start, R_Z_stop), p_Z)
Z_sol = DE.solve(Z_prob, DE.Tsit5())


R_grid = range(R_start, R_stop, length=100)

Sigmadot_star_grid = GalacticWind.Sigma_star_ks.(R_grid, R_g, sigma_g0, A, N)
Sigmadot_land_grid = landing_sol.(R_grid)
Mdot_land_cumulative_grid = [2*pi*quadgk(r -> r * landing_sol(r), R_nucl, R)[1] for R in R_grid]
Sigma_g_grid = GalacticWind.Sigma_g.(R_grid, R_g, sigma_g0)
Z_grid = Z_sol.(R_grid)
Z_land = GalacticWind.Z_land_mixing(Z_nucl, Z_CGM, mu)
Z_eq_grid = Z_land .+ y .* Sigmadot_star_grid ./ Sigmadot_land_grid
v_R_grid = GalacticWind.v_R_forward.(R_grid, Sigmadot_land_grid, R_g, sigma_g0, R_nucl, mu, beta)
Mdot_acc_grid = GalacticWind.Mdot_acc_forward.(R_grid, Sigmadot_land_grid, R_nucl, mu, beta)
t_inflow_grid = R_grid ./ abs.(v_R_grid) ./ 1e9 #Gyr
t_depletion_grid = Sigma_g_grid ./ Sigmadot_star_grid ./ 1e9 #Gyr

j_disk_grid = GalacticWind.j_rotcurve_flat.(R_grid, v_c)
j_CGM_grid = GalacticWind.j_cgm.(R_grid, beta, v_c)
j_land_grid = GalacticWind.j_land_mixing.(R_grid, R_nucl, mu, beta, v_c)
j_CGM_ratio_grid = j_CGM_grid ./ j_disk_grid
j_land_ratio_grid = j_land_grid ./ j_disk_grid

v_R_grid .*= 9.7779222168e8 #kpc/yr to km/s


output_dir = joinpath(@__DIR__, "..", "output", splitext(basename(@__FILE__))[1])
mkpath(output_dir)

# Z and Z_eq vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"Z")
lines!(ax, R_grid, Z_grid, label = L"Z")
lines!(ax, R_grid, Z_eq_grid, label = L"Z_{\mathrm{eq}}")
axislegend(ax)
save(joinpath(output_dir, "metallicity.pdf"), fig)

# v_R vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"v_R\;[\mathrm{km\,s^{-1}}]")
lines!(ax, R_grid, v_R_grid)
save(joinpath(output_dir, "radial_velocity.pdf"), fig)

# inflow and gas depletion timescales vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"t\;[\mathrm{Gyr}]", yscale = log10)
lines!(ax, R_grid, t_inflow_grid, label = L"t_{\mathrm{inflow}}")
lines!(ax, R_grid, t_depletion_grid, label = L"t_{\mathrm{depletion}}")
axislegend(ax)
save(joinpath(output_dir, "timescales.pdf"), fig)

# surface rates from SFR and fountain landing vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"\dot{\Sigma}\;[M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}]", yscale = log10)
lines!(ax, R_grid, Sigmadot_star_grid, label = L"\dot{\Sigma}_\star")
lines!(ax, R_grid, Sigmadot_land_grid, label = L"\dot{\Sigma}_{\mathrm{land}}")
axislegend(ax)
save(joinpath(output_dir, "surface_rates.pdf"), fig)

# cumulative landing rate vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"\dot{M}_{\mathrm{land}}(<R)\;[M_\odot\,\mathrm{yr}^{-1}]")
lines!(ax, R_grid, Mdot_land_cumulative_grid, label = L"\mathrm{integrated}")
hlines!(ax, [mdot_land], linestyle = :dash, color = :black, label = L"\mathrm{target}")
axislegend(ax)
save(joinpath(output_dir, "cumulative_landing_rate.pdf"), fig)

# accretion rate vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"\dot{M}_{\mathrm{acc}}\;[M_\odot\,\mathrm{yr}^{-1}]")
lines!(ax, R_grid, Mdot_acc_grid)
save(joinpath(output_dir, "accretion_rate.pdf"), fig)

# angular momentum ratios vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"j/j_{\mathrm{disk}}")
lines!(ax, R_grid, j_CGM_ratio_grid, label = L"j_{\mathrm{CGM}}/j_{\mathrm{disk}}")
lines!(ax, R_grid, j_land_ratio_grid, label = L"j_{\mathrm{land}}/j_{\mathrm{disk}}")
hlines!(ax, [1.0], linestyle = :dash, color = :black)
axislegend(ax)
save(joinpath(output_dir, "angular_momentum_ratio.pdf"), fig)
