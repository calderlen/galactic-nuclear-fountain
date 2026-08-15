import GalacticWind
import DifferentialEquations as DE
using CairoMakie
using LaTeXStrings

# given an assumed landing profile, what j_land is req'd?

# model parameters
R_g = 3.0 #kpc
M_g = 1e9 #Msun between R_nucl and R_out
R_nucl = 0.5 #kpc    
R_out = 15.0 #kpc
Mdot_nucl = 1.0 #Msun/yr
f = 0.8                            
n = 0.7                            
N = 7//5                         
A = 2.5e-4 / 1e6^N #KS coefficient for Sigma_g in Msun/kpc^2
Z_land = 0.02                     
y = 0.015                       
v_c = 200.0 #km/s

# boundary conditions
Z0 = 0.02183
R_stop = R_nucl
R_start = 14.9 #kpc


p = (
    R_g = R_g,
    M_g = M_g,
    R_nucl = R_nucl,
    R_out = R_out,
    A = A,
    N = N,
    Mdot_nucl = Mdot_nucl,
    f = f,
    n = n,
    Z_land = Z_land,
    y = y
)



prob = DE.ODEProblem(GalacticWind.metallicity_ode, Z0, (R_start,R_stop), p)
sol = DE.solve(prob, DE.Tsit5())

R_grid = range(R_start, R_stop, length=100)
Z_grid = sol.(R_grid)


sigma_g0 = GalacticWind.Sigma_g0(M_g, R_g, R_nucl, R_out)
mdot_land = GalacticWind.Mdot_land(Mdot_nucl, f, n)
Sigmadot_star_grid = GalacticWind.Sigma_star_ks.(R_grid, R_g, sigma_g0, A, N)
Sigmadot_land_grid = GalacticWind.Sigmadot_land.(R_grid, mdot_land, R_nucl, R_out)
v_R_grid = GalacticWind.v_R.(R_grid, R_g, sigma_g0, A, N,mdot_land, R_nucl, R_out)
j_ratio_grid = GalacticWind.j_ratio_flat.( R_grid, R_g, sigma_g0, A, N, mdot_land,R_nucl, R_out, v_c)
Z_eq_grid = Z_land .+ y .* (Sigmadot_star_grid ./ Sigmadot_land_grid)
v_R_grid .*= 9.7779222168e8 # kpc/yr to km/s

output_dir = joinpath(@__DIR__, "..", "output", splitext(basename(@__FILE__))[1])
mkpath(output_dir)

# Z and Z_eq vs. R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"Z")
lines!(ax, reverse(R_grid), reverse(Z_grid), label = L"Z")
lines!(ax, reverse(R_grid), reverse(Z_eq_grid), label = L"Z_{\mathrm{eq}}")
axislegend(ax)
save(joinpath(output_dir, "metallicity.pdf"), fig)

# v_R vs R.
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"v_R\;[\mathrm{km\,s^{-1}}]")
lines!(ax, reverse(R_grid), reverse(v_R_grid))
save(joinpath(output_dir, "radial_velocity.pdf"), fig)

# surface rates from SFR and fountain landing vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"\dot{\Sigma}\;[M_\odot\,\mathrm{yr}^{-1}\,\mathrm{kpc}^{-2}]", yscale = log10)
lines!(ax, reverse(R_grid), reverse(Sigmadot_star_grid), label = L"\dot{\Sigma}_\star")
lines!(ax, reverse(R_grid), reverse(Sigmadot_land_grid), label = L"\dot{\Sigma}_{\mathrm{land}}")
axislegend(ax)
save(joinpath(output_dir, "surface_rates.pdf"), fig)

# j_land & j_nucl vs R
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"R\;[\mathrm{kpc}]", ylabel = L"j_{\mathrm{land}}/j_{\mathrm{nucl}}")
lines!(ax, reverse(R_grid), reverse(j_ratio_grid))
hlines!(ax, [1.0], linestyle = :dash)
save(joinpath(output_dir, "angular_momentum_ratio.pdf"), fig)
