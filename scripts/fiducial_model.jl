import GalacticWind
import DifferentialEquations as DE
using CairoMakie
using Unitful
using UnitfulAstro


# model parameters
R_g = 3.0u"kpc"
M_g = 1e9u"Msun"
R_nucl = 0.5u"kpc"
R_out = 15.0u"kpc"
Mdot_nucl = 1.0u"Msun/yr"
f = 0.8
n = 0.7
N = 7//5
A=(2.5e-4u"Msun/yr/kpc^2"/(1.0u"Msun/pc^2")^N)
Z_land = 0.02
y = 0.015
v_c = 200.0u"km/s"

# boundary conditions
Z0 = 0.02183
R_stop = R_nucl
R_start = 14.9u"kpc"


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


Sigmadot_star_grid = GalacticWind.Sigma_star_ks.(
    R_grid,
    R_g,
    sigma_g0,
    A,
    N
)

Sigmadot_land_grid = GalacticWind.Sigmadot_land.(
    R_grid,
    mdot_land,
    R_nucl,
    R_out
)

v_R_grid = GalacticWind.v_R.(
    R_grid,
    R_g,
    sigma_g0,
    A,
    N,
    mdot_land,
    R_nucl,
    R_out
)

j_ratio_grid = GalacticWind.j_ratio_flat.(
    R_grid,
    R_g,
    sigma_g0,
    A,
    N,
    mdot_land,
    R_nucl,
    R_out,
    v_c
)


Z_eq_grid = Z_land .+ y .* ustrip.(
    uconvert.(
        NoUnits,
        Sigmadot_star_grid ./ Sigmadot_land_grid
    )
)

v_R_grid = uconvert.(u"km/s", v_R_grid)

Sigmadot_star_grid = uconvert.(
    u"Msun/yr/kpc^2",
    Sigmadot_star_grid
)

Sigmadot_land_grid = uconvert.(
    u"Msun/yr/kpc^2",
    Sigmadot_land_grid
)

j_ratio_grid = ustrip.(
    uconvert.(
        NoUnits,
        j_ratio_grid
    )
)



output_dir = joinpath(@__DIR__, "..", "output")
mkpath(output_dir)



fig = Figure()

ax = Axis(
    fig[1, 1],
    xlabel = "R",
    ylabel = "Z"
)

lines!(ax, reverse(R_grid), reverse(Z_grid), label = "Z")
lines!(ax, reverse(R_grid), reverse(Z_eq_grid), label = "Z_eq")

axislegend(ax)

save(joinpath(output_dir, "fiducial_metallicity.pdf"), fig)



fig = Figure()

ax = Axis(
    fig[1, 1],
    xlabel = "R",
    ylabel = "v_R"
)

lines!(ax, reverse(R_grid), reverse(v_R_grid))

save(joinpath(output_dir, "fiducial_radial_velocity.pdf"), fig)



fig = Figure()

ax = Axis(
    fig[1, 1],
    xlabel = "R",
    ylabel = "mass deposition / formation rate per unit area",
    yscale = log10
)

lines!(ax, reverse(R_grid), reverse(Sigmadot_star_grid), label = "SFR")
lines!(ax, reverse(R_grid), reverse(Sigmadot_land_grid), label = "fountain landing")

axislegend(ax)

save(joinpath(output_dir, "fiducial_surface_rates.pdf"), fig)



fig = Figure()

ax = Axis(
    fig[1, 1],
    xlabel = "R",
    ylabel = "j_land / j_nucl"
)

lines!(ax, reverse(R_grid), reverse(j_ratio_grid))

hlines!(ax, [1.0], linestyle = :dash)

save(joinpath(output_dir, "fiducial_j_ratio.pdf"), fig)