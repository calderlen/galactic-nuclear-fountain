module GalacticWind

using QuadGK
import DifferentialEquations as DE

# (1) disk gas models 

function Sigma_g(R, R_g, sigma_g0)
    # exponential gas surface density profile
    return sigma_g0 * exp(-R / R_g)
end


function Sigma_g0(M_g, R_g, R_nucl, R_out)
    # normalization of gas surface density profile (!) this implies that M_g is the gas mass betweeen R_nucl and R_out, not the total gas mass of the disk
    F(R) = (1+R/R_g) * exp(-R/R_g)
    return M_g / (2*pi*R_g^2 * (F(R_nucl) - F(R_out)))
end


function Sigma_star_ks(R, R_g, sigma_g0, A, N)
    # Kennicutt-Schmidt SFR surface density
    return A * Sigma_g(R, R_g, sigma_g0)^N
end


# (3) CGM processing models

function Mdot_land(Mdot_nucl, f, n)
    # mass flux of incident fountain gas
    return f * n * Mdot_nucl
end


# (4) landing models

function Sigmadot_land(R, Mdot_land, R_nucl, R_out)
    # mass deposition rate per unit area of incident fountain gas
    return Mdot_land / (2 * pi * R^2 * log(R_out / R_nucl))
end


function v_R(R, R_g, sigma_g0, A, N, Mdot_land, R_nucl, R_out)
    # radial velocity of disk gas induced by mass and angular momnetum deposition of incident fountain gas
    integrand(r) = r * (Sigmadot_land(r, Mdot_land, R_nucl, R_out) - Sigma_star_ks(r, R_g, sigma_g0, A, N))

    integral, error = quadgk(integrand, R_out, R)

    return integral / (R*Sigma_g(R, R_g, sigma_g0))
end


# (5) disk response models

function j_land(R, R_g, sigma_g0, A, N, Mdot_land, R_nucl, R_out, v_c)
    # required specific angular momentum of incident fountain gas
    return v_c*(R + Sigma_g(R, R_g, sigma_g0) * v_R(R, R_g, sigma_g0, A, N, Mdot_land, R_nucl, R_out) / Sigmadot_land(R, Mdot_land, R_nucl, R_out))
end


function j_rotcurve_flat(R, v_c)
    # flat rotation curve
    return v_c * R
end

function j_ratio_flat(R, R_g, sigma_g0, A, N, Mdot_land, R_nucl, R_out, v_c)
        # ratio of specific angular momentum of incident fountain gas to that of circular orbit launched from the nuclear region radius
    return j_land(R, R_g, sigma_g0, A, N, Mdot_land, R_nucl, R_out, v_c) / j_rotcurve_flat(R_nucl, v_c)
end

function dZ_dR(Z, R, R_g, M_g, R_nucl, R_out, A, N, Mdot_nucl, f, n, Z_land, y)

    sigma_g0 = Sigma_g0(M_g, R_g, R_nucl, R_out)
    mdot_land = Mdot_land(Mdot_nucl, f, n)

    dZ_dR = (
        Sigmadot_land(R, mdot_land, R_nucl, R_out) * (Z_land - Z)
        +
        y * Sigma_star_ks(R, R_g, sigma_g0, A, N)
    ) / (
        Sigma_g(R, R_g, sigma_g0)
        *
        v_R(R, R_g, sigma_g0, A, N, mdot_land, R_nucl, R_out)
    )

    return dZ_dR
end


function metallicity_ode(Z,p,R)

    return dZ_dR(Z, R, p.R_g, p.M_g, p.R_nucl, p.R_out, p.A, p.N, p.Mdot_nucl, p.f, p.n, p.Z_land, p.y)

end







end # module