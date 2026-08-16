function Sigmadot_land(R, Mdot_land, R_nucl, R_out)
    # mass deposition rate per unit area of incident fountain gas
    return Mdot_land / (2 * pi * R^2 * log(R_out / R_nucl))
end


function v_R(R, R_g, sigma_g0, A, N, Mdot_land, R_nucl, R_out)
    # radial velocity of disk gas induced by mass and angular #momnetum deposition of incident fountain gas
    integrand(r) = r * (Sigmadot_land(r, Mdot_land, R_nucl, R_out) - Sigma_star_ks(r, R_g, sigma_g0, A, N))

    integral, error = quadgk(integrand, R_out, R)

    return integral / (R*Sigma_g(R, R_g, sigma_g0))
end


function j_land(R, R_g, sigma_g0, A, N, Mdot_land, R_nucl, R_out, v_c)
    # required specific angular momentum of incident fountain gas
    return v_c*(R + Sigma_g(R, R_g, sigma_g0) * v_R(R, R_g, sigma_g0, A, N, Mdot_land, R_nucl, R_out) / Sigmadot_land(R, Mdot_land, R_nucl, R_out))
end


function j_ratio_flat(R, R_g, sigma_g0, A, N, Mdot_land, R_nucl, R_out, v_c)
        # ratio of specific angular momentum of incident fountain gas to that of circular orbit launched from the nuclear region radius
    return j_land(R, R_g, sigma_g0, A, N, Mdot_land, R_nucl, R_out, v_c) / j_rotcurve_flat(R_nucl, v_c)
end


"""Radial velocity from observed gas and SFR profiles, in kpc/yr."""
function v_R_empirical(R, Sigma_g, Sigmadot_star, Sigmadot_land, R_out)
    integrand(r) = r * (Sigmadot_land(r) - Sigmadot_star(r))
    integral, error = quadgk(integrand, R_out, R)
    return integral / (R * Sigma_g(R))
end


"""Required landing angular momentum for a general rotation curve."""
function j_land_required(R, sigma_g, v_R, sigmadot_land, v_c, dv_c_dR)
    j_disk = R * v_c
    dj_disk_dR = v_c + R * dv_c_dR
    return j_disk + sigma_g * v_R / sigmadot_land * dj_disk_dR
end
