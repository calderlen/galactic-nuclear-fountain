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