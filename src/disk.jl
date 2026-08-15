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


function j_rotcurve_flat(R, v_c)
    # flat rotation curve
    return v_c * R
end