function dSigmadot_land_dR(Sigmadot_land, R, R_g, sigma_g0, A, N, R_nucl, mu, beta)
    #
    Sigmadot_star = Sigma_star_ks(R, R_g, sigma_g0, A, N)
    return -Sigmadot_land * ((mu*beta - 2*(1+mu))/(R_nucl + mu*beta*R - R*(1+mu)) + 1/R) - Sigmadot_star *(1+mu)/(R_nucl + mu*beta*R - R*(1+mu))
end

function Sigmadot_land_ode(Sigmadot_land, p, R)
    #
    return dSigmadot_land_dR(Sigmadot_land, R, p.R_g, p.sigma_g0, p.A, p.N, p.R_nucl, p.mu, p.beta)
end

function v_R_forward(R, Sigmadot_land, R_g, sigma_g0, R_nucl, mu, beta)
    #
    return Sigmadot_land/Sigma_g(R, R_g, sigma_g0)* ((R_nucl+R*(mu*(beta-1)-1))/(1+mu))
end

function Mdot_acc_forward(R, Sigmadot_land, R_nucl, mu, beta)
    #
    return -2*pi*R*Sigmadot_land * (R_nucl + R*(mu*(beta-1)-1))/(1+mu)
end


# General forward-model relations for observed disk profiles and a non-flat
# rotation curve. With F = R * Sigma_g * v_R, mass and angular-momentum
# conservation reduce to F' = F * j_disk' / (j_land - j_disk) - R * Sigmadot_star.
# The inward-positive accretion rate through a ring is -2pi * F.

j_disk_empirical(R, v_c) = R * v_c(R)

dj_disk_dR_empirical(R, v_c, dv_c_dR) = v_c(R) + R * dv_c_dR(R)


function j_land_mixing_empirical(R, R_nucl, mu, beta, v_c)
    j_launch = j_disk_empirical(R_nucl, v_c)
    j_CGM = beta * j_disk_empirical(R, v_c)
    return (j_launch + mu * j_CGM) / (1 + mu)
end


function radial_mass_flux_ode(F, p, R)
    j_disk = j_disk_empirical(R, p.v_c)
    j_land = j_land_mixing_empirical(R, p.R_nucl, p.mu, p.beta, p.v_c)
    dj_disk_dR = dj_disk_dR_empirical(R, p.v_c, p.dv_c_dR)
    return F * dj_disk_dR / (j_land - j_disk) - R * p.Sigmadot_star(R)
end


function Sigmadot_land_forward_empirical(R, F, R_nucl, mu, beta, v_c, dv_c_dR)
    j_disk = j_disk_empirical(R, v_c)
    j_land = j_land_mixing_empirical(R, R_nucl, mu, beta, v_c)
    dj_disk_dR = dj_disk_dR_empirical(R, v_c, dv_c_dR)
    return F * dj_disk_dR / (R * (j_land - j_disk))
end


v_R_forward_empirical(R, F, Sigma_g) = F / (R * Sigma_g(R))

Mdot_acc_forward_empirical(F) = -2 * pi * F
