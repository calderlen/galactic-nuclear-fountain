
function dSigmadot_land_dR(Sigmadot_land, R, R_g, sigma_g0, A, N, R_nucl, mu, beta)

    Sigmadot_star = Sigma_star_ks(R, R_g, sigma_g0, A, N)

    return -Sigmadot_land * ((mu*beta - 2*(1+mu))/(R_nucl + mu*beta*R - R*(1+mu)) + 1/R) - Sigmadot_star *(1+mu)/(R_nucl + mu*beta*R - R*(1+mu))
end


function landing_ode(Sigmadot_land, p, R)
    return dSigmadot_land_dR(Sigmadot_land, R, p.R_g, p.sigma_g0, p.A, p.N, p.R_nucl, p.mu, p.beta)
end

function v_R_forward(R, Sigmadot_land, R_g, sigma_g0, R_nucl, mu, beta)
    return Sigmadot_land/Sigma_g(R, R_g, sigma_g0)* ((R_nucl+R*(mu*(beta-1)-1))/(1+mu))
end

function Mdot_acc_forward(R, Sigmadot_land, R_nucl, mu, beta)

    return -2*pi*R*Sigmadot_land * (R_nucl + R*(mu*(beta-1)-1))/(1+mu)

end