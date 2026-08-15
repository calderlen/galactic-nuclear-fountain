function Mdot_land(Mdot_nucl, f, n)
    # mass flux of incident fountain gas
    return f * n * Mdot_nucl
end

function Mdot_land_mixing(Mdot_nucl, f, n, mu)
    # mass flux of incident fountain gas after mixing with CGM
    return Mdot_land(Mdot_nucl, f, n) * (1 + mu)
end

function j_nucl(R_nucl, v_c)
    return R_nucl * v_c
end

function j_cgm(R, beta, v_c)
    return beta * R * v_c
end

function j_land_mixing(R, R_nucl, mu, beta, v_c)
    # specific angular momentum of incident fountain gas after mixing with CGM
    return (j_nucl(R_nucl, v_c) + j_cgm(R, beta, v_c) * mu )/ (1+mu)
end

function dj_land_dR_mixing(mu, beta, v_c)
    return mu*beta*v_c/(1+mu)
end