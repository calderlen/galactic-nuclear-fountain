module GalacticFountain


# disk structure

function Sigma_g(R, R_g, sigma_g0)
    # exponential gas surface density profile
    return sigma_g0 * exp(-R / R_g)
end


function ks_Sigma_star(R, R_g, sigma_g0, A, N)
    # Kennicutt-Schmidt SFR surface density
    return A * Sigma_g(R, R_g, sigma_g0)^N
end


# CGM interaction (currently unused)


# fountain landing


## NOTE: DONT THINK THIS IS RIGHT

function Sigma_dot_land(R, Mdot_land, R_out, R_w)
    #incident mass flux of fountain gas landing on the disk, where Mdot_land is the total mass landing between R_w and R_out, and R is the radius at which we wish to compute the mass flux
    return Mdot_land / (2 * pi * R^2 * log(R_out / R_w))
end



# mass conservation
function Mdot_star(R_g, sigma_g0, A, N, R_w, R_out)
    F(R) = exp(-N * R / R_g) * (1 + N * R / R_g)
    prefactor = 2 * pi * A * sigma_g0^N * R_g^2 / N^2
    return prefactor * (F(R_w) - F(R_out))
end


function Sigmadot_star()
   return
end


# angular momentum conservation

function v_r(R, sigma_g0, A, N, Mdot_land, R_out, R_w)  )
    return
end

function j_land
    return
end


# diagnostics

function j_ratio
    return
end






end #module
