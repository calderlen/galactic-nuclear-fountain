const DEFAULT_INPUT_DIR = normpath(joinpath(@__DIR__, "..", "input"))

const GALAXY_ALIASES = Dict(
    "HOI" => "UGC5139",
    "HOII" => "UGC4305",
)


function normalize_galaxy_name(name)
    compact = uppercase(replace(strip(string(name)), r"[\s,-]" => ""))
    match_result = match(r"^(NGC|UGC|IC|DDO)0*(\d+)$", compact)

    if match_result !== nothing
        compact = match_result.captures[1] * string(parse(Int, match_result.captures[2]))
    end

    return get(GALAXY_ALIASES, compact, compact)
end


"""Return a piecewise-linear profile that refuses to extrapolate."""
function linear_profile(x_values, y_values)
    order = sortperm(x_values)
    x = Float64.(x_values[order])
    y = Float64.(y_values[order])

    length(x) >= 2 || error("A radial profile needs at least two points")
    all(diff(x) .> 0) || error("Radial coordinates must be unique and increasing")

    function profile(R)
        (R < x[1] || R > x[end]) &&
            throw(DomainError(R, "requested radius is outside [$(x[1]), $(x[end])] kpc"))

        R == x[end] && return y[end]

        index = searchsortedlast(x, R)
        fraction = (R - x[index]) / (x[index + 1] - x[index])
        return y[index] + fraction * (y[index + 1] - y[index])
    end

    return profile
end


rising_rotation_curve(R, Vflat, lflat) = Vflat * (-expm1(-R / lflat))

rising_rotation_curve_derivative(R, Vflat, lflat) =
    Vflat / lflat * exp(-R / lflat)


"""
Fit v(R) = Vflat * (1 - exp(-R/lflat)) to a rotation curve.

For each trial lflat, Vflat is solved by weighted linear least squares. A
dense logarithmic search is sufficient for this two-parameter science fit and
keeps the implementation transparent.
"""
function fit_rising_rotation_curve(R_values, V_values, error_values)
    R = Float64.(R_values)
    V = Float64.(V_values)
    errors = Float64.(error_values)

    valid = isfinite.(R) .& isfinite.(V) .& isfinite.(errors) .&
        (R .> 0) .& (errors .> 0)
    R = R[valid]
    V = V[valid]
    errors = errors[valid]

    length(R) >= 3 || error("A SPARC rotation-curve fit needs at least three points")

    lflat_min = max(minimum(R) / 20, 1e-3)
    lflat_max = max(20 * maximum(R), 100 * lflat_min)
    weights = 1.0 ./ errors.^2

    best_chi2 = Inf
    best_Vflat = NaN
    best_lflat = NaN

    for log_lflat in range(log(lflat_min), log(lflat_max), length=4000)
        lflat = exp(log_lflat)
        shape = -expm1.(-R ./ lflat)
        Vflat = sum(weights .* shape .* V) / sum(weights .* shape.^2)
        chi2 = sum(weights .* (V .- Vflat .* shape).^2)

        if Vflat > 0 && chi2 < best_chi2
            best_chi2 = chi2
            best_Vflat = Vflat
            best_lflat = lflat
        end
    end

    dof = length(R) - 2
    return (
        Vflat = best_Vflat,
        lflat = best_lflat,
        chi2 = best_chi2,
        dof = dof,
        reduced_chi2 = dof > 0 ? best_chi2 / dof : NaN,
    )
end


function required_metadata(metadata, field)
    value = getproperty(metadata, field)
    ismissing(value) && error("Missing $(field) for $(metadata.galaxy)")
    return Float64(value)
end


"""
Load one galaxy's Leroy disk profiles and both Leroy and SPARC kinematics.

The adopted geometry is Leroy/THINGS because the gas and SFR profiles were
constructed with that geometry. SPARC radii and velocities are transformed to
it before the SPARC curve is fitted.
"""
function load_galaxy_observations(name; input_dir=DEFAULT_INPUT_DIR)
    galaxy = normalize_galaxy_name(name)

    catalog = CSV.read(joinpath(input_dir, "galaxies_triple_overlap.csv"), DataFrame)
    catalog = catalog[catalog.galaxy .== galaxy, :]
    nrow(catalog) == 1 || error("Expected one triple-overlap row for $galaxy; found $(nrow(catalog))")
    metadata = NamedTuple(catalog[1, :])

    radial = CSV.read(joinpath(input_dir, "leroy_radial_profiles.csv"), DataFrame)
    radial = radial[radial.galaxy .== galaxy, :]
    radial = radial[.!ismissing.(radial.R_kpc) .&
        .!ismissing.(radial.SigmaHI_Msun_pc2) .&
        .!ismissing.(radial.SigmaSFR_Msun_yr_kpc2), :]
    sort!(radial, :R_kpc)
    nrow(radial) >= 2 || error("Not enough Leroy radial data for $galaxy")

    has_radial_H2 = any(.!ismissing.(radial.SigmaH2_Msun_pc2))
    H2_treatment = has_radial_H2 ?
        "measured H2; blank radial bins set to zero" :
        "HI only; radial H2 is unknown"
    SigmaH2_used = coalesce.(radial.SigmaH2_Msun_pc2, 0.0)
    e_SigmaHI = coalesce.(radial.e_SigmaHI_Msun_pc2, 0.0)
    e_SigmaH2 = coalesce.(radial.e_SigmaH2_Msun_pc2, 0.0)

    radial.SigmaH2_used_Msun_pc2 = SigmaH2_used
    radial.Sigma_g_Msun_kpc2 =
        1e6 .* (radial.SigmaHI_Msun_pc2 .+ SigmaH2_used)
    radial.e_Sigma_g_Msun_kpc2 =
        1e6 .* sqrt.(e_SigmaHI.^2 .+ e_SigmaH2.^2)

    Sigma_g = linear_profile(radial.R_kpc, radial.Sigma_g_Msun_kpc2)
    Sigmadot_star = linear_profile(radial.R_kpc, radial.SigmaSFR_Msun_yr_kpc2)

    sparc = CSV.read(joinpath(input_dir, "sparc_rotation_curves.csv"), DataFrame)
    sparc = sparc[sparc.galaxy .== galaxy, :]
    sparc = sparc[.!ismissing.(sparc.R_kpc) .&
        .!ismissing.(sparc.Vobs_kms) .&
        .!ismissing.(sparc.e_Vobs_kms), :]
    sort!(sparc, :R_kpc)
    nrow(sparc) >= 3 || error("Not enough SPARC rotation-curve data for $galaxy")

    D_sparc = required_metadata(metadata, :D_sparc_Mpc)
    D_adopted = required_metadata(metadata, :D_leroy_Mpc)
    inclination_sparc = required_metadata(metadata, :inc_sparc_deg)
    inclination_adopted = required_metadata(metadata, :inc_leroy_deg)

    distance_scale = D_adopted / D_sparc
    velocity_scale = sind(inclination_sparc) / sind(inclination_adopted)

    sparc.R_sparc_kpc = copy(sparc.R_kpc)
    sparc.Vobs_sparc_kms = copy(sparc.Vobs_kms)
    sparc.e_Vobs_sparc_kms = copy(sparc.e_Vobs_kms)
    sparc.R_adopted_kpc = sparc.R_kpc .* distance_scale
    sparc.Vobs_adopted_kms = sparc.Vobs_kms .* velocity_scale
    sparc.e_Vobs_adopted_kms = sparc.e_Vobs_kms .* abs(velocity_scale)

    sparc_fit = fit_rising_rotation_curve(
        sparc.R_adopted_kpc,
        sparc.Vobs_adopted_kms,
        sparc.e_Vobs_adopted_kms,
    )

    leroy_Vflat = required_metadata(metadata, :Vflat_leroy_kms)
    leroy_lflat = required_metadata(metadata, :lflat_leroy_kpc)

    leroy_rotation = (
        Vflat = leroy_Vflat,
        lflat = leroy_lflat,
        v = R -> rising_rotation_curve(R, leroy_Vflat, leroy_lflat),
        dv_dR = R -> rising_rotation_curve_derivative(R, leroy_Vflat, leroy_lflat),
    )

    sparc_rotation = (
        Vflat = sparc_fit.Vflat,
        lflat = sparc_fit.lflat,
        chi2 = sparc_fit.chi2,
        dof = sparc_fit.dof,
        reduced_chi2 = sparc_fit.reduced_chi2,
        v = R -> rising_rotation_curve(R, sparc_fit.Vflat, sparc_fit.lflat),
        dv_dR = R -> rising_rotation_curve_derivative(R, sparc_fit.Vflat, sparc_fit.lflat),
    )

    radial_bounds = (minimum(radial.R_kpc), maximum(radial.R_kpc))
    sparc_bounds = (minimum(sparc.R_adopted_kpc), maximum(sparc.R_adopted_kpc))
    common_bounds = (max(radial_bounds[1], sparc_bounds[1]),
        min(radial_bounds[2], sparc_bounds[2]))
    common_bounds[1] < common_bounds[2] || error("No common radial range for $galaxy")

    return (
        galaxy = galaxy,
        metadata = metadata,
        radial = radial,
        sparc = sparc,
        Sigma_g = Sigma_g,
        Sigmadot_star = Sigmadot_star,
        leroy_rotation = leroy_rotation,
        sparc_rotation = sparc_rotation,
        radial_bounds = radial_bounds,
        sparc_bounds = sparc_bounds,
        common_bounds = common_bounds,
        has_radial_H2 = has_radial_H2,
        H2_treatment = H2_treatment,
        geometry = (
            D_sparc_Mpc = D_sparc,
            D_adopted_Mpc = D_adopted,
            inclination_sparc_deg = inclination_sparc,
            inclination_adopted_deg = inclination_adopted,
            distance_scale = distance_scale,
            velocity_scale = velocity_scale,
        ),
    )
end
