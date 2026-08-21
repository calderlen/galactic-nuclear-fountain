module Observations

using CSV
using DataFrames
using Dierckx
using Statistics

const DEFAULT_INPUT_DIR = normpath(joinpath(@__DIR__, "..", "input"))

const GALAXY_ALIASES = Dict(
    "HOI" => "UGC5139",
    "HOII" => "UGC4305",
)
const SFR_CROSSCALIBRATION_R_R25 = (0.9, 1.2)


function normalize_galaxy_name(name)
    compact = uppercase(replace(strip(string(name)), r"[\s,-]" => ""))
    match_result = match(r"^(NGC|UGC|IC|DDO)0*(\d+)$", compact)

    if match_result !== nothing
        compact = match_result.captures[1] * string(parse(Int, match_result.captures[2]))
    end

    return get(GALAXY_ALIASES, compact, compact)
end


"""Fit an uncertainty-weighted cubic smoothing spline without extrapolation."""
function fit_uncertainty_weighted_smoothing_spline(
    R_values,
    values,
    error_values;
    description,
)
    R = Float64.(R_values)
    y = Float64.(values)
    errors = Float64.(error_values)

    all(isfinite.(R)) || error("$description spline radii must be finite")
    all(isfinite.(y)) || error("$description spline values must be finite")
    all(isfinite.(errors) .& (errors .> 0)) ||
        error("$description spline uncertainties must be finite and positive")
    all(diff(R) .> 0) || error("$description spline radii must be strictly increasing")

    weights = 1.0 ./ errors
    spline = Spline1D(
        R,
        y;
        w=weights,
        k=3,
        s=length(R),
        bc="error",
    )
    fitted = spline(R)
    fitted_derivative = derivative(spline, R)

    return (
        support=(first(R), last(R)),
        fitted=fitted,
        derivative_at_measurements=fitted_derivative,
        v=R_query -> spline(R_query),
        dv_dR=R_query -> derivative(spline, R_query),
    )
end


fit_sparc_smoothing_spline(R_values, V_values, error_values) =
    fit_uncertainty_weighted_smoothing_spline(
        R_values,
        V_values,
        error_values;
        description="SPARC rotation-curve",
    )


fit_gas_smoothing_spline(R_values, Sigma_values, error_values) =
    fit_uncertainty_weighted_smoothing_spline(
        R_values,
        Sigma_values,
        error_values;
        description="gas-profile",
    )


function fit_sfr_smoothing_spline(R_values, Sigma_values, error_values)
    values = Float64.(Sigma_values)
    errors = Float64.(error_values)
    all(values .> 0) ||
        error("Star-formation surface densities must be positive for the log spline")

    # The combined inner/outer profile spans several decades. Fit ln(SigmaSFR)
    # with propagated fractional errors so the cubic spline remains positive.
    log_spline = fit_uncertainty_weighted_smoothing_spline(
        R_values,
        log.(values),
        errors ./ values;
        description="log star-formation-profile",
    )
    fitted = exp.(log_spline.fitted)
    return (
        support=log_spline.support,
        fitted=fitted,
        derivative_at_measurements=fitted .* log_spline.derivative_at_measurements,
        v=R_query -> exp(log_spline.v(R_query)),
        dv_dR=R_query -> exp(log_spline.v(R_query)) * log_spline.dv_dR(R_query),
    )
end


"""Linearly interpolate a positive uncertainty profile without extrapolation."""
function interpolate_sorted(x_values, y_values, query)
    x = Float64.(x_values)
    y = Float64.(y_values)
    x[1] <= query <= x[end] ||
        throw(DomainError(query, "interpolation query is outside the measured support"))
    query == x[end] && return y[end]
    index = searchsortedlast(x, query)
    fraction = (query - x[index]) / (x[index + 1] - x[index])
    return y[index] + fraction * (y[index + 1] - y[index])
end


"""
Cross-calibrate Bigiel's FUV-only SFR scale to Leroy's FUV+24 micron scale.

For each common galaxy, the Leroy-only log-SFR spline is evaluated at Bigiel
annuli inside 0.9 <= R/R25 <= 1.2 and inside the measured Leroy support. A
weighted mean log ratio is applied separately to each galaxy so different
tracer combinations and resolutions do not create a discontinuity at R25. An
equal-galaxy population summary is retained for diagnostics only.
"""
function cross_calibrate_bigiel_sfr(leroy_radial, bigiel_radial)
    diagnostic_rows = NamedTuple[]
    lower_window, upper_window = SFR_CROSSCALIBRATION_R_R25

    for galaxy in sort(collect(intersect(
        Set(String.(leroy_radial.galaxy)),
        Set(String.(bigiel_radial.galaxy)),
    )))
        leroy = leroy_radial[leroy_radial.galaxy .== galaxy, :]
        leroy = leroy[.!ismissing.(leroy.R_R25) .&
            .!ismissing.(leroy.SigmaSFR_Msun_yr_kpc2) .&
            .!ismissing.(leroy.e_SigmaSFR_Msun_yr_kpc2), :]
        sort!(leroy, :R_R25)
        nrow(leroy) >= 4 || continue

        leroy_spline = fit_sfr_smoothing_spline(
            leroy.R_R25,
            leroy.SigmaSFR_Msun_yr_kpc2,
            leroy.e_SigmaSFR_Msun_yr_kpc2,
        )
        lower = max(lower_window, leroy_spline.support[1])
        upper = min(upper_window, leroy_spline.support[2])
        lower < upper || continue

        bigiel = bigiel_radial[bigiel_radial.galaxy .== galaxy, :]
        bigiel = bigiel[(bigiel.R_R25 .>= lower) .& (bigiel.R_R25 .<= upper), :]
        sort!(bigiel, :R_R25)

        for row in eachrow(bigiel)
            R_R25 = Float64(row.R_R25)
            leroy_value = leroy_spline.v(R_R25)
            leroy_error = interpolate_sorted(
                leroy.R_R25,
                leroy.e_SigmaSFR_Msun_yr_kpc2,
                R_R25,
            )
            bigiel_value = Float64(row.SigmaSFR_Msun_yr_kpc2)
            bigiel_error = Float64(row.e_SigmaSFR_Msun_yr_kpc2)
            log_ratio = log(leroy_value / bigiel_value)
            log_ratio_error = hypot(
                leroy_error / leroy_value,
                bigiel_error / bigiel_value,
            )
            push!(diagnostic_rows, (
                galaxy=galaxy,
                R_R25=R_R25,
                SigmaSFR_Leroy_Msun_yr_kpc2=leroy_value,
                e_SigmaSFR_Leroy_Msun_yr_kpc2=leroy_error,
                SigmaSFR_Bigiel_raw_Msun_yr_kpc2=bigiel_value,
                e_SigmaSFR_Bigiel_raw_Msun_yr_kpc2=bigiel_error,
                log_Leroy_over_Bigiel=log_ratio,
                e_log_Leroy_over_Bigiel=log_ratio_error,
                weight=inv(log_ratio_error^2),
            ))
        end
    end

    diagnostics = DataFrame(diagnostic_rows)
    nrow(diagnostics) > 0 || error("No Leroy/Bigiel SFR overlap is available")

    galaxy_calibrations = Dict{String,NamedTuple}()
    for group in groupby(diagnostics, :galaxy)
        galaxy = String(first(group.galaxy))
        weight_sum = sum(group.weight)
        log_factor =
            sum(group.weight .* group.log_Leroy_over_Bigiel) / weight_sum
        formal_uncertainty = sqrt(inv(weight_sum))
        log_scatter = nrow(group) > 1 ? sqrt(
            sum(group.weight .* (group.log_Leroy_over_Bigiel .- log_factor).^2) /
            weight_sum
        ) : 0.0
        log_uncertainty = hypot(
            formal_uncertainty,
            log_scatter / sqrt(nrow(group)),
        )
        galaxy_calibrations[galaxy] = (
            factor=exp(log_factor),
            log_factor=log_factor,
            log_uncertainty=log_uncertainty,
            log_scatter=log_scatter,
            point_count=nrow(group),
        )
    end

    length(galaxy_calibrations) >= 2 ||
        error("SFR cross-calibration requires at least two overlap galaxies")
    galaxy_log_offsets = [value.log_factor for value in values(galaxy_calibrations)]
    global_log_factor = mean(galaxy_log_offsets)
    galaxy_scatter = std(galaxy_log_offsets; corrected=true)
    global_log_uncertainty = galaxy_scatter / sqrt(length(galaxy_log_offsets))
    global_factor = exp(global_log_factor)

    diagnostics.galaxy_log_ratio_mean =
        [galaxy_calibrations[String(galaxy)].log_factor for galaxy in diagnostics.galaxy]
    diagnostics.galaxy_overlap_point_count =
        [galaxy_calibrations[String(galaxy)].point_count for galaxy in diagnostics.galaxy]
    diagnostics.galaxy_crosscal_factor =
        [galaxy_calibrations[String(galaxy)].factor for galaxy in diagnostics.galaxy]
    diagnostics.galaxy_crosscal_log_uncertainty =
        [galaxy_calibrations[String(galaxy)].log_uncertainty for galaxy in diagnostics.galaxy]
    diagnostics.galaxy_log_ratio_scatter =
        [galaxy_calibrations[String(galaxy)].log_scatter for galaxy in diagnostics.galaxy]
    diagnostics.global_summary_crosscal_factor = fill(global_factor, nrow(diagnostics))
    diagnostics.global_crosscal_log_uncertainty =
        fill(global_log_uncertainty, nrow(diagnostics))
    diagnostics.galaxy_to_galaxy_log_scatter =
        fill(galaxy_scatter, nrow(diagnostics))
    diagnostics.calibration_R_R25_min = fill(lower_window, nrow(diagnostics))
    diagnostics.calibration_R_R25_max = fill(upper_window, nrow(diagnostics))

    return (
        global_summary_factor=global_factor,
        global_summary_log_uncertainty=global_log_uncertainty,
        galaxy_scatter=galaxy_scatter,
        galaxy_count=length(galaxy_calibrations),
        point_count=nrow(diagnostics),
        by_galaxy=galaxy_calibrations,
        diagnostics=diagnostics,
    )
end


"""
Load one galaxy's Leroy and Bigiel disk profiles and SPARC kinematics.

The adopted geometry is Leroy because the gas and SFR profiles were constructed
with that geometry. Bigiel radii are supplied as R/R25 and converted using the
adopted Leroy R25. SPARC radii and velocities are transformed to the adopted
geometry before the SPARC spline is fitted.
"""
function load_galaxy_observations(name; input_dir=DEFAULT_INPUT_DIR)
    galaxy = normalize_galaxy_name(name)

    catalog = CSV.read(joinpath(input_dir, "galaxies_triple_overlap.csv"), DataFrame)
    catalog = catalog[catalog.galaxy .== galaxy, :]
    metadata = NamedTuple(catalog[1, :])

    radial_all = CSV.read(joinpath(input_dir, "leroy_radial_profiles.csv"), DataFrame)
    bigiel_all = CSV.read(joinpath(input_dir, "bigiel_radial_profiles.csv"), DataFrame)
    sfr_crosscalibration = cross_calibrate_bigiel_sfr(radial_all, bigiel_all)
    applied_sfr_crosscalibration = get(
        sfr_crosscalibration.by_galaxy,
        galaxy,
        (
            factor=NaN,
            log_factor=NaN,
            log_uncertainty=NaN,
            log_scatter=NaN,
            point_count=0,
        ),
    )

    radial = radial_all[radial_all.galaxy .== galaxy, :]
    radial = radial[.!ismissing.(radial.R_kpc) .&
        .!ismissing.(radial.SigmaHI_Msun_pc2) .&
        .!ismissing.(radial.SigmaSFR_Msun_yr_kpc2), :]
    sort!(radial, :R_kpc)

    has_radial_H2 = any(.!ismissing.(radial.SigmaH2_Msun_pc2))
    H2_sources = sort(unique(String.(collect(skipmissing(radial.SigmaH2_source)))))
    H2_source = isempty(H2_sources) ? "none" : join(H2_sources, "+")
    H2_treatment = has_radial_H2 ?
        "measured H2 from $H2_source via Leroy2008; blank radial bins set to zero" :
        "HI only; radial H2 is unknown"
    SigmaH2_used = coalesce.(radial.SigmaH2_Msun_pc2, 0.0)
    e_SigmaHI = coalesce.(radial.e_SigmaHI_Msun_pc2, 0.0)
    e_SigmaH2 = coalesce.(radial.e_SigmaH2_Msun_pc2, 0.0)

    radial.SigmaH2_used_Msun_pc2 = SigmaH2_used
    radial.Sigma_g_Msun_kpc2 =
        1e6 .* (radial.SigmaHI_Msun_pc2 .+ SigmaH2_used)
    radial.e_Sigma_g_Msun_kpc2 =
        1e6 .* sqrt.(e_SigmaHI.^2 .+ e_SigmaH2.^2)

    bigiel = bigiel_all[bigiel_all.galaxy .== galaxy, :]
    sort!(bigiel, :R_R25)
    if nrow(bigiel) > 0
        R25_adopted = Float64(metadata.R25_leroy_kpc)
        bigiel.R_kpc = bigiel.R_R25 .* R25_adopted
        bigiel.Sigma_g_Msun_kpc2 = 1e6 .* bigiel.SigmaHI_Msun_pc2
        bigiel.e_Sigma_g_Msun_kpc2 = 1e6 .* bigiel.e_SigmaHI_Msun_pc2
        bigiel.SigmaSFR_Bigiel_raw_Msun_yr_kpc2 =
            copy(bigiel.SigmaSFR_Msun_yr_kpc2)
        bigiel.e_SigmaSFR_stat_Bigiel_raw_Msun_yr_kpc2 =
            copy(bigiel.e_SigmaSFR_stat_Msun_yr_kpc2)
        bigiel.e_SigmaSFR_Bigiel_raw_Msun_yr_kpc2 =
            copy(bigiel.e_SigmaSFR_Msun_yr_kpc2)
        isfinite(applied_sfr_crosscalibration.factor) ||
            error("No SFR cross-calibration is available for $galaxy")
        factor = applied_sfr_crosscalibration.factor
        log_uncertainty = applied_sfr_crosscalibration.log_uncertainty
        bigiel.SigmaSFR_Msun_yr_kpc2 .*= factor
        bigiel.e_SigmaSFR_stat_Msun_yr_kpc2 .*= factor
        bigiel.e_SigmaSFR_Msun_yr_kpc2 = factor .* hypot.(
            bigiel.e_SigmaSFR_Bigiel_raw_Msun_yr_kpc2,
            bigiel.SigmaSFR_Bigiel_raw_Msun_yr_kpc2 .* log_uncertainty,
        )
        bigiel.SFR_crosscal_factor = fill(factor, nrow(bigiel))
        bigiel.SFR_crosscal_log_uncertainty = fill(log_uncertainty, nrow(bigiel))
    else
        bigiel.R_kpc = Float64[]
        bigiel.Sigma_g_Msun_kpc2 = Float64[]
        bigiel.e_Sigma_g_Msun_kpc2 = Float64[]
        bigiel.SigmaSFR_Bigiel_raw_Msun_yr_kpc2 = Float64[]
        bigiel.e_SigmaSFR_stat_Bigiel_raw_Msun_yr_kpc2 = Float64[]
        bigiel.e_SigmaSFR_Bigiel_raw_Msun_yr_kpc2 = Float64[]
        bigiel.SFR_crosscal_factor = Float64[]
        bigiel.SFR_crosscal_log_uncertainty = Float64[]
    end
    if nrow(bigiel) > 0
        H2_treatment *= "; Bigiel outer-disk points treated as HI-dominated"
    end

    leroy_gas_source = has_radial_H2 ?
        "Leroy2008_THINGS+$H2_source" : "Leroy2008_THINGS"
    gas_measurements = DataFrame(
        source=fill(leroy_gas_source, nrow(radial)),
        R_kpc=Float64.(radial.R_kpc),
        value=Float64.(radial.Sigma_g_Msun_kpc2),
        error=Float64.(radial.e_Sigma_g_Msun_kpc2),
    )
    append!(gas_measurements, DataFrame(
        source=fill("Bigiel2010", nrow(bigiel)),
        R_kpc=Float64.(bigiel.R_kpc),
        value=Float64.(bigiel.Sigma_g_Msun_kpc2),
        error=Float64.(bigiel.e_Sigma_g_Msun_kpc2),
    ))
    sort!(gas_measurements, :R_kpc)

    sfr_measurements = DataFrame(
        source=fill("Leroy2008", nrow(radial)),
        R_kpc=Float64.(radial.R_kpc),
        value=Float64.(radial.SigmaSFR_Msun_yr_kpc2),
        error=Float64.(radial.e_SigmaSFR_Msun_yr_kpc2),
    )
    append!(sfr_measurements, DataFrame(
        source=fill("Bigiel2010_crosscal", nrow(bigiel)),
        R_kpc=Float64.(bigiel.R_kpc),
        value=Float64.(bigiel.SigmaSFR_Msun_yr_kpc2),
        error=Float64.(bigiel.e_SigmaSFR_Msun_yr_kpc2),
    ))
    sort!(sfr_measurements, :R_kpc)

    gas_spline = fit_gas_smoothing_spline(
        gas_measurements.R_kpc,
        gas_measurements.value,
        gas_measurements.error,
    )
    sfr_spline = fit_sfr_smoothing_spline(
        sfr_measurements.R_kpc,
        sfr_measurements.value,
        sfr_measurements.error,
    )
    sparc = CSV.read(joinpath(input_dir, "sparc_rotation_curves.csv"), DataFrame)
    sparc = sparc[sparc.galaxy .== galaxy, :]
    sparc = sparc[.!ismissing.(sparc.R_kpc) .&
        .!ismissing.(sparc.Vobs_kms) .&
        .!ismissing.(sparc.e_Vobs_kms), :]
    sort!(sparc, :R_kpc)

    D_sparc = Float64(metadata.D_sparc_Mpc)
    D_adopted = Float64(metadata.D_leroy_Mpc)
    inclination_sparc = Float64(metadata.inc_sparc_deg)
    inclination_adopted = Float64(metadata.inc_leroy_deg)

    distance_scale = D_adopted / D_sparc
    velocity_scale = sind(inclination_sparc) / sind(inclination_adopted)

    sparc.R_sparc_kpc = copy(sparc.R_kpc)
    sparc.Vobs_sparc_kms = copy(sparc.Vobs_kms)
    sparc.e_Vobs_sparc_kms = copy(sparc.e_Vobs_kms)
    sparc.R_adopted_kpc = sparc.R_kpc .* distance_scale
    sparc.Vobs_adopted_kms = sparc.Vobs_kms .* velocity_scale
    sparc.e_Vobs_adopted_kms = sparc.e_Vobs_kms .* abs(velocity_scale)

    sparc_spline = fit_sparc_smoothing_spline(
        sparc.R_adopted_kpc,
        sparc.Vobs_adopted_kms,
        sparc.e_Vobs_adopted_kms,
    )

    sparc_bounds = (minimum(sparc.R_adopted_kpc), maximum(sparc.R_adopted_kpc))
    common_bounds = (
        max(gas_spline.support[1], sfr_spline.support[1], sparc_bounds[1]),
        min(gas_spline.support[2], sfr_spline.support[2], sparc_bounds[2]),
    )
    common_bounds[1] < common_bounds[2] || error("No common radial range for $galaxy")

    return (
        galaxy = galaxy,
        metadata = metadata,
        radial = radial,
        bigiel = bigiel,
        sparc = sparc,
        gas_measurements = gas_measurements,
        sfr_measurements = sfr_measurements,
        gas_spline = gas_spline,
        sfr_spline = sfr_spline,
        sparc_spline = sparc_spline,
        common_bounds = common_bounds,
        has_radial_H2 = has_radial_H2,
        H2_source = H2_source,
        H2_treatment = H2_treatment,
        sfr_crosscalibration = sfr_crosscalibration,
        applied_sfr_crosscalibration = applied_sfr_crosscalibration,
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

end # module
