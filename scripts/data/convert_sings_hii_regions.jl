using CSV
using DataFrames
using Downloads

include(joinpath(@__DIR__, "..", "..", "observations", "observations.jl"))
using .Observations

# Moustakas et al. 2010, ApJS, 190, 233
# VizieR catalog J/ApJS/190/233, DOI: 10.26093/cds/vizier.21900233
#
# The paper and CDS ReadMe disagree about whether flag `a` means an ambiguous
# R23 branch or an undefined abundance. For a conservative science-ready
# sample, either nonblank flag is retained but marked unusable here.

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const INPUT = joinpath(ROOT, "input")
const RAW_DIR = joinpath(INPUT, "sings")

const TABLE10_FILE = joinpath(RAW_DIR, "table10.dat")
const TABLE1_FILE = joinpath(RAW_DIR, "table1.dat")
const REFS_FILE = joinpath(RAW_DIR, "refs.dat")
const README_FILE = joinpath(RAW_DIR, "ReadMe")
const LOCAL_GALAXIES_FILE = joinpath(INPUT, "galaxies_triple_overlap.csv")

const OUTPUT_FILE = joinpath(INPUT, "sings_hii_regions.csv")
const COVERAGE_FILE = joinpath(INPUT, "sings_hii_coverage.csv")

const CDS_BASE = "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/190/233"
const RAW_FILES = Dict(
    TABLE10_FILE => "$(CDS_BASE)/table10.dat",
    TABLE1_FILE => "$(CDS_BASE)/table1.dat",
    REFS_FILE => "$(CDS_BASE)/refs.dat",
    README_FILE => "$(CDS_BASE)/ReadMe",
)

# The analysis sample is the Q=1 SPARC/THINGS/Leroy overlap restricted to
# galaxies with Moustakas et al. (2010) H II-region metallicity measurements.
const SAMPLE_GALAXIES = [
    "NGC2403",
    "NGC2841",
    "NGC3198",
    "NGC3521",
    "NGC5055",
    "NGC6946",
    "NGC7331",
    "NGC7793",
]

const EXPECTED_COUNTS = Dict(
    "NGC2403" => 46,
    "NGC2841" => 5,
    "NGC3198" => 14,
    "NGC3521" => 13,
    "NGC5055" => 5,
    "NGC6946" => 8,
    "NGC7331" => 10,
    "NGC7793" => 12,
)


function fixed_field(line, first_byte, last_byte)
    padded = ncodeunits(line) < last_byte ? rpad(line, last_byte) : line
    return strip(padded[first_byte:last_byte])
end


function parse_float_field(line, first_byte, last_byte)
    value = fixed_field(line, first_byte, last_byte)
    return isempty(value) ? missing : something(tryparse(Float64, value), missing)
end


function parse_int_field(line, first_byte, last_byte)
    value = fixed_field(line, first_byte, last_byte)
    return isempty(value) ? missing : something(tryparse(Int, value), missing)
end


function parse_string_field(line, first_byte, last_byte)
    value = fixed_field(line, first_byte, last_byte)
    return isempty(value) ? missing : value
end


function ensure_raw_catalog()
    mkpath(RAW_DIR)

    for (path, url) in RAW_FILES
        if !isfile(path)
            @info "Downloading Moustakas et al. (2010) catalog file" url path
            Downloads.download(url, path)
        end
    end
end


function read_hii_regions(path)
    rows = NamedTuple[]

    for line in eachline(path)
        source_seq = parse_int_field(line, 1, 3)
        ismissing(source_seq) && continue

        name_sings = fixed_field(line, 5, 13)
        flag_KK04 = parse_string_field(line, 80, 80)
        flag_PT05 = parse_string_field(line, 92, 92)
        OH12_KK04 = parse_float_field(line, 70, 73)
        OH12_PT05 = parse_float_field(line, 82, 85)
        e_OH12_KK04 = parse_float_field(line, 75, 78)
        e_OH12_PT05 = parse_float_field(line, 87, 90)

        offRA = parse_int_field(line, 118, 121)
        offDE = parse_int_field(line, 123, 126)
        offRA = isequal(offRA, -999) ? missing : offRA
        offDE = isequal(offDE, -999) ? missing : offDE

        RA_hours = parse_int_field(line, 97, 98)
        RA_minutes = parse_int_field(line, 100, 101)
        RA_seconds = parse_float_field(line, 103, 106)
        DE_sign = fixed_field(line, 108, 108)
        DE_degrees = parse_int_field(line, 109, 110)
        DE_minutes = parse_int_field(line, 112, 113)
        DE_seconds = parse_int_field(line, 115, 116)

        center_RAJ2000_deg = 15 * (
            RA_hours + RA_minutes / 60 + RA_seconds / 3600
        )
        center_DEJ2000_deg = (DE_sign == "-" ? -1 : 1) * (
            DE_degrees + DE_minutes / 60 + DE_seconds / 3600
        )

        push!(rows, (
            source_seq = source_seq,
            galaxy = Observations.normalize_galaxy_name(name_sings),
            name_sings = name_sings,
            hii_region = fixed_field(line, 15, 31),
            R_R25 = parse_float_field(line, 33, 36),
            R23 = parse_float_field(line, 38, 42),
            e_R23 = parse_float_field(line, 44, 47),
            O32 = parse_float_field(line, 49, 53),
            e_O32 = parse_float_field(line, 55, 58),
            P = parse_float_field(line, 60, 63),
            e_P = parse_float_field(line, 65, 68),
            OH12_KK04_dex = OH12_KK04,
            e_OH12_KK04_dex = e_OH12_KK04,
            flag_OH12_KK04 = flag_KK04,
            usable_OH12_KK04 = !ismissing(OH12_KK04) &&
                !ismissing(e_OH12_KK04) && ismissing(flag_KK04),
            OH12_PT05_dex = OH12_PT05,
            e_OH12_PT05_dex = e_OH12_PT05,
            flag_OH12_PT05 = flag_PT05,
            usable_OH12_PT05 = !ismissing(OH12_PT05) &&
                !ismissing(e_OH12_PT05) && ismissing(flag_PT05),
            reference_id = parse_int_field(line, 94, 95),
            center_RAJ2000_deg = center_RAJ2000_deg,
            center_DEJ2000_deg = center_DEJ2000_deg,
            offRA_arcsec = offRA,
            offDE_arcsec = offDE,
        ))
    end

    return DataFrame(rows)
end


function read_sings_galaxies(path)
    rows = Dict{String, NamedTuple}()

    for line in eachline(path)
        name_sings = fixed_field(line, 1, 8)
        isempty(name_sings) && continue

        galaxy = Observations.normalize_galaxy_name(name_sings)
        rows[galaxy] = (
            D_sings_Mpc = parse_float_field(line, 64, 69),
            R25_sings_arcmin = parse_float_field(line, 26, 30),
            inc_sings_deg = parse_int_field(line, 32, 33),
            PA_sings_deg = parse_int_field(line, 37, 39),
        )
    end

    return rows
end


function read_references(path)
    rows = Dict{Int, NamedTuple}()

    for line in eachline(path)
        reference_id = parse_int_field(line, 1, 2)
        ismissing(reference_id) && continue

        rows[reference_id] = (
            reference_bibcode = fixed_field(line, 4, 22),
            reference_authors = fixed_field(line, 24, 48),
            reference_comment = parse_string_field(line, 50, 95),
        )
    end

    return rows
end


function physical_radius_kpc(angle_arcmin, distance_Mpc)
    any(ismissing, (angle_arcmin, distance_Mpc)) && return missing
    return 1_000 * distance_Mpc * tan(deg2rad(angle_arcmin / 60))
end


function add_metadata(hii, sings_galaxies, references, local_galaxies)
    local_by_galaxy = Dict(
        string(row.galaxy) => NamedTuple(row) for row in eachrow(local_galaxies)
    )
    rows = NamedTuple[]

    for row in eachrow(hii)
        row.galaxy in SAMPLE_GALAXIES || continue

        survey = sings_galaxies[row.galaxy]
        reference = references[row.reference_id]
        adopted = local_by_galaxy[row.galaxy]

        R25_sings_kpc = physical_radius_kpc(
            survey.R25_sings_arcmin,
            survey.D_sings_Mpc,
        )
        R_sings_kpc = ismissing(row.R_R25) ? missing : row.R_R25 * R25_sings_kpc

        R_adopted_kpc = ismissing(row.R_R25) ?
            missing : row.R_R25 * adopted.R25_leroy_kpc
        push!(rows, (
            galaxy = row.galaxy,
            name_sings = row.name_sings,
            source_seq = row.source_seq,
            hii_region = row.hii_region,
            R_R25 = row.R_R25,
            R_sings_kpc = R_sings_kpc,
            R_adopted_kpc = R_adopted_kpc,
            D_sings_Mpc = survey.D_sings_Mpc,
            R25_sings_arcmin = survey.R25_sings_arcmin,
            R25_sings_kpc = R25_sings_kpc,
            inc_sings_deg = survey.inc_sings_deg,
            PA_sings_deg = survey.PA_sings_deg,
            D_adopted_Mpc = adopted.D_leroy_Mpc,
            D_sparc_Mpc = adopted.D_sparc_Mpc,
            R25_adopted_kpc = adopted.R25_leroy_kpc,
            R23 = row.R23,
            e_R23 = row.e_R23,
            O32 = row.O32,
            e_O32 = row.e_O32,
            P = row.P,
            e_P = row.e_P,
            OH12_KK04_dex = row.OH12_KK04_dex,
            e_OH12_KK04_dex = row.e_OH12_KK04_dex,
            flag_OH12_KK04 = row.flag_OH12_KK04,
            usable_OH12_KK04 = row.usable_OH12_KK04,
            OH12_PT05_dex = row.OH12_PT05_dex,
            e_OH12_PT05_dex = row.e_OH12_PT05_dex,
            flag_OH12_PT05 = row.flag_OH12_PT05,
            usable_OH12_PT05 = row.usable_OH12_PT05,
            reference_id = row.reference_id,
            reference_bibcode = reference.reference_bibcode,
            reference_authors = reference.reference_authors,
            reference_comment = reference.reference_comment,
            center_RAJ2000_deg = row.center_RAJ2000_deg,
            center_DEJ2000_deg = row.center_DEJ2000_deg,
            offRA_arcsec = row.offRA_arcsec,
            offDE_arcsec = row.offDE_arcsec,
        ))
    end

    sort!(rows; by=row -> (
        row.galaxy,
        coalesce(row.R_R25, Inf),
        row.source_seq,
    ))
    return DataFrame(rows)
end


function make_coverage(sample)
    rows = NamedTuple[]

    for galaxy in SAMPLE_GALAXIES
        subset = sample[sample.galaxy .== galaxy, :]
        has_radius = .!ismissing.(subset.R_R25)
        usable_KK04 = subset.usable_OH12_KK04
        usable_PT05 = subset.usable_OH12_PT05

        push!(rows, (
            galaxy = galaxy,
            in_sings_point_catalog = nrow(subset) > 0,
            n_rows = nrow(subset),
            n_with_radius = count(identity, has_radius),
            n_usable_KK04 = count(identity, usable_KK04),
            n_fit_ready_KK04 = count(identity, has_radius .& usable_KK04),
            n_usable_PT05 = count(identity, usable_PT05),
            n_fit_ready_PT05 = count(identity, has_radius .& usable_PT05),
        ))
    end

    return DataFrame(rows)
end


function validate_catalog(raw, sample, coverage)
    nrow(raw) == 561 || error("Expected 561 rows in table10.dat; found $(nrow(raw))")
    allunique(raw.source_seq) || error("SINGS source sequence numbers are not unique")
    expected_sample_rows = sum(values(EXPECTED_COUNTS))
    nrow(sample) == expected_sample_rows ||
        error("Expected $expected_sample_rows matched rows; found $(nrow(sample))")

    counts = Dict(row.galaxy => row.n_rows for row in eachrow(coverage))
    counts == EXPECTED_COUNTS || error("Unexpected matched-galaxy row counts: $counts")

    radii = collect(skipmissing(sample.R_R25))
    all(isfinite, radii) && all(>=(0), radii) || error("Invalid R/R25 value")

    for column in (:OH12_KK04_dex, :OH12_PT05_dex)
        values = collect(skipmissing(sample[!, column]))
        all(value -> 7 <= value <= 10, values) ||
            error("Implausible oxygen abundance in $column")
    end
end


function main()
    ensure_raw_catalog()

    raw = read_hii_regions(TABLE10_FILE)
    sings_galaxies = read_sings_galaxies(TABLE1_FILE)
    references = read_references(REFS_FILE)
    local_galaxies = CSV.read(LOCAL_GALAXIES_FILE, DataFrame)

    sample = add_metadata(raw, sings_galaxies, references, local_galaxies)
    coverage = make_coverage(sample)
    validate_catalog(raw, sample, coverage)

    CSV.write(OUTPUT_FILE, sample)
    CSV.write(COVERAGE_FILE, coverage)

    println("SINGS HII-region rows:       $(nrow(raw))")
    println("Matched galaxies:            $(count(coverage.in_sings_point_catalog)) / $(nrow(coverage))")
    println("Matched catalog rows:        $(nrow(sample))")
    println("KK04 fit-ready points:       $(sum(coverage.n_fit_ready_KK04))")
    println("PT05 fit-ready points:       $(sum(coverage.n_fit_ready_PT05))")
    println("Wrote: $(relpath(OUTPUT_FILE, ROOT))")
    println("Wrote: $(relpath(COVERAGE_FILE, ROOT))")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
