using CSV
using DataFrames

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const INPUT = joinpath(ROOT, "input")

const SPARC_FILE = joinpath(INPUT, "sparc", "SPARC_Lelli2016c.mrt")
const SPARC_RC_FILE = joinpath(INPUT, "sparc", "MassModels_Lelli2016c.mrt")
const THINGS_FILE = joinpath(INPUT, "things", "Galaxies.dat")
const LEROY_FILE = joinpath(INPUT, "leroy2008", "GlobalGalaxyQuantities.dat")
const LEROY_RADIAL = joinpath(INPUT, "leroy2008", "RadialBins.dat")
const BIGIEL_RADIAL = joinpath(INPUT, "bigiel2010", "RadialProfiles.csv")

const OUTDIR = INPUT

const ALIASES = Dict("HOI"  => "UGC5139", "HOII" => "UGC4305") # necessary because 
# Leroy et al. (2008) derive their spiral-galaxy H2 columns from HERACLES
# CO(2-1), with BIMA SONG/other CO used outside this target list.
const HERACLES_GALAXIES = Set([
    "DDO154",
    "UGC5139",
    "UGC4305",
    "IC2574",
    "NGC628",
    "NGC925",
    "NGC2841",
    "NGC2903",
    "NGC2976",
    "NGC3184",
    "NGC3198",
    "NGC3351",
    "NGC3521",
    "NGC4214",
    "NGC4736",
    "NGC5055",
    "NGC6946",
    "NGC7331",
])

function normalize_name(name)
    s = uppercase(replace(strip(name), r"[\s,-]" => ""))

    m = match(r"^(NGC|UGC|IC|DDO)0*(\d+)$", s)
    if m !== nothing
        s = m.captures[1] * string(parse(Int, m.captures[2]))
    end

    get(ALIASES, s, s)
end

function field(line, a, b)
    line = length(line) < b ? rpad(line, b) : line
    strip(line[a:b])
end

floatfield(line, a, b) =
    something(tryparse(Float64, field(line, a, b)), missing)

intfield(line, a, b) =
    something(tryparse(Int, field(line, a, b)), missing)

pow10(x) = ismissing(x) ? missing : 10.0^x

function read_sparc(path)
    rows = NamedTuple[]

    for line in eachline(path)
        x = split(strip(line))
        length(x) < 18 && continue

        T     = tryparse(Int, x[2])
        D     = tryparse(Float64, x[3])
        inc   = tryparse(Float64, x[6])
        Reff  = tryparse(Float64, x[10])
        Rdisk = tryparse(Float64, x[12])
        MHI   = tryparse(Float64, x[14])
        RHI   = tryparse(Float64, x[15])
        Vflat = tryparse(Float64, x[16])
        Q     = tryparse(Int, x[18])

        any(isnothing, (T, D, inc, Reff, Rdisk, MHI, RHI, Vflat, Q)) && continue

        name = x[1]

        push!(rows, (
            galaxy = normalize_name(name),
            name_sparc = name,
            T_sparc = T,
            D_sparc_Mpc = D,
            inc_sparc_deg = inc,
            Reff_sparc_kpc = Reff,
            Rdisk_sparc_kpc = Rdisk,
            MHI_sparc_Msun = MHI * 1e9,
            RHI_sparc_kpc = RHI,
            Vflat_sparc_kms = Vflat,
            Q_sparc = Q,
        ))
    end

    DataFrame(rows)
end

function read_things(path)
    rows = NamedTuple[]

    for line in eachline(path)
        name = field(line, 1, 8)
        D = floatfield(line, 51, 54)

        isempty(name) && continue
        ismissing(D) && continue

        push!(rows, (
            galaxy = normalize_name(name),
            name_things = name,
            aliases_things = field(line, 10, 24),
            D_things_Mpc = D,
            inc_things_deg = intfield(line, 81, 82),
            PA_things_deg = intfield(line, 84, 86),
            metallicity_things = floatfield(line, 93, 96),
            SFR_things_Msun_yr = floatfield(line, 102, 106),
            T_things = intfield(line, 112, 113),
        ))
    end

    DataFrame(rows)
end

function read_leroy(path)
    rows = NamedTuple[]

    for line in eachline(path)
        name = field(line, 1, 8)
        D = floatfield(line, 10, 13)

        isempty(name) && continue
        ismissing(D) && continue

        logMstar = floatfield(line, 47, 50)
        logMHI = floatfield(line, 52, 55)
        logMH2 = floatfield(line, 59, 61)

        push!(rows, (
            galaxy = normalize_name(name),
            name_leroy = name,
            D_leroy_Mpc = D,
            inc_leroy_deg = intfield(line, 15, 16),
            PA_leroy_deg = intfield(line, 18, 20),
            morph_leroy = field(line, 22, 25),
            MB_leroy = floatfield(line, 27, 31),
            R25_leroy_kpc = floatfield(line, 33, 36),
            Vflat_leroy_kms = floatfield(line, 38, 40),
            lflat_leroy_kpc = floatfield(line, 42, 45),

            logMstar_leroy = logMstar,
            logMHI_leroy = logMHI,
            logMH2_leroy = logMH2,

            Mstar_leroy_Msun = pow10(logMstar),
            MHI_leroy_Msun = pow10(logMHI),
            MH2_leroy_Msun = pow10(logMH2),

            MH2_upper_limit = field(line, 57, 58) == "<=",

            SFR_leroy_Msun_yr = floatfield(line, 64, 68),
            lstar_leroy_kpc = floatfield(line, 70, 72),
            lSFR_leroy_kpc = floatfield(line, 74, 76),
            lCO_leroy_kpc = floatfield(line, 78, 80),
        ))
    end

    DataFrame(rows)
end

function read_sparc_rotation_curves(path)
    rows = NamedTuple[]

    for line in eachline(path)
        R = floatfield(line, 20, 25)
        ismissing(R) && continue

        name = field(line, 1, 11)

        push!(rows, (
            galaxy = normalize_name(name),
            R_kpc = R,
            Vobs_kms = floatfield(line, 27, 32),
            e_Vobs_kms = floatfield(line, 34, 38),
            Vgas_kms = floatfield(line, 40, 45),
            Vdisk_kms = floatfield(line, 47, 52),
            Vbul_kms = floatfield(line, 54, 59),
            SBdisk_Lsun_pc2 = floatfield(line, 61, 67),
            SBbul_Lsun_pc2 = floatfield(line, 69, 76),
        ))
    end

    DataFrame(rows)
end

function read_leroy_radial(path)
    rows = NamedTuple[]

    for line in eachline(path)
        R = floatfield(line, 10, 13)
        ismissing(R) && continue

        name = field(line, 1, 8)
        galaxy = normalize_name(name)

        sfr = floatfield(line, 56, 61)
        e_sfr = floatfield(line, 63, 68)
        sfr_fuv = floatfield(line, 70, 75)
        sfr_24 = floatfield(line, 77, 82)
        sigma_H2 = floatfield(line, 30, 35)

        push!(rows, (
            galaxy = galaxy,
            R_kpc = R,
            R_R25 = floatfield(line, 15, 18),

            SigmaHI_Msun_pc2 = floatfield(line, 20, 24),
            e_SigmaHI_Msun_pc2 = floatfield(line, 26, 28),

            SigmaH2_Msun_pc2 = sigma_H2,
            e_SigmaH2_Msun_pc2 = floatfield(line, 37, 40),
            SigmaH2_source = ismissing(sigma_H2) ? missing :
                galaxy in HERACLES_GALAXIES ? "HERACLES" : "BIMA_SONG_or_other",

            SigmaStar_Msun_pc2 = floatfield(line, 42, 48),
            e_SigmaStar_Msun_pc2 = floatfield(line, 50, 54),

            SigmaSFR_Msun_yr_kpc2 =
                ismissing(sfr) ? missing : sfr * 1e-4,

            e_SigmaSFR_Msun_yr_kpc2 =
                ismissing(e_sfr) ? missing : e_sfr * 1e-4,

            SigmaSFR_FUV_Msun_yr_kpc2 =
                ismissing(sfr_fuv) ? missing : sfr_fuv * 1e-4,

            SigmaSFR_24_Msun_yr_kpc2 =
                ismissing(sfr_24) ? missing : sfr_24 * 1e-4,
        ))
    end

    DataFrame(rows)
end

function read_bigiel_radial(path)
    radial = CSV.read(path, DataFrame)
    required = [
        :galaxy,
        :R_R25,
        :SigmaHI_Msun_pc2,
        :e_SigmaHI_Msun_pc2,
        :SigmaSFR_Msun_yr_kpc2,
        :e_SigmaSFR_stat_Msun_yr_kpc2,
        :e_SigmaSFR_Msun_yr_kpc2,
    ]
    missing_columns = setdiff(required, propertynames(radial))
    isempty(missing_columns) ||
        error("Bigiel radial table is missing columns: $(join(missing_columns, ", "))")

    radial.galaxy = normalize_name.(radial.galaxy)
    all(radial.R_R25 .>= 1.0) ||
        error("Bigiel radial table must contain only outer-disk points at R/R25 >= 1")
    return radial
end

sparc = read_sparc(SPARC_FILE)
things = read_things(THINGS_FILE)
leroy = read_leroy(LEROY_FILE)
bigiel_radial = read_bigiel_radial(BIGIEL_RADIAL)

galaxies = outerjoin(sparc, things, on=:galaxy)
galaxies = outerjoin(galaxies, leroy, on=:galaxy)

galaxies.in_sparc = .!ismissing.(galaxies.name_sparc)
galaxies.in_things = .!ismissing.(galaxies.name_things)
galaxies.in_leroy = .!ismissing.(galaxies.name_leroy)
galaxies.in_bigiel2010 = in.(galaxies.galaxy, Ref(Set(bigiel_radial.galaxy)))
galaxies.in_heracles = in.(galaxies.galaxy, Ref(HERACLES_GALAXIES))

sort!(galaxies, :galaxy)

triple = filter(
    row -> row.in_sparc && row.in_things && row.in_leroy && row.Q_sparc == 1,
    galaxies,
)
all(triple.Q_sparc .== 1) || error("The overlap sample contains a SPARC galaxy with Q != 1")

rotation_curves = read_sparc_rotation_curves(SPARC_RC_FILE)
radial_profiles = read_leroy_radial(LEROY_RADIAL)

sort!(rotation_curves, [:galaxy, :R_kpc])
sort!(radial_profiles, [:galaxy, :R_kpc])
sort!(bigiel_radial, [:galaxy, :R_R25])

mkpath(OUTDIR)

CSV.write(joinpath(OUTDIR, "galaxies.csv"), galaxies)
CSV.write(joinpath(OUTDIR, "galaxies_triple_overlap.csv"), triple)
CSV.write(joinpath(OUTDIR, "sparc_rotation_curves.csv"), rotation_curves)
CSV.write(joinpath(OUTDIR, "leroy_radial_profiles.csv"), radial_profiles)
CSV.write(joinpath(OUTDIR, "bigiel_radial_profiles.csv"), bigiel_radial)
