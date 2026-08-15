module GalacticWind

using QuadGK
import DifferentialEquations as DE

include("disk.jl")
include("cgm.jl")
include("inverse_model.jl")
include("forward_model.jl")
include("metallicity.jl")

end # module