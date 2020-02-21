# This file is part of the TaylorIntegration.jl package; MIT licensed

module TaylorIntegration

using Reexport
@reexport using TaylorSeries
using LinearAlgebra
using Markdown
using Requires
using InteractiveUtils: methodswith


export taylorinteg, lyap_taylorinteg, @taylorize

include("explicitode.jl")
include("lyapunovspectrum.jl")
include("rootfinding.jl")
include("parse_eqs.jl")

function __init__()
    @require DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e" include("common.jl")
end

end #module
