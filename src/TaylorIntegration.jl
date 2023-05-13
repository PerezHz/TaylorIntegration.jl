# This file is part of the TaylorIntegration.jl package; MIT licensed

module TaylorIntegration

using Reexport
@reexport using TaylorSeries
using LinearAlgebra
using Markdown
using Requires
using InteractiveUtils: methodswith


export taylorinteg, lyap_taylorinteg, @taylorize

include("parse_eqs.jl")
include("explicitode.jl")
include("lyapunovspectrum.jl")
include("rootfinding.jl")

function __init__()
    @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" include("common.jl")
end

end #module
