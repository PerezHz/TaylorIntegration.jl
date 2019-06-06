# This file is part of the TaylorIntegration.jl package; MIT licensed

module TaylorIntegration

using Reexport
@reexport using TaylorSeries
using LinearAlgebra
using Markdown
using Requires


export taylorinteg, lyap_taylorinteg, @taylorize

include("explicitode.jl")
include("lyapunovspectrum.jl")
include("rootfinding.jl")
include("parse_eqs.jl")

function __init__()
    # Remember to delete this message when we release the next minor/major version
    @warn("""\n\n
        # Breaking changes

        `TaylorIntegration.jl` follows now (≥ v0.5.0) the convention
        of `DifferentialEquations.jl` for the function containing
        the differential equation to be integrated. The function
        must have the form `f(x, p, t)` for one dependent variable,
        or `f!(dx, x, p, t)` for several dependent variables, where
        `dx` is mutated.\n
    """)

    @require DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e" include("common.jl")
end

end #module
