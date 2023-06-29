# This file is part of the TaylorIntegration.jl package; MIT licensed

module TaylorIntegration

using Reexport
@reexport using TaylorSeries
using LinearAlgebra
using Markdown
using InteractiveUtils: methodswith
if !isdefined(Base, :get_extension)
    using Requires
end

export taylorinteg, lyap_taylorinteg, @taylorize

include("parse_eqs.jl")
include("explicitode.jl")
include("lyapunovspectrum.jl")
include("rootfinding.jl")
include("common.jl")

function __init__()

    ### Breaking change warning added in v0.14.0; to be deleted in next minor version (v0.15.0)
    @warn("""\n\n
        # Breaking changes
        When dense output from `taylorinteg` is requested by the user,
        the first row of the dense output polynomials matrix now represents
        the Taylor expansion of the solution around the initial condition
        w.r.t. the independent variable in the ODE. Previously it represented
        the initial condition plus a Taylor1 perturbation.\n
    """)

    @static if !isdefined(Base, :get_extension)
        @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
            include("../ext/TaylorIntegrationDiffEq.jl")
        end
    end
end

end #module
