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
include("integrator.jl")
include("lyapunovspectrum.jl")
include("rootfinding.jl")
include("common.jl")
include("adsbinarynode.jl")
include("adstaylorinteg.jl")

function __init__()

    @static if !isdefined(Base, :get_extension)
        @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
            include("../ext/TaylorIntegrationDiffEq.jl")
        end
    end
end

end #module
