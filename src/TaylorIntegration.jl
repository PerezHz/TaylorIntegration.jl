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
    @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
        @info("`@require` OrdinaryDiffEq")

        @require DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e" begin
            @info("`@require` DiffEqBase")
            using OrdinaryDiffEq
            include("common.jl")
        end
    end

    @require DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e" begin
        using OrdinaryDiffEq
    end

end

end #module
