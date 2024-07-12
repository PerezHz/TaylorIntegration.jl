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

function __init__()

    @static if !isdefined(Base, :get_extension)
        @require OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed" begin
            include("../ext/TaylorIntegrationDiffEq.jl")
        end
    end
end

@inline function diffeq!(a::Taylor1{T}, b::Taylor1{T}, k::Int) where {T<:TaylorSeries.NumberNotSeries}
    a[k] = b[k-1]/k
    return nothing
end

@inline function diffeq!(res::Taylor1{Taylor1{T}}, a::Taylor1{Taylor1{T}},
        k::Int) where {T<:TaylorSeries.NumberNotSeries}
    for l in eachindex(a[k-1])
        res[k][l] = a[k-1][l]/k
    end
    return nothing
end

@inline function diffeq!(res::Taylor1{TaylorN{T}}, a::Taylor1{TaylorN{T}},
        k::Int) where {T<:TaylorSeries.NumberNotSeries}
    for l in eachindex(a[k-1])
        for m in eachindex(a[k-1][l])
            res[k][l][m] = a[k-1][l][m]/k
        end
    end
    return nothing
end

end #module
