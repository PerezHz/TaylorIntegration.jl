# This file is part of the TaylorIntegration.jl package; MIT licensed

module TaylorIntegration

using Reexport
@reexport using TaylorSeries
using LinearAlgebra
using Markdown
using InteractiveUtils #: methodswith
using Parameters

export TaylorSolution, taylorinteg, lyap_taylorinteg, @taylorize, 
       taylorinteg_wrap, taylorinteg_optim!, taylorinteg_wrap_optim!,
       taylorinteg_ps!

include("parse_eqs.jl")
include("integrator/cache.jl")
include("integrator/jetcoeffs.jl")
include("integrator/stepsize.jl")
include("integrator/taylorstep.jl")
include("integrator/taylorsolution.jl")
include("integrator/taylorinteg.jl")
include("integrator/taylorinteg_wrap.jl")
include("integrator/taylorinteg_optim.jl")
include("lyapunovspectrum.jl")
include("rootfinding.jl")
include("rootfinding_wrap.jl")
include("poincare_section.jl")
include("common.jl")


@inline function solcoeff!(
    a::Taylor1{T},
    b::Taylor1{T},
    k::Int,
) where {T<:TaylorSeries.NumberNotSeries}
    @inbounds a[k] = b[k-1] / k
    return nothing
end

@inline function solcoeff!(
    res::Taylor1{Taylor1{T}},
    a::Taylor1{Taylor1{T}},
    k::Int,
) where {T<:TaylorSeries.NumberNotSeriesN}
    @inbounds for l in eachindex(a[k-1])
        # res[k][l] = a[k-1][l] / k
        TS.div!(res[k], a[k-1], k, l)
    end
    return nothing
end

@inline function solcoeff!(
    res::Taylor1{TaylorN{T}},
    a::Taylor1{TaylorN{T}},
    k::Int,
) where {T<:TaylorSeries.NumberNotSeries}
    @inbounds for l in eachindex(a[k-1])
        for m in eachindex(a[k-1][l])
            res[k][l][m] = a[k-1][l][m] / k
        end
    end
    return nothing
end

end #module
