# This file is part of the TaylorIntegration.jl package; MIT licensed

abstract type AbstractTaylorSolution{T<:Real, U<:Number} end

struct TaylorSolution{T, U, N, VT<:AbstractVector{T}, AX<:AbstractArray{U,N},
        P<:Union{Nothing, Array{Taylor1{U}, N}}} <: AbstractTaylorSolution{T, U}
    t::VT
    x::AX
    p::P
    function TaylorSolution{T, U, N, VT, AX, P}(t::VT, x::AX, p::P) where {T, U, 
            VT, AX, P, N}
        @assert length(t) == size(x, 1)
        !isnothing(p) && begin
            @assert size(x, 1) - 1 == size(p, 1)
            @assert size(x)[2:end] == size(p)[2:end]
        end
        return new{T, U, N, VT, AX, P}(t, x, p)
    end
end
TaylorSolution(t::VT, x::AX, p::P) where {T, U, N, VT<:AbstractVector{T}, 
    AX<:AbstractArray{U,N}, P<:Union{Nothing, Array{Taylor1{U},N}}} = 
    TaylorSolution{T, U, N, VT, AX, P}(t, x, p)
TaylorSolution(t, x) = TaylorSolution(t, x, nothing)
