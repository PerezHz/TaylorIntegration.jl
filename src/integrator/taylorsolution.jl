# This file is part of the TaylorIntegration.jl package; MIT licensed

abstract type AbstractTaylorSolution{T<:Real, U<:Number} end

struct TaylorSolution{T, U, N, VT<:AbstractVector{T}, AX<:AbstractArray{U,N},
        P<:Union{Nothing, AbstractArray{Taylor1{U}, N}}} <: AbstractTaylorSolution{T, U}
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
    AX<:AbstractArray{U,N}, P<:Union{Nothing, AbstractArray{Taylor1{U},N}}} =
    TaylorSolution{T, U, N, VT, AX, P}(t, x, p)
TaylorSolution(t, x) = TaylorSolution(t, x, nothing)

vecsol(p::Nothing, n::Int) = nothing
vecsol(v::AbstractVector, n::Int) = view(v, 1:n)

matsol(p::Nothing, n::Int) = nothing
matsol(m::Matrix, n::Int) = view(transpose(view(m,:,1:n)),1:n,:)

build_solution(t::AbstractVector{T}, x::Vector{U}, p::Union{Nothing, Vector{Taylor1{U}}}, nsteps::Int) where {T, U} =
TaylorSolution(vecsol(t, nsteps), vecsol(x, nsteps), isnothing(p) ? p : vecsol(p, nsteps-1))
build_solution(t::AbstractVector{T}, x::Matrix{U}, p::Union{Nothing, Matrix{Taylor1{U}}}, nsteps::Int) where {T, U} =
TaylorSolution(vecsol(t, nsteps), matsol(x, nsteps), isnothing(p) ? p : matsol(p, nsteps-1))

build_solution(t::AbstractVector{T}, x::Vector{U}) where {T, U} = TaylorSolution(t, x)
build_solution(t::AbstractVector{T}, x::Matrix{U}) where {T, U} = TaylorSolution(t, transpose(x))
