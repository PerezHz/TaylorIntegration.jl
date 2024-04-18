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
        @assert issorted(t) || issorted(t, rev = true)
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

vecsol(::Nothing, ::Int) = nothing
vecsol(v::AbstractVector, n::Int) = view(v, 1:n)

matsol(::Nothing, ::Int) = nothing
matsol(m::Matrix, n::Int) = view(transpose(view(m,:,1:n)),1:n,:)

build_solution(t::AbstractVector{T}, x::Vector{U}, p::Union{Nothing, Vector{Taylor1{U}}}, nsteps::Int) where {T, U} =
TaylorSolution(vecsol(t, nsteps), vecsol(x, nsteps), isnothing(p) ? p : vecsol(p, nsteps-1))
build_solution(t::AbstractVector{T}, x::Matrix{U}, p::Union{Nothing, Matrix{Taylor1{U}}}, nsteps::Int) where {T, U} =
TaylorSolution(vecsol(t, nsteps), matsol(x, nsteps), isnothing(p) ? p : matsol(p, nsteps-1))

build_solution(t::AbstractVector{T}, x::Vector{U}) where {T, U} = TaylorSolution(t, x)
build_solution(t::AbstractVector{T}, x::Matrix{U}) where {T, U} = TaylorSolution(t, transpose(x))

# Custom print
function Base.show(io::IO, sol::TaylorSolution)
    tspan = minmax(sol.t[1], sol.t[end])
    S = eltype(sol.x)
    nvars = size(sol.x, 2)
    plural = nvars > 1 ? "s" : ""
    print(io, "tspan: ", tspan, ", x: ", nvars, " ", S, " variable"*plural)
end

@doc raw"""
    timeindex(sol::TaylorSolution, t::TT) where TT

Return the index of `sol.t` corresponding to `t` and the time elapsed from `sol.t0`
to `t`.
"""
function timeindex(sol::TaylorSolution, t::TT) where TT
    t0 = sol.t[1]
    _t = constant_term(constant_term(t))  # Current time
    tmin, tmax = minmax(sol.t[end], t0)   # Min and max time in sol

    @assert tmin ≤ _t ≤ tmax "Evaluation time outside range of interpolation"

    if _t == sol.t[end]        # Compute solution at final time from last step expansion
        ind = lastindex(sol.t) - 1
    elseif issorted(sol.t)       # Forward integration
        ind = searchsortedlast(sol.t, _t)
    elseif issorted(sol.t, rev=true) # Backward integration
        ind = searchsortedlast(sol.t, _t, rev=true)
    end
    # Time since the start of the ind-th timestep
    δt = t - sol.t[ind]
    # Return index and elapsed time since i-th timestep
    return ind::Int, δt::TT
end

# Function-like (callability) methods

const TaylorSolutionCallingArgs{T,U} = Union{T, U, Taylor1{U}, TaylorN{U}, Taylor1{TaylorN{U}}} where {T,U}

@doc raw"""
    (sol::TaylorSolution{T, U, 1})(t::T) where {T, U}
    (sol::TaylorSolution{T, U, 1})(t::TT) where {T, U, TT<:TaylorSolutionCallingArgs{T,U}}
    (sol::TaylorSolution{T, U, 2})(t::T) where {T, U}
    (sol::TaylorSolution{T, U, 2})(t::TT) where {T, U, TT<:TaylorSolutionCallingArgs{T,U}}

Evaluate `sol.x` at time `t`.

See also [`timeindex`](@ref).
"""
function (sol::TaylorSolution{T, U, 1})(t::T) where {T, U}
    # Get index of sol.x that interpolates at time t
    ind::Int, δt::T = timeindex(sol, t)
    # Evaluate sol.x[ind] at δt
    return (sol.p[ind])(δt)::U
end
function (sol::TaylorSolution{T, U, 1})(t::TT) where {T, U, TT<:TaylorSolutionCallingArgs{T,U}}
    # Get index of sol.x that interpolates at time t
    ind::Int, δt::TT = timeindex(sol, t)
    # Evaluate sol.x[ind] at δt
    return (sol.p[ind])(δt)::TT
end

function (sol::TaylorSolution{T, U, 2})(t::T) where {T, U}
    # Get index of sol.x that interpolates at time t
    ind::Int, δt::T = timeindex(sol, t)
    # Evaluate sol.x[ind] at δt
    return view(sol.p, ind, :)(δt)::Vector{U}
end
function (sol::TaylorSolution{T, U, 2})(t::TT) where {T, U, TT<:TaylorSolutionCallingArgs{T,U}}
    # Get index of sol.x that interpolates at time t
    ind::Int, δt::TT = timeindex(sol, t)
    # Evaluate sol.x[ind] at δt
    return view(sol.p, ind, :)(δt)::Vector{TT}
end