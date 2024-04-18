# This file is part of the TaylorIntegration.jl package; MIT licensed

abstract type AbstractTaylorSolution{T<:Real, U<:Number} end

## Constructors

struct TaylorSolution{T, U, N, VT<:AbstractVector{T}, AX<:AbstractArray{U,N},
        P<:Union{Nothing, AbstractArray{Taylor1{U}, N}},
        VTE<:Union{Nothing, AbstractVector{U}},
        AXE<:Union{Nothing, AbstractArray{U, N}}} <: AbstractTaylorSolution{T, U}
    t::VT
    x::AX
    p::P
    tevents::VTE
    xevents::AXE
    gresids::VTE
    function TaylorSolution{T, U, N, VT, AX, P, VTE, AXE}(t::VT, x::AX, p::P, tevents::VTE, xevents::AXE, gresids::VTE) where {T, U,
            N, VT, AX, P, VTE, AXE}
        @assert length(t) == size(x, 1)
        @assert issorted(t) || issorted(t, rev = true)
        !isnothing(p) && begin
            @assert size(x, 1) - 1 == size(p, 1)
            @assert size(x)[2:end] == size(p)[2:end]
        end
        @assert isnothing(tevents) == isnothing(xevents) == isnothing(gresids) "`Nothing`-ness across `tevents`, `xevents` and `gresids` must be consistent."
        !isnothing(tevents) && begin
            @assert length(tevents) == size(xevents, 1)
            @assert size(xevents, 2) == size(x, 2)
            @assert length(tevents) == length(gresids)
        end
        return new{T, U, N, VT, AX, P, VTE, AXE}(t, x, p, tevents, xevents, gresids)
    end
end
TaylorSolution(t::VT, x::AX, p::P, tevents::VTE, xevents::AXE, gresids::VTE) where {T, U, N, VT<:AbstractVector{T},
    AX<:AbstractArray{U,N}, P<:Union{Nothing, AbstractArray{Taylor1{U},N}}, VTE<:Union{Nothing, AbstractVector{U}},
    AXE<:Union{Nothing, AbstractArray{U, N}}} =
    TaylorSolution{T, U, N, VT, AX, P, VTE, AXE}(t, x, p, tevents, xevents, gresids)

# 4-arg constructor (event-related fields are nothing)
TaylorSolution(t, x, p, ::Nothing) = TaylorSolution(t, x, p, nothing, nothing, nothing)
# 3-arg constructor (event-related fields are nothing; helps not to write too many nothings)
TaylorSolution(t, x, p) = TaylorSolution(t, x, p, nothing)
# 2-arg constructor (dense polynomial and event-related fields are nothing)
TaylorSolution(t, x) = TaylorSolution(t, x, nothing)

### Solution construction auxiliary methods

vecsol(::Nothing, ::Int) = nothing
vecsol(v::AbstractVector, n::Int) = view(v, 1:n)

matsol(::Nothing, ::Int) = nothing
matsol(m::Matrix, n::Int) = view( transpose(view(m, :, 1:n)), 1:n, : )

### `build_solution`: a helper function for constructing `TaylorSolution`s

build_solution(t::AbstractVector{T}, x::Vector{U}, p::Union{Nothing, Vector{Taylor1{U}}}, nsteps::Int) where {T, U} =
    TaylorSolution(vecsol(t, nsteps), vecsol(x, nsteps), vecsol(p, nsteps-1))
build_solution(t::AbstractVector{T}, x::Matrix{U}, p::Union{Nothing, Matrix{Taylor1{U}}}, nsteps::Int) where {T, U} =
    TaylorSolution(vecsol(t, nsteps), matsol(x, nsteps), matsol(p, nsteps-1))

build_solution(t::AbstractVector{T}, x::Vector{U}) where {T, U} = TaylorSolution(t, x)
build_solution(t::AbstractVector{T}, x::Matrix{U}) where {T, U} = TaylorSolution(t, transpose(x))

### `build_solution` method for root-finding

build_solution(t::AbstractVector{T},
        x::Matrix{U},
        p::Union{Nothing, Matrix{Taylor1{U}}},
        tevents::AbstractVector{U},
        xevents::Matrix{U},
        gresids::AbstractVector{U},
        nsteps::Int,
        nevents::Int) where {T, U} =
    TaylorSolution(vecsol(t, nsteps),
        matsol(x, nsteps),
        matsol(p, nsteps-1),
        vecsol(tevents, nevents-1),
        matsol(xevents, nevents-1),
        vecsol(gresids, nevents-1))

#### `build_solution` method for root-finding with time-ranges

build_solution(t::AbstractVector{T},
        x::Matrix{U},
        tevents::AbstractVector{U},
        xevents::Matrix{U},
        gresids::AbstractVector{U},
        nevents::Int) where {T, U} =
    TaylorSolution(t,
        transpose(x),
        nothing,
        vecsol(tevents, nevents-1),
        matsol(xevents, nevents-1),
        vecsol(gresids, nevents-1))

### Custom print

function Base.show(io::IO, sol::TaylorSolution)
    tspan = minmax(sol.t[1], sol.t[end])
    S = eltype(sol.x)
    nvars = size(sol.x, 2)
    plural = nvars > 1 ? "s" : ""
    print(io, "tspan: ", tspan, ", x: ", nvars, " ", S, " variable"*plural)
end

### Callability ("functor") methods

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