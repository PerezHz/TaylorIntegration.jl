# This file is part of the TaylorIntegration.jl package; MIT licensed

abstract type AbstractTaylorSolution{T<:Real,U<:Number} end

## Constructors

"""
TaylorSolution{T, U, N, VT<:AbstractVector{T}, AX<:AbstractArray{U,N},
        P<:Union{Nothing, AbstractArray{Taylor1{U}, N}},
        VTE<:Union{Nothing, AbstractVector{U}},
        AXE<:Union{Nothing, AbstractArray{U, N}},
        VΛ<:Union{Nothing, AbstractArray{U,N}}} <: AbstractTaylorSolution{T, U}

This `struct` represents the return type for `taylorinteg`. Fields `t` and `x` represent,
respectively, a vector with the values of time (independent variable), and a vector with the
computed values of the dependent variable(s). When `taylorinteg` is called with `dense=true`,
then field `p` stores the Taylor polynomial expansion computed at each time step. Fields
`tevents`, `xevents` and `gresids` are related to root-finding methods of `taylorinteg`, while
`λ` is related to the output of [lyap_taylorinteg](@ref).
"""
@auto_hash_equals struct TaylorSolution{
    T,
    U,
    N,
    VT<:AbstractVector{T},
    AX<:AbstractArray{U,N},
    P<:Union{Nothing,AbstractArray{Taylor1{U},N}},
    VTE<:Union{Nothing,AbstractVector{U}},
    AXE<:Union{Nothing,AbstractArray{U,N}},
    VΛ<:Union{Nothing,AbstractArray{U,N}},
} <: AbstractTaylorSolution{T,U}
    t::VT
    x::AX
    p::P
    tevents::VTE
    xevents::AXE
    gresids::VTE
    λ::VΛ
    function TaylorSolution{T,U,N,VT,AX,P,VTE,AXE,VΛ}(
        t::VT,
        x::AX,
        p::P,
        tevents::VTE,
        xevents::AXE,
        gresids::VTE,
        λ::VΛ,
    ) where {T,U,N,VT,AX,P,VTE,AXE,VΛ}
        @assert length(t) == size(x, 1)
        @assert issorted(t) || issorted(t, rev = true)
        !isnothing(p) && begin
            @assert size(x, 1) - 1 == size(p, 1)
            @assert size(x)[2:end] == size(p)[2:end]
        end
        @assert isnothing(tevents) == isnothing(xevents) == isnothing(gresids) "`Nothing`-ness must be consistent across `tevents`, `xevents` and `gresids`."
        !isnothing(tevents) && begin
            @assert length(tevents) == size(xevents, 1)
            @assert size(xevents, 2) == size(x, 2)
            @assert length(tevents) == length(gresids)
        end
        !isnothing(λ) && @assert size(λ) == size(x)
        return new{T,U,N,VT,AX,P,VTE,AXE,VΛ}(t, x, p, tevents, xevents, gresids, λ)
    end
end
TaylorSolution(
    t::VT,
    x::AX,
    p::P,
    tevents::VTE,
    xevents::AXE,
    gresids::VTE,
    λ::VΛ,
) where {
    T,
    U,
    N,
    VT<:AbstractVector{T},
    AX<:AbstractArray{U,N},
    P<:Union{Nothing,AbstractArray{Taylor1{U},N}},
    VTE<:Union{Nothing,AbstractVector{U}},
    AXE<:Union{Nothing,AbstractArray{U,N}},
    VΛ<:Union{Nothing,AbstractArray{U,N}},
} = TaylorSolution{T,U,N,VT,AX,P,VTE,AXE,VΛ}(t, x, p, tevents, xevents, gresids, λ)

# 4-arg constructor (root-finding and Lyapunov fields are nothing)
TaylorSolution(t, x, p, ::Nothing) =
    TaylorSolution(t, x, p, nothing, nothing, nothing, nothing)
# 3-arg constructor (root-finding and Lyapunov fields are nothing; helps not to write too many nothings)
TaylorSolution(t, x, p) = TaylorSolution(t, x, p, nothing)
# 2-arg constructor (dense polynomial, root-finding and Lyapunov fields are nothing)
TaylorSolution(t, x) = TaylorSolution(t, x, nothing)

function _empty_polynomial_array_error()
    return ArgumentError(
        "Cannot infer solution values from an empty polynomial array `p`; construct with `TaylorSolution(t, x, p)` instead.",
    )
end

function _solution_values(t::AbstractVector{T}, p::AbstractArray{Taylor1{U},1}) where {T,U}
    @assert length(t) == length(p) + 1
    isempty(p) && throw(_empty_polynomial_array_error())
    x = Vector{U}(undef, length(t))
    x[1] = p[1](zero(T))
    for i in eachindex(p)
        x[i+1] = p[i](t[i+1] - t[i])
    end
    return x
end

function _solution_values(
    t::AbstractVector{T},
    p::AbstractArray{Taylor1{U},N},
) where {T,U,N}
    @assert length(t) == size(p, 1) + 1
    isempty(p) && throw(_empty_polynomial_array_error())
    x = Array{U,N}(undef, length(t), size(p)[2:end]...)
    selectdim(x, 1, 1) .= selectdim(p, 1, 1)(zero(T))
    for i in axes(p, 1)
        selectdim(x, 1, i + 1) .= selectdim(p, 1, i)(t[i+1] - t[i])
    end
    return x
end

function TaylorSolution(t::AbstractVector, p::AbstractArray{<:Taylor1})
    if length(t) == size(p, 1) + 1
        return TaylorSolution(t, _solution_values(t, p), p)
    else
        return TaylorSolution(t, p, nothing)
    end
end

# arraysol: auxiliary function for solution construction

arraysol(::Nothing, ::Int) = nothing
arraysol(v::AbstractVector, n::Int) = view(v, 1:n)
arraysol(m::AbstractMatrix, n::Int) = view(transpose(view(m, :, 1:n)), 1:n, :)

# build_solution

"""
    build_solution(t, x, p, nsteps)
    build_solution(t, x)

Helper function to build a [`TaylorSolution`](@ref) from a call to
[`taylorinteg`](@ref).

"""
build_solution(t::AbstractVector{T}, x::Vector{U}, ::Nothing, nsteps::Int) where {T,U} =
    TaylorSolution(arraysol(t, nsteps), arraysol(x, nsteps))
build_solution(
    t::AbstractVector{T},
    x::Vector{U},
    p::Vector{Taylor1{U}},
    nsteps::Int,
) where {T,U} =
    TaylorSolution(arraysol(t, nsteps), arraysol(x, nsteps), arraysol(p, nsteps - 1))
build_solution(t::AbstractVector{T}, x::Matrix{U}, ::Nothing, nsteps::Int) where {T,U} =
    TaylorSolution(arraysol(t, nsteps), arraysol(x, nsteps))
build_solution(
    t::AbstractVector{T},
    x::Matrix{U},
    p::Matrix{Taylor1{U}},
    nsteps::Int,
) where {T,U} =
    TaylorSolution(arraysol(t, nsteps), arraysol(x, nsteps), arraysol(p, nsteps - 1))

build_solution(t::AbstractVector{T}, x::Vector{U}) where {T,U} = TaylorSolution(t, x)
build_solution(t::AbstractVector{T}, x::Matrix{U}) where {T,U} =
    TaylorSolution(t, transpose(x))

### Custom print

function Base.show(io::IO, sol::TaylorSolution)
    tspan = minmax(sol.t[1], sol.t[end])
    S = eltype(sol.x)
    nvars = size(sol.x, 2)
    plural = nvars > 1 ? "s" : ""
    print(io, "tspan: ", tspan, ", x: ", nvars, " ", S, " variable" * plural)
end

# Function-like (callability or "functor") methods

@doc doc"""
    timeindex(sol::TaylorSolution, t::TT) where TT

Return the index of `sol.t` corresponding to `t` and the time elapsed from `sol.t0`
to `t`.
"""
function timeindex(sol::TaylorSolution, t::TT) where {TT}
    t0 = sol.t[1]
    _t = constant_term(constant_term(t))  # Current time
    tmin, tmax = minmax(sol.t[end], t0)   # Min and max time in sol

    @assert tmin ≤ _t ≤ tmax "Evaluation time outside range of interpolation"

    if _t == sol.t[end]        # Compute solution at final time from last step expansion
        ind = lastindex(sol.t) - 1
    elseif issorted(sol.t)       # Forward integration
        ind = searchsortedlast(sol.t, _t)
    elseif issorted(sol.t, rev = true) # Backward integration
        ind = searchsortedlast(sol.t, _t, rev = true)
    end
    # Time since the start of the ind-th timestep
    δt = t - sol.t[ind]
    # Return index and elapsed time since i-th timestep
    return ind::Int, δt::TT
end

const TaylorSolutionCallingArgs{T,U} =
    Union{T,U,Taylor1{U},TaylorN{U},Taylor1{TaylorN{U}}} where {T,U}

@doc doc"""
    (sol::TaylorSolution{T, U, 1})(t::T) where {T, U}
    (sol::TaylorSolution{T, U, 1})(t::TT) where {T, U, TT<:TaylorSolutionCallingArgs{T,U}}
    (sol::TaylorSolution{T, U, 2})(t::T) where {T, U}
    (sol::TaylorSolution{T, U, 2})(t::TT) where {T, U, TT<:TaylorSolutionCallingArgs{T,U}}

Evaluate `sol.x` at time `t`.

See also [`timeindex`](@ref).
"""
(sol::TaylorSolution)(t) = sol(Val(isnothing(sol.p)), t)

(sol::TaylorSolution{T,U,N})(
    ::Val{true},
    t::TT,
) where {T,U,N,TT<:TaylorSolutionCallingArgs{T,U}} = error(
    "`TaylorSolution` objects computed from calls to `taylorinteg` with `dense=false` are not callable.",
)

function (sol::TaylorSolution{T,U,1})(::Val{false}, t::T) where {T,U}
    # Get index of sol.x that interpolates at time t
    ind::Int, δt::T = timeindex(sol, t)
    # Evaluate sol.x[ind] at δt
    return (sol.p[ind])(δt)::U
end
function (sol::TaylorSolution{T,U,1})(
    ::Val{false},
    t::TT,
) where {T,U,TT<:TaylorSolutionCallingArgs{T,U}}
    # Get index of sol.x that interpolates at time t
    ind::Int, δt::TT = timeindex(sol, t)
    # Evaluate sol.x[ind] at δt
    return (sol.p[ind])(δt)::TT
end

function (sol::TaylorSolution{T,U,2})(::Val{false}, t::T) where {T,U}
    # Get index of sol.x that interpolates at time t
    ind::Int, δt::T = timeindex(sol, t)
    # Evaluate sol.x[ind] at δt
    return view(sol.p, ind, :)(δt)::Vector{U}
end
function (sol::TaylorSolution{T,U,2})(
    ::Val{false},
    t::TT,
) where {T,U,TT<:TaylorSolutionCallingArgs{T,U}}
    # Get index of sol.x that interpolates at time t
    ind::Int, δt::TT = timeindex(sol, t)
    # Evaluate sol.x[ind] at δt
    return view(sol.p, ind, :)(δt)::Vector{TT}
end

function _dense_polynomials(sol::TaylorSolution)
    isnothing(sol.p) && error(
        "`TaylorSolution` objects computed from calls to `taylorinteg` with `dense=false` do not store Taylor polynomials.",
    )
    return sol.p
end

TaylorSeries.order(sol::TaylorSolution) = TaylorSeries.order(first(_dense_polynomials(sol)))

_convert_numtype(::Type{T}, ::Type{<:Real}) where {T<:Real} = T
_convert_numtype(::Type{T}, ::Type{Complex{S}}) where {T<:Real,S<:Real} = Complex{T}
_convert_numtype(::Type{T}, ::Type{Taylor1{S}}) where {T<:Real,S<:Number} =
    Taylor1{_convert_numtype(T, S)}
_convert_numtype(::Type{T}, ::Type{TaylorN{S}}) where {T<:Real,S<:Number} =
    TaylorN{_convert_numtype(T, S)}
_convert_numtype(::Type{T}, ::Type{HomogeneousPolynomial{S}}) where {T<:Real,S<:Number} =
    HomogeneousPolynomial{_convert_numtype(T, S)}

_convert_number(::Type{T}, x::S) where {T<:Real,S<:Number} =
    convert(_convert_numtype(T, S), x)

_convert_array(::Type{T}, x::Nothing) where {T<:Real} = nothing
_convert_array(::Type{T}, x::AbstractArray) where {T<:Real} =
    map(y -> _convert_number(T, y), x)

function convert(::Type{T}, sol::TaylorSolution) where {T<:Real}
    return TaylorSolution(
        T.(sol.t),
        _convert_array(T, sol.x),
        _convert_array(T, sol.p),
        _convert_array(T, sol.tevents),
        _convert_array(T, sol.xevents),
        _convert_array(T, sol.gresids),
        _convert_array(T, sol.λ),
    )
end

function zero(
    ::Type{TaylorSolution{T,U,N,VT,AX,P,Nothing,Nothing,Nothing}},
) where {T<:Real,U<:Number,N,VT<:AbstractVector{T},AX<:AbstractArray{U,N},
    P<:Union{Nothing,AbstractArray{Taylor1{U},N}}}
    xdims = (1, ntuple(_ -> 0, Val(N - 1))...)
    t = zeros(T, 1)
    x = Array{U,N}(undef, xdims)
    p = P <: Nothing ? nothing : Array{Taylor1{U},N}(undef, (0, xdims[2:end]...)...)
    return TaylorSolution(t, x, p)
end

iszero(sol::TaylorSolution) = sol == zero(typeof(sol))

function reverse(sol::TaylorSolution{T,U,1}) where {T<:Real,U<:Number}
    p = _dense_polynomials(sol)
    prev = similar(p)
    ord = TaylorSeries.order(sol)
    prev[1] = sol(sol.t[end] + Taylor1(ord))
    for i in 2:length(p)
        prev[i] = p[end-i+2]
    end
    return TaylorSolution(reverse(sol.t), prev)
end

function reverse(sol::TaylorSolution{T,U,N}) where {T<:Real,U<:Number,N}
    p = _dense_polynomials(sol)
    prev = similar(p)
    ord = TaylorSeries.order(sol)
    selectdim(prev, 1, 1) .= sol(sol.t[end] + Taylor1(ord))
    for i in 2:size(p, 1)
        selectdim(prev, 1, i) .= selectdim(p, 1, size(p, 1)-i+2)
    end
    return TaylorSolution(reverse(sol.t), prev)
end

function flipsign(sol::TaylorSolution)
    p = _dense_polynomials(sol)
    t = flipsign.(sol.t, -one(eltype(sol.t)))
    t[1] = sol.t[1]
    return TaylorSolution(t, p(-Taylor1(TaylorSeries.order(sol))))
end

function join(bwd::TaylorSolution{T,U,N}, fwd::TaylorSolution{T,U,N}) where {T<:Real,U<:Number,N}
    @assert bwd.t[1] == fwd.t[1] "Initial time must be the same for both TaylorSolution"
    @assert TaylorSeries.order(bwd) == TaylorSeries.order(fwd) "Expansion order must be the same for both TaylorSolution"
    bwd_rev = reverse(bwd)
    t = vcat(bwd_rev.t, fwd.t[2:end])
    p = vcat(bwd_rev.p, fwd.p)
    return TaylorSolution(t, p)
end

struct TaylorSolutionSerialization{T,N}
    order::Int
    dims::NTuple{N,Int}
    t::Vector{T}
    p::Vector{T}
end

TaylorSeries.order(x::TaylorSolutionSerialization) = x.order

function writeas(
    ::Type{<:TaylorSolution{T,T,N,Vector{T},Array{T,N},Array{Taylor1{T},N},Nothing,Nothing,Nothing}},
) where {T<:Real,N}
    return TaylorSolutionSerialization{T,N}
end

function convert(
    ::Type{TaylorSolutionSerialization{T,N}},
    sol::TaylorSolution{T,T,N,Vector{T},Array{T,N},Array{Taylor1{T},N},Nothing,Nothing,Nothing},
) where {T<:Real,N}
    ord = TaylorSeries.order(sol)
    k = ord + 1
    dims = size(sol.p)
    p = Vector{T}(undef, k * length(sol.p))
    for (i, idx) in enumerate(eachindex(sol.p))
        p[(i-1)*k+1:i*k] = sol.p[idx].coeffs
    end
    return TaylorSolutionSerialization{T,N}(ord, dims, collect(sol.t), p)
end

function convert(
    ::Type{<:TaylorSolution{T,T,N}},
    sol::TaylorSolutionSerialization{T,N},
) where {T<:Real,N}
    ord = TaylorSeries.order(sol)
    k = ord + 1
    p = Array{Taylor1{T},N}(undef, sol.dims)
    for i in eachindex(p)
        p[i] = Taylor1{T}(sol.p[(i-1)*k+1:i*k], ord)
    end
    return TaylorSolution(sol.t, p)
end

convert(::Type{TaylorSolution}, sol::TaylorSolutionSerialization{T,N}) where {T<:Real,N} =
    convert(TaylorSolution{T,T,N}, sol)

struct TaylorSolutionNSerialization{T,N}
    vars::Vector{String}
    order::Int
    varorder::Int
    dims::NTuple{N,Int}
    t::Vector{T}
    p::Vector{T}
end

TaylorSeries.order(x::TaylorSolutionNSerialization) = x.order

function writeas(
    ::Type{<:TaylorSolution{T,TaylorN{T},N,Vector{T},Array{TaylorN{T},N},Array{Taylor1{TaylorN{T}},N},Nothing,Nothing,Nothing}},
) where {T<:Real,N}
    return TaylorSolutionNSerialization{T,N}
end

function convert(
    ::Type{TaylorSolutionNSerialization{T,Ndim}},
    sol::TaylorSolution{T,TaylorN{T},Ndim,Vector{T},Array{TaylorN{T},Ndim},Array{Taylor1{TaylorN{T}},Ndim},Nothing,Nothing,Nothing},
) where {T<:Real,Ndim}
    coeff = sol.p[1].coeffs[1]
    vars = TS.get_variable_names(TS.space(coeff))
    nvars = length(vars)
    dims = size(sol.p)
    ord = TaylorSeries.order(sol)
    k = ord + 1
    varorder = TaylorSeries.order(coeff)
    ncoeffs = binomial(nvars + varorder, varorder)
    p = Vector{T}(undef, length(sol.p) * k * ncoeffs)

    i = 1
    for ip in eachindex(sol.p)
        for it in 1:k
            for ivarord in 0:varorder
                for ihom in 1:binomial(nvars + ivarord - 1, ivarord)
                    p[i] = sol.p[ip].coeffs[it].coeffs[ivarord+1].coeffs[ihom]
                    i += 1
                end
            end
        end
    end

    return TaylorSolutionNSerialization{T,Ndim}(
        vars,
        ord,
        varorder,
        dims,
        collect(sol.t),
        p,
    )
end

function convert(
    ::Type{<:TaylorSolution{T,TaylorN{T},Ndim}},
    sol::TaylorSolutionNSerialization{T,Ndim},
) where {T<:Real,Ndim}
    vars = sol.vars
    nvars = length(vars)
    ord = TaylorSeries.order(sol)
    k = ord + 1
    varorder = sol.varorder
    ncoeffs = varorder + 1

    if TS.get_variable_names() != vars || TS.order() != varorder
        TS.variables!(T, vars, order = varorder)
    end

    p = Array{Taylor1{TaylorN{T}},Ndim}(undef, sol.dims)
    i = 1
    for ip in eachindex(p)
        taylor1_coeffs = Vector{TaylorN{T}}(undef, k)
        for it in 1:k
            taylorn_coeffs = Vector{HomogeneousPolynomial{T}}(undef, ncoeffs)
            for ivarord in 0:varorder
                nhom = binomial(nvars + ivarord - 1, ivarord)
                taylorn_coeffs[ivarord+1] =
                    HomogeneousPolynomial(sol.p[i:i+nhom-1], ivarord)
                i += nhom
            end
            taylor1_coeffs[it] = TaylorN(taylorn_coeffs, varorder)
        end
        p[ip] = Taylor1{TaylorN{T}}(taylor1_coeffs, ord)
    end

    return TaylorSolution(sol.t, p)
end

convert(::Type{TaylorSolution}, sol::TaylorSolutionNSerialization{T,N}) where {T<:Real,N} =
    convert(TaylorSolution{T,TaylorN{T},N}, sol)
