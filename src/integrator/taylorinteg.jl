# This file is part of the TaylorIntegration.jl package; MIT licensed

"""
    _stored_value(x::Taylor1)
    _stored_value(x::TaylorN)
    _stored_value(x::T) where {T<:Number}

Return a value suitable for storage in solution arrays and integration caches.

Taylor polynomials are copied into fresh coefficient storage with
`TaylorSeries.identity!`, `isbits` numbers are stored directly, and other
`Number` subtypes fall back to `deepcopy`. This gives mutable number-like
values snapshot semantics while avoiding unnecessary copies for immutable scalar
values.
"""
@inline function _stored_value(x::Taylor1{U}) where {U<:Number}
    y = zero(x)
    TS.identity!(y, x)
    return y
end

@inline function _stored_value(x::TaylorN{U}) where {U<:Number}
    y = zero(x)
    TS.identity!(y, x)
    return y
end

@inline _stored_value(x::T) where {T<:Number} =
    Base.isbitstype(T) ? x : deepcopy(x)

"""
    _copy_value!(dest::Taylor1{U}, src::Taylor1{U}) where {U<:Number}
    _copy_value!(dest::TaylorN{U}, src::TaylorN{U}) where {U<:Number}
    _copy_value!(dest::T, src::T) where {T<:Number}

Copy `src` into reusable storage `dest` when possible and return the stored
value.

Matching `Taylor1` and `TaylorN` values with compatible lengths are updated in
place with `TaylorSeries.identity!`, preserving the object already held by the
cache. If the Taylor polynomial storage is incompatible, or if the value is an
immutable scalar-like number, the method returns a fresh [`_stored_value`](@ref)
instead. Use [`_store_value!`](@ref) when the caller has an array slot that can
be updated directly.
"""
@inline function _copy_value!(dest::Taylor1{U}, src::Taylor1{U}) where {U<:Number}
    length(dest) == length(src) || return _stored_value(src)
    TS.identity!(dest, src)
    return dest
end

@inline function _copy_value!(dest::TaylorN{U}, src::TaylorN{U}) where {U<:Number}
    length(dest) == length(src) || return _stored_value(src)
    TS.identity!(dest, src)
    return dest
end

@inline _copy_value!(dest::T, src::T) where {T<:Number} = _stored_value(src)

"""
    _store_value!(dest::AbstractVector{U}, src::U, i) where {U<:Number}
    _store_value!(dest::AbstractMatrix{U}, src::U, i, j) where {U<:Number}

Store `src` in `dest[i]` or `dest[i, j]`, reusing an already assigned mutable
Taylor slot when possible.

This is the slot-aware wrapper around [`_copy_value!`](@ref). If the element
type is `isbits` or the slot has not been assigned yet, a fresh
[`_stored_value`](@ref) is assigned. Otherwise, the existing slot value is
updated in place when possible, and replaced only when necessary.
"""
@inline function _store_value!(dest::AbstractVector{U}, src::U, i) where {U<:Number}
    if Base.isbitstype(U) || !isassigned(dest, i)
        @inbounds dest[i] = _stored_value(src)
    else
        @inbounds dest[i] = _copy_value!(dest[i], src)
    end
    return nothing
end

@inline function _store_value!(dest::AbstractMatrix{U}, src::U, i, j) where {U<:Number}
    if Base.isbitstype(U) || !isassigned(dest, i, j)
        @inbounds dest[i, j] = _stored_value(src)
    else
        @inbounds dest[i, j] = _copy_value!(dest[i, j], src)
    end
    return nothing
end

"""
    _stored_state(q0::AbstractVector{U}) where {U<:Number}

Return a freshly allocated state vector whose entries are independent stored
values copied from `q0`.

This is used for working initial conditions that may contain mutable Taylor
polynomials. It preserves the values from `q0` without aliasing the user's input
or later cache mutations.
"""
function _stored_state(q0::AbstractVector{U}) where {U<:Number}
    x0 = similar(q0)
    @inbounds for i in eachindex(q0)
        _store_value!(x0, q0[i], i)
    end
    return x0
end

"""
    _copy_state!(dest::AbstractVector{U}, src::AbstractVector{U}) where {U<:Number}

Copy `src` into the reusable state vector `dest`.

Each element is stored through [`_store_value!`](@ref), so compatible Taylor
entries are updated in place and incompatible or scalar entries are replaced in
the vector slot.
"""
function _copy_state!(dest::AbstractVector{U}, src::AbstractVector{U}) where {U<:Number}
    @inbounds for i in eachindex(src)
        _store_value!(dest, src[i], i)
    end
    return dest
end

"""
    _store_state_column!(
        xv::AbstractMatrix{U},
        nsteps::Int,
        x0::AbstractVector{U},
    ) where {U<:Number}

Store the state vector `x0` in column `nsteps` of `xv`. Previously assigned
non-isbits entries are reused when possible, so cache reuse does not require
allocating new Taylor polynomial objects for every stored state.

The integrator stores vector-valued states internally as `variables x steps`.
For immutable `isbits` element types, values are assigned directly. For mutable
Taylor-valued states, unassigned slots receive independent stored copies and
assigned slots are updated in place through [`_store_value!`](@ref).
"""
function _store_state_column!(xv::AbstractMatrix{U}, nsteps::Int,
        x0::AbstractVector{U}) where {U<:Number}
    @inbounds for i in eachindex(x0)
        _store_value!(xv, x0[i], i, nsteps)
    end
    return nothing
end

# set_psol!
@doc doc"""
    set_psol!(::Val{true}, psol::Array{Taylor1{U},1}, nsteps::Int, x::Taylor1{U}) where {U<:Number}
    set_psol!(::Val{true}, psol::Array{Taylor1{U},2}, nsteps::Int, x::Vector{Taylor1{U}}) where {U<:Number}
    set_psol!(::Val{false}, args...)

Auxiliary function to save Taylor polynomials in a call to [`taylorinteg`](@ref).

When the first argument is `Val(true)`, the method stores the current working
Taylor polynomial(s) into `psol`, the dense-output cache backing field `:p` in
[`TaylorSolution`](@ref). Scalar integrations store one polynomial per step;
vector integrations store one polynomial per component and step. Assigned
`psol` slots are reused in place through [`_store_value!`](@ref), so repeated
uses of a preallocated cache do not allocate fresh dense polynomials for every
stored step. When the first argument is `Val(false)`, the method returns
`nothing`.

See also [`init_psol`](@ref).
"""
@inline function set_psol!(
    ::Val{true},
    psol::Array{Taylor1{U},1},
    nsteps::Int,
    x::Taylor1{U},
) where {U<:Number}
    _store_value!(psol, x, nsteps)
    return nothing
end
@inline function set_psol!(
    ::Val{true},
    psol::Array{Taylor1{U},2},
    nsteps::Int,
    x::Vector{Taylor1{U}},
) where {U<:Number}
    @inbounds for j in eachindex(x)
        _store_value!(psol, x[j], j, nsteps)
    end
    return nothing
end
@inline set_psol!(::Val{false}, args...) = nothing

# init_psol
@doc doc"""
    init_psol(::Val{true}, maxsteps::Int, ::Int, ::Taylor1{U}) where {U<:Number}
    init_psol(::Val{true}, maxsteps::Int, dof::Int, ::Array{Taylor1{U},1}) where {U<:Number}
    init_psol(::Val{false}, ::Int, ::Int, ::Taylor1{U}) where {U<:Number}
    init_psol(::Val{false}, ::Int, ::Int, ::Array{Taylor1{U},1}) where {U<:Number}

Auxiliary function to initialize the dense-output storage `psol` during a call
to [`taylorinteg`](@ref).

When the first argument is `Val(true)`, the method allocates the array that will
store Taylor polynomials for dense output and later become field `:p` in
[`TaylorSolution`](@ref). Scalar integrations use a vector of length
`maxsteps`; vector integrations use a `dof x maxsteps` matrix. The storage is
left uninitialized so [`set_psol!`](@ref) can populate only the steps that are
actually taken. When the first argument is `Val(false)`, the method returns
`nothing`.
"""
@inline function init_psol(
    ::Val{true},
    maxsteps::Int,
    ::Int,
    ::Taylor1{U},
) where {U<:Number}
    return Array{Taylor1{U}}(undef, maxsteps)
end
@inline function init_psol(
    ::Val{true},
    maxsteps::Int,
    dof::Int,
    ::Array{Taylor1{U},1},
) where {U<:Number}
    return Array{Taylor1{U}}(undef, dof, maxsteps)
end

@inline init_psol(::Val{false}, ::Int, ::Int, ::Taylor1{U}) where {U<:Number} = nothing
@inline init_psol(::Val{false}, ::Int, ::Int, ::Array{Taylor1{U},1}) where {U<:Number} =
    nothing

# taylorinteg
function taylorinteg(
    f,
    x0::U,
    t0::T,
    tmax::T,
    order::Int,
    abstol::T,
    params = nothing;
    maxsteps::Int = 500,
    parse_eqs::Bool = true,
    dense::Bool = true,
    reltol::T = zero(T),
    minstepsize::T = zero(T),
    maxstepsize::T = T(Inf)
) where {T<:Real,U<:Number}

    # Allocation
    cache = init_cache(Val(dense), t0, x0, maxsteps, order, f, params; parse_eqs)

    return taylorinteg!(Val(dense), f, x0, t0, tmax, abstol, cache, params; maxsteps,
                        reltol, minstepsize, maxstepsize, copy_solution=Val(false))
end

function taylorinteg!(
    dense::Val{D},
    f,
    x0::U,
    t0::T,
    tmax::T,
    abstol::T,
    cache::ScalarCache,
    params;
    maxsteps::Int = 500,
    reltol::T = zero(T),
    minstepsize::T = zero(T),
    maxstepsize::T = T(Inf),
    copy_solution::Val = Val(true)
) where {T<:Real,U<:Number,D}

    (; tv, xv, psol, t, x, rv, parse_eqs) = cache

    # Initial conditions
    update_cache!(cache, t0, x0)
    nsteps = 1
    @inbounds tv[1] = t0
    @inbounds xv[1] = x0
    sign_tstep = copysign(1, tmax - t0)

    # Integration
    while sign_tstep * t0 < sign_tstep * tmax
        δt = taylorstep!(Val(parse_eqs), f, t, x, abstol, params, rv, reltol,
                         minstepsize, maxstepsize) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        if iszero(δt)
            @warn("""The step-size is zero; aborting integration.""")
            break
        end
        δt = sign_tstep * min(δt, sign_tstep * (tmax - t0))
        x0 = evaluate(x, δt) # new initial condition
        set_psol!(dense, psol, nsteps, x) # Store the Taylor polynomial solution
        t0 += δt
        update_cache!(cache, t0, x0)
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds xv[nsteps] = x0
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return build_solution(copy_solution, tv, xv, psol, nsteps)
end


function taylorinteg(
    f!,
    q0::Vector{U},
    t0::T,
    tmax::T,
    order::Int,
    abstol::T,
    params = nothing;
    maxsteps::Int = 500,
    parse_eqs::Bool = true,
    dense::Bool = true,
    reltol::T = zero(T),
    minstepsize::T = zero(T),
    maxstepsize::T = T(Inf)
) where {T<:Real,U<:Number}

    # Allocation
    cache = init_cache(Val(dense), t0, q0, maxsteps, order, f!, params; parse_eqs)

    return taylorinteg!(Val(dense), f!, q0, t0, tmax, abstol, cache, params; maxsteps,
                        reltol, minstepsize, maxstepsize, copy_solution=Val(false))
end

function taylorinteg!(
    dense::Val{D},
    f!,
    q0::Array{U,1},
    t0::T,
    tmax::T,
    abstol::T,
    cache::VectorCache,
    params;
    maxsteps::Int = 500,
    reltol::T = zero(T),
    minstepsize::T = zero(T),
    maxstepsize::T = T(Inf),
    copy_solution::Val = Val(true)
) where {T<:Real,U<:Number,D}

    (; tv, xv, psol, xaux, t, x, dx, rv, parse_eqs) = cache

    # Initial conditions
    x0 = _stored_state(q0)
    update_cache!(cache, t0, x0)
    @inbounds tv[1] = t0
    _store_state_column!(xv, 1, q0)
    sign_tstep = copysign(1, tmax - t0)

    # Integration
    nsteps = 1
    while sign_tstep * t0 < sign_tstep * tmax
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv, reltol,
                         minstepsize, maxstepsize) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        if iszero(δt)
            @warn("""The step-size is zero; aborting integration.""")
            break
        end
        δt = sign_tstep * min(δt, sign_tstep * (tmax - t0))
        evaluate!(x, δt, x0) # new initial condition
        set_psol!(dense, psol, nsteps, x) # Store the Taylor polynomial solution
        t0 += δt
        update_cache!(cache, t0, x0)
        nsteps += 1
        @inbounds tv[nsteps] = t0
        _store_state_column!(xv, nsteps, x0)
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return build_solution(copy_solution, tv, xv, psol, nsteps)
end

@doc doc"""
    taylorinteg(f, x0, t0, tmax, order, abstol, params[=nothing]; kwargs... )

General-purpose Taylor integrator for the explicit ODE ``\dot{x}=f(x, p, t)``,
where `p` are the parameters encoded in `params`.
The initial conditions are specified by `x0` at time `t0`; `x0` may be of type `T<:Number`
or `Vector{T}`, with `T` including `TaylorN{T}`; the latter case
is of interest for [jet transport applications](@ref jettransport).

The equations of motion are specified by the function `f`; we follow the same
convention of `DifferentialEquations.jl` to define this function, i.e.,
`f(x, p, t)` or `f!(dx, x, p, t)`; see the examples below.

The functions returns a `TaylorSolution`, whose fields are `t` and `x`; they represent,
respectively, a vector with the values of time (independent variable),
and a vector with the computed values of
the dependent variable(s). When the keyword argument `dense` is set to `true`, it also
outputs in the field `p` the Taylor polynomial expansion computed at each time step.
The integration stops when time is larger than `tmax`, in which case the last returned
value(s) correspond to `tmax`, or when the number of saved steps is larger
than `maxsteps`.

The integration method uses polynomial expansions on the independent variable
of order `order`; the parameter `abstol` serves to define the
time step using the last two Taylor coefficients of the expansions.
Make sure you use a *large enough* `order` to assure convergence.

Currently, the recognized keyword arguments are:
- `maxsteps[=500]`: maximum number of integration steps.
- `parse_eqs[=true]`: use the specialized method of `jetcoeffs!` created
    with [`@taylorize`](@ref).
- `dense[=true]`: output the Taylor polynomial expansion at each time step.

## Examples

For one dependent variable the function `f` defines the RHS of the equation of
motion, returning the value of ``\dot{x}``. The arguments of
this function are `(x, p, t)`, where `x` are the dependent variables, `p` are
the paremeters and `t` is the independent variable.

For several (two or more) dependent variables, the function `f!` defines
the RHS of the equations of motion, mutating (in-place) the (preallocated) vector
with components of ``\dot{x}``. The arguments of this function are `(dx, x, p, t)`,
where `dx` is the preallocated vector of ``\dot{x}``, `x` are the dependent
variables, `p` are the paremeters entering the ODEs and `t` is the independent
variable. The function may return this vector or simply `nothing`.

```julia
using TaylorIntegration

f(x, p, t) = x^2

sol = taylorinteg(f, 3, 0.0, 0.3, 25, 1.0e-20, maxsteps=100 )

function f!(dx, x, p, t)
    for i in eachindex(x)
        dx[i] = x[i]^2
    end
    return nothing
end

sol = taylorinteg(f!, [3, 3], 0.0, 0.3, 25, 1.0e-20, maxsteps=100 )

sol = taylorinteg(f!, [3, 3], 0.0, 0.3, 25, 1.0e-20, maxsteps=100, dense=true )
```

""" taylorinteg


# Integrate and return results evaluated at given time
@doc doc"""
    taylorinteg(f, x0, trange, order, abstol, params[=nothing]; kwargs... )

General-purpose Taylor integrator for the explicit ODE
``\dot{x}=f(x,p,t)`` with initial condition specified by `x0::{T<:Number}`
or `x0::Vector{T}` at time `t0`.

The method returns a `TaylorSolution` whose field `x` represents the computed values of
the dependent variable(s), evaluated *only* at the times specified by
the range `trange`.

## Examples

```julia
sol = taylorinteg(f, 3, 0.0:0.001:0.3, 25, 1.0e-20, maxsteps=100 )

sol = taylorinteg(f!, [3, 3], 0.0:0.001:0.3, 25, 1.0e-20, maxsteps=100 );

```

"""
function taylorinteg(
    f,
    x0::U,
    trange::AbstractVector{T},
    order::Int,
    abstol::T,
    params = nothing;
    maxsteps::Int = 500,
    parse_eqs::Bool = true,
    reltol::T = zero(T),
    minstepsize::T = zero(T),
    maxstepsize::T = T(Inf)
) where {T<:Real,U<:Number}

    # Check if trange is increasingly or decreasingly sorted
    @assert (issorted(trange) || issorted(trange, rev = true)) "`trange` or `reverse(trange)` must be sorted"

    # Allocation
    cache = init_cache(Val(false), trange, x0, maxsteps, order, f, params; parse_eqs)

    return taylorinteg!(f, x0, trange, abstol, cache, params; maxsteps, reltol,
                        minstepsize, maxstepsize, copy_solution=Val(false))
end

function taylorinteg!(
    f,
    x0::U,
    trange::AbstractVector{T},
    abstol::T,
    cache::ScalarCache,
    params;
    maxsteps::Int = 500,
    reltol::T = zero(T),
    minstepsize::T = zero(T),
    maxstepsize::T = T(Inf),
    copy_solution::Val = Val(true)
) where {T<:Real,U<:Number}

    (; xv, t, x, rv, parse_eqs) = cache

    # Initial conditions
    @inbounds t0, t1, tmax = trange[1], trange[2], trange[end]
    update_cache!(cache, t0, x0)
    sign_tstep = copysign(1, tmax - t0)
    @inbounds xv[1] = x0

    # Integration
    iter = 2
    nsteps = 1
    while sign_tstep * t0 < sign_tstep * tmax
        δt = taylorstep!(Val(parse_eqs), f, t, x, abstol, params, rv, reltol,
                         minstepsize, maxstepsize)# δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        if iszero(δt)
            @warn("""The step-size is zero; aborting integration.""")
            break
        end
        δt = sign_tstep * min(δt, sign_tstep * (tmax - t0))
        x0 = evaluate(x, δt) # new initial condition
        tnext = t0 + δt
        # Evaluate solution at times within convergence radius
        while sign_tstep * t1 < sign_tstep * tnext
            x1 = evaluate(x, t1 - t0)
            @inbounds xv[iter] = x1
            iter += 1
            @inbounds t1 = trange[iter]
        end
        if δt == tmax - t0
            @inbounds xv[iter] = x0
            break
        end
        t0 = tnext
        update_cache!(cache, t0, x0)
        nsteps += 1
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end
    return build_solution(copy_solution, trange, xv)
end

function taylorinteg(
    f!,
    q0::Vector{U},
    trange::AbstractVector{T},
    order::Int,
    abstol::T,
    params = nothing;
    maxsteps::Int = 500,
    parse_eqs::Bool = true,
    reltol::T = zero(T),
    minstepsize::T = zero(T),
    maxstepsize::T = T(Inf)
) where {T<:Real,U<:Number}

    # Check if trange is increasingly or decreasingly sorted
    @assert (issorted(trange) || issorted(trange, rev = true)) "`trange` or `reverse(trange)` must be sorted"

    # Allocation
    cache = init_cache(Val(false), trange, q0, maxsteps, order, f!, params; parse_eqs)

    return taylorinteg!(f!, q0, trange, abstol, cache, params; maxsteps, reltol,
                        minstepsize, maxstepsize, copy_solution=Val(false))
end

function taylorinteg!(
    f!,
    q0::Vector{U},
    trange::AbstractVector{T},
    abstol::T,
    cache::VectorTRangeCache,
    params;
    maxsteps::Int = 500,
    reltol::T = zero(T),
    minstepsize::T = zero(T),
    maxstepsize::T = T(Inf),
    copy_solution::Val = Val(true)
) where {T<:Real,U<:Number}

    (; xv, xaux, x0, x1, t, x, dx, rv, parse_eqs) = cache

    # Initial conditions
    @inbounds t0, t1, tmax = trange[1], trange[2], trange[end]
    sign_tstep = copysign(1, tmax - t0)
    _copy_state!(x0, q0)
    update_cache!(cache, t0, x0)
    _store_state_column!(xv, 1, q0)

    # Integration
    iter = 2
    nsteps = 1
    while sign_tstep * t0 < sign_tstep * tmax
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv, reltol,
                         minstepsize, maxstepsize) # δt is positive!
        if iszero(δt)
            @warn("""The step-size is zero; aborting integration.""")
            break
        end
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep * (tmax - t0))
        evaluate!(x, δt, x0) # new initial condition
        tnext = t0 + δt
        # Evaluate solution at times within convergence radius
        while sign_tstep * t1 < sign_tstep * tnext
            evaluate!(x, t1 - t0, x1)
            _store_state_column!(xv, iter, x1)
            iter += 1
            @inbounds t1 = trange[iter]
        end
        if δt == tmax - t0
            _store_state_column!(xv, iter, x0)
            break
        end
        t0 = tnext
        update_cache!(cache, t0, x0)
        nsteps += 1
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return build_solution(copy_solution, trange, xv)
end

@doc doc"""
    taylorinteg!(dense::Val, f, x0, t0, tmax, abstol, cache, params; kwargs...)
    taylorinteg!(dense::Val, f!, q0, t0, tmax, abstol, cache, params; kwargs...)
    taylorinteg!(f, x0, trange, abstol, cache, params; kwargs...)
    taylorinteg!(f!, q0, trange, abstol, cache, params; kwargs...)
    taylorinteg!(dense::Val, f!, g, q0, t0, tmax, abstol, cache, params; kwargs...)
    taylorinteg!(f!, g, q0, trange, abstol, cache, params; kwargs...)

Integrate using a preallocated `cache`, returning a [`TaylorSolution`](@ref).
This is the in-place, cache-mutating counterpart of [`taylorinteg`](@ref); callers
that pass the same cache repeatedly can avoid rebuilding the working Taylor polynomials.
The last two signatures are the root-finding variants.

The keyword `copy_solution` controls whether the returned solution owns its
storage:
- `copy_solution=Val(true)`: copy the returned arrays and dense polynomials out
  of the cache. This is the default for `taylorinteg!` and is safe when the cache
  will be reused while earlier solutions are still needed.
- `copy_solution=Val(false)`: return views or borrowed arrays backed by the
  cache. This avoids the final solution copy, but the returned solution may be
  overwritten by later reuse of the same cache.

Use [`taylorinteg`](@ref) for one-shot runs; it creates a private cache
and uses `copy_solution=Val(false)` internally to minimize allocations.
""" taylorinteg!


# Generic functions
for R in (:Number, :Integer)
    @eval begin

        function taylorinteg(
            f,
            xx0::S,
            tt0::T,
            ttmax::U,
            order::Int,
            aabstol::V,
            params = nothing;
            dense::Bool = true,
            maxsteps::Int = 500,
            parse_eqs::Bool = true,
            reltol::V = zero(V),
            minstepsize::V = zero(V),
            maxstepsize::V = V(Inf)
        ) where {S<:$R,T<:Real,U<:Real,V<:Real}

            # In order to handle mixed input types, we promote types before integrating:
            t0, tmax, abstol, rreltol, rminstepsize, rmaxstepsize, _ = promote(tt0, ttmax, aabstol,
                reltol, minstepsize, maxstepsize, one(Float64))
            x0, _ = promote(xx0, t0)

            return taylorinteg(
                f,
                x0,
                t0,
                tmax,
                order,
                abstol,
                params,
                dense = dense,
                maxsteps = maxsteps,
                parse_eqs = parse_eqs,
                reltol = rreltol,
                minstepsize = rminstepsize,
                maxstepsize = rmaxstepsize
            )
        end

        function taylorinteg(
            f,
            q0::Array{S,1},
            tt0::T,
            ttmax::U,
            order::Int,
            aabstol::V,
            params = nothing;
            dense::Bool = true,
            maxsteps::Int = 500,
            parse_eqs::Bool = true,
            reltol::V = zero(V),
            minstepsize::V = zero(V),
            maxstepsize::V = V(Inf)
        ) where {S<:$R,T<:Real,U<:Real,V<:Real}

            #promote to common type before integrating:
            t0, tmax, abstol, rreltol, rminstepsize, rmaxstepsize, _ = promote(tt0, ttmax, aabstol,
                reltol, minstepsize, maxstepsize, one(Float64))
            elq0, _ = promote(q0[1], t0)
            #convert the elements of q0 to the common, promoted type:
            q0_ = convert(Array{typeof(elq0)}, q0)

            return taylorinteg(
                f,
                q0_,
                t0,
                tmax,
                order,
                abstol,
                params,
                dense = dense,
                maxsteps = maxsteps,
                parse_eqs = parse_eqs,
                reltol = rreltol,
                minstepsize = rminstepsize,
                maxstepsize = rmaxstepsize
            )
        end

        function taylorinteg(
            f,
            xx0::S,
            trange::AbstractVector{T},
            order::Int,
            aabstol::U,
            params = nothing;
            maxsteps::Int = 500,
            parse_eqs::Bool = true,
            reltol::U = zero(U),
            minstepsize::U = zero(U),
            maxstepsize::U = U(Inf)
        ) where {S<:$R,T<:Real,U<:Real}

            t0, abstol, rreltol, rminstepsize, rmaxstepsize, _ = promote(trange[1], aabstol,
                reltol, minstepsize, maxstepsize, one(Float64))
            x0, _ = promote(xx0, t0)

            return taylorinteg(
                f,
                x0,
                trange .* one(t0),
                order,
                abstol,
                params,
                maxsteps = maxsteps,
                parse_eqs = parse_eqs,
                reltol = rreltol,
                minstepsize = rminstepsize,
                maxstepsize = rmaxstepsize
            )
        end

        function taylorinteg(
            f,
            q0::Array{S,1},
            trange::AbstractVector{T},
            order::Int,
            aabstol::U,
            params = nothing;
            maxsteps::Int = 500,
            parse_eqs::Bool = true,
            reltol::U = zero(U),
            minstepsize::U = zero(U),
            maxstepsize::U = U(Inf)
        ) where {S<:$R,T<:Real,U<:Real}

            t0, abstol, rreltol, rminstepsize, rmaxstepsize, _ = promote(trange[1], aabstol,
                reltol, minstepsize, maxstepsize, one(Float64))
            elq0, _ = promote(q0[1], t0)
            q0_ = convert(Array{typeof(elq0)}, q0)

            return taylorinteg(
                f,
                q0_,
                trange .* one(t0),
                order,
                abstol,
                params,
                maxsteps = maxsteps,
                parse_eqs = parse_eqs,
                reltol = rreltol,
                minstepsize = rminstepsize,
                maxstepsize = rmaxstepsize
            )
        end

    end
end
