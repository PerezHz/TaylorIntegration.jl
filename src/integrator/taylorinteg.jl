# This file is part of the TaylorIntegration.jl package; MIT licensed
@inline function setpsol(::Type{Val{true}}, psol::Array{Taylor1{U},1}, nsteps::Int, x::Taylor1{U}) where {U<:Number}
    @inbounds psol[nsteps] = deepcopy(x)
    return nothing
end
@inline function setpsol(::Type{Val{false}}, ::Array{Taylor1{U},1}, ::Int, ::Taylor1{U}) where {U<:Number}
    return nothing
end
@inline function setpsol(::Type{Val{true}}, psol::Array{Taylor1{U},2}, nsteps::Int, x::Vector{Taylor1{U}}) where {U<:Number}
    @inbounds psol[:,nsteps] .= deepcopy.(x)
    return nothing
end
@inline function setpsol(::Type{Val{false}}, ::Array{Taylor1{U},2}, ::Int, ::Vector{Taylor1{U}}) where {U<:Number}
    return nothing
end

# taylorinteg
function taylorinteg(f, x0::U, t0::T, tmax::T, order::Int, abstol::T, ::Val{S}, params = nothing;
        maxsteps::Int=500, parse_eqs::Bool=true) where {T<:Real, U<:Number, S}

    # Initialize the Taylor1 expansions
    t = t0 + Taylor1( T, order )
    x = Taylor1( x0, order )

    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f, t, x, params)

    # Re-initialize the Taylor1 expansions
    t = t0 + Taylor1( T, order )
    x = Taylor1( x0, order )
    return _taylorinteg!(f, t, x, x0, t0, tmax, abstol, rv,
        Val(parse_eqs), params, maxsteps=maxsteps)
end

function _taylorinteg!(f, t::Taylor1{T}, x::Taylor1{U},
        x0::U, t0::T, tmax::T, abstol::T, rv::RetAlloc{Taylor1{U}}, ::Val{S}, params;
        maxsteps::Int=500) where {T<:Real, U<:Number, S}

    # Allocation
    tv = Array{T}(undef, maxsteps+1)
    xv = Array{U}(undef, maxsteps+1)
    psol = Array{Taylor1{U}}(undef, maxsteps)

    # Initial conditions
    nsteps = 1
    @inbounds t[0] = t0
    @inbounds tv[1] = t0
    @inbounds xv[1] = x0
    sign_tstep = copysign(1, tmax-t0)

    # Integration
    while sign_tstep*t0 < sign_tstep*tmax
        δt = taylorstep!(Val{S}, f, t, x, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        x0 = evaluate(x, δt) # new initial condition
        setpsol(Val{S}, psol, nsteps, x) # Store the Taylor polynomial solution
        @inbounds x[0] = x0
        t0 += δt
        @inbounds t[0] = t0
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

    return build_solution(tv, xv, S == true ? psol : nothing, nsteps)
end


function taylorinteg(f!, q0::Array{U,1}, t0::T, tmax::T, order::Int, abstol::T, ::Val{S}, params = nothing;
        maxsteps::Int=500, parse_eqs::Bool=true) where {T<:Real, U<:Number}

    # Initialize the vector of Taylor1 expansions
    dof = length(q0)
    t = t0 + Taylor1( T, order )
    x = Array{Taylor1{U}}(undef, dof)
    dx = Array{Taylor1{U}}(undef, dof)
    @inbounds for i in eachindex(q0)
        @inbounds x[i] = Taylor1( q0[i], order )
        @inbounds dx[i] = Taylor1( zero(q0[i]), order )
    end

    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f!, t, x, dx, params)

    # Re-initialize the Taylor1 expansions
    t = t0 + Taylor1( T, order )
    x .= Taylor1.( q0, order )
    dx .= Taylor1.( zero.(q0), order)
    return _taylorinteg!(f!, t, x, dx, q0, t0, tmax, abstol, rv,
        Val(parse_eqs), params; maxsteps)
end

function _taylorinteg!(f!, t::Taylor1{T}, x::Array{Taylor1{U},1}, dx::Array{Taylor1{U},1},
        q0::Array{U,1}, t0::T, tmax::T, abstol::T, rv::RetAlloc{Taylor1{U}}, ::Val{S}, params;
        maxsteps::Int=500) where {T<:Real, U<:Number}

    # Initialize the vector of Taylor1 expansions
    dof = length(q0)

    # Allocation of output
    tv = Array{T}(undef, maxsteps+1)
    xv = Array{U}(undef, dof, maxsteps+1)
    psol = Array{Taylor1{U}}(undef, dof, maxsteps)

    # Initial conditions
    @inbounds t[0] = t0
    x0 = deepcopy(q0)
    @inbounds tv[1] = t0
    @inbounds xv[:,1] .= q0
    sign_tstep = copysign(1, tmax-t0)

    # Integration
    nsteps = 1
    while sign_tstep*t0 < sign_tstep*tmax
        δt = taylorstep!(f!, t, x, dx, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        evaluate!(x, δt, x0) # new initial condition
        setpsol(Val{S}, psol, nsteps, x) # Store the Taylor polynomial solution
        @inbounds for i in eachindex(x0)
            x[i][0] = x0[i]
            dx[i][0] = zero(x0[i])
        end
        t0 += δt
        @inbounds t[0] = t0
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds xv[:,nsteps] .= deepcopy.(x0)
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return build_solution(tv, xv, S == true ? psol : nothing, nsteps)
end

@doc doc"""
    taylorinteg(f, x0, t0, tmax, order, abstol, params[=nothing]; kwargs... )
    taylorinteg(f, x0, t0, tmax, order, abstol, Val(false), params[=nothing]; kwargs... )
    taylorinteg(f, x0, t0, tmax, order, abstol, Val(true), params[=nothing]; kwargs... )

General-purpose Taylor integrator for the explicit ODE ``\dot{x}=f(x, p, t)``,
where `p` are the parameters encoded in `params`.
The initial conditions are specified by `x0` at time `t0`; `x0` may be of type `T<:Number`
or `Vector{T}`, with `T` including `TaylorN{T}`; the latter case
is of interest for [jet transport applications](@ref jettransport).

The equations of motion are specified by the function `f`; we follow the same
convention of `DifferentialEquations.jl` to define this function, i.e.,
`f(x, p, t)` or `f!(dx, x, p, t)`; see the examples below.

The functions return a vector with the values of time (independent variable),
and a vector with the computed values of
the dependent variable(s), and if the method used involves `Val(true)` it also
outputs the Taylor polynomial solutions obtained at each time step.
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

tv, xv = taylorinteg(f, 3, 0.0, 0.3, 25, 1.0e-20, maxsteps=100 )

function f!(dx, x, p, t)
    for i in eachindex(x)
        dx[i] = x[i]^2
    end
    return nothing
end

tv, xv = taylorinteg(f!, [3, 3], 0.0, 0.3, 25, 1.0e-20, maxsteps=100 )

tv, xv, psol = taylorinteg(f!, [3, 3], 0.0, 0.3, 25, 1.0e-20, maxsteps=100, Val(true) )
```

""" taylorinteg


# Integrate and return results evaluated at given time
@doc doc"""
    taylorinteg(f, x0, trange, order, abstol, params[=nothing]; kwargs... )

General-purpose Taylor integrator for the explicit ODE
``\dot{x}=f(x,p,t)`` with initial condition specified by `x0::{T<:Number}`
or `x0::Vector{T}` at time `t0`.

The method returns a vector with the computed values of
the dependent variable(s), evaluated *only* at the times specified by
the range `trange`.

## Examples

```julia
xv = taylorinteg(f, 3, 0.0:0.001:0.3, 25, 1.0e-20, maxsteps=100 )

xv = taylorinteg(f!, [3, 3], 0.0:0.001:0.3, 25, 1.0e-20, maxsteps=100 );

```

"""
function taylorinteg(f, x0::U, trange::AbstractVector{T},
        order::Int, abstol::T, params = nothing;
        maxsteps::Int=500, parse_eqs::Bool=true) where {T<:Real, U<:Number}

    # Check if trange is increasingly or decreasingly sorted
    @assert (issorted(trange) ||
        issorted(reverse(trange))) "`trange` or `reverse(trange)` must be sorted"

    # Initialize the Taylor1 expansions
    t0 = trange[1]
    t = t0 + Taylor1( T, order )
    x = Taylor1( x0, order )

    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f, t, x, params)

    if parse_eqs
        # Re-initialize the Taylor1 expansions
        t = t0 + Taylor1( T, order )
        x = Taylor1( x0, order )
        return _taylorinteg!(f, t, x, x0, trange, abstol, rv,
            params, maxsteps=maxsteps)
    else
        return _taylorinteg!(f, t, x, x0, trange, abstol, params, maxsteps=maxsteps)
    end
end

function _taylorinteg!(f, t::Taylor1{T}, x::Taylor1{U}, x0::U, trange::AbstractVector{T},
        abstol::T, params; maxsteps::Int=500) where {T<:Real, U<:Number}

    # Allocation
    nn = length(trange)
    xv = Array{U}(undef, nn)
    fill!(xv, T(NaN))

    # Initial conditions
    @inbounds t0, t1, tmax = trange[1], trange[2], trange[end]
    sign_tstep = copysign(1, tmax-t0)
    @inbounds t[0] = t0
    @inbounds xv[1] = x0

    # Integration
    iter = 2
    nsteps = 1
    while sign_tstep*t0 < sign_tstep*tmax
        δt = taylorstep!(f, t, x, abstol, params)# δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        x0 = evaluate(x, δt) # new initial condition
        tnext = t0+δt
        # Evaluate solution at times within convergence radius
        while sign_tstep*t1 < sign_tstep*tnext
            x1 = evaluate(x, t1-t0)
            @inbounds xv[iter] = x1
            iter += 1
            @inbounds t1 = trange[iter]
        end
        if δt == tmax-t0
            @inbounds xv[iter] = x0
            break
        end
        @inbounds x[0] = x0
        t0 = tnext
        @inbounds t[0] = t0
        nsteps += 1
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end
    return build_solution(trange, xv, nothing, nn)
end
function _taylorinteg!(f, t::Taylor1{T}, x::Taylor1{U}, x0::U, trange::AbstractVector{T},
        abstol::T, rv::RetAlloc{Taylor1{U}}, params; maxsteps::Int=500) where {T<:Real, U<:Number}

    # Allocation
    nn = length(trange)
    xv = Array{U}(undef, nn)
    fill!(xv, T(NaN))

    # Initial conditions
    @inbounds t0, t1, tmax = trange[1], trange[2], trange[end]
    sign_tstep = copysign(1, tmax-t0)
    @inbounds t[0] = t0
    @inbounds xv[1] = x0

    # Integration
    iter = 2
    nsteps = 1
    while sign_tstep*t0 < sign_tstep*tmax
        δt = taylorstep!(f, t, x, abstol, params, rv)# δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        x0 = evaluate(x, δt) # new initial condition
        tnext = t0+δt
        # Evaluate solution at times within convergence radius
        while sign_tstep*t1 < sign_tstep*tnext
            x1 = evaluate(x, t1-t0)
            @inbounds xv[iter] = x1
            iter += 1
            @inbounds t1 = trange[iter]
        end
        if δt == tmax-t0
            @inbounds xv[iter] = x0
            break
        end
        @inbounds x[0] = x0
        t0 = tnext
        @inbounds t[0] = t0
        nsteps += 1
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end
    return build_solution(trange, xv, nothing, nn)
end

function taylorinteg(f!, q0::Array{U,1}, trange::AbstractVector{T},
        order::Int, abstol::T, params = nothing;
        maxsteps::Int=500, parse_eqs::Bool=true) where {T<:Real, U<:Number}

    # Check if trange is increasingly or decreasingly sorted
    @assert (issorted(trange) ||
        issorted(reverse(trange))) "`trange` or `reverse(trange)` must be sorted"

    # Initialize the vector of Taylor1 expansions
    dof = length(q0)
    t0 = trange[1]
    t = t0 + Taylor1( T, order )
    x = Array{Taylor1{U}}(undef, dof)
    dx = Array{Taylor1{U}}(undef, dof)
    x .= Taylor1.( q0, order )
    dx .= Taylor1.( zero.(q0), order )

    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f!, t, x, dx, params)

    if parse_eqs
        # Re-initialize the Taylor1 expansions
        t = t0 + Taylor1( T, order )
        x .= Taylor1.( q0, order )
        dx .= Taylor1.( zero.(q0), order )
        return _taylorinteg!(f!, t, x, dx, q0, trange, abstol, rv,
            params, maxsteps=maxsteps)
    else
        return _taylorinteg!(f!, t, x, dx, q0, trange, abstol, params, maxsteps=maxsteps)
    end
end

function _taylorinteg!(f!, t::Taylor1{T}, x::Array{Taylor1{U},1}, dx::Array{Taylor1{U},1},
        q0::Array{U,1}, trange::AbstractVector{T}, abstol::T, params;
        maxsteps::Int=500) where {T<:Real, U<:Number}

    # Allocation
    nn = length(trange)
    dof = length(q0)
    x0 = similar(q0, eltype(q0), dof)
    x1 = similar(x0)
    fill!(x0, T(NaN))
    xv = Array{eltype(q0)}(undef, dof, nn)
    for ind in 1:nn
        @inbounds xv[:,ind] .= x0
    end
    xaux = Array{Taylor1{U}}(undef, dof)

    # Initial conditions
    @inbounds t[0] = trange[1]
    @inbounds t0, t1, tmax = trange[1], trange[2], trange[end]
    sign_tstep = copysign(1, tmax-t0)
    # x .= Taylor1.(q0, order)
    @inbounds x0 .= q0
    @inbounds xv[:,1] .= q0

    # Integration
    iter = 2
    nsteps = 1
    while sign_tstep*t0 < sign_tstep*tmax
        δt = taylorstep!(f!, t, x, dx, xaux, abstol, params) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        evaluate!(x, δt, x0) # new initial condition
        tnext = t0+δt
        # Evaluate solution at times within convergence radius
        while sign_tstep*t1 < sign_tstep*tnext
            evaluate!(x, t1-t0, x1)
            @inbounds xv[:,iter] .= x1
            iter += 1
            @inbounds t1 = trange[iter]
        end
        if δt == tmax-t0
            @inbounds xv[:,iter] .= x0
            break
        end
        @inbounds for i in eachindex(x0)
            x[i][0] = x0[i]
            dx[i][0] = zero(x0[i])
        end
        t0 = tnext
        @inbounds t[0] = t0
        nsteps += 1
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return build_solution(trange, xv, nothing, nn)
end
function _taylorinteg!(f!, t::Taylor1{T}, x::Array{Taylor1{U},1}, dx::Array{Taylor1{U},1},
        q0::Array{U,1}, trange::AbstractVector{T}, abstol::T, rv::RetAlloc{Taylor1{U}}, params;
        maxsteps::Int=500) where {T<:Real, U<:Number}

    # Allocation
    nn = length(trange)
    dof = length(q0)
    x0 = similar(q0, eltype(q0), dof)
    x1 = similar(x0)
    fill!(x0, T(NaN))
    xv = Array{eltype(q0)}(undef, dof, nn)
    for ind in 1:nn
        @inbounds xv[:,ind] .= x0
    end

    # Initial conditions
    @inbounds t[0] = trange[1]
    @inbounds t0, t1, tmax = trange[1], trange[2], trange[end]
    sign_tstep = copysign(1, tmax-t0)
    # x .= Taylor1.(q0, order)
    @inbounds x0 .= q0
    @inbounds xv[:,1] .= q0

    # Integration
    iter = 2
    nsteps = 1
    while sign_tstep*t0 < sign_tstep*tmax
        δt = taylorstep!(f!, t, x, dx, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        evaluate!(x, δt, x0) # new initial condition
        tnext = t0+δt
        # Evaluate solution at times within convergence radius
        while sign_tstep*t1 < sign_tstep*tnext
            evaluate!(x, t1-t0, x1)
            @inbounds xv[:,iter] .= x1
            iter += 1
            @inbounds t1 = trange[iter]
        end
        if δt == tmax-t0
            @inbounds xv[:,iter] .= x0
            break
        end
        @inbounds for i in eachindex(x0)
            x[i][0] = x0[i]
            dx[i][0] = zero(x0[i])
        end
        t0 = tnext
        @inbounds t[0] = t0
        nsteps += 1
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return build_solution(trange, xv, nothing, nn)
end


# Generic functions
for R in (:Number, :Integer)
    @eval begin
        taylorinteg(f, xx0::S, tt0::T, ttmax::U, order::Int, aabstol::V, params = nothing;
                maxsteps::Int=500, parse_eqs::Bool=true) where {S<:$R, T<:Real, U<:Real, V<:Real} =
            taylorinteg(f, xx0, tt0, ttmax, order, aabstol, Val(false), params,
                maxsteps=maxsteps, parse_eqs=parse_eqs)

        taylorinteg(f, q0::Array{S,1}, tt0::T, ttmax::U, order::Int, aabstol::V, params = nothing;
                maxsteps::Int=500, parse_eqs::Bool=true) where {S<:$R, T<:Real, U<:Real, V<:Real} =
            taylorinteg(f, q0, tt0, ttmax, order, aabstol, Val(false), params,
                maxsteps=maxsteps, parse_eqs=parse_eqs)

        function taylorinteg(f, xx0::S, trange::AbstractVector{T}, order::Int, aabstol::U, params = nothing;
                maxsteps::Int=500, parse_eqs::Bool=true) where {S<:$R, T<:Real, U<:Real}

            t0, abstol, _ = promote(trange[1], aabstol, one(Float64))
            x0, _ = promote(xx0, t0)

            return taylorinteg(f, x0, trange .* one(t0), order, abstol, params,
                maxsteps=maxsteps, parse_eqs=parse_eqs)
        end

        function taylorinteg(f, q0::Array{S,1}, trange::AbstractVector{T}, order::Int, aabstol::U, params = nothing;
                maxsteps::Int=500, parse_eqs::Bool=true) where {S<:$R, T<:Real, U<:Real}

            t0, abstol, _ = promote(trange[1], aabstol, one(Float64))
            elq0, _ = promote(q0[1], t0)
            q0_ = convert(Array{typeof(elq0)}, q0)

            return taylorinteg(f, q0_, trange .* one(t0), order, abstol, params,
                maxsteps=maxsteps, parse_eqs=parse_eqs)
        end

        function taylorinteg(f, xx0::S, tt0::T, ttmax::U, order::Int, aabstol::V,
                ::Val{true}, params = nothing; maxsteps::Int=500, parse_eqs::Bool=true) where
                {S<:$R, T<:Real, U<:Real, V<:Real}

            # In order to handle mixed input types, we promote types before integrating:
            t0, tmax, abstol, _ = promote(tt0, ttmax, aabstol, one(Float64))
            x0, _ = promote(xx0, t0)

            return taylorinteg(f, x0, t0, tmax, order, abstol, Val(true), params,
                maxsteps=maxsteps, parse_eqs=parse_eqs)
        end

        function taylorinteg(f, q0::Array{S,1}, tt0::T, ttmax::U, order::Int, aabstol::V,
                ::Val{true}, params = nothing; maxsteps::Int=500, parse_eqs::Bool=true) where
                {S<:$R, T<:Real, U<:Real, V<:Real}

            #promote to common type before integrating:
            t0, tmax, abstol, _ = promote(tt0, ttmax, aabstol, one(Float64))
            elq0, _ = promote(q0[1], t0)
            #convert the elements of q0 to the common, promoted type:
            q0_ = convert(Array{typeof(elq0)}, q0)

            return taylorinteg(f, q0_, t0, tmax, order, abstol, Val(true), params,
                maxsteps=maxsteps, parse_eqs=parse_eqs)
        end

        function taylorinteg(f, xx0::S, tt0::T, ttmax::U, order::Int, aabstol::V,
                ::Val{false}, params = nothing; maxsteps::Int=500, parse_eqs::Bool=true) where
                {S<:$R, T<:Real, U<:Real, V<:Real}

            # In order to handle mixed input types, we promote types before integrating:
            t0, tmax, abstol, _ = promote(tt0, ttmax, aabstol, one(Float64))
            x0, _ = promote(xx0, t0)

            return taylorinteg(f, x0, t0, tmax, order, abstol, Val(false), params,
                maxsteps=maxsteps, parse_eqs=parse_eqs)
        end

        function taylorinteg(f, q0::Array{S,1}, tt0::T, ttmax::U, order::Int, aabstol::V,
                ::Val{false}, params = nothing; maxsteps::Int=500, parse_eqs::Bool=true) where
                {S<:$R, T<:Real, U<:Real, V<:Real}

            #promote to common type before integrating:
            t0, tmax, abstol, _ = promote(tt0, ttmax, aabstol, one(Float64))
            elq0, _ = promote(q0[1], t0)
            #convert the elements of q0 to the common, promoted type:
            q0_ = convert(Array{typeof(elq0)}, q0)

            return taylorinteg(f, q0_, t0, tmax, order, abstol, Val(false), params,
                maxsteps=maxsteps, parse_eqs=parse_eqs)
        end
    end
end
