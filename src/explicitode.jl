# This file is part of the TaylorIntegration.jl package; MIT licensed

# jetcoeffs!
doc"""
    jetcoeffs!(eqsdiff, t, x, vT)

Returns an updated `x` using the recursion relation of the
derivatives obtained from the differential equations
$\dot{x}=dx/dt=f(t,x)$.

`eqsdiff` is the function defining the RHS of the ODE,
`x` contains the Taylor1 expansion of the dependent variable(s) and
`t` is the independent variable. See [`taylorinteg`](@ref) for examples
and structure of `eqsdiff`.
Note that `x` is of type `Taylor1{T}` or `Taylor1{TaylorN{T}}`.
`vT::Vector{T}` is a pre-allocated
vector used for time-dependent differential equations.

Initially, `x` contains only the 0-th order Taylor coefficient of
the current system state (the initial conditions), and `jetcoeffs!`
computes recursively the high-order derivates back into `x`.

"""
function jetcoeffs!{T<:Real, U<:Number}(eqsdiff, t::Taylor1{T},
        x::Taylor1{U})
    order = x.order
    for ord in 1:order
        ordnext = ord+1

        # Set `taux`, `xaux`, auxiliary Taylor1 variables to order `ord`
        @inbounds taux = Taylor1( t.coeffs[1:ord] )
        @inbounds xaux = Taylor1( x.coeffs[1:ord] )

        # Equations of motion
        # TODO! define a macro to optimize the eqsdiff
        dx = eqsdiff(taux, xaux)

        # Recursion relation
        @inbounds x[ordnext] = dx[ord]/ord
    end
    nothing
end

doc"""
    jetcoeffs!(eqsdiff!, t, x, dx, xaux, vT)

Returns an updated `x` using the recursion relation of the
derivatives obtained from the differential equations
$\dot{x}=dx/dt=f(t,x)$.

`eqsdiff!` is the function defining the RHS of the ODE,
`x` contains the Taylor1 expansion of the dependent variables and
`t` is the independent variable. See [`taylorinteg`](@ref) for examples
and structure of `eqsdiff!`. Note that `x` is of type `Vector{Taylor1{T}}`
or `Vector{Taylor1{TaylorN{T}}}`. In this case, two auxiliary containers
`dx` and `xaux` (both of the same type as `x`) are needed to avoid
allocations. `vT::Vector{T}` is a pre-allocated
vector used for time-dependent differential equations.

Initially, `x` contains only the 0-th order Taylor coefficient of
the current system state (the initial conditions), and `jetcoeffs!`
computes recursively the high-order derivates back into `x`.

"""
function jetcoeffs!{T<:Real, U<:Number}(eqsdiff!, t::Taylor1{T}, x::Vector{Taylor1{U}},
        dx::Vector{Taylor1{U}}, xaux::Vector{Taylor1{U}})
    order = x[1].order
    for ord in 1:order
        ordnext = ord+1

        # Set `taux`, auxiliary Taylor1 variable to order `ord`
        @inbounds taux = Taylor1( t.coeffs[1:ord] )
        # Set xaux`, auxiliary vector of Taylor1 to order `ord`
        for j in eachindex(x)
            @inbounds xaux[j] = Taylor1( x[j].coeffs[1:ord] )
        end

        # Equations of motion
        # TODO! define a macro to optimize the eqsdiff
        eqsdiff!(taux, xaux, dx)

        # Recursion relations
        for j in eachindex(x)
            @inbounds x[j][ordnext] = dx[j][ord]/ord
        end
    end
    nothing
end

# stepsize
doc"""
    stepsize(x, epsilon)

Returns a maximum time-step for a the Taylor expansion `x`
using a prescribed absolute tolerance `epsilon` and the last two
Taylor coefficients of (each component of) `x`.

Note that `x` is of type `Taylor1{T}` or `Vector{Taylor1{T}}`, including
also the cases `Taylor1{TaylorN{T}}` and `Vector{Taylor1{TaylorN{T}}}`.

"""
function stepsize{T<:Real, U<:Number}(x::Taylor1{U}, epsilon::T)
    ord = x.order
    h = convert(T, Inf)
    for k in (ord-1, ord)
        @inbounds aux = norm( x[k+1], Inf)
        aux == zero(T) && continue
        aux = epsilon / aux
        kinv = one(T)/k
        aux = aux^kinv
        h = min(h, aux)
    end
    return h
end

function stepsize{T<:Real, U<:Number}(q::Array{Taylor1{U},1}, epsilon::T)
    h = convert(T, Inf)
    for i in eachindex(q)
        @inbounds hi = stepsize( q[i], epsilon )
        h = min( h, hi )
    end
    return h
end

#taylorstep
doc"""
    taylorstep!(f, x, t0, t1, x0, order, abstol, vT)

One-step Taylor integration for the ODE $\dot{x}=dx/dt=f(t, x)$
with initial conditions $x(t_0)=x_0$, computed from `t0` up to
`t1`, returning the time-step of the actual integration carried out
and the updated value of `x0`.

Here, `f` is the function defining the RHS of the ODE (see
[`taylorinteg`](@ref) for examples and structure of `f`), `x` contains
the Taylor expansion of the dependent variable, `x0` is the initial
value of the dependent variable, `order`
is the degree  used for the `Taylor1` polynomials during the integration
and `abstol` is the absolute tolerance used to determine the time step
of the integration. Note that `x0` is of type `Taylor1{T<:Number}` or
`Taylor1{TaylorN{T}}`. If the time step is larger than `t1-t0`, that
difference is used as the time step. `vT::Vector{T}` is a pre-allocated
vector used for time-dependent differential equations.

"""
function taylorstep!{T<:Real, U<:Number}(f, t::Taylor1{T}, x::Taylor1{U},
        t0::T, t1::T, x0::U, order::Int, abstol::T)
    @assert t1 > t0

    # Compute the Taylor coefficients
    jetcoeffs!(f, t, x)

    # Compute the step-size of the integration using `abstol`
    δt = stepsize(x, abstol)
    δt = min(δt, t1-t0)

    x0 = evaluate(x, δt)
    return δt, x0
end

doc"""
    taylorstep!(f!, x, dx, xaux, t0, t1, x0, order, abstol, vT)

One-step Taylor integration for the ODE $\dot{x}=dx/dt=f(t, x)$
with initial conditions $x(t_0)=x_0$, computed from `t0` up to
`t1`, returning the time-step of the actual integration carried out
and updating (in-place) `x0`.

Here, `f!` is the function defining the RHS of the ODE (see
[`taylorinteg`](@ref) for examples and structure of `f!`), `x` contains
the Taylor expansion of the dependent variables, `x0` corresponds
to the initial (and updated) dependent variables and is of
type `Vector{Taylor1{T<:Number}}` or `Vector{Taylor1{TaylorN{T}}}`, `order`
is the degree used for the `Taylor1` polynomials during the integration
and `abstol` is the absolute tolerance used to determine the time step
of the integration.  `dx` and `xaux`, both of the same type as `x0`,
are needed to avoid allocations; `vT::Vector{T}` is a pre-allocated
vector used for time-dependent differential equations.


"""
function taylorstep!{T<:Real, U<:Number}(f!, t::Taylor1{T},
        x::Vector{Taylor1{U}}, dx::Vector{Taylor1{U}},
        xaux::Vector{Taylor1{U}}, t0::T, t1::T, x0::Array{U,1},
        order::Int, abstol::T)
    @assert t1 > t0

    # Compute the Taylor coefficients
    jetcoeffs!(f!, t, x, dx, xaux)

    # Compute the step-size of the integration using `abstol`
    δt = stepsize(x, abstol)
    δt = min(δt, t1-t0)

    evaluate!(x, δt, x0)
    return δt
end

# taylorinteg
doc"""
    taylorinteg(f, x0, t0, tmax, order, abstol; keyword... )

General-purpose Taylor integrator for the explicit ODE
$\dot{x}=f(t,x)$ with initial condition specified by `x0`
at time `t0`. The initial condition `x0` may be of type `T<:Number`
or a `Vector{T}`, with `T` including `TaylorN{T}`; the latter case
is of interest for jet transport applications.

It returns a vector with the values of time (independent variable),
and a vector (of type `typeof(x0)`) with the computed values of
the dependent variable(s). The integration stops when time
is larger than `tmax` (in which case the last returned values are
`t_max`, `x(t_max)`), or else when the number of saved steps is larger
than `maxsteps`.

The integration uses polynomial expansions on the independent variable
of order `order`; the parameter `abstol` serves to define the
time step using the last two Taylor coefficients of the expansions.
Make sure you use a *large enough* `order` to assure convergence.

The current keyword argument is `maxsteps=500`.

**Examples**:

- One dependent variable: The function `f` defines the equation of motion.

```julia
    using TaylorIntegration

    f(t, x) = x^2

    tv, xv = taylorinteg(f, 3, 0.0, 0.3, 25, 1.0e-20, maxsteps=100 )
```

- Many (two or more) dependent variable: The function `f!` defines
    the equation of motion.

```julia
    using TaylorIntegration

    function f!(t, x, dx)
        for i in eachindex(x)
            dx[i] = x[i]^2
        end
    end

    tv, xv = taylorinteg(f!, [3.0,3.0], 0.0, 0.3, 25, 1.0e-20, maxsteps=100 )
```
Note that `f!` updates (mutates) the pre-allocated vector `dx`.

"""
function taylorinteg{S<:Number, T<:Real, U<:Real, V<:Real}(f, x0::S,
        t0::T, tmax::U, order::Int, abstol::V; maxsteps::Int=500)

    #in order to handle mixed input types, we promote types before integrating:
    x0, t0, tmax, abstol, afloat = promote(x0, t0, tmax, abstol, one(Float64))

    taylorinteg(f, x0, t0, tmax, order, abstol, maxsteps=maxsteps)
end

function taylorinteg{S<:Number, T<:Real, U<:Real, V<:Real}(f,
        q0::Array{S,1}, t0::T, tmax::U, order::Int, abstol::V; maxsteps::Int=500)

    #promote to common type before integrating:
    elq0, t0, tmax, abstol, afloat = promote(q0[1], t0, tmax, abstol, one(Float64))
    #convert the elements of q0 to the common, promoted type:
    q0_ = convert(Array{typeof(elq0)}, q0)

    taylorinteg(f, q0_, t0, tmax, order, abstol, maxsteps=maxsteps)
end

function taylorinteg{T<:Real, U<:Number}(f, x0::U, t0::T, tmax::T, order::Int,
        abstol::T; maxsteps::Int=500)

    # Allocation
    const tv = Array{T}(maxsteps+1)
    const xv = Array{U}(maxsteps+1)

    # Initialize the Taylor1 expansions
    const t = Taylor1( T, order )
    const x = Taylor1( x0, order )

    # Initial conditions
    nsteps = 1
    @inbounds t[1] = t0
    @inbounds tv[1] = t0
    @inbounds xv[1] = x0

    # Integration
    while t0 < tmax
        δt, x0 = taylorstep!(f, t, x, t0, tmax, x0, order, abstol)
        @inbounds x[1] = x0
        t0 += δt
        @inbounds t[1] = t0
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds xv[nsteps] = x0
        if nsteps > maxsteps
            warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    #return tv, xv
    return view(tv,1:nsteps), view(xv,1:nsteps)
end

function taylorinteg{T<:Real, U<:Number}(f!, q0::Array{U,1}, t0::T, tmax::T,
        order::Int, abstol::T; maxsteps::Int=500)

    # Allocation
    const tv = Array{T}(maxsteps+1)
    dof = length(q0)
    const xv = Array{U}(dof, maxsteps+1)

    # Initialize the vector of Taylor1 expansions
    const t = Taylor1(T, order)
    const x = Array{Taylor1{U}}(dof)
    const dx = Array{Taylor1{U}}(dof)
    const xaux = Array{Taylor1{U}}(dof)

    # Initial conditions
    @inbounds t[1] = t0
    x .= Taylor1.(q0, order)
    x0 = deepcopy(q0)
    @inbounds tv[1] = t0
    @inbounds xv[:,1] .= q0

    # Integration
    nsteps = 1
    while t0 < tmax
        δt = taylorstep!(f!, t, x, dx, xaux, t0, tmax, x0, order, abstol)
        for i in eachindex(x0)
            @inbounds x[i][1] = x0[i]
        end
        t0 += δt
        @inbounds t[1] = t0
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds xv[:,nsteps] .= x0
        if nsteps > maxsteps
            warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return view(tv,1:nsteps), view(transpose(view(xv,:,1:nsteps)),1:nsteps,:)
end

# Integrate and return results evaluated at given time
doc"""
    taylorinteg(f, x0, t0, trange, order, abstol; keyword... )

General-purpose Taylor integrator for the explicit ODE
$\dot{x}=f(t,x)$ with initial condition specified by `x0::{T<:Number}`
or `x0::Vector{T}` at time `t0`.
It returns a vector with the values of time (independent variable),
and a vector (of type `typeof(x0)`) with the computed values of
the dependent variable(s), evaluated *only* at the times specified by
the range `trange`. The integration stops at `tmax=trange[end]`
(in which case the last returned values are `t_max`, `x(t_max)`), or
else when the number of computed time steps is larger than `maxsteps`.

The integration uses polynomial expansions on the independent variable
of order `order`; the parameter `abstol` serves to define the
time step using the last two Taylor coefficients of the expansions.
Make sure you use a *large enough* `order` to assure convergence.

The current keyword argument is `maxsteps=500`.

**Examples**:

- One dependent variable: The function `f` defines the equation of motion.

```julia
    using TaylorIntegration

    f(t, x) = x^2

    xv = taylorinteg(f, 3.0, 0.0:0.001:0.3, 25, 1.0e-20, maxsteps=100 )
```

- Many (two or more) dependent variable: The function f! defines the
    equation of motion.

```julia
    using TaylorIntegration

    function f!(t, x, dx)
        for i in eachindex(x)
            dx[i] = x[i]^2
        end
    end

    xv = taylorinteg(f!, [3.0, 3.0], 0.0:0.001:0.3, 25, 1.0e-20, maxsteps=100 )
```
Note that f! updates (mutates) the pre-allocated vector dx.

- Jet transport for the simple pendulum.

```julia
    using TaylorSeries, TaylorIntegration

    function pendulum!(t, x, dx) #the simple pendulum ODE
        dx[1] = x[2]
        dx[2] = -sin(x[1])
    end

    p = set_variables("ξ", numvars=2, order=5) #TaylorN set-up, order 5
    q0 = [1.3, 0.0]    # initial conditions
    q0TN = q0 + p      # parametrization of a neighbourhood around q0
    tr = 0.0:0.125:6pi

    @time xv = taylorinteg(pendulum!, q0TN, tr, 28, 1e-20, maxsteps=100);
```
Note that the initial conditions `q0TN` are of type `TaylorN{Float64}`.

"""
function taylorinteg{T<:Real, U<:Number}(f, x0::U, trange::Range{T},
        order::Int, abstol::T; maxsteps::Int=500)

    # Allocation
    nn = length(trange)
    const xv = Array{U}(nn)
    fill!(xv, T(NaN))

    # Initialize the Taylor1 expansions
    const t = Taylor1( T, order )
    const x = Taylor1( x0, order )

    # Initial conditions
    @inbounds t[1] = trange[1]
    @inbounds xv[1] = x0

    # Integration
    iter = 1
    while iter < nn
        @inbounds t0, t1 = trange[iter], trange[iter+1]
        nsteps = 0
        while nsteps < maxsteps
            δt, x0 = taylorstep!(f, t, x, t0, t1, x0, order, abstol)
            @inbounds x[1] = x0
            t0 += δt
            @inbounds t[1] = t0
            t0 ≥ t1 && break
            nsteps += 1
        end
        if nsteps ≥ maxsteps && t0 != t1
            warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
        iter += 1
        @inbounds xv[iter] = x0
    end

    return xv
end

function taylorinteg{T<:Real, U<:Number}(f!, q0::Array{U,1}, trange::Range{T},
        order::Int, abstol::T; maxsteps::Int=500)

    # Allocation
    nn = length(trange)
    dof = length(q0)
    const x0 = similar(q0, eltype(q0), dof)
    fill!(x0, T(NaN))
    const xv = Array{eltype(q0)}(dof, nn)
    for ind in 1:nn
        @inbounds xv[:,ind] .= x0
    end

    # Initialize the vector of Taylor1 expansions
    const t = Taylor1( T, order )
    const x = Array{Taylor1{U}}(dof)
    const dx = Array{Taylor1{U}}(dof)
    const xaux = Array{Taylor1{U}}(dof)

    # Initial conditions
    @inbounds t[1] = trange[1]
    x .= Taylor1.(q0, order)
    @inbounds x0 .= q0
    @inbounds xv[:,1] .= q0

    # Integration
    iter = 1
    while iter < nn
        t0, t1 = trange[iter], trange[iter+1]
        nsteps = 0
        while nsteps < maxsteps
            δt = taylorstep!(f!, t, x, dx, xaux, t0, t1, x0, order, abstol)
            for i in eachindex(x0)
                @inbounds x[i][1] = x0[i]
            end
            t0 += δt
            t0 ≥ t1 && break
            nsteps += 1
        end
        if nsteps ≥ maxsteps && t0 != t1
            warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
        iter += 1
        @inbounds xv[:,iter] .= x0
    end

    return transpose(xv)
end
