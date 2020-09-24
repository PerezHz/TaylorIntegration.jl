"""
    surfacecrossing(g_old, g_now, eventorder::Int)

Detect if the solution crossed a root of event function `g`. `g_old` represents
the last-before-current value of event function `g`, and `g_now` represents the
current one; these are `Tuple{Bool,Taylor1{T}}`s. `eventorder` is the order of the derivative
of the event function `g` whose root we are trying to find. Returns `true` if
the constant terms of `g_old[2]` and `g_now[2]` have different signs (i.e.,
if one is positive and the other
one is negative). Otherwise, if `g_old[2]` and `g_now[2]` have the same sign or if
the first component of either of them is `false`, then it returns `false`.
"""
function surfacecrossing(g_old::Tuple{Bool,Taylor1{T}}, g_now::Tuple{Bool,Taylor1{T}},
        eventorder::Int) where {T <: Number}
    (g_old[1] && g_now[1]) || return false
    g_product = constant_term(g_old[2][eventorder])*constant_term(g_now[2][eventorder])
    return g_product < zero(g_product)
end


"""
    nrconvergencecriterion(g_val, nrabstol::T, nriter::Int, newtoniter::Int) where {T<:Real}

A rudimentary convergence criterion for the Newton-Raphson root-finding process.
`g_val` may be either a `Real`, `Taylor1{T}` or a `TaylorN{T}`, where `T<:Real`.
Returns `true` if: 1) the absolute value of `g_val`, the value of the event function
`g` evaluated at the
current estimated root by the Newton-Raphson process, is less than the `nrabstol`
tolerance; and 2) the number of iterations `nriter` of the Newton-Raphson process
is less than the maximum allowed number of iterations, `newtoniter`; otherwise,
returns `false`.
"""
nrconvergencecriterion(g_val::U, nrabstol::T, nriter::Int,
        newtoniter::Int) where {U<:Number, T<:Real} = abs(constant_term(g_val)) > nrabstol && nriter ≤ newtoniter

"""
    findroot!(t, x, dx, g_tupl_old, g_tupl, eventorder, tvS, xvS, gvS,
        t0, δt_old, x_dx, x_dx_val, g_dg, g_dg_val, nrabstol,
        newtoniter, nevents) -> nevents

Internal root-finding subroutine, based on Newton-Raphson process. If there is
a crossing, then the crossing data is stored in `tvS`, `xvS` and `gvS` and
`nevents`, the number of events/crossings, is updated. Here
`t` is a `Taylor1` polynomial which represents the independent
variable; `x` is an array of `Taylor1` variables which represent the vector of
dependent variables; `dx` is an array of `Taylor1` variables which represent the
LHS of the ODE; `g_tupl_old` is the last-before-current value returned by event function
`g` and `g_tupl` is the current one; `eventorder` is
the order of the derivative of `g` whose roots the user is interested in finding;
`tvS` stores the surface-crossing instants; `xvS` stores the value of the
solution at each of the crossings; `gvS` stores the values of the event function
`g` (or its `eventorder`-th derivative) at each of the crossings; `t0` is the
current time; `δt_old` is the last time-step size; `x_dx`, `x_dx_val`, `g_dg`,
`g_dg_val` are auxiliary variables; `nrabstol` is the Newton-Raphson process
tolerance; `newtoniter` is the maximum allowed number of Newton-Raphson
iteration; `nevents` is the current number of detected events/crossings.
"""
function findroot!(t, x, dx, g_tupl_old, g_tupl, eventorder, tvS, xvS, gvS,
        t0, δt_old, x_dx, x_dx_val, g_dg, g_dg_val, nrabstol,
        newtoniter, nevents)

    if surfacecrossing(g_tupl_old, g_tupl, eventorder)
        #auxiliary variables
        g_val = g_tupl[2]
        g_val_old = g_tupl_old[2]
        nriter = 1
        dof = length(x)

        #first guess: linear interpolation
        slope = (g_val[eventorder]-g_val_old[eventorder])/δt_old
        dt_li = -(g_val[eventorder]/slope)

        x_dx[1:dof] = x
        x_dx[dof+1:2dof] = dx
        g_dg[1] = derivative(g_val, eventorder)
        g_dg[2] = derivative(g_dg[1])

        #Newton-Raphson iterations
        dt_nr = dt_li
        evaluate!(g_dg, dt_nr, view(g_dg_val,:))

        while nrconvergencecriterion(g_dg_val[1], nrabstol, nriter, newtoniter)
            dt_nr = dt_nr-g_dg_val[1]/g_dg_val[2]
            evaluate!(g_dg, dt_nr, view(g_dg_val,:))
            nriter += 1
        end
        nriter == newtoniter+1 && @warn("""
        Newton-Raphson did not converge for prescribed tolerance and maximum allowed iterations.
        """)
        evaluate!(x_dx, dt_nr, view(x_dx_val,:))

        tvS[nevents] = t0+dt_nr
        xvS[:,nevents] .= view(x_dx_val,1:dof)
        gvS[nevents] = g_dg_val[1]

        nevents += 1
    end

    return nevents
end

"""
    taylorinteg(f, g, x0, t0, tmax, order, abstol, params[=nothing]; kwargs... )

    taylorinteg(f, g, x0, trange, order, abstol; kwargs... )

Root-finding method of `taylorinteg`.

Given a function `g(dx, x, params, t)::Tuple{Bool, Taylor1{T}}`,
called the event function, `taylorinteg` checks for the occurrence of a root
or event defined by `cond2 == 0` (`cond2::Taylor1{T}`)
if `cond1::Bool` is satisfied (`true`); `g` is thus assumed to return the
tuple (cond1, cond2). Then, `taylorinteg` attempts to find that
root (or event, or crossing) by performing a Newton-Raphson process. When
called with the `eventorder=n` keyword argument, `taylorinteg` searches for the
roots of the `n`-th derivative of `cond2`, which is computed via automatic
differentiation.

The current keyword arguments are:
- `maxsteps[=500]`: maximum number of integration steps.
- `parse_eqs[=true]`: use the specialized method of `jetcoeffs!` created
    with [`@taylorize`](@ref).
- `eventorder[=0]`: order of the derivative of `g` whose roots are computed.
- `newtoniter[=10]`: maximum Newton-Raphson iterations per detected root.
- `nrabstol[=eps(T)]`: allowed tolerance for the Newton-Raphson process; T is the common
    type of `t0`, `tmax` (or `eltype(trange)`) and `abstol`.


## Examples:

```julia
using TaylorIntegration

function pendulum!(dx, x, params, t)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
    nothing
end

g(dx, x, params, t) = (true, x[2])

x0 = [1.3, 0.0]

# find the roots of `g` along the solution
tv, xv, tvS, xvS, gvS = taylorinteg(pendulum!, g, x0, 0.0, 22.0, 28, 1.0E-20)

# find the roots of the 2nd derivative of `g` along the solution
tv, xv, tvS, xvS, gvS = taylorinteg(pendulum!, g, x0, 0.0, 22.0, 28, 1.0E-20; eventorder=2)

# times at which the solution will be returned
tv = 0.0:1.0:22.0

# find the roots of `g` along the solution; return the solution *only* at each value of `tv`
xv, tvS, xvS, gvS = taylorinteg(pendulum!, g, x0, tv, 28, 1.0E-20)

# find the roots of the 2nd derivative of `g` along the solution; return the solution *only* at each value of `tv`
xv, tvS, xvS, gvS = taylorinteg(pendulum!, g, x0, tv, 28, 1.0E-20; eventorder=2)
```

"""
function taylorinteg(f!, g, q0::Array{U,1}, t0::T, tmax::T, order::Int,
        abstol::T, params = nothing; maxsteps::Int=500, parse_eqs::Bool=true,
        dense::Bool=false, eventorder::Int=0, newtoniter::Int=10,
        nrabstol::T=eps(T)) where {T <: Real,U <: Number}

    @assert order ≥ eventorder "`eventorder` must be less than or equal to `order`"

    # Allocation
    tv = Array{T}(undef, maxsteps+1)
    dof = length(q0)
    xv = Array{U}(undef, dof, maxsteps+1)
    if dense
        xv_interp = Array{Taylor1{U}}(undef, dof, maxsteps+1)
    end

    # Initialize the vector of Taylor1 expansions
    t = Taylor1(T, order)
    x = Array{Taylor1{U}}(undef, dof)
    dx = Array{Taylor1{U}}(undef, dof)
    xaux = Array{Taylor1{U}}(undef, dof)
    dx = Array{Taylor1{U}}(undef, dof)

    # Initial conditions
    @inbounds t[0] = t0
    x .= Taylor1.(q0, order)
    dx .= zero.(x)
    x0 = deepcopy(q0)
    @inbounds tv[1] = t0
    @inbounds xv[:,1] .= q0
    sign_tstep = copysign(1, tmax-t0)

    # Determine if specialized jetcoeffs! method exists
    parse_eqs = _determine_parsing!(parse_eqs, f!, t, x, dx, params)

    # Some auxiliary arrays for root-finding/event detection/Poincaré surface of section evaluation
    g_tupl = g(dx, x, params, t)
    g_tupl_old = g(dx, x, params, t)
    # g_val = zero(gg_tupl[2])
    # g_val_old = zero(g_val)
    slope = zero(x[1])
    dt_li = zero(x[1])
    dt_nr = zero(x[1])
    δt = zero(x[1])
    δt_old = zero(x[1])

    x_dx = vcat(x, dx)
    # g_dg = vcat(g_val, g_val_old)
    g_dg = vcat(g_tupl[2], g_tupl_old[2])
    x_dx_val = Array{U}(undef, length(x_dx) )
    # g_dg_val = vcat(evaluate(g_val), evaluate(g_val_old))
    g_dg_val = vcat(evaluate(g_tupl[2]), evaluate(g_tupl_old[2]))

    tvS = Array{U}(undef, maxsteps+1)
    xvS = similar(xv)
    gvS = similar(tvS)

    # Integration
    nsteps = 1
    nevents = 1 #number of detected events
    while sign_tstep*t0 < sign_tstep*tmax
        δt_old = δt
        δt = taylorstep!(f!, t, x, dx, xaux, abstol, params, parse_eqs) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep*(tmax-t0))
        evaluate!(x, δt, x0) # new initial condition
        # g_val = g(dx, x, params, t)
        # nevents = findroot!(t, x, dx, g_val_old, g_val, eventorder,
        #     tvS, xvS, gvS, t0, δt_old, x_dx, x_dx_val, g_dg, g_dg_val,
        #     nrabstol, newtoniter, nevents)
        # g_val_old = deepcopy(g_val)
        g_tupl = g(dx, x, params, t)
        nevents = findroot!(t, x, dx, g_tupl_old, g_tupl, eventorder,
        tvS, xvS, gvS, t0, δt_old, x_dx, x_dx_val, g_dg, g_dg_val,
        nrabstol, newtoniter, nevents)
        if dense
            xv_interp[:,nsteps] .= deepcopy(x)
        end
        g_tupl_old = deepcopy(g_tupl)
        for i in eachindex(x0)
            @inbounds x[i][0] = x0[i]
        end
        t0 += δt
        @inbounds t[0] = t0
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds xv[:,nsteps] .= x0
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end
    if dense
        return TaylorInterpolant(tv[1], view(tv.-tv[1],1:nsteps), view(transpose(view(xv_interp,:,1:nsteps-1)),1:nsteps-1,:)), view(tvS,1:nevents-1), view(transpose(view(xvS,:,1:nevents-1)),1:nevents-1,:), view(gvS,1:nevents-1)
    else
        return view(tv,1:nsteps), view(transpose(view(xv,:,1:nsteps)),1:nsteps,:), view(tvS,1:nevents-1), view(transpose(view(xvS,:,1:nevents-1)),1:nevents-1,:), view(gvS,1:nevents-1)
    end
end

function taylorinteg(f!, g, q0::Array{U,1}, trange::AbstractVector{T},
        order::Int, abstol::T, params = nothing; maxsteps::Int=500, parse_eqs::Bool=true,
        eventorder::Int=0, newtoniter::Int=10, nrabstol::T=eps(T)) where {T <: Real,U <: Number}

    @assert order ≥ eventorder "`eventorder` must be less than or equal to `order`"

    # Check if trange is increasingly or decreasingly sorted
    @assert (issorted(trange) ||
        issorted(reverse(trange))) "`trange` or `reverse(trange)` must be sorted"

    # Allocation
    nn = length(trange)
    dof = length(q0)
    x0 = similar(q0, eltype(q0), dof)
    fill!(x0, T(NaN))
    xv = Array{eltype(q0)}(undef, dof, nn)
    for ind in 1:nn
        @inbounds xv[:,ind] .= x0
    end

    # Initialize the vector of Taylor1 expansions
    t = Taylor1( T, order )
    x = Array{Taylor1{U}}(undef, dof)
    dx = Array{Taylor1{U}}(undef, dof)
    xaux = Array{Taylor1{U}}(undef, dof)
    for i in eachindex(q0)
        @inbounds x[i] = Taylor1( q0[i], order )
        @inbounds dx[i] = Taylor1( zero(q0[i]), order )
    end

    # Initial conditions
    @inbounds t[0] = trange[1]
    @inbounds t0, t1, tmax = trange[1], trange[2], trange[end]
    sign_tstep = copysign(1, tmax-t0)
    x0 = deepcopy(q0)
    x1 = similar(x0)
    x .= Taylor1.(q0, order)
    @inbounds xv[:,1] .= q0

    # Determine if specialized jetcoeffs! method exists
    parse_eqs = _determine_parsing!(parse_eqs, f!, t, x, dx, params)

    # Some auxiliary arrays for root-finding/event detection/Poincaré surface of section evaluation
    g_tupl = g(dx, x, params, t)
    g_tupl_old = g(dx, x, params, t)
    # g_val = zero(gg_tupl[2])
    # g_val_old = zero(g_val)
    slope = zero(U)
    dt_li = zero(U)
    dt_nr = zero(U)
    δt = zero(U)
    δt_old = zero(U)

    x_dx = vcat(x, dx)
    # g_dg = vcat(g_val, g_val_old)
    g_dg = vcat(g_tupl[2], g_tupl_old[2])
    x_dx_val = Array{U}(undef, length(x_dx) )
    # g_dg_val = vcat(evaluate(g_val), evaluate(g_val_old))
    g_dg_val = vcat(evaluate(g_tupl[2]), evaluate(g_tupl_old[2]))

    tvS = Array{U}(undef, maxsteps+1)
    xvS = similar(xv)
    gvS = similar(tvS)

    # Integration
    iter = 2
    nsteps = 1
    nevents = 1 #number of detected events
    while sign_tstep*t0 < sign_tstep*tmax
        δt_old = δt
        δt = taylorstep!(f!, t, x, dx, xaux, abstol, params, parse_eqs) # δt is positive!
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
        # g_val = g(dx, x, params, t)
        # nevents = findroot!(t, x, dx, g_val_old, g_val, eventorder,
        #     tvS, xvS, gvS, t0, δt_old, x_dx, x_dx_val, g_dg, g_dg_val,
        #     nrabstol, newtoniter, nevents)
        # g_val_old = deepcopy(g_val)
        g_tupl = g(dx, x, params, t)
        nevents = findroot!(t, x, dx, g_tupl_old, g_tupl, eventorder,
            tvS, xvS, gvS, t0, δt_old, x_dx, x_dx_val, g_dg, g_dg_val,
            nrabstol, newtoniter, nevents)
        g_tupl_old = deepcopy(g_tupl)
        for i in eachindex(x0)
            @inbounds x[i][0] = x0[i]
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

    return transpose(xv), view(tvS,1:nevents-1), view(transpose(view(xvS,:,1:nevents-1)),1:nevents-1,:), view(gvS,1:nevents-1)
end
