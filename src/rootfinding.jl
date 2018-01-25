doc"""
    surfacecrossing(g_old, g_now, eventorder::Int)

Detect if the solution crossed a root of event function `g`. `g_old` represents
the last-before-current value of event function `g`; `g_now` represents the
current value of event function `g`; `eventorder` is the order of the derivative
of the event function `g` whose root we are trying to find. Returns `true` if
`g_old` and `g_now` have different signs (i.e., if one is positive and the other
one is negative); otherwise returns `false`.
"""
surfacecrossing(g_old::Taylor1{T}, g_now::Taylor1{T}, eventorder::Int) where {T <: Real} = g_old[eventorder]*g_now[eventorder] < zero(T)

surfacecrossing(g_old::Taylor1{Taylor1{T}}, g_now::Taylor1{Taylor1{T}}, eventorder::Int) where {T <: Real} = g_old[eventorder][0]*g_now[eventorder][0] < zero(T)

surfacecrossing(g_old::Taylor1{TaylorN{T}}, g_now::Taylor1{TaylorN{T}}, eventorder::Int) where {T <: Real} = g_old[eventorder][0][1]*g_now[eventorder][0][1] < zero(T)

doc"""
    nrconvergencecriterion(g_val, nrabstol::T, nriter::Int, newtoniter::Int) where {T<:Real}

A rudimentary convergence criterion for the Newton-Raphson root-finding process.
`g_val` may be either a `Real`, `Taylor1{T}` or a `TaylorN{T}`, where `T<:Real`.
Returns `true` if: 1) the absolute value of `g_val`, the event function `g` evaluated at the
current estimated root by the Newton-Raphson process, is less than the `nrabstol`
tolerance; and 2) the number of iterations `nriter` of the Newton-Raphson process
is less than the maximum allowed number of iterations, `newtoniter`; otherwise,
returns `false`.
"""
nrconvergencecriterion(g_val::T, nrabstol::T, nriter::Int, newtoniter::Int) where {T<:Real} = abs(g_val) > nrabstol && nriter ≤ newtoniter

nrconvergencecriterion(g_val::Taylor1{T}, nrabstol::T, nriter::Int, newtoniter::Int) where {T<:Real} = abs(g_val[0]) > nrabstol && nriter ≤ newtoniter

nrconvergencecriterion(g_val::TaylorN{T}, nrabstol::T, nriter::Int, newtoniter::Int) where {T<:Real} = abs(g_val[0][1]) > nrabstol && nriter ≤ newtoniter

doc"""
    findroot!(g, t, x, dx, g_val_old, g_val, eventorder, tvS, xvS, gvS,
        t0, δt_old, x_dx, x_dx_val, g_dg, g_dg_val, nrabstol,
        newtoniter, nevents) -> nevents

Internal root-finding subroutine, based on Newton-Raphson process. If there is
a crossing, then the crossing data is stored in `tvS`, `xvS` and `gvS` and
`nevents`, the number of events/crossings, is updated. `g` is the
event function, `t` is a `Taylor1` polynomial which represents the independent
variable; `x` is an array of `Taylor1` variables which represent the vector of
dependent variables; `dx` is an array of `Taylor1` variables which represent the
LHS of the ODE; `g_val_old` is the last-before-current value of event function
`g`; `g_val` is the current value of the event function `g`; `eventorder` is
the order of the derivative of `g` whose roots the user is interested in finding;
`tvS` stores the surface-crossing instants; `xvS` stores the value of the
solution at each of the crossings; `gvS` stores the values of the event function
`g` (or its `eventorder`-th derivative) at each of the crossings; `t0` is the
current time; `δt_old` is the last time-step size; `x_dx`, `x_dx_val`, `g_dg`,
`g_dg_val` are auxiliary variables; `nrabstol` is the Newton-Raphson process
tolerance; `newtoniter` is the maximum allowed number of Newton-Raphson
iteration; `nevents` is the current number of detected events/crossings.
"""
function findroot!(g, t, x, dx, g_val_old, g_val, eventorder, tvS, xvS, gvS,
        t0, δt_old, x_dx, x_dx_val, g_dg, g_dg_val, nrabstol,
        newtoniter, nevents)

    if surfacecrossing(g_val_old, g_val, eventorder)
        #auxiliary variables
        const nriter = 1
        const dof = length(x)

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
        nriter == newtoniter+1 && warn("""
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

doc"""
    taylorinteg(f, g, x0, t0, tmax, order, abstol; kwargs... )

Root-finding method of `taylorinteg`. Given a function `g(t, x, dx)`,
called the event function, `taylorinteg` checks for the occurrence of a root
of `g` evaluated at the solution; that is, it checks for the occurrence of an
event or condition specified by `g=0`. Then, `taylorinteg` attempts to find that
root (or event, or crossing) by performing a Newton-Raphson process. When
called with the `eventorder=n` keyword argument, `taylorinteg` searches for the
roots of the `n`-th derivative of `g`, which is computed via automatic
differentiation.

`maxsteps` is the maximum number of allowed time steps; `eventorder` is the
order of the derivatives of `g` whose roots the user is interested in finding;
`newtoniter` is the maximum number of Newton-Raphson iterations per detected
root; `nrabstol` is the allowed tolerance for the Newton-Raphson process.

The current keyword arguments are `maxsteps=500`, `eventorder=0`,
`newtoniter=10`, and `nrabstol=eps(T)`, where `T` is the common type of `t0`,
`tmax` and `abstol`.

For more details about conventions in `taylorinteg`, please see [`taylorinteg`](@ref).

**Examples**:

```julia
    using TaylorIntegration
    function pendulum!(t, x, dx)
        dx[1] = x[2]
        dx[2] = -sin(x[1])
        nothing
    end
    g(t, x, dx) = x[2]
    x0 = [1.3, 0.0]
    # find the roots of `g` along the solution
    tv, xv, tvS, xvS, gvS = taylorinteg(pendulum!, g, x0, 0.0, 22.0, 28, 1.0E-20);
    # find the roots of the 2nd derivative of `g` along the solution
    tv, xv, tvS, xvS, gvS = taylorinteg(pendulum!, g, x0, 0.0, 22.0, 28, 1.0E-20; eventorder=2);
"""
function taylorinteg(f!, g, q0::Array{U,1}, t0::T, tmax::T,
        order::Int, abstol::T; maxsteps::Int=500, eventorder::Int=0,
        newtoniter::Int=10, nrabstol::T=eps(T)) where {T <: Real,U <: Number}

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
    @inbounds t[0] = t0
    x0 = deepcopy(q0)
    x .= Taylor1.(q0, order)
    @inbounds tv[1] = t0
    @inbounds xv[:,1] .= q0

    #Some auxiliary arrays for root-finding/event detection/Poincaré surface of section evaluation
    const g_val = zero(g(t,x,x))
    const g_val_old = zero(g_val)
    const slope = zero(U)
    const dt_li = zero(U)
    const dt_nr = zero(U)
    const δt = zero(U)
    const δt_old = zero(U)

    const x_dx = vcat(x, dx)
    const g_dg = vcat(g_val, g_val_old)
    const x_dx_val = Array{U}( length(x_dx) )
    const g_dg_val = vcat(evaluate(g_val), evaluate(g_val_old))

    const tvS = Array{U}(maxsteps+1)
    const xvS = similar(xv)
    const gvS = similar(tvS)

    # Integration
    const nsteps = 1
    const nevents = 1 #number of detected events
    while t0 < tmax
        δt_old = δt
        δt = taylorstep!(f!, t, x, dx, xaux, t0, tmax, x0, order, abstol)
        g_val = g(t, x, dx)
        nevents = findroot!(g, t, x, dx, g_val_old, g_val, eventorder,
            tvS, xvS, gvS, t0, δt_old, x_dx, x_dx_val, g_dg, g_dg_val,
            nrabstol, newtoniter, nevents)
        g_val_old = deepcopy(g_val)
        for i in eachindex(x0)
            @inbounds x[i][0] = x0[i]
        end
        t0 += δt
        @inbounds t[0] = t0
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

    return view(tv,1:nsteps), view(transpose(view(xv,:,1:nsteps)),1:nsteps,:), view(tvS,1:nevents-1), view(transpose(view(xvS,:,1:nevents-1)),1:nevents-1,:), view(gvS,1:nevents-1)
end
