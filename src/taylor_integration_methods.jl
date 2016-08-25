# This file is part of the TaylorIntegration.jl package; MIT licensed

doc"""
    jetcoeffs!(t, x, f)

Returns an updated `x` using the recursion relation of the
derivatives from the ODE $\dot{x}=f(t,x)$.

`f` is the function describing the RHS of the ODE, `x` is
a `TaylorSeries.Taylor1{T}` containing the Taylor expansion
of the dependent variable of the ODE and `t` is the independent
variable.
Initially, `x` contains only the 0-th order Taylor coefficients of
the current system state (the initial conditions), and `jetcoeffs!`
fills recursively the high-order derivates back into `x`.
"""
function jetcoeffs!{T<:Number}(t0::T, x::Taylor1{T}, eqsdiff)
    order = x.order
    for ord in 1:order
        ordnext = ord+1

        # Set `xaux`, auxiliary Taylor1 variable to order `ord`
        xaux = Taylor1( x.coeffs[1:ord] )

        # Equations of motion
        # TODO! define a macro to optimize the eqsdiff
        F = eqsdiff(t0, xaux)

        # Recursion relation
        @inbounds x.coeffs[ordnext] = F.coeffs[ord]/ord
    end
    nothing
end

doc"""
    jetcoeffs!(t, x, f)

Returns an updated `x` using the recursion relation of the
derivatives from the ODE $\dot{x}=f(t,x)$.

`f` is the function describing the RHS of the ODE, `x` is
a vector of `TaylorSeries.Taylor1{T}` containing the Taylor expansions
of the dependent variables of the ODE.
Initially, `x` contains only the 0-th order Taylor coefficients of
the current system state (the initial conditions), and `jetcoeffs!`
fills recursively the high-order derivates back into `x`.
"""
function jetcoeffs!{T<:Number}(t0::T, x::Vector{Taylor1{T}}, eqsdiff)
    order = x[1].order
    xaux = similar(x)
    for ord in 1:order
        ordnext = ord+1

        # Set `xaux`, auxiliary vector of Taylor1 to order `ord`
        for j in eachindex(x)
            @inbounds xaux[j] = Taylor1( x[j].coeffs[1:ord] )
        end

        # Equations of motion
        # TODO! define a macro to optimize the eqsdiff
        F = eqsdiff(t0, xaux)

        # Recursion relations
        for j in eachindex(x)
            @inbounds x[j].coeffs[ordnext] = F[j].coeffs[ord]/ord
        end
    end
    nothing
end


doc"""
    stepsize(x, epsilon)

Returns a time-step for a `x::TaylorSeries.Taylor1{T}` using a
prescribed absolute tolerance `epsilon`.
"""
function stepsize{T<:Number}(x::Taylor1{T}, epsilon::T)
    ord = x.order
    h = T(Inf)
    for k in [ord-1, ord]
        kinv = one(T)/k
        @inbounds aux = abs( x.coeffs[k+1] )
        h = min(h, (epsilon/aux)^kinv)
    end
    return h
end

doc"""
    stepsize(q, epsilon)

Returns the minimum time-step for `q::Array{TaylorSeries.Taylor1{T},1}`,
using a prescribed absolute tolerance `epsilon`.
"""
function stepsize{T<:Number}(q::Array{Taylor1{T},1}, epsilon::T)
    h = T(Inf)
    for i in eachindex(q)
        @inbounds hi = stepsize( q[i], epsilon )
        h = min( h, hi )
    end
    return h
end


doc"""
    evaluate(x, δt)

Evaluates each element of `x::Array{Taylor1{T},1}`, representing
the dependent variables of an ODE, at *time* δt.
"""
function evaluate{T<:Number}(x::Array{Taylor1{T},1}, δt::T)
    xnew = Array{T,1}( length(x) )
    for i in eachindex(x)
        @inbounds xnew[i] = evaluate( x[i], δt )
    end
    return xnew
end


doc"""
    taylorstep!(t0, x0, abs_tol, order, f)

Compute one-step Taylor integration for the ODE $\dot{x}=f(t, x)$
with initial condition $x(t_0)=x0$, updating `x0` and returning
the step-size of the integration carried out.

Here, `x0` is the initial (and updated) dependent variable, `order`
is the degree used for the `Taylor1` polynomials during the integration
and `abs_tol` is the absolute tolerance used to determine the time step
of the integration.
"""
function taylorstep!{T<:Number}(t0::T, x0::T, order::Int, abs_tol::T, f)
    # Inizialize the Taylor1 expansions
    xT = Taylor1( x0, order )
    # Compute the Taylor coefficients
    jetcoeffs!(t0, xT, f)
    # Compute the step-size of the integration using `abs_tol`
    δt = stepsize(xT, abs_tol)
    x0 = evaluate(xT, δt)
    t0 += δt
    return nothing
end
function taylorstep!{T<:Number}(t0::T, x0::Array{T,1}, order::Int, abs_tol::T, f)
    # Inizialize the vector of Taylor1 expansions
    xT = Array{Taylor1{T},1}(length(x0))
    for i in eachindex(x0)
        @inbounds xT[i] = Taylor1( x0[i], order )
    end
    # Compute the Taylor coefficients
    jetcoeffs!(xT, f)
    # Compute the step-size of the integration using `abs_tol`
    δt = stepsize(xT, abs_tol)
    x0 = evaluate(xT, δt)
    t0 += δt
    return nothing
end


doc"""
    taylorinteg!{T<:Number}(initial_state, abs_tol, order, t_max, datalog, params, f)

This is a general-purpose Taylor integrator for the explicit ODE
$\dot{x}=f(x)$ with initial condition specified by `initial_state::Array{T,1}`.
Returns final state up to time `t_max`, storing the system history into `datalog`. The Taylor expansion order
is specified by `order`, and `abs_tol` is the absolute tolerance. Time-step
control must be provided by the user via the `timestep_method` argument.

NOTE: this integrator assumes that the independent variable is included as the
first component of the `initial_state` array, and its evolution ṫ=1 must be included
in the equations of motion as well.
"""
function taylorinteg{T<:Number}(x0::T, t0::T, t_max::T,
        order::Int, abs_tol::T, f, maxsteps=500)

    datalog = Array{T,2}()
    t = t0
    nsteps = 0
    # push!(datalog, t0, x0)
    while (t0 < t_max) && (nsteps < maxsteps)
        dt = taylorstep!(x0, order, abs_tol, f)
        t += dt
        # push!(datalog, t0, x0)
        nsteps += 1
    end

    return state
end

function taylorinteg{T<:Number}(initial_state::Array{T,1},
    abs_tol::T, order::Int, t0::T, t_max::T,
    datalog::Array{Array{T,1},1}, f)

    @assert length(initial_state) == length(datalog)-1 "`length(initial_state)` must be equal to `length(datalog)` minus one"

    initial_stateT = Array{Taylor1{T},1}(length(initial_state))
    for i in eachindex(initial_state)
        @inbounds initial_stateT[i] = Taylor1( initial_state[i], order )
    end #for

    @assert length( f(initial_stateT) ) == length( initial_state ) "`length(f(initial_stateT, params))` must be equal to `length(initial_state)`"

    state = initial_state #`state` stores the current system state

    elapsed_time = Ref(zero(T)) #this `Base.RefValue{T}` variable stores elapsed time, so that we can change its .x field inside `taylorstep!`

    push!(datalog[1], t0)

    for i in 2:length(datalog)
        push!(datalog[i], state[i-1])
    end #for

    while datalog[1][end]<t_max

        state = taylorstep!(state, elapsed_time, state, abs_tol, order, f)

        @inbounds push!( datalog[1], t0+elapsed_time.x )

        for i in 2:length(datalog)
            @inbounds push!(datalog[i], state[i-1])
        end #for

    end #while

    return state
end
