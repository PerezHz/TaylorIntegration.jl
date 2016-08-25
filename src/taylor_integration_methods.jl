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
    taylorinteg(x0, t0, tmax, order, abs_tol, f[, maxsteps::Int=500])

This is a general-purpose Taylor integrator for the explicit ODE
$\dot{x}=f(x)$ with initial condition specified by `x0` at time `t0`.
It returns a vector with the values of time (independent variable),
and a vector (of of type `typeof(x0)`) with the computed values of
the dependent variables. The integration stops when time
is larger than `tmax`, or the number of saved steps is larger
than `maxsteps`.

The integrator uses polynomial expansions on the independent variable
of order `order` and the parameter `abs_tol` serves to define the
time step using the last two Taylor coefficients of the expansions.
"""
function taylorinteg{T<:Number}(x0::T, t0::T, t_max::T,
        order::Int, abs_tol::T, f, maxsteps::Int=500)

    tv = [t0]
    xv = [x0]
    nsteps = 0
    while (t0 < t_max) && (nsteps < maxsteps)
        taylorstep!(t0, x0, order, abs_tol, f)
        # t += dt
        push!(tv, t0)
        push!(xv, x0)
        nsteps += 1
    end

    return tv, xv
end

function taylorinteg{T<:Number}(x0::Array{T,1}, t0::T, t_max::T,
    order::Int, abs_tol::T, f, maxsteps::Int=500)

    tv = [t0]
    xv = Array{typeof(x0),1}()
    push!(xv, x0)

    while (t0 < t_max) && (nsteps < maxsteps)
        taylorstep!(t0, x0, order, abs_tol, f)
        push!(tv, t0)
        push!(xv, x0)
        nsteps += 1
    end

    return tv, xv
end
