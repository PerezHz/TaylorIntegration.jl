# This file is part of the TaylorIntegration.jl package; MIT licensed

doc"""
    jetcoeffs!(f, t, x)

Returns an updated `x` using the recursion relation of the
derivatives from the ODE $\dot{x}=dx/dt=f(t,x)$.

`f` is the function defining the RHS of the ODE, `x` is a `Taylor1{T}`
or a vector of that type, containing the Taylor expansion
of the dependent variables of the ODE and `t` is the independent
variable.
Initially, `x` contains only the 0-th order Taylor coefficients of
the current system state (the initial conditions), and `jetcoeffs!`
computes recursively the high-order derivates back into `x`.
"""
function jetcoeffs!{T<:Number}(eqsdiff, t0::T, x::Taylor1{T})
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

function jetcoeffs!{T<:Number}(eqsdiff, t0::T, x::Vector{Taylor1{T}})
    order = x[1].order
    xaux = similar(x)
    for ord in 1:order
        ordnext = ord+1

        # Set `xaux`, auxiliary vector of Taylor1 to order `ord`
        @inbounds for j in eachindex(x)
            xaux[j] = Taylor1( x[j].coeffs[1:ord] )
        end

        # Equations of motion
        # TODO! define a macro to optimize the eqsdiff
        F = eqsdiff(t0, xaux)

        # Recursion relations
        @inbounds for j in eachindex(x)
            x[j].coeffs[ordnext] = F[j].coeffs[ord]/ord
        end
    end
    nothing
end


doc"""
    stepsize(x, epsilon)

Returns a time-step for a `x::Taylor1{T}` using a
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

Returns the minimum time-step for `q::Array{Taylor1{T},1}`,
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
    evaluate!(x, δt, xnew)
    return xnew
end

doc"""
    evaluate!(x, δt, x0)

Evaluates each element of `x::Array{Taylor1{T},1}`, representing
the Taylor expansion for the dependent variables of an ODE at
*time* δt; it updates the vector `x0` with the computed values.
"""
function evaluate!{T<:Number}(x::Array{Taylor1{T},1}, δt::T, x0::Array{T,1})
    @assert length(x) == length(x0)
    @inbounds for i in eachindex(x)
        x0[i] = evaluate( x[i], δt )
    end
    nothing
end


doc"""
    taylorstep(f, t0, x0, abs_tol, order)

Compute one-step Taylor integration for the ODE $\dot{x}=dx/dt=f(t, x)$
with initial condition $x(t_0)=x0$, returning `x0` and
the step-size of the integration carried out.

Here, `x0` is the initial (and updated) dependent variable, `order`
is the degree used for the `Taylor1` polynomials during the integration
and `abs_tol` is the absolute tolerance used to determine the time step
of the integration.
"""
function taylorstep{T<:Number}(f, t0::T, x0::T, order::Int, abs_tol::T)
    # Inizialize the Taylor1 expansions
    xT = Taylor1( x0, order )
    # Compute the Taylor coefficients
    jetcoeffs!(f, t0, xT)
    # Compute the step-size of the integration using `abs_tol`
    δt = stepsize(xT, abs_tol)
    x0 = evaluate(xT, δt)
    return δt, x0
end

doc"""
    taylorstep!(f, t0, x0, abs_tol, order)

Compute one-step Taylor integration for the ODE $\dot{x}=dx/dt=f(t, x)$
with initial conditions $x(t_0)=x0$, a vector of type T, updating `x0`
and returning the step-size of the integration carried out.

Here, `x0` is the initial (and updated) dependent variables, `order`
is the degree used for the `Taylor1` polynomials during the integration
and `abs_tol` is the absolute tolerance used to determine the time step
of the integration.
"""
function taylorstep!{T<:Number}(f, t0::T, x0::Array{T,1}, order::Int, abs_tol::T)
    # Inizialize the vector of Taylor1 expansions
    xT = Array{Taylor1{T},1}(length(x0))
    for i in eachindex(x0)
        @inbounds xT[i] = Taylor1( x0[i], order )
    end
    # Compute the Taylor coefficients
    jetcoeffs!(f, t0, xT)
    # Compute the step-size of the integration using `abs_tol`
    δt = stepsize(xT, abs_tol)
    evaluate!(xT, δt, x0)
    return δt
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
function taylorinteg{T<:Number}(f, x0::T, t0::T, t_max::T,
        order::Int, abs_tol::T, maxsteps::Int=500)

    tv = [t0]
    xv = [x0]
    nsteps = 0
    while (t0 < t_max) && (nsteps < maxsteps)
        δt, x0 = taylorstep(f, t0, x0, order, abs_tol)
        t0 += δt
        push!(tv, t0)
        push!(xv, x0)
        nsteps += 1
    end

    return tv, xv
end

function taylorinteg{T<:Number}(f, x0::Array{T,1}, t0::T, t_max::T,
        order::Int, abs_tol::T, maxsteps::Int=500)

    tv = [t0]
    xv = Array{typeof(x0),1}()
    push!(xv, x0)
    nsteps = 0
    while (t0 < t_max) && (nsteps < maxsteps)
        δt = taylorstep!(f, t0, x0, order, abs_tol)
        t0 += δt
        push!(tv, t0)
        push!(xv, x0)
        nsteps += 1
    end

    return tv, xv
end
