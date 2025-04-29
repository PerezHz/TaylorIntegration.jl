# This file is part of the TaylorIntegration.jl package; MIT licensed

#taylorstep
@doc doc"""
    taylorstep!(::Val{false}, f, t, x, abstol, params, rv, parse_eqs=true) -> δt
    taylorstep!(::Val{true}, f, t, x, abstol, params, rv, parse_eqs=true) -> δt
    taylorstep!(::Val{false}, f!, t, x, dx, xaux, abstol, params, rv, parse_eqs=true) -> δt
    taylorstep!(::Val{true}, f!, t, x, dx, xaux, abstol, params, rv, parse_eqs=true) -> δt

One-step Taylor integration for the one-dependent variable ODE ``\dot{x}=dx/dt=f(x, p, t)``
with initial conditions ``x(t_0)=x_0``.
Returns the time-step `δt` of the actual integration carried out (δt is positive).

Here, `f` represents the function defining the RHS of the ODE (see
[`taylorinteg`](@ref)), `t` is the
independent variable, `x` contains the Taylor expansion of the dependent
variable, `order` is
the degree  used for the `Taylor1` polynomials during the integration
`abstol` is the absolute tolerance used to determine the time step
of the integration, and `params` are the parameters entering the ODE
functions.
For several variables, `dx` and `xaux`, both of the same type as `x`,
are needed to save allocations. Finally, `parse_eqs` is a switch
to force *not* using (`parse_eqs=false`) the specialized method of `jetcoeffs!`
created with [`@taylorize`](@ref); the default is `true` (parse the equations).
Finally, `parse_eqs` is a switch
to force *not* using (`parse_eqs=false`) the specialized method of `jetcoeffs!`
created with [`@taylorize`](@ref); the default is `true` (parse the equations).
The first argument in the function call `Val{bool}` (`bool::Bool`) controls whether
a specialized [`jetcoeffs!](@ref) method is being used or not.

"""
function taylorstep!(
    ::Val{V},
    f,
    t::Taylor1{T},
    x::Taylor1{U},
    abstol::T,
    params,
    rv::RetAlloc{Taylor1{U}},
) where {T<:Real,U<:Number,V}

    # Compute the Taylor coefficients
    __jetcoeffs!(Val(V), f, t, x, params, rv)

    # Compute the step-size of the integration using `abstol`
    δt = stepsize(x, abstol)
    if isinf(δt)
        δt = _second_stepsize(x, abstol)
    end

    return δt
end

function taylorstep!(
    ::Val{V},
    f!,
    t::Taylor1{T},
    x::Vector{Taylor1{U}},
    dx::Vector{Taylor1{U}},
    xaux::Vector{Taylor1{U}},
    abstol::T,
    params,
    rv::RetAlloc{Taylor1{U}},
) where {T<:Real,U<:Number,V}

    # Compute the Taylor coefficients
    __jetcoeffs!(Val(V), f!, t, x, dx, xaux, params, rv)

    # Compute the step-size of the integration using `abstol`
    δt = stepsize(x, abstol)

    return δt
end