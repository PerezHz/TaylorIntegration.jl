# This file is part of the TaylorIntegration.jl package; MIT licensed

# stepsize
"""
    stepsize(x, epsilon) -> h

Returns a maximum time-step for a the Taylor expansion `x`
using a prescribed absolute tolerance `epsilon` and the last two
Taylor coefficients of (each component of) `x`.

Note that `x` is of type `Taylor1{U}` or `Vector{Taylor1{U}}`, including
also the cases `Taylor1{TaylorN{U}}` and `Vector{Taylor1{TaylorN{U}}}`.

Depending of `eltype(x)`, i.e., `U<:Number`, it may be necessary to overload
`stepsize`, specializing it on the type `U`, to avoid type instabilities.
"""
function stepsize(x::Taylor1{U}, epsilon::T) where {T<:Real,U<:Number}
    R = promote_type(typeof(norm(constant_term(x), Inf)), T)
    ord = x.order
    h = typemax(R)
    for k in (ord - 1, ord)
        @inbounds aux = norm(x[k], Inf)
        TS._isthinzero(aux) && continue
        aux1 = _stepsize(aux, epsilon, k)
        h = min(h, aux1)
    end
    return h::R
end

function stepsize(q::AbstractArray{Taylor1{U},N}, epsilon::T) where {T<:Real,U<:Number,N}
    R = promote_type(typeof(norm(constant_term(q[1]), Inf)), T)
    h = typemax(R)
    for i in eachindex(q)
        @inbounds hi = stepsize(q[i], epsilon)
        h = min(h, hi)
    end

    # If `isinf(h)==true`, we use the maximum (finite)
    # step-size obtained from all coefficients as above.
    # Note that the time step is independent from `epsilon`.
    if isinf(h)
        h = zero(R)
        for i in eachindex(q)
            @inbounds hi = _second_stepsize(q[i], epsilon)
            h = max(h, hi)
        end
    end
    return h::R
end

"""
    _stepsize(aux1, epsilon, k)

Helper function to avoid code repetition.
Returns `(epsilon/aux1)^(1/k)`.
"""
@inline function _stepsize(aux1::U, epsilon::T, k::Int) where {T<:Real,U<:Number}
    aux = epsilon / aux1
    kinv = 1 / k
    return aux^kinv
end

"""
    _second_stepsize(x, epsilon)

Corresponds to the "second stepsize control" in Jorba and Zou
(2005) paper. We use it if [`stepsize`](@ref) returns `Inf`.
"""
function _second_stepsize(x::Taylor1{U}, epsilon::T) where {T<:Real,U<:Number}
    R = promote_type(typeof(norm(constant_term(x), Inf)), T)
    iszero(x) && return convert(R, Inf)
    ord = x.order
    u = one(R)
    h = zero(R)
    for k = 1:ord-2
        @inbounds aux = norm(x[k], Inf)
        TS._isthinzero(aux) && continue
        aux1 = _stepsize(aux, u, k)
        h = max(h, aux1)
    end
    return h::R
end