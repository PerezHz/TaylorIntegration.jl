# This file is part of the TaylorIntegration.jl package; MIT licensed

"""
    _norminf(x::Number)
    _norminf(x::Taylor1)
    _norminf(x::HomogeneousPolynomial)
    _norminf(x::TaylorN)

Return the infinity norm used by the time-step controllers.

For ordinary numbers this delegates to `norm(x, Inf)`. For nested Taylor
objects, it recursively takes the maximum infinity norm over all stored
coefficients, reducing values such as `Taylor1{TaylorN}` to a scalar norm.
"""
@inline _norminf(x::Number) = norm(x, Inf)

function _norminf(x::Taylor1)
    coeffs = x.coeffs
    @inbounds h = zero(_norminf(coeffs[firstindex(coeffs)]))
    @inbounds for c in coeffs
        h = max(h, _norminf(c))
    end
    return h
end

function _norminf(x::HomogeneousPolynomial)
    coeffs = x.coeffs
    @inbounds h = zero(_norminf(coeffs[firstindex(coeffs)]))
    @inbounds for c in coeffs
        h = max(h, _norminf(c))
    end
    return h
end

function _norminf(x::TaylorN)
    coeffs = x.coeffs
    @inbounds h = zero(_norminf(coeffs[firstindex(coeffs)]))
    @inbounds for c in coeffs
        h = max(h, _norminf(c))
    end
    return h
end

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
function stepsize(x::Taylor1{U}, absepsilon::T,
                  relepsilon::T = zero(T)) where {T<:Real,U<:Number}
    x0 = constant_term(x)
    x0norm = _norminf(x0)
    R = promote_type(typeof(x0norm), T)
    ord = order(x)
    h = typemax(R)
    for k in (ord - 1, ord)
        @inbounds aux = _norminf(x[k])
        TS._isthinzero(aux) && continue
        #eq. 3-3 Jorba and Zou (2005)
        if absepsilon ≥ relepsilon * x0norm
            aux1 = _stepsize(aux, absepsilon, k)
        else
            aux1 = _stepsize(aux, relepsilon * x0norm, k)
        end
        h = min(h, aux1)
    end
    return h::R
end

function stepsize(q::AbstractArray{Taylor1{U},N}, absepsilon::T,
                  relepsilon::T = zero(T),) where {T<:Real,U<:Number,N}
    R = promote_type(typeof(_norminf(constant_term(q[1]))), T)
    h = typemax(R)
    for i in eachindex(q)
        @inbounds hi = stepsize(q[i], absepsilon, relepsilon)
        h = min(h, hi)
    end

    # If `isinf(h)==true`, we use the maximum (finite)
    # step-size obtained from all coefficients as above.
    # Note that the time step is independent from `epsilon`.
    if isinf(h)
        h = zero(R)
        for i in eachindex(q)
            @inbounds hi = _second_stepsize(q[i], absepsilon)
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
    R = promote_type(typeof(_norminf(constant_term(x))), T)
    TS._isthinzero(x) && return typemax(R)#convert(R, Inf)
    ord = order(x)
    u = one(R)
    h = zero(R)
    for k = 1:ord-2
        @inbounds aux = _norminf(x[k])
        TS._isthinzero(aux) && continue
        aux1 = _stepsize(aux, u, k)
        h = max(h, aux1)
    end
    return h::R
end


# The following is used to get an exact, floating number representation,
# of the step size δt and of the accumulated time
# Knuth's TwoSum: s + e == a + b exactly
@inline function _two_sum(a::T, b::T) where {T<:AbstractFloat}
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e
end

"""
    _exact_step(t0, δt, tmax, sign_tstep) -> δt′

Largest step with `t0 + δt′` exact in floating point and `|δt′| ≤ |δt|`, except that
a step reaching `tmax` lands exactly on it. `sign_tstep = ±1` (forward/backward).

If `|δt|` is below the resolution of `t0` (no representable step `≤ |δt|` exists),
it returns `zero(δt)`: callers must check `iszero(δt′)`, otherwise the time variable
does not advance.
"""
function _exact_step(t0::T, δt::T, tmax::T, sign_tstep::Int) where {T<:AbstractFloat}
    t1 = t0 + δt
    sign_tstep*t1 >= sign_tstep*tmax && (t1 = tmax)
    for _ in 1:8
        δt1 = t1 - t0
        s, e = _two_sum(t0, δt1)
        if s == t1 && iszero(e)                          # t0 + δt1 == t1 exactly
            # `δt1 == 0` here means no representable step `≤ |δt|` (below the resolution of `t0`)
            (t1 == tmax || sign_tstep*δt1 <= sign_tstep*δt) && return δt1
            t1 = sign_tstep > 0 ? prevfloat(t1) : nextfloat(t1)    # ½-ulp overshoot
        else
            # t1 - t0 not representable (t1 not within a factor 2 of t0): take an
            # exact shorter step; later steps reach tmax exactly (Sterbenz).
            if sign_tstep*t0 > 0                         # away from 0: at most doubling
                δt = sign_tstep * min(abs(δt1), abs(t0))
            elseif sign_tstep*tmax >= 0 && abs(t0) <= abs(δt)   # 0 lies on the way
                δt = -t0                                 # stop exactly at 0
            else                                         # toward 0, tmax before 0
                δt = sign_tstep * min(abs(δt1), abs(t0)) / 2
            end
            t1 = t0 + δt
        end
    end
    error("_exact_step: no exact step found (t0 = $t0, δt = $δt)")
end
# exact types (e.g. Rational): the sum is exact, only the clamp to `tmax` is needed
_exact_step(t0::Real, δt::Real, tmax::Real, sign_tstep::Int) =
    sign_tstep * min(sign_tstep * δt, sign_tstep * (tmax - t0))