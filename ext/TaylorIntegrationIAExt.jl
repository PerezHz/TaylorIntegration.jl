# This file is part of the TaylorIntegration.jl package; MIT licensed

module TaylorIntegrationIAExt

using TaylorIntegration

using IntervalArithmetic

const TI = TaylorIntegration
const IA = IntervalArithmetic

# stepsize
function TI.stepsize(x::Taylor1{Interval{U}}, absepsilon::T,
        relepsilon::T=zero(T), minstep::T = zero(T),
        maxstep::T = T(Inf)) where {T<:Real,U<:Number}
    x0 = constant_term(x)
    R = promote_type(IA.numtype(x0), T)#promote_type(typeof(norm(x0, Inf)), T)
    ord = get_order(x)
    h = typemax(R)
    for k in (ord - 1, ord)
        @inbounds aux = sup(norm(x[k], Inf))
        TS._isthinzero(aux) && continue
        #eq. 3-3 Jorba and Zou (2005)
        if absepsilon ≥ relepsilon * sup(norm(x0, Inf))
            aux1 = TI._stepsize(aux, absepsilon, k)
        else
            aux1 = TI._stepsize(aux, relepsilon * sup(norm(x0, Inf)), k)
        end
        h = min(h, aux1)
    end
    h = clamp(h, minstep, maxstep)
    return h::R
end

function TI.stepsize(q::AbstractArray{Taylor1{Interval{U}},N}, epsilon::T,
                     relepsilon::T=zero(T), minstep::T = zero(T),
                     maxstep::T = T(Inf)) where {T<:Real,U<:IA.NumTypes,N}
    R = promote_type(IA.numtype(constant_term(q[1])), T)#promote_type(typeof(norm(x0, Inf)), T)
    h = typemax(R)
    for i in eachindex(q)
        @inbounds hi = TI.stepsize(q[i], epsilon, relepsilon)
        h = min(h, hi)
    end

    # If `isinf(h)==true`, we use the maximum (finite)
    # step-size obtained from all coefficients as above.
    # Note that the time step is independent from `epsilon`.
    if isinf(h)
        h = zero(R)
        for i in eachindex(q)
            @inbounds hi = TI._second_stepsize(q[i], epsilon)
            h = max(h, inf(hi))
        end
    end
    h = clamp(h, minstep, maxstep)
    return h::R
end

# @inline function TI._stepsize(aux1::Interval{T}, epsilon::Interval{T}, k::Int) where
#         {T<:Real}
#     aux = epsilon / aux1
#     kinv = inv(interval(k))
#     return aux^kinv
# end

end
