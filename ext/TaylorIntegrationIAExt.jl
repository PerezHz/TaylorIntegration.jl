# This file is part of the TaylorIntegration.jl package; MIT licensed

module TaylorIntegrationIAExt

using TaylorIntegration

using IntervalArithmetic

const TI = TaylorIntegration
const IA = IntervalArithmetic

# stepsize
function TI.stepsize(q::AbstractArray{Taylor1{Interval{U}},N}, epsilon::T,
        relepsilon::T=zero(T)) where {T<:Real,U<:IA.NumTypes,N}
    R = promote_type(typeof(norm(constant_term(q[1]), Inf)), T)
    h = typemax(R)
    for i in eachindex(q)
        @inbounds hi = TI.stepsize(q[i], epsilon, relepsilon)
        h = min(h, hi)
    end

    # If `isinf(h)==true`, we use the maximum (finite)
    # step-size obtained from all coefficients as above.
    # Note that the time step is independent from `epsilon`.
    if isequal_interval(h, typemax(R))
        h = zero(R)
        for i in eachindex(q)
            @inbounds hi = TI._second_stepsize(q[i], epsilon)
            h = max(h, hi)
        end
    end
    return h::R
end

end
