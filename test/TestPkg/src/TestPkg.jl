module TestPkg

# __precompile__(false)

using TaylorIntegration
# using BenchmarkTools
using InteractiveUtils

order = 25
N = 200
x0 = 10randn(2N)
μ = 1e-7rand(N)
x0[N] = -sum(x0[1:N-1] .* μ[1:N-1]) / μ[N]
x0[2N] = -sum(x0[N+1:2N-1] .* μ[1:N-1]) / μ[N]

@taylorize xdot1(x, p, t) = b1 - x^2

@taylorize function f1!(dq, q, params, t)
    local N = Int(length(q) / 2)
    local _eltype_q_ = eltype(q)
    local μ = params
    X = Array{_eltype_q_}(undef, N, N)
    accX = Array{_eltype_q_}(undef, N) #acceleration
    for j = 1:N
        accX[j] = zero(q[1])
        dq[j] = q[N+j]
    end
    #compute accelerations
    for j = 1:N
        for i = 1:N
            if i == j
            else
                X[i, j] = q[i] - q[j]
                temp_001 = accX[j] + (μ[i] * X[i, j])
                accX[j] = temp_001
            end #if i != j
        end #for, i
    end #for, j
    for i = 1:N
        dq[N+i] = accX[i]
    end
    nothing
end

function f2!(dq, q, params, t)
    local N = Int(length(q) / 2)
    local _eltype_q_ = eltype(q)
    local μ = params
    X = Array{_eltype_q_}(undef, N, N)
    accX = Array{_eltype_q_}(undef, N) #acceleration
    for j = 1:N
        accX[j] = zero(q[1])
        dq[j] = q[N+j]
    end
    #compute accelerations
    for j = 1:N
        for i = 1:N
            if i == j
            else
                X[i, j] = q[i] - q[j]
                temp_001 = accX[j] + (μ[i] * X[i, j])
                accX[j] = temp_001
            end #if i != j
        end #for, i
    end #for, j
    for i = 1:N
        dq[N+i] = accX[i]
    end
    nothing
end

function TaylorIntegration.jetcoeffs!(
    ::Val{f2!},
    t::Taylor1{_T},
    q::AbstractVector{Taylor1{_S}},
    dq::AbstractVector{Taylor1{_S}},
    params,
) where {_T<:Real,_S<:Number}
    order = t.order
    local N = Int(length(q) / 2)
    local _eltype_q_ = eltype(q)
    local μ = params
    X = Array{_eltype_q_}(undef, N, N)
    accX = Array{_eltype_q_}(undef, N)
    for j = 1:N
        accX[j] = Taylor1(zero(constant_term(q[1])), order)
        dq[j] = Taylor1(identity(constant_term(q[N+j])), order)
    end
    tmp337 = Array{Taylor1{_S}}(undef, size(X))
    tmp337 .= Taylor1(zero(_S), order)
    temp_001 = Array{Taylor1{_S}}(undef, size(tmp337))
    temp_001 .= Taylor1(zero(_S), order)
    for j = 1:N
        for i = 1:N
            if i == j
            else
                X[i, j] = Taylor1(constant_term(q[i]) - constant_term(q[j]), order)
                tmp337[i, j] = Taylor1(constant_term(μ[i]) * constant_term(X[i, j]), order)
                temp_001[i, j] =
                    Taylor1(constant_term(accX[j]) + constant_term(tmp337[i, j]), order)
                accX[j] = Taylor1(identity(constant_term(temp_001[i, j])), order)
            end
        end
    end
    for i = 1:N
        dq[N+i] = Taylor1(identity(constant_term(accX[i])), order)
    end
    for __idx in eachindex(q)
        (q[__idx]).coeffs[2] = (dq[__idx]).coeffs[1]
    end
    for ord = 1:order-1
        ordnext = ord + 1
        for j = 1:N
            TaylorSeries.zero!(accX[j], q[1], ord)
            TaylorSeries.identity!(dq[j], q[N+j], ord)
        end
        for j = 1:N
            for i = 1:N
                if i == j
                else
                    TaylorSeries.subst!(X[i, j], q[i], q[j], ord)
                    TaylorSeries.mul!(tmp337[i, j], μ[i], X[i, j], ord)
                    TaylorSeries.add!(temp_001[i, j], accX[j], tmp337[i, j], ord)
                    TaylorSeries.identity!(accX[j], temp_001[i, j], ord)
                end
            end
        end
        for i = 1:N
            TaylorSeries.identity!(dq[N+i], accX[i], ord)
        end
        for __idx in eachindex(q)
            (q[__idx]).coeffs[ordnext+1] = (dq[__idx]).coeffs[ordnext] / ordnext
        end
    end
    return nothing
end

ex1 = :(function f3!(dq, q, params, t)
    local N = Int(length(q) / 2)
    local _eltype_q_ = eltype(q)
    local μ = params
    X = Array{_eltype_q_}(undef, N, N)
    accX = Array{_eltype_q_}(undef, N) #acceleration
    for j = 1:N
        accX[j] = zero(q[1])
        dq[j] = q[N+j]
    end
    #compute accelerations
    for j = 1:N
        for i = 1:N
            if i == j
            else
                X[i, j] = q[i] - q[j]
                temp_001 = accX[j] + (μ[i] * X[i, j])
                accX[j] = temp_001
            end #if i != j
        end #for, i
    end #for, j
    for i = 1:N
        dq[N+i] = accX[i]
    end
    nothing
end)

ex2 = :(xdot2(x, p, t) = b1 - x^2)

ex3 = :(function harm_osc!(dx, x, p, t)
    local ω = p[1]
    local ω2 = ω^2
    dx[1] = x[2]
    dx[2] = -(ω2 * x[1])
    return nothing
end)

nex1, narr1 = TaylorIntegration._make_parsed_jetcoeffs(ex1)
nex2, narr2 = TaylorIntegration._make_parsed_jetcoeffs(ex2)
nex3, narr3 = TaylorIntegration._make_parsed_jetcoeffs(ex3)

greet(f, parse_eqs) = begin
    t = Taylor1(order)
    x = Taylor1.(x0, t.order)
    dx = similar(x)

    @show @which TaylorIntegration.__jetcoeffs!(Val(parse_eqs), f, t, x, dx, similar(x), μ)
    @show @which TaylorIntegration.jetcoeffs!(Val(f), t, x, dx, μ)
    @show methods(TaylorIntegration.jetcoeffs!)
    # @btime TaylorIntegration.__jetcoeffs!(Val($parse_eqs), $f, $t, $x, $dx, similar($x), μ)
end

end # module
