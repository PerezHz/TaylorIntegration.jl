# ==========
# Classical many-spin system
# ==========

using Random

# ==========
# Equations of motion
# ==========
@taylorize function spin_odes!(dq, q, params, t)
    local nn = params[1]
    local B = params[2]
    local J = params[3]

    # Initilize vars
    x = Array{eltype(q),1}(undef, nn)
    y = Array{eltype(q),1}(undef, nn)
    z = Array{eltype(q),1}(undef, nn)
    for i = 1:nn
        x[i] = q[3i-2]
        y[i] = q[3i-1]
        z[i] = q[3i  ]
    end

    local zz = zero(x[1])
    sum_s = Array{eltype(q),1}(undef, nn)
    aux   = Array{eltype(q),1}(undef, nn)
    for i = 1:nn
        sum_s[i] = zz
        for a = 1:nn
            aux[i] = sum_s[i]
            xJ = J[a,i] * x[a]
            sum_s[i] = aux[i] + xJ
        end
        dq[3i-2] = -B[i] * y[i]
        dq[3i-1] =  B[i] * x[i] + z[i] * sum_s[i]
        dq[3i  ] =              - y[i] * sum_s[i]
    end

    return dq
end

# ==========
# Some functions
# ==========
function fix_B(nn)
    @assert nn < 21
    local B20 = [0.80000305175781250, 0.85261507034301753, 1.1022420883178712, 0.98346004486083982,
        1.0131068229675293, 0.88758363723754885, 0.81881780624389644, 1.0715457916259765,
        1.0717185020446778, 1.1738771438598632, 0.95340080261230464, 1.0077665328979493,
        1.1323861122131347, 0.81382875442504887, 0.82138462066650386, 1.0118800163269044,
        1.0684597015380859, 0.80307922363281248, 0.95336618423461916, 0.82673683166503908]
        return B20[1:nn]
end
function fix_J(nn; α=1.4)
    JJ = zeros(nn, nn)
    @inbounds for d = 1:nn-1
        val = 1.0 / d^α
        @inbounds for i = 1:nn-d
            JJ[i, i+d] = val
            JJ[i+d, i] = val
        end
    end
    return JJ
end
function ini_conditions!(q, sz)
    # sz = similar(q, length(q)//3)
    @. sz = 2.0 * (rand() - 0.5)

    @inbounds for i in eachindex(sz)
        as = sqrt(1.0 - sz[i]*sz[i])
        pha = 2.0 * pi * rand()
        q[3i-2] = as * cos(pha)
        q[3i-1] = as * sin(pha)
        q[3i  ] = sz[i]
    end

    return nothing
end

# ==========
# Run benchmarks
# ==========
let
    local _abstol = 1.0e-20
    local _order = 26
    local L1 = 3
    local pars1 = (L1, fix_B(L1), fix_J(L1))
    local L2 = 7
    local pars2 = (L2, fix_B(L2), fix_J(L2))
    local t0 = 0.0
    local tf1 = 100.0
    local tf2 = 1000.0
    local maxsteps = 100_000
    Random.seed!(1023)
    local q1 = zeros(3*L1)
    ini_conditions!(q1, similar(q1, L1))
    Random.seed!(1023)
    local q2 = zeros(3*L2)
    ini_conditions!(q2, similar(q2, L2))

    SUITE["ManySpin"] = BenchmarkGroup()

    SUITE["ManySpin"]["manyspin1-1"] = @benchmarkable taylorinteg(
        spin_odes!, $q1, $t0, $tf1, $_order, $_abstol, $pars1, maxsteps=$maxsteps)
    SUITE["ManySpin"]["manyspin1-2"] = @benchmarkable taylorinteg(
        spin_odes!, $q1, $t0, $tf2, $_order, $_abstol, $pars1, maxsteps=$maxsteps)

    SUITE["ManySpin"]["manyspin2-1"] = @benchmarkable taylorinteg(
        spin_odes!, $q2, $t0, $tf1, $_order, $_abstol, $pars2, maxsteps=$maxsteps)
    SUITE["ManySpin"]["manyspin2-2"] = @benchmarkable taylorinteg(
        spin_odes!, $q2, $t0, $tf2, $_order, $_abstol, $pars2, maxsteps=$maxsteps)
end
