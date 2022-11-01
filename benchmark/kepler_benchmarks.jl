# ==========
# Kepler problem
# ==========

# ==========
# Define equations of motion
# ==========
# Using `^`
@taylorize function kepler1!(dq, q, p, t)
    local μ = p
    r_p3d2 = (q[1]^2+q[2]^2)^1.5
    #
    dq[1] = q[3]
    dq[2] = q[4]
    dq[3] = μ * q[1]/r_p3d2
    dq[4] = μ * q[2]/r_p3d2
    #
    return nothing
end

@taylorize function kepler2!(dq, q, p, t)
    nn, μ = p
    r2 = zero(q[1])
    for i = 1:nn
        r2_aux = r2 + q[i]^2
        r2 = r2_aux
    end
    r_p3d2 = r2^(3/2)
    for j = 1:nn
        dq[j] = q[nn+j]
        dq[nn+j] = μ*q[j]/r_p3d2
    end
    nothing
end

# Using `sqrt`
@taylorize function kepler5!(dq, q, p, t)
    local μ = -1.0
    r = sqrt(q[1]^2+q[2]^2)
    r_p3d2 = r^3
    dq[1] = q[3]
    dq[2] = q[4]
    dq[3] = μ * q[1] / r_p3d2
    dq[4] = μ * q[2] / r_p3d2
    return nothing
end

@taylorize function kepler6!(dq, q, p, t)
    local NN = 2
    local μ = p
    r2 = zero(q[1])
    for i = 1:NN
        r2_aux = r2 + q[i]^2
        r2 = r2_aux
    end
    r = sqrt(r2)
    r_p3d2 = r^3
    for j = 1:NN
        dq[j] = q[NN+j]
        dq[NN+j] =  μ * q[j] / r_p3d2
    end
    nothing
end


# ==========
# Run benchmarks
# ==========
# const TI = TaylorIntegration
const _abstol = 1.0e-20
const _order = 28
const t0 = 0.0
const tf1 = 2π*1000.0
const maxsteps1 = 500_000
const tf2 = 2π*10_000.0
const maxsteps2 = 50_000_000

pars = (2, -1.0)
q0 = [0.2, 0.0, 0.0, 3.0]

# ==========
# Kepler_parsed
# ==========
SUITE["Kepler"] = BenchmarkGroup()

SUITE["Kepler"]["kepler1-1"] = @benchmarkable taylorinteg(
    kepler1!, $q0, $t0, $tf1, $_order, $_abstol, -1.0, maxsteps=$maxsteps1)
SUITE["Kepler"]["kepler2-1"] = @benchmarkable taylorinteg(
    kepler2!, $q0, $t0, $tf1, $_order, $_abstol, $pars, maxsteps=$maxsteps1)

SUITE["Kepler"]["kepler5-1"] = @benchmarkable taylorinteg(
    kepler5!, $q0, $t0, $tf1, $_order, $_abstol, maxsteps=$maxsteps1)
SUITE["Kepler"]["kepler6-1"] = @benchmarkable taylorinteg(
    kepler6!, $q0, $t0, $tf1, $_order, $_abstol, -1.0, maxsteps=$maxsteps1)

SUITE["Kepler"]["kepler1-2"] = @benchmarkable taylorinteg(
    kepler1!, $q0, $t0, $tf2, $_order, $_abstol, -1.0, maxsteps=$maxsteps2)
SUITE["Kepler"]["kepler2-2"] = @benchmarkable taylorinteg(
    kepler2!, $q0, $t0, $tf2, $_order, $_abstol, $pars, maxsteps=$maxsteps2)

SUITE["Kepler"]["kepler5-2"] = @benchmarkable taylorinteg(
    kepler5!, $q0, $t0, $tf2, $_order, $_abstol, maxsteps=$maxsteps2)
SUITE["Kepler"]["kepler6-2"] = @benchmarkable taylorinteg(
    kepler6!, $q0, $t0, $tf2, $_order, $_abstol, -1.0, maxsteps=$maxsteps2)


# # ==========
# # KeplerNotParsed
# # ==========
# SUITE["KeplerNotParsed"] = BenchmarkGroup()

# SUITE["KeplerNotParsed"]["kepler1"] = @benchmarkable taylorinteg(
#     kepler1!, $q0, $t0, $tf, $_order, $_abstol, -1.0, maxsteps=$maxsteps,
#     parse_eqs=false)
# SUITE["KeplerNotParsed"]["kepler2"] = @benchmarkable taylorinteg(
#     kepler2!, $q0, $t0, $tf, $_order, $_abstol, $pars, maxsteps=$maxsteps,
#     parse_eqs=false)
#
# SUITE["KeplerNotParsed"]["kepler5"] = @benchmarkable taylorinteg(
#     kepler5!, $q0, $t0, $tf, $_order, $_abstol, maxsteps=$maxsteps,
#     parse_eqs=false)
# SUITE["KeplerNotParsed"]["kepler6"] = @benchmarkable taylorinteg(
#     kepler6!, $q0, $t0, $tf, $_order, $_abstol, -1.0, maxsteps=$maxsteps,
#     parse_eqs=false)
#
