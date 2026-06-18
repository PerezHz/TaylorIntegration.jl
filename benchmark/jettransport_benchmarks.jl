# ==========
# Jet transport dense output
# ==========

@taylorize function jt_pendulum!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
    return nothing
end

let
    local _abstol = 1.0e-20
    local _order = 20
    local t0 = 0.0
    local tf = 100.0
    local maxsteps = 10_000
    local q0 = [1.3, 0.0]

    local q0T1 = [
        Taylor1([q0[1], 1.0], 3),
        Taylor1([q0[2], 1.0], 3),
    ]

    variables!("xi", numvars = 2, order = 3)
    local q0TN = q0 + variables()

    SUITE["JetTransport"] = BenchmarkGroup()

    SUITE["JetTransport"]["Float64 dense"] = @benchmarkable taylorinteg(
        jt_pendulum!,
        $q0,
        $t0,
        $tf,
        $_order,
        $_abstol,
        maxsteps = $maxsteps,
        dense = true,
    )

    SUITE["JetTransport"]["Taylor1 dense"] = @benchmarkable taylorinteg(
        jt_pendulum!,
        $q0T1,
        $t0,
        $tf,
        $_order,
        $_abstol,
        maxsteps = $maxsteps,
        dense = true,
    )

    SUITE["JetTransport"]["TaylorN dense"] = @benchmarkable taylorinteg(
        jt_pendulum!,
        $q0TN,
        $t0,
        $tf,
        $_order,
        $_abstol,
        maxsteps = $maxsteps,
        dense = true,
    )
end
