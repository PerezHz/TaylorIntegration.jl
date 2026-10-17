# ==========
# Jet transport (dense output), pendulum
# ==========
# The `TaylorN` initial condition is built in `setup`, since the number of `TaylorN` variables is
# global state (also changed by other benchmarks).

let
    q0, t0, tf = PEND_Q0, 0.0, JT_TF
    order, abstol, maxsteps = JT_ORDER, PEND_ABSTOL, JT_MAXSTEPS
    jo = JT_JETORDER
    q0T1 = [Taylor1([q0[1], 1.0], jo), Taylor1([q0[2], 1.0], jo)]
    steps(q) = nsteps(taylorinteg(bench_pendulum!, q, t0, tf, order, abstol; maxsteps, dense = true))

    add!("JetTransport", "Float64 dense", @benchmarkable(
        taylorinteg(bench_pendulum!, $q0, $t0, $tf, $order, $abstol; maxsteps = $maxsteps, dense = true)),
        steps = steps(q0))
    add!("JetTransport", "Taylor1 dense", @benchmarkable(
        taylorinteg(bench_pendulum!, $q0T1, $t0, $tf, $order, $abstol; maxsteps = $maxsteps, dense = true)),
        steps = steps(q0T1))
    qTN = q0 + variables!("ξ", numvars = 2, order = jo, nowarn = true)
    add!("JetTransport", "TaylorN dense", @benchmarkable(
        taylorinteg(bench_pendulum!, qTN, $t0, $tf, $order, $abstol; maxsteps = $maxsteps, dense = true),
        setup = (qTN = $q0 + variables!("ξ", numvars = 2, order = $jo, nowarn = true)), evals = 1),
        steps = steps(qTN))
end