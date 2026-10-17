# ==========
# Pendulum with root finding (events: velocity = 0)
# ==========

let
    q0, t0, tf = PEND_Q0, PEND_T0, PEND_TF
    order, abstol, maxsteps = PEND_ORDER, PEND_ABSTOL, PEND_MAXSTEPS

    for dense in (false, true)
        pe = true
        label = "parsed" * (dense ? " dense" : "")
        n = nsteps(taylorinteg(bench_pendulum!, pendulum_event, q0, t0, tf, order, abstol;
            maxsteps, parse_eqs = pe, dense = dense))
        add!("Pendulum", "events/$label", @benchmarkable(
            taylorinteg(bench_pendulum!, pendulum_event, $q0, $t0, $tf, $order, $abstol;
                maxsteps = $maxsteps, parse_eqs = $pe, dense = $dense)), steps = n)
    end
end