# ==========
# Scalar problem: x' = b - x^2
# ==========
# Very short integrations: dominated by the cost per call (cache construction, parsing check).

let
    x0, t0, tf = SCALAR_X0, SCALAR_T0, SCALAR_TF
    order, abstol, maxsteps = SCALAR_ORDER, SCALAR_ABSTOL, SCALAR_MAXSTEPS
    steps(f, pe) = nsteps(taylorinteg(f, x0, t0, tf, order, abstol; maxsteps, parse_eqs = pe, dense = false))

    for pe in (true, false)
        label = pe ? "parsed" : "noparse"
        add!("Scalar", "riccati/$label", @benchmarkable(
            taylorinteg(scalar_riccati, $x0, $t0, $tf, $order, $abstol;
                maxsteps = $maxsteps, parse_eqs = $pe, dense = false)), steps = steps(scalar_riccati, pe))
    end
    add!("Scalar", "riccati/plain", @benchmarkable(
        taylorinteg(scalar_plain, $x0, $t0, $tf, $order, $abstol; maxsteps = $maxsteps, dense = false)),
        steps = steps(scalar_plain, false))
    add!("Scalar", "riccati/parsed dense", @benchmarkable(
        taylorinteg(scalar_riccati, $x0, $t0, $tf, $order, $abstol;
            maxsteps = $maxsteps, parse_eqs = true, dense = true)), steps = steps(scalar_riccati, true))
end