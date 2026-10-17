# ==========
# Lyapunov spectrum of the Lorenz system
# ==========
# `lyap_taylorinteg` needs the number of `TaylorN` variables to be the dimension of the system; it is
# set in `setup`, since other benchmarks (jet transport) change this global state.

let
    q0, t0, tf = LORENZ_Q0, LORENZ_T0, LORENZ_TF
    order, abstol, maxsteps = LORENZ_ORDER, LORENZ_ABSTOL, LORENZ_MAXSTEPS
    pars = LORENZ_PARS
    variables!("δ", order = 1, numvars = 3, nowarn = true)

    n = nsteps(lyap_taylorinteg(lorenz!, q0, t0, tf, order, abstol, pars; maxsteps, parse_eqs = true))
    add!("Lyapunov", "lorenz/parsed", @benchmarkable(
        lyap_taylorinteg(lorenz!, $q0, $t0, $tf, $order, $abstol, $pars;
            maxsteps = $maxsteps, parse_eqs = true),
        setup = (variables!("δ", order = 1, numvars = 3, nowarn = true)), evals = 1), steps = n)
end