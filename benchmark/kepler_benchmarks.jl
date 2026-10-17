# ==========
# Kepler problem (parsed, `@taylorize`)
# ==========
# Names: "<problem>/parsed" (`parse_eqs = true`, `dense = false`); dense; trange; inplace (`taylorinteg!`
# with a preallocated cache: isolates the integration loop). The not parsed path is benchmarked only
# in the (cheap) scalar problem.

let
    q0, t0, tf = KEPLER_Q0, KEPLER_T0, KEPLER_TF
    order, abstol, maxsteps = KEPLER_ORDER, KEPLER_ABSTOL, KEPLER_MAXSTEPS
    trange = collect(range(t0, tf, length = 1001))
    steps(f, p) = nsteps(taylorinteg(f, q0, t0, tf, order, abstol, p; maxsteps, parse_eqs = true, dense = false))

    for pb in KEPLER_PROBLEMS
        pb.taylorized || continue
        f, p, name = pb.f, pb.params, pb.name
        add!("Kepler", "$name/parsed", @benchmarkable(
            taylorinteg($f, $q0, $t0, $tf, $order, $abstol, $p;
                maxsteps = $maxsteps, parse_eqs = true, dense = false)), steps = steps(f, p))
    end

    f1, p1 = kepler1!, KEPLER_MU
    n1 = steps(f1, p1)
    add!("Kepler", "kepler1/parsed dense", @benchmarkable(
        taylorinteg($f1, $q0, $t0, $tf, $order, $abstol, $p1;
            maxsteps = $maxsteps, parse_eqs = true, dense = true)), steps = n1)
    add!("Kepler", "kepler1/parsed trange", @benchmarkable(
        taylorinteg($f1, $q0, $trange, $order, $abstol, $p1; maxsteps = $maxsteps, parse_eqs = true)),
        steps = n1)

    cache = TI.init_cache(Val(false), t0, q0, maxsteps, order, f1, p1; parse_eqs = true)
    add!("Kepler", "kepler1/parsed inplace", @benchmarkable(
        TI.taylorinteg!(Val(false), $f1, $q0, $t0, $tf, $abstol, $cache, $p1;
            maxsteps = $maxsteps, copy_solution = Val(false))), steps = n1)
    cachet = TI.init_cache(Val(false), trange, q0, maxsteps, order, f1, p1; parse_eqs = true)
    add!("Kepler", "kepler1/parsed trange inplace", @benchmarkable(
        TI.taylorinteg!($f1, $q0, $trange, $abstol, $cachet, $p1;
            maxsteps = $maxsteps, copy_solution = Val(false))), steps = n1)
end