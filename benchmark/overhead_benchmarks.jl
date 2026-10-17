# ==========
# Per-call overhead: cache construction, `_determine_parsing!`; and `_exact_step`
# ==========

using InteractiveUtils: methodswith

let
    q0, t0, order = KEPLER_Q0, KEPLER_T0, KEPLER_ORDER
    p = KEPLER_MU
    maxsteps = 100
    t, x, dx = TI.init_expansions(t0, q0, order)
    trange = collect(range(t0, KEPLER_TF, length = 1001))

    # The check done (before v0.18.19) in `_determine_parsing!`, for reference
    add!("Overhead", "methodswith", @benchmarkable(
        !isempty(methodswith(Val{kepler1!}, TI.jetcoeffs!)) &&
        !isempty(methodswith(Val{kepler1!}, TI._allocate_jetcoeffs!)), evals = 10))
    add!("Overhead", "_determine_parsing! parsed", @benchmarkable(
        TI._determine_parsing!(true, kepler1!, $t, $x, $dx, $p), evals = 10))
    add!("Overhead", "_determine_parsing! plain", @benchmarkable(
        TI._determine_parsing!(true, kepler_plain!, $t, $x, $dx, $p), evals = 10))
    for (label, f, pe) in (("parsed", kepler1!, true), ("noparse", kepler1!, false), ("plain", kepler_plain!, true))
        add!("Overhead", "init_cache $label", @benchmarkable(
            TI.init_cache(Val(false), $t0, $q0, $maxsteps, $order, $f, $p; parse_eqs = $pe), evals = 10))
    end
    add!("Overhead", "issorted trange", @benchmarkable(issorted($trange), evals = 10))
    for (label, pe) in (("parsed", true), ("noparse", false))
        add!("Overhead", "init_cache trange $label", @benchmarkable(
            TI.init_cache(Val(false), $trange, $q0, $maxsteps, $order, kepler1!, $p; parse_eqs = $pe),
            evals = 10))
    end

    # Cost of making the step exact (v0.18.19), per call: a typical step of a Kepler orbit
    if isdefined(TI, :_exact_step)
        tt, δt, tmax = 123.456, 0.8, 1000.0
        add!("Overhead", "_exact_step", @benchmarkable(TI._exact_step($tt, $δt, $tmax, 1), evals = 1000))
    end
end