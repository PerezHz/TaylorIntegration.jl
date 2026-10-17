# This file is part of the TaylorIntegration.jl package; MIT licensed

# Accuracy of the step-size conventions of Taylor integration.
#
# After evaluating the Taylor series at the step `δt`, the clock is advanced to `t + δt`, which is
# rounded. Three conventions are compared:
#   :raw      the series is evaluated at `δt`, the clock is `t += δt`      (before v0.18.19)
#   :nearest  the series is evaluated at `(t + δt) - t`                     (OrdinaryDiffEqCore)
#   :exact    the series is evaluated at `_exact_step(t, δt, tf, 1)`:
#             largest step `≤ δt` with `t + dt` exact                       (`taylorinteg`, v0.18.19)
# so that, in the last two, the recorded pair `(t_k, x_k)` is consistent with the recorded times.
#
# For each problem the Float64 integrations are compared with a reference computed with `BigFloat`
# at the *recorded* times `t_k`. The ensemble has several members: the order of the Taylor
# polynomial around the optimal one for `abstol` (Jorba-Zou: ⌈-log(abstol)/2⌉ + 1) and tiny changes of
# `abstol` (different rounding patterns; the truncation error is negligible). Conservative problems:
# energy. The errors are reported as maxima over time windows, since they oscillate along an orbit.
# Jet transport (Kepler, `TaylorN` of order 2 in 4 variables): relative errors of the jets of order 1
# (the Jacobian) and 2, for 1, 1.75 and 2.25 periods (`JET_PERIODS`; independent of `scale`).
#
#   julia --project=. benchmark/accuracy/accuracy.jl [scale] [nojets] [checkref]
#
# `scale` multiplies the integration time (default 0.1: a short run; use 1 or more for the real one).

using TaylorIntegration, Printf, Statistics
const TI = TaylorIntegration

# Optional: exact pendulum solution (Elliptic.jl) to validate the reference
const HAS_ELLIPTIC = try
    @eval using Elliptic
    true
catch
    false
end

optimal_order(abstol) = ceil(Int, -log(abstol) / 2) + 1

# ---------- problems (generic in the number type) ----------

function pendulum!(dq, q, p, t)
    dq[1] = q[2]
    dq[2] = -sin(q[1])
    return nothing
end

function kepler!(dq, q, p, t)                 # p = μ < 0
    r2 = q[1]^2 + q[2]^2
    r3 = r2 * sqrt(r2)
    dq[1] = q[3]
    dq[2] = q[4]
    dq[3] = p * q[1] / r3
    dq[4] = p * q[2] / r3
    return nothing
end

function duffing!(dq, q, p, t)                # forced (non-autonomous) nonlinear oscillator
    ε, γ, ω = p
    dq[1] = q[2]
    dq[2] = -q[1] - ε * q[1]^3 + γ * cos(ω * t)
    return nothing
end

# Parameters are Float64 values converted to the number type (the same values in all runs)
const PROBLEMS = (
    (name = "pendulum", f! = pendulum!, q0 = [1.3, 0.0], tf = 700.0, abstol = 1e-20,
        params = T -> nothing, energy = (q, p) -> q[2]^2 / 2 - cos(q[1])),
    (name = "kepler (e = 0.8)", f! = kepler!, q0 = [0.2, 0.0, 0.0, 3.0], tf = 2π * 100, abstol = 1e-20,
        params = T -> T(-1.0), energy = (q, p) -> (q[3]^2 + q[4]^2) / 2 + p / sqrt(q[1]^2 + q[2]^2)),
    (name = "forced Duffing", f! = duffing!, q0 = [0.5, 0.0], tf = 600.0, abstol = 1e-20,
        params = T -> (T(0.1), T(0.05), T(1.2)), energy = nothing),
)

# ---------- the three conventions ----------

function _step(mode::Symbol, t, δt, tf)
    if mode === :raw
        return min(δt, tf - t)
    elseif mode === :nearest
        dt = min(δt, tf - t)
        return (t + dt) - t
    else
        return TI._exact_step(t, δt, tf, 1)
    end
end

# Taylor integration (vector case, not parsed), as in `taylorinteg!`, with the chosen convention.
# `q0` may contain numbers or jets (`TaylorN`); the time is a Float64.
function integrate(f!, q0, params, t0, tf, order, abstol, mode; maxsteps = 1_000_000)
    cache = TI.init_cache(Val(false), t0, q0, 10, order, f!, params; parse_eqs = false)
    # `deepcopy`: jets (`TaylorN`) are mutable; with a shallow `copy` the stored states would alias
    # the state being updated in place
    x0 = deepcopy(q0)
    ts = typeof(t0)[t0]
    xs = Vector{eltype(q0)}[deepcopy(x0)]
    t = t0
    while t < tf
        TI.update_cache!(cache, t, x0)
        δt = TI.taylorstep!(cache.parse_eqs, f!, cache.t, cache.x, cache.dx, cache.xaux,
            abstol, params, cache.rv)
        dt = _step(mode, t, δt, tf)
        iszero(dt) && error("step below the resolution of the time variable")
        evaluate!(cache.x, dt, x0)
        t += dt
        push!(ts, t)
        push!(xs, deepcopy(x0))
        length(ts) > maxsteps && error("maxsteps exceeded")
    end
    return ts, xs
end

# ---------- references (BigFloat) ----------

reference(pb, tf; order_ref = 50, abstol_ref = 1e-40) =
    taylorinteg(pb.f!, big.(pb.q0), big(0.0), big(tf), order_ref, big(abstol_ref), pb.params(BigFloat);
        dense = true, maxsteps = 1_000_000, parse_eqs = false)

# ---------- analysis ----------

const NSAMPLES = 100
const NWINDOWS = 10

# Maxima over time windows (the errors oscillate along an orbit)
function windowmax(v, nw = NWINDOWS)
    n = length(v)
    return [maximum(v[floor(Int, (j - 1) * n / nw)+1:floor(Int, j * n / nw)]) for j = 1:nw]
end

# Slope of log(window maximum) against log(time) (least squares, skipping the first window)
function growth_slope(wmax, wtimes)
    lx, ly = log.(wtimes[2:end]), log.(wmax[2:end])
    return cov(lx, ly) / var(lx)
end

# Errors (and energy drift) at ~NSAMPLES recorded times
function analyze(ts, xs, solref, pb, E0)
    idxs = unique(round.(Int, range(1, length(ts), length = NSAMPLES)))
    errs = Float64[]
    dEs = Float64[]
    pbig = pb.params(BigFloat)
    for k in idxs
        xb = solref(big(ts[k]))
        push!(errs, Float64(maximum(abs.(big.(xs[k]) .- xb))))
        pb.energy === nothing || push!(dEs, Float64(abs(pb.energy(big.(xs[k]), pbig) - E0)))
    end
    return idxs, errs, dEs
end

function members(pb; offsets = -2:2, jitters = (0.0, 0.003))
    p0 = optimal_order(pb.abstol)
    return [(order = p0 + o, abstol = pb.abstol * (1 + j)) for o in offsets for j in jitters]
end

function run_problem(pb, scale)
    tf = pb.tf * scale
    ens = members(pb)
    println("== ", pb.name, "   (tf = ", round(tf, digits = 2), ", ", length(ens), " members: orders ",
        extrema(e.order for e in ens), ")")
    # Self-check: the harness with `:exact` reproduces `taylorinteg`
    p0 = optimal_order(pb.abstol)
    solti = taylorinteg(pb.f!, pb.q0, 0.0, tf, p0, pb.abstol, pb.params(Float64);
        maxsteps = 1_000_000, parse_eqs = false, dense = false)
    ts, xs = integrate(pb.f!, pb.q0, pb.params(Float64), 0.0, tf, p0, pb.abstol, :exact)
    same = solti.t == ts && solti.x == permutedims(reduce(hcat, xs))
    println("   harness (:exact) reproduces taylorinteg bit by bit: ", same, "   (", length(ts) - 1, " steps)")
    print("   reference (BigFloat)... ")
    tref = @elapsed solref = reference(pb, tf)
    @printf("%.1f s\n", tref)
    E0 = pb.energy === nothing ? nothing : pb.energy(big.(pb.q0), pb.params(BigFloat))

    @printf("   %-9s %11s %11s %11s %11s %11s %7s\n", "mode", "err last", "(min)", "(max)",
        "err max", "max |ΔE|", "slope")
    for mode in (:raw, :nearest, :exact)
        elast = Float64[]; emax = Float64[]; dEmax = Float64[]; slopes = Float64[]
        for m in ens
            ts, xs = integrate(pb.f!, pb.q0, pb.params(Float64), 0.0, tf, m.order, m.abstol, mode)
            idxs, errs, dEs = analyze(ts, xs, solref, pb, E0)
            w = windowmax(errs)
            wt = [mean(ts[idxs[floor(Int, (j - 1) * length(idxs) / NWINDOWS)+1:floor(Int, j * length(idxs) / NWINDOWS)]])
                  for j = 1:NWINDOWS]
            push!(elast, w[end]); push!(emax, maximum(errs)); push!(slopes, growth_slope(w, wt))
            isempty(dEs) || push!(dEmax, maximum(dEs))
        end
        @printf("   %-9s %11.3e %11.3e %11.3e %11.3e %11s %7.2f\n", mode, median(elast), minimum(elast),
            maximum(elast), median(emax), isempty(dEmax) ? "-" : @sprintf("%.3e", median(dEmax)),
            median(slopes))
    end
    println()
end

# ---------- jet transport (Kepler) ----------

# Relative error of the jets of order 0, 1, 2 (norm Inf of the coefficients); the Float64 jets are
# compared coefficient by coefficient with the BigFloat ones (no arithmetic between different types)
function jet_errors(xk, refk)
    out = zeros(3)
    for ord = 0:2
        num = zero(BigFloat)
        den = zero(BigFloat)
        for i in eachindex(xk)
            c = xk[i][ord].coeffs
            r = refk[i][ord].coeffs
            num = max(num, maximum(abs.(big.(c) .- r)))
            den = max(den, maximum(abs.(r)))
        end
        out[ord+1] = iszero(den) ? Float64(num) : Float64(num / den)    # den == 0 at t = 0 (order 2)
    end
    return out
end

# Durations of the jet transport study, in periods of the orbit. The initial condition is at
# perihelion, where the flow is most sensitive: integer numbers of periods end there, so fractions
# (1.75 and 2.25 periods) sample a more typical phase.
const JET_PERIODS = (1.0, 1.75, 2.25)

function run_jets(checkref = false)
    pb = PROBLEMS[2]
    tfmax = 2π * maximum(JET_PERIODS)
    abstol, p0 = pb.abstol, optimal_order(pb.abstol)
    println("== jet transport, ", pb.name, "   (TaylorN of order 2 in 4 variables; durations in periods: ",
        JET_PERIODS, ")")
    function jetreference(order_ref, abstol_ref, bits)
        return setprecision(BigFloat, bits) do
            pbig = variables!(BigFloat, "ξ"; order = 2, numvars = 4, nowarn = true)
            taylorinteg(pb.f!, big.(pb.q0) .+ pbig, big(0.0), big(tfmax), order_ref, big(abstol_ref),
                pb.params(BigFloat); dense = true, maxsteps = 1_000_000, parse_eqs = false)
        end
    end
    print("   reference (BigFloat, 128 bits, order 35)... ")
    tref = @elapsed solref = jetreference(35, 1e-30, 128)
    @printf("%.1f s\n", tref)
    if checkref
        print("   check: reference of higher accuracy (192 bits, order 45)... ")
        tref2 = @elapsed solref2 = jetreference(45, 1e-40, 192)
        d = maximum(maximum(jet_errors(solref(big(t)), solref2(big(t)))[2:3]) for t in [0.25, 0.5, 0.75, 1.0] .* tfmax)
        @printf("%.1f s;  max relative difference (orders 1, 2) = %.2e\n", tref2, d)
    end
    pvars = variables!(Float64, "ξ"; order = 2, numvars = 4, nowarn = true)
    q0TN = pb.q0 .+ pvars
    ens = [(order = p0 + o, abstol = abstol * (1 + j)) for o in (-1, 0, 1) for j in (0.0, 0.003)]
    println("   median over ", length(ens), " members of the relative error of the jets (order 0 is the nominal trajectory)")
    @printf("   %8s %-9s | %12s %12s | %12s %12s\n", "periods", "mode", "end: order 1", "order 2",
        "max: order 1", "order 2")
    for periods in JET_PERIODS
        tf = 2π * periods
        for mode in (:raw, :nearest, :exact)
            e_end = zeros(length(ens), 3)
            e_max = zeros(length(ens), 3)
            for (i, m) in enumerate(ens)
                ts, xs = integrate(pb.f!, q0TN, pb.params(Float64), 0.0, tf, m.order, m.abstol, mode)
                idxs = unique(round.(Int, range(1, length(ts), length = NSAMPLES)))
                for k in idxs
                    e_max[i, :] = max.(e_max[i, :], jet_errors(xs[k], solref(big(ts[k]))))
                end
                e_end[i, :] = jet_errors(xs[end], solref(big(ts[end])))
            end
            @printf("   %8.2f %-9s | %12.3e %12.3e | %12.3e %12.3e\n", periods, mode,
                median(e_end[:, 2]), median(e_end[:, 3]), median(e_max[:, 2]), median(e_max[:, 3]))
        end
    end
    println()
end

# ---------- optional check of the pendulum reference with the exact solution (Elliptic.jl) ----------

function check_pendulum_reference(scale)
    HAS_ELLIPTIC || return println("Pendulum: check with Elliptic.jl skipped (package not available)\n")
    pb = PROBLEMS[1]
    try
        θ0 = pb.q0[1]
        k = sin(θ0 / 2)
        m = k^2
        K = Elliptic.K(m)
        exact(t) = 2asin(k * Elliptic.Jacobi.sn(K + t, m))
        solref = reference(pb, pb.tf * scale)
        d = maximum(abs(Float64(solref(big(t))[1]) - exact(t)) for t in [0.25, 0.5, 0.75, 1.0] .* (pb.tf * scale))
        @printf("Pendulum: |BigFloat reference - Elliptic.jl| = %.2e (Float64 accuracy of Elliptic.jl)\n\n", d)
    catch err
        println("Pendulum: check with Elliptic.jl skipped (", sprint(showerror, err), ")\n")
    end
end

# Optional: the BigFloat references against references of higher order and smaller tolerance
function check_references(scale)
    for pb in PROBLEMS
        tf = pb.tf * scale
        r1 = reference(pb, tf)
        r2 = reference(pb, tf; order_ref = 60, abstol_ref = 1e-48)
        d = maximum(Float64(maximum(abs.(r1(big(t)) .- r2(big(t))))) for t in [0.25, 0.5, 0.75, 1.0] .* tf)
        @printf("Reference of %-18s: max difference with a more accurate one = %.2e\n", pb.name, d)
    end
    println()
end

scale = isempty(ARGS) ? 0.1 : parse(Float64, ARGS[1])
"checkref" in ARGS && check_references(scale)
check_pendulum_reference(scale)
for pb in PROBLEMS
    run_problem(pb, scale)
end
"nojets" in ARGS || run_jets("checkref" in ARGS)