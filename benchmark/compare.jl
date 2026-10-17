# This file is part of the TaylorIntegration.jl package; MIT licensed

# Compare two result files produced by `run_local.jl`.
#
#   julia --project=. benchmark/compare.jl <base> <target> [min]
#
# `<base>` and `<target>` are labels (files `benchmark/results/<label>.json`) or paths to json
# files. Times are *medians* (or minima, with the third argument `min`); all ratios are
# target/base, so a ratio < 1 means that `target` is better. Reported:
#   1. per benchmark: time (flag with 5% tolerance), allocations and memory ratios; and the
#      geometric means per group and overall;
#   2. per step: time/step and allocations/step (benchmarks with a step count in both runs);
#   3. the overhead of the common interface: fixed cost per call `a` and cost per step `b`
#      (time = a + b * steps) of each interface, and their ratios (group `Interface`).

using BenchmarkTools, Printf, Statistics

length(ARGS) in (2, 3) || error("usage: compare.jl <base> <target> [min]")
estimator = length(ARGS) == 3 && ARGS[3] == "min" ? minimum : median

resultspath(arg) = isfile(arg) ? arg : joinpath(@__DIR__, "results", arg * ".json")
loadresults(arg) = BenchmarkTools.load(resultspath(arg))[1]

function loadsteps(arg)
    path = replace(resultspath(arg), r"\.json$" => ".steps.tsv")
    d = Dict{Tuple{String,String},Int}()
    isfile(path) || return d
    for line in eachline(path)
        g, name, s = split(line, '\t')
        d[(String(g), String(name))] = parse(Int, s)
    end
    return d
end

fmttime(ns) =
    ns < 1e3 ? @sprintf("%.1f ns", ns) :
    ns < 1e6 ? @sprintf("%.2f μs", ns / 1e3) :
    ns < 1e9 ? @sprintf("%.2f ms", ns / 1e6) : @sprintf("%.2f s", ns / 1e9)

ratio(a, b) = (iszero(a) && iszero(b)) ? 1.0 : (iszero(b) ? NaN : a / b)
fmtratio(r) = isnan(r) ? "      -" : @sprintf("%7.3f", r)
geomean(v) = (v = filter(x -> isfinite(x) && x > 0, v); isempty(v) ? NaN : exp(mean(log.(v))))

base = estimator(loadresults(ARGS[1]))
target = estimator(loadresults(ARGS[2]))
steps_base, steps_target = loadsteps(ARGS[1]), loadsteps(ARGS[2])

inboth(g, b) = haskey(base, g) && haskey(base[g], b)

# ---------- 1. per benchmark ----------
println("base = ", ARGS[1], "   target = ", ARGS[2], "   (", estimator === median ? "median" : "minimum", ")\n")
@printf("%-32s %11s %11s %7s %-8s %7s %7s  %s\n", "benchmark", "base", "target", "time", "", "allocs",
    "memory", "allocs (base → target)")
println("-"^108)
all_t = Float64[]; all_a = Float64[]; all_m = Float64[]
for gname in sort!(collect(keys(target)))
    gt, ga, gm = Float64[], Float64[], Float64[]
    println(gname)
    for bname in sort!(collect(keys(target[gname])))
        t = target[gname][bname]
        if !inboth(gname, bname)
            @printf("  %-30s %11s %11s %7s %-8s %7s %7s  %d (new)\n", bname, "-", fmttime(time(t)), "", "new",
                "", "", allocs(t))
            continue
        end
        b = base[gname][bname]
        # Classify by *time* only (`isimprovement`/`isregression` also look at memory)
        tj = BenchmarkTools.time(judge(t, b))
        flag = tj == :improvement ? "faster" : tj == :regression ? "SLOWER" : "~"
        rt, ra, rm = ratio(time(t), time(b)), ratio(allocs(t), allocs(b)), ratio(memory(t), memory(b))
        push!(gt, rt); push!(ga, ra); push!(gm, rm)
        @printf("  %-30s %11s %11s %s %-8s %s %s  %d → %d\n", bname, fmttime(time(b)), fmttime(time(t)),
            fmtratio(rt), flag, fmtratio(ra), fmtratio(rm), allocs(b), allocs(t))
    end
    @printf("  %-30s %11s %11s %s %-8s %s %s\n", "geometric mean", "", "", fmtratio(geomean(gt)), "",
        fmtratio(geomean(ga)), fmtratio(geomean(gm)))
    append!(all_t, gt); append!(all_a, ga); append!(all_m, gm)
end
println("-"^108)
@printf("%-32s %11s %11s %s %-8s %s %s\n", "overall geometric mean", "", "", fmtratio(geomean(all_t)), "",
    fmtratio(geomean(all_a)), fmtratio(geomean(all_m)))

# ---------- 2. per step ----------
rows = []
for gname in sort!(collect(keys(target))), bname in sort!(collect(keys(target[gname])))
    inboth(gname, bname) || continue
    sb, st = get(steps_base, (gname, bname), 0), get(steps_target, (gname, bname), 0)
    (sb > 0 && st > 0) || continue
    b, t = base[gname][bname], target[gname][bname]
    push!(rows, (gname, bname, time(b) / 1e3 / sb, time(t) / 1e3 / st, allocs(b) / sb, allocs(t) / st))
end
if !isempty(rows)
    println("\nPer step\n")
    @printf("%-40s %9s %9s %7s   %10s %10s %7s\n", "benchmark", "μs/step", "μs/step", "ratio",
        "allocs/step", "allocs/step", "ratio")
    @printf("%-40s %9s %9s %7s   %10s %10s %7s\n", "", "base", "target", "", "base", "target", "")
    println("-"^100)
    for (g, n, tb, tt, ab, at) in rows
        @printf("%-40s %9.3f %9.3f %s   %10.2f %10.2f %s\n", g * "/" * n, tb, tt, fmtratio(ratio(tt, tb)), ab, at,
            fmtratio(ratio(at, ab)))
    end
end

# ---------- 3. overhead of the common interface ----------
function interface_fit(est, steps)
    g = haskey(est, "Interface") ? est["Interface"] : return nothing
    res = Dict{String,Tuple{Float64,Float64}}()
    for iface in ("native", "common")
        ks = filter(k -> startswith(k, iface * "/") && get(steps, ("Interface", k), 0) > 0, collect(keys(g)))
        length(ks) >= 2 || return nothing
        ns = [Float64(steps[("Interface", k)]) for k in ks]
        ts = [time(g[k]) / 1e3 for k in ks]                  # μs
        b = cov(ns, ts) / var(ns)
        res[iface] = (mean(ts) - b * mean(ns), b)            # (a [μs], b [μs/step])
    end
    return res
end
fits = [("base", interface_fit(base, steps_base)), ("target", interface_fit(target, steps_target))]
if any(f -> f[2] !== nothing, fits)
    println("\nCommon interface vs. native (Kepler, parsed): time = a + b * steps\n")
    @printf("%-8s %-10s %14s %14s\n", "run", "interface", "a (μs)", "b (μs/step)")
    for (name, f) in fits
        f === nothing && continue
        for iface in ("native", "common")
            @printf("%-8s %-10s %14.1f %14.3f\n", name, iface, f[iface]...)
        end
        @printf("%-8s %-10s %14.2f %14.2f   (ratios common/native)\n", name, "", f["common"][1] / f["native"][1],
            f["common"][2] / f["native"][2])
    end
end