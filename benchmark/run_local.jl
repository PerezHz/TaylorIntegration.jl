# This file is part of the TaylorIntegration.jl package; MIT licensed

# Run the benchmark suite on the *current working tree* (no commit needed) and save the results in
# `benchmark/results/<label>.json`, and the number of steps of each benchmark in
# `benchmark/results/<label>.steps.tsv`.
#
#   julia --project=. benchmark/run_local.jl <label> [Group1 Group2 ...]
#
# Groups: Kepler, Scalar, Pendulum, Lyapunov, JetTransport, Overhead, Interface (default: all).
# `BenchmarkTools` must be available (e.g. installed in the default environment).

using BenchmarkTools

label = isempty(ARGS) ? "local" : ARGS[1]
groups = ARGS[2:end]

include(joinpath(@__DIR__, "benchmarks.jl"))

suite = SUITE
if !isempty(groups)
    suite = BenchmarkGroup()
    for g in groups
        suite[g] = SUITE[g]
    end
end

println("Warm-up (compilation)...")
warmup(suite; verbose = false)

println("Running benchmarks: \"$label\"")
results = run(suite; verbose = true)

resdir = joinpath(@__DIR__, "results")
mkpath(resdir)
path = joinpath(resdir, label * ".json")
BenchmarkTools.save(path, results)
open(joinpath(resdir, label * ".steps.tsv"), "w") do io
    for (g, d) in STEPS, (name, s) in d
        haskey(suite, g) && println(io, g, '\t', name, '\t', s)
    end
end
println("\nResults saved in ", path)