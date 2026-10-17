# This file is part of the TaylorIntegration.jl package; MIT licensed

using BenchmarkTools, TaylorIntegration

const BT = BenchmarkTools
const TI = TaylorIntegration

# The package under test must be the checkout that contains this directory, not a registered release
# (the benchmark environment has `TaylorIntegration` as a dependency: use `Pkg.develop(path = ".")`)
let root = realpath(joinpath(@__DIR__, ".."))
    loaded = realpath(pkgdir(TaylorIntegration))
    loaded == root || error("TaylorIntegration is loaded from $loaded, not from $root: " *
        "in the benchmark environment, run `using Pkg; Pkg.develop(path = \"$root\")`")
end
println("Julia ", VERSION, ", TaylorIntegration ", pkgversion(TaylorIntegration))

# Each benchmark is a full (ms-scale) integration: one evaluation per sample, no `tune!`
BT.DEFAULT_PARAMETERS.samples = 15
BT.DEFAULT_PARAMETERS.seconds = 30
BT.DEFAULT_PARAMETERS.evals = 1

include(joinpath(@__DIR__, "problems.jl"))

SUITE = BenchmarkGroup()

# Number of steps of the integration performed by each benchmark: `STEPS[group][name]`.
# `run_local.jl` saves it next to the results, and `compare.jl` uses it to report the time and the
# allocations *per step* (which stay comparable if a change alters the number of steps).
const STEPS = Dict{String,Dict{String,Int}}()

# Register a benchmark (and, optionally, its number of steps)
function add!(group::String, name::String, b; steps::Int = 0)
    haskey(SUITE, group) || (SUITE[group] = BenchmarkGroup())
    SUITE[group][name] = b
    steps > 0 && (get!(STEPS, group, Dict{String,Int}())[name] = steps)
    return b
end

nsteps(sol) = length(sol.t) - 1

for file in ("kepler", "scalar", "pendulum", "lyapunov", "jettransport", "overhead", "interface")
    include(joinpath(@__DIR__, file * "_benchmarks.jl"))
end