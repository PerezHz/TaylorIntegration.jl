using BenchmarkTools, TaylorIntegration

const directorypath = dirname(@__FILE__)

const BT = BenchmarkTools

SUITE = BenchmarkGroup()

# ==========
#  Include files to benchmarks with parameters
# ==========
#
BT.DEFAULT_PARAMETERS.samples = 25
# BT.DEFAULT_PARAMETERS.seconds = 12

include("kepler_benchmarks.jl")
include("manyspin_benchmarks.jl")
include("Lyap_benchmark.jl")

# ToDo:
# pendulum
# Lyapunov spectrum
# jet transport
# common interface
# Poincare maps
# @taylorize with @threads

# ==========
# Tune
# ==========
println("Load `tune.json`")
paramspath = joinpath(directorypath, "tune.json")
if isfile(paramspath)
    loadparams!(SUITE, BT.load(paramspath)[1], :evals, :samples)
else
    tune!(SUITE)
    BT.save(paramspath, params(SUITE))
end
