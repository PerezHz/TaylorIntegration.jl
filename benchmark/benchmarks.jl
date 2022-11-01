using BenchmarkTools, TaylorIntegration
using Dates

const directorypath = dirname(@__FILE__)

const BT = BenchmarkTools

# ==========
#  Parameters
# ==========
BT.DEFAULT_PARAMETERS.samples = 25
# BT.DEFAULT_PARAMETERS.seconds = 12

SUITE = BenchmarkGroup()

# ==========
# Include files to benchmark
# ==========
include("kepler_benchmarks.jl")

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
paramspath = joinpath(directorypath, "tune.json")
if isfile(paramspath)
    loadparams!(SUITE, BT.load(paramspath)[1], :evals, :samples);
else
    tune!(SUITE)
    BT.save(paramspath, params(SUITE));
end

