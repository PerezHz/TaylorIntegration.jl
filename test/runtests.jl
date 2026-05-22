# This file is part of the TaylorIntegration.jl package; MIT licensed

using Logging

macro test_no_logs(args...)
    isempty(args) && error("@test_no_logs requires an expression to evaluate")
    expr = args[end]
    return quote
        Logging.with_logger(Logging.NullLogger()) do
            $(esc(expr))
        end
    end
end

testfiles = (
    "solution.jl",
    "one_ode.jl",
    "many_ode.jl",
    "complex.jl",
    "jettransport.jl",
    "lyapunov.jl",
    "bigfloats.jl",
    "common.jl",
    "rootfinding.jl",
    "taylorize.jl",
    "interval.jl",
    "aqua.jl",
)

for file in testfiles
    include(file)
end
