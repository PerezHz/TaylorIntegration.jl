using Documenter
using TaylorIntegration

makedocs(
    modules  = [TaylorIntegration],
    format   = :html,
    sitename = "TaylorIntegration.jl",
    pages    = [
        "Home" => "index.md",
        "Background" => "taylor_method.md",
        "Tutorial" => ["example1.md",
            "kepler.md"],
        "API"  => "api.md",
    ]
)

deploydocs(
    repo   = "github.com/PerezHz/TaylorIntegration.jl.git",
    target = "build",
    julia  = "0.6",
    osname = "linux",
    deps   = nothing,
    make   = nothing
)
