using Documenter
using TaylorIntegration

makedocs(
    modules  = [TaylorIntegration],
    format   = :html,
    sitename = "TaylorIntegration.jl",
    pages    = [
        "Home" => "index.md",
        "Taylor's method" => "taylor_method.md",
        "Examples" => [ "Example 1" => "example1.md"],
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
