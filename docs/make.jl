using Documenter
using TaylorIntegration

makedocs(
    modules  = [TaylorIntegration],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"),
    clean = false,
    sitename = "TaylorIntegration.jl",
    authors  = "Jorge A. Pérez-Hernández and Luis Benet",
    pages    = [
        "Home" => "index.md",
        "Background" => [
            "taylor_method.md",
            "lyapunov_spectrum.md",
            "jet_transport.md"
            ],
        "Examples" => [
            "simple_example.md",
            "kepler.md",
            "lorenz_lyapunov.md",
            "pendulum.md",
            "root_finding.md",
            "common.md"
            ],
        "Optimizing: `@taylorize`" => "taylorize.md",
        "API"  => "api.md",
        ]
)

deploydocs(
    repo   = "github.com/PerezHz/TaylorIntegration.jl.git",
    target = "build",
    # julia  = "1.0",
    # osname = "linux",
    deps   = nothing,
    make   = nothing
)
