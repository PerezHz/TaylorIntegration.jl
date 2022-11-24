using PkgBenchmark, Dates

const PB = PkgBenchmark
const directorypath = dirname(@__FILE__)

# Adjust the following line and commit before running the benchmarks.
# Adapted from: https://stackoverflow.com/questions/41088615/get-branch-name-of-an-installed-package-directly-from-the-julia-package-manager
const target_branch = PB.LibGit2.shortname(
    PB.LibGit2.head(PB.LibGit2.GitRepo(directorypath * "/..")))
const base_branch = "master"

# ==========
# Run PkgBenchmark `judge`
# ==========
compara = judge("TaylorIntegration", target_branch, base_branch)
# PB.benchmarkgroup(compara)

# ==========
# Save results
# ==========
hoy = string(today())
comparapath = joinpath(directorypath, hoy * "_comparacion.md")
export_markdown(comparapath, compara, export_invariants=true)
#
comparapath = joinpath(directorypath, hoy * "_resultadosTarg.json")
PB.writeresults(comparapath, PB.target_result(compara))
#
comparapath = joinpath(directorypath, hoy * "_resultadosBase.json")
PB.writeresults(comparapath, PB.baseline_result(compara))
