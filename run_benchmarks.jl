using PkgBenchmark, Dates

const PB = PkgBenchmark
const directorypath = dirname(@__FILE__)

# ==========
# Run PkgBenchmark `judge`
# ==========
compara = judge("TaylorIntegration", "test/benchmarking", "test/v0.8.12")
# PB.benchmarkgroup(compara)

# ==========
# Save results
# ==========
hoy = string(today())
comparapath = joinpath(directorypath, "benchmark", hoy * "_comparacion.md")
export_markdown(comparapath, compara, export_invariants=true)
#
comparapath = joinpath(directorypath, "benchmark", hoy * "_resultadosTarg.json")
PB.writeresults(comparapath, PB.target_result(compara))
#
comparapath = joinpath(directorypath, "benchmark", hoy * "_resultadosBase.json")
PB.writeresults(comparapath, PB.baseline_result(compara))
