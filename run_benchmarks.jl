using PkgBenchmark

const PB = PkgBenchmark

# ==========
# Run PkgBenchmark `judge`
# ==========
compara = judge("TaylorIntegration", "test/benchmarking", "test/v0.8.12")
# PB.benchmarkgroup(compara)

# ==========
# Save results
# ==========
comparapath = joinpath(dirname(@__FILE__), "benchmark/comparacion.md")
export_markdown(comparapath, compara, export_invariants=true)
#
comparapath = joinpath(dirname(@__FILE__), "benchmark/resultadosT.json")
PB.writeresults(comparapath, PB.target_result(compara))
#
comparapath = joinpath(dirname(@__FILE__), "benchmark/resultadosB.json")
PB.writeresults(comparapath, PB.baseline_result(compara))
