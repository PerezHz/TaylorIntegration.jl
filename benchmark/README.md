# Benchmarks

The simplest way to run the benchmarks is to use the command-line
and, being at the root-directory of the package, run the command
```bash
$ julia --project=. run_benchmarks.jl
```

The results are stored in three files in the `benchmark/` directory:
`yyyy-mm-dd_comparacion.md` contains a markdown file comparing the
target and base versions, and further information of the parameters
used for the benchmarks. The actual benchmarks are stored in
`yyyy-mm-dd_resultadosBase.json` and `yyyy-mm-dd_resultadosTarg.json`,
for the base and target branches.
