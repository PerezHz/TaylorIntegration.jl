# Benchmarks

`benchmarks.jl` is the entry point for a `BenchmarkTools` `SUITE`; the groups are defined in the
`*_benchmarks.jl` files and the problems in `problems.jl`:

| group | what it measures |
|---|---|
| `Kepler` | parsed; dense; `trange`; in-place (isolates the integration loop) |
| `Scalar` | parsed, not parsed and no macro (the only group with the not parsed path: cheap) |
| `Pendulum` (root finding), `Lyapunov`, `JetTransport` | the other integrators (parsed) |
| `Overhead` | per-call costs: cache construction, `_determine_parsing!`, `_exact_step` |
| `Interface` | native vs. common interface (needs `OrdinaryDiffEqCore` and `SciMLBase`) |

Besides times, the suite records the number of steps of each benchmark, so that the comparison reports
time and allocations *per step*, and the fixed cost per call and the cost per step of the common
interface. Time is a soft signal (noisy); allocation counts are deterministic.

## Environment

`benchmark/Project.toml` lists the dependencies, `BenchmarkTools` and the packages needed to load the extension for
the common interface. `TaylorIntegration` must be the checkout being benchmarked, not a registered release:
the first time (and in each checkout)

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
```

(`benchmarks.jl` checks it and stops with a message otherwise). Use the environment with `--project=benchmark`.
It works with any Julia version; to compare versions with `juliaup`: `julia +lts --project=benchmark
benchmark/run_local.jl lts`, `julia +1 --project=benchmark benchmark/run_local.jl one`, and
`compare.jl lts one`; use a separate manifest per version if `Pkg` complains.

## Locally benchmarking uncommitted changes

```bash
# 1. Baseline: stash the changes in src/ and ext/ (the benchmark/ files stay)
git stash push -- src ext
julia --project=benchmark benchmark/run_local.jl baseline
git stash pop

# 2. Current working tree
julia --project=benchmark benchmark/run_local.jl current

# 3. Compare (ratios are target/base; < 1 means that `target` is better)
julia --project=benchmark benchmark/compare.jl baseline current
```

Optionally restrict to some groups: `run_local.jl current Kepler Interface`. Results are saved in
`benchmark/results/`, which should not to be committed.

## Accuracy

`accuracy/accuracy.jl` compares the accuracy of the step-size conventions against `BigFloat` references
(pendulum, Kepler, forced Duffing, jet transport); see the header of the file. `accuracy/tolerance_calibration.jl`
calibrates the tolerances used in the tests that compare the native and the common interface.

## Continuous integration

`.github/workflows/benchmark.yml` runs on the pull requests that touch `src/`, `ext/` or `benchmark/`, but
it is not triggered by pushes to the default branch; it can be run manually from the Actions tab, selecting
a branch, and optionally a Julia version; the default is the latest stable release running on `ubuntu-latest`.
It benchmarks the base and the head of the pull request on the same runner, with the benchmark
suite of the pull request, and posts `compare.jl`'s report as a comment and in the job summary.

A commit whose message contains `[skip bench]` does not trigger the benchmarks, just as
`[skip ci]` skips *all* the workflows, including the tests.