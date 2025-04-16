# ==========
# Multiple pendula
# ==========

# ==========
# Define equations of motion
# ==========
@taylorize function multpendula1!(dx, x, p, t)
    for i in p[2]
        dx[i] = x[p[1]+i]
        dx[i+p[1]] = -sin(x[i])
    end
    return nothing
end

# ==========
# Constants
# ==========
const _abstol = 1.0e-20
const _order = 20
const pars = (3, 1:3)
const t0 = 0.0
const tf1 = 1000.0
const tf2 = 10_000.0
const maxsteps = 10_000
const q0 = [pi - 0.001, 0.0, pi - 0.001, 0.0, pi - 0.001, 0.0]

# ==========
# Run benchmarks
# ==========
SUITE["Pendumum"] = BenchmarkGroup()

SUITE["Pendumum"]["pendulum1-1"] = @benchmarkable taylorinteg(
    multpendula1!,
    $q0,
    $t0,
    $tf1,
    $_order,
    $_abstol,
    $pars,
    maxsteps = $maxsteps,
)
SUITE["Pendumum"]["pendulum1-2"] = @benchmarkable taylorinteg(
    multpendula1!,
    $q0,
    $t0,
    $tf2,
    $_order,
    $_abstol,
    $pars,
    maxsteps = $maxsteps,
)

SUITE["Pendumum"]["pendulum1-1"] = @benchmarkable taylorinteg(
    multpendula1!,
    $q0,
    $t0,
    $tf1,
    $_order,
    $_abstol,
    $pars,
    maxsteps = $maxsteps,
)
SUITE["Pendumum"]["pendulum1-2"] = @benchmarkable taylorinteg(
    multpendula1!,
    $q0,
    $t0,
    $tf2,
    $_order,
    $_abstol,
    $pars,
    maxsteps = $maxsteps,
)
