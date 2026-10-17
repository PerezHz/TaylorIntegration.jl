# This file is part of the TaylorIntegration.jl package; MIT licensed

# Equations of motion and parameters shared by the benchmark suite
# (`benchmarks.jl`) and by `consistency.jl`.

using TaylorIntegration

# ==========
# Kepler problem (several formulations)
# ==========

# Using `^`; the parameter `p` is μ
@taylorize function kepler1!(dq, q, p, t)
    local μ = p
    r_p3d2 = (q[1]^2 + q[2]^2)^1.5
    dq[1] = q[3]
    dq[2] = q[4]
    dq[3] = μ * q[1] / r_p3d2
    dq[4] = μ * q[2] / r_p3d2
    return nothing
end

# Using `^` and loops; the parameter is `(nn, μ)`
@taylorize function kepler2!(dq, q, p, t)
    nn, μ = p
    r2 = zero(q[1])
    for i = 1:nn
        r2_aux = r2 + q[i]^2
        r2 = r2_aux
    end
    r_p3d2 = r2^(3 / 2)
    for j = 1:nn
        dq[j] = q[nn+j]
        dq[nn+j] = μ * q[j] / r_p3d2
    end
    nothing
end

# Using `sqrt`; μ is hard-coded
@taylorize function kepler5!(dq, q, p, t)
    local μ = -1.0
    r = sqrt(q[1]^2 + q[2]^2)
    r_p3d2 = r^3
    dq[1] = q[3]
    dq[2] = q[4]
    dq[3] = μ * q[1] / r_p3d2
    dq[4] = μ * q[2] / r_p3d2
    return nothing
end

# Using `sqrt` and loops; the parameter `p` is μ
@taylorize function kepler6!(dq, q, p, t)
    local NN = 2
    local μ = p
    r2 = zero(q[1])
    for i = 1:NN
        r2_aux = r2 + q[i]^2
        r2 = r2_aux
    end
    r = sqrt(r2)
    r_p3d2 = r^3
    for j = 1:NN
        dq[j] = q[NN+j]
        dq[NN+j] = μ * q[j] / r_p3d2
    end
    nothing
end

# Same equations as `kepler1!`, but *without* `@taylorize`: no parsed method
# exists, so only the hard-coded `jetcoeffs!` can be used.
function kepler_plain!(dq, q, p, t)
    local μ = p
    r_p3d2 = (q[1]^2 + q[2]^2)^1.5
    dq[1] = q[3]
    dq[2] = q[4]
    dq[3] = μ * q[1] / r_p3d2
    dq[4] = μ * q[2] / r_p3d2
    return nothing
end

const KEPLER_Q0 = [0.2, 0.0, 0.0, 3.0]
const KEPLER_MU = -1.0
const KEPLER_ORDER = 28
const KEPLER_ABSTOL = 1.0e-20
const KEPLER_T0 = 0.0
const KEPLER_TF = 2π * 20       # 20 periods
const KEPLER_MAXSTEPS = 20_000

# `taylorized` tells whether a parsed method of `jetcoeffs!` exists
const KEPLER_PROBLEMS = (
    (name = "kepler1", f = kepler1!, params = KEPLER_MU, taylorized = true),
    (name = "kepler2", f = kepler2!, params = (2, KEPLER_MU), taylorized = true),
    (name = "kepler5", f = kepler5!, params = nothing, taylorized = true),
    (name = "kepler6", f = kepler6!, params = KEPLER_MU, taylorized = true),
    (name = "plain", f = kepler_plain!, params = KEPLER_MU, taylorized = false),
)

# ==========
# Scalar problem: x' = b - x^2
# ==========
const SCALAR_B = 3.0

@taylorize scalar_riccati(x, p, t) = SCALAR_B - x^2
scalar_plain(x, p, t) = SCALAR_B - x^2

const SCALAR_X0 = 1.0
const SCALAR_ORDER = 28
const SCALAR_ABSTOL = 1.0e-20
const SCALAR_T0 = 0.0
const SCALAR_TF = 10.0
const SCALAR_MAXSTEPS = 1_000

# ==========
# Pendulum (also used for root finding and jet transport)
# ==========
@taylorize function bench_pendulum!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
    return nothing
end

# Event: velocity crosses zero
pendulum_event(dx, x, p, t) = (true, x[2])

const PEND_Q0 = [1.3, 0.0]
const PEND_ORDER = 28
const PEND_ABSTOL = 1.0e-20
const PEND_T0 = 0.0
const PEND_TF = 50 * 7.019250311844546   # ~ 50 periods
const PEND_MAXSTEPS = 10_000

# Jet transport
const JT_ORDER = 20
const JT_TF = 100.0
const JT_MAXSTEPS = 10_000
const JT_JETORDER = 3

# ==========
# Lorenz system (Lyapunov spectrum)
# ==========
@taylorize function lorenz!(dx, x, param, t)
    σ, ρ, β = param
    dx[1] = σ * (x[2] - x[1])
    dx[2] = x[1] * (ρ - x[3]) - x[2]
    dx[3] = x[1] * x[2] - β * x[3]
    nothing
end

const LORENZ_Q0 = [19.0, 20.0, 50.0]
const LORENZ_PARS = [16.0, 45.92, 4.0]
const LORENZ_ORDER = 28
const LORENZ_ABSTOL = 1.0e-20
const LORENZ_T0 = 0.0
const LORENZ_TF = 20.0
const LORENZ_MAXSTEPS = 20_000