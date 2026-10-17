# This file is part of the TaylorIntegration.jl package; MIT licensed

# Calibrate the tolerances of the three `<` comparisons of test/common.jl that now fail
# (Kepler problem of "Test consistency with taylorinteg": order 25, abstol 1e-20,
# tspan = (0, 10π); jets of order 2 in 4 variables).
#
# For several times it reports the difference between `taylorinteg` and the common
# interface, and, as a yardstick for the intrinsic sensitivity of the problem, the
# difference between `taylorinteg` and itself when the initial condition is perturbed
# by ONE ulp. If the first is comparable to the second, the interfaces differ by what
# the problem amplifies from rounding-level perturbations.
#
#   julia --project=. benchmark/tolerance_calibration.jl

using TaylorIntegration, OrdinaryDiffEqCore, Printf
using SciMLBase: ODEProblem, solve
using LinearAlgebra: norm

@taylorize function kepler1!(dq, q, p, t)
    local μ = -1.0
    r_p2 = q[1]^2 + q[2]^2
    r_p3d2 = r_p2^1.5
    dq[1] = q[3]
    dq[2] = q[4]
    newtonianCoeff = μ / r_p3d2
    dq[3] = q[1] * newtonianCoeff
    dq[4] = q[2] * newtonianCoeff
    return nothing
end

const order, abstol, tspan = 25, 1e-20, (0.0, 10pi)
q0 = [0.2, 0.0, 0.0, 3.0]
q0pert = copy(q0)
q0pert[1] = nextfloat(q0[1])         # one ulp
times = tspan[2] .* (0.25, 0.5, 0.75, 1.0)

println("== Nominal trajectory (Float64)")
solp = taylorinteg(kepler1!, q0, tspan[1], tspan[2], order, abstol; dense = true, maxsteps = 2000)
solpp = taylorinteg(kepler1!, q0pert, tspan[1], tspan[2], order, abstol; dense = true, maxsteps = 2000)
sol1 = solve(ODEProblem(kepler1!, q0, tspan), TaylorMethod(order), abstol = abstol,
    maxiters = 2000, parse_eqs = false)
@printf("   %10s  %22s  %22s\n", "t", "|taylorinteg - interface|", "|taylorinteg - 1ulp-perturbed|")
for t in times
    t = min(t, tspan[2])
    @printf("   %10.4f  %22.3e  %22.3e\n", t, norm(solp(t) - sol1(t), Inf), norm(solp(t) - solpp(t), Inf))
end

println("\n== Jets (TaylorN, order 2, 4 variables)")
p = variables!("ξ", numvars = 4, order = 2, nowarn = true)
q0TN = q0 + p
q0TNpert = q0pert + p
solTN = taylorinteg(kepler1!, q0TN, tspan[1], tspan[2], order, abstol; dense = true, maxsteps = 2000)
solTNp = taylorinteg(kepler1!, q0TNpert, tspan[1], tspan[2], order, abstol; dense = true, maxsteps = 2000)
sol2TN = solve(ODEProblem(kepler1!, q0TN, tspan), TaylorMethod(order), abstol = abstol,
    parse_eqs = true, dense = true, maxiters = 2000)
@printf("   %10s  %12s  %12s  %12s  %12s  %12s\n", "t", "|jet|", "|Δ| iface", "rel iface",
    "|Δ| 1ulp", "rel 1ulp")
for t in times
    t = min(t, tspan[2])
    nj = norm(solTN(t), Inf)
    di = norm(solTN(t) - sol2TN(t), Inf)
    dp = norm(solTN(t) - solTNp(t), Inf)
    @printf("   %10.4f  %12.3e  %12.3e  %12.3e  %12.3e  %12.3e\n", t, nj, di, di / nj, dp, dp / nj)
end
println("\n(constant terms) |taylorinteg - interface| at t = tf/2: ",
    norm(solTN(tspan[2] / 2)() - sol2TN(tspan[2] / 2)(), Inf))