# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorSeries
include("../src/TaylorIntegration.jl")
using TaylorIntegration
using FastAnonymous
using FactCheck

const _order = 28
const _abs_tol = 1.0E-20

facts("Tests: dot{x}=x^2, x(0) = 1") do
    eqs_mov(t, x) = x^2
    eqs_mov_anon = @anon (t, x) -> eqs_mov(t, x)

    t0 = 0.0
    x0 = 1.0
    x0T = Taylor1(x0, _order)
    jetcoeffs!(eqs_mov, t0, x0T)
    @fact x0T.coeffs[end] == 1.0 --> true
    δt = _abs_tol^inv(_order-1)
    @fact stepsize(x0T, _abs_tol) == δt --> true
end

facts("Tests: dot{x}=x.^2, x(0) = 3") do
    eqs_mov(t, x) = x.^2
    eqs_mov_anon = @anon (t, x) -> eqs_mov(t, x)

    t0 = 0.0
    q0 = [3.0]
    q0T = Array{Taylor1{Float64},1}(1)
    q0T[1] = Taylor1(q0[1], _order)
    jetcoeffs!(eqs_mov, t0, q0T)
    @fact q0T[1].coeffs[end] == 3.0^(_order+1) --> true
    δt = (_abs_tol/q0T[1].coeffs[end-1])^inv(_order-1)
    @fact stepsize(q0T, _abs_tol) == δt --> true
    @fact evaluate(q0T, δt)[1] - 3.0/(1.0-3.0*δt) < eps(1.0) --> true
end

exitstatus()
