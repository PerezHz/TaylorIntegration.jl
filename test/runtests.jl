# This file is part of the TaylorIntegration.jl package; MIT licensed

# using TaylorSeries
include("../src/TaylorIntegration.jl")
using TaylorIntegration
using FactCheck
FactCheck.setstyle(:compact)

const _order = 28
const _abs_tol = 1.0E-20

facts("Tests: dot{x}=x^2, x(0) = 1") do
    eqs_mov(t, x) = x^2
    t0 = 0.0
    x0 = 1.0
    x0T = TaylorSeries.Taylor1(x0, _order)
    jetcoeffs!(eqs_mov, t0, x0T)
    @fact x0T.coeffs[end] --> 1.0
    δt = _abs_tol^inv(_order-1)
    @fact stepsize(x0T, _abs_tol) --> δt

    tv, xv = taylorinteg(eqs_mov, x0, 0.0, 1.0, _order, _abs_tol)
    @fact length(tv) --> 501
    @fact xv[1] --> x0
    @fact tv[end] < 1.0 --> true

    trange = 0.0:1/8:1.0
    xv = taylorinteg(eqs_mov, x0, trange, _order, _abs_tol)
    @fact length(xv) --> length(trange)
    @fact typeof(xv) --> Array{typeof(x0),1}
    @fact xv[1] --> x0
    @fact isnan(xv[end]) --> true
    @fact abs(xv[5] - 2.0) ≤ eps(2.0) --> true
end

facts("Tests: dot{x}=x.^2, x(0) = [3.0,1.0]") do
    eqs_mov(t, x) = x.^2
    exactsol(t, x0) = x0/(1.0-x0*t)
    t0 = 0.0
    q0 = [3.0, 1.0]
    q0T = [TaylorSeries.Taylor1(q0[1], _order), TaylorSeries.Taylor1(q0[2], _order)]
    jetcoeffs!(eqs_mov, t0, q0T)
    @fact q0T[1].coeffs[end] --> 3.0^(_order+1)
    @fact q0T[2].coeffs[end] --> 1.0
    δt = (_abs_tol/q0T[1].coeffs[end-1])^inv(_order-1)
    @fact stepsize(q0T, _abs_tol) --> δt
    @fact evaluate(q0T, δt)[1] - 3.0/(1.0-3.0*δt) < eps(1.0) --> true

    tv, xv = taylorinteg(eqs_mov, q0, 0.0, 0.5, _order, _abs_tol)
    @fact length(tv) --> 501
    @fact xv[1] --> q0
    @fact tv[end] < 1/3 --> true

    trange = 0.0:1/8:1.0
    xv = taylorinteg(eqs_mov, q0, trange, _order, _abs_tol)
    @fact size(xv) --> (9,)
    @fact q0 --> [3.0, 1.0]
    @fact length(xv) --> length(trange)
    @fact typeof(xv) --> Array{typeof(q0),1}
    @fact xv[1] --> [3.0, 1.0]
    @fact (isnan(xv[4][1]) && isnan(xv[4][2])) --> true
    @fact (isnan(xv[end][1]) && isnan(xv[end][2])) --> true
    @fact abs(xv[3][2] - 4/3) ≤ eps(4/3) --> true
    @fact abs(xv[2][1] - 4.8) ≤ eps(4.8) --> true
end

exitstatus()
