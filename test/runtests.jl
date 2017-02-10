# This file is part of the TaylorIntegration.jl package; MIT licensed

include("../src/TaylorIntegration.jl")
using TaylorIntegration
using FactCheck
# FactCheck.setstyle(:compact)

const _order = 28
const _abs_tol = 1.0E-20

facts("Tests: dot{x}=x^2, x(0) = 1") do
    eqs_mov(t, x) = x^2
    t0 = 0.0
    x0 = 1.0
    x0T = TaylorSeries.Taylor1(x0, _order)
    TaylorIntegration.jetcoeffs!(eqs_mov, t0, x0T)
    @fact x0T.coeffs[end] --> 1.0
    δt = _abs_tol^inv(_order-1)
    @fact TaylorIntegration.stepsize(x0T, _abs_tol) --> δt

    tv, xv = taylorinteg(eqs_mov, x0, 0.0, 1.0, _order, _abs_tol)
    @fact length(tv) --> 501
    @fact length(xv) --> 501
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

facts("Tests: dot{x}=x^2, x(0) = 3; nsteps <= maxsteps") do
    eqs_mov(t, x) = x.^2 #the ODE (i.e., the equations of motion)
    exactsol(t, x0) = x0/(1.0-x0*t) #the analytical solution
    t0 = 0.0
    tmax = 0.3
    x0 = 3.0
    q0 = [3.0, 3.0]
    x0T = TaylorSeries.Taylor1(x0, _order)
    TaylorIntegration.jetcoeffs!(eqs_mov, t0, x0T)
    @fact x0T.coeffs[end] --> 3.0^(_order+1)
    δt = (_abs_tol/x0T.coeffs[end-1])^inv(_order-1)
    @fact TaylorIntegration.stepsize(x0T, _abs_tol) --> δt

    tv, xv = taylorinteg(eqs_mov, x0, 0.0, tmax, _order, _abs_tol)
    @fact length(tv) < 501 --> true
    @fact length(xv) < 501 --> true
    @fact length(tv) --> 14
    @fact length(xv) --> 14
    @fact xv[1] --> x0
    @fact tv[end] < 1/3 --> true
    @fact tv[end] --> tmax
    @fact abs(xv[end]-exactsol(tv[end], xv[1])) < 2e-14 --> true

    tv, xv = taylorinteg(eqs_mov, q0, 0.0, tmax, _order, _abs_tol)
    @fact length(tv) < 501 --> true
    @fact length(xv[:,1]) < 501 --> true
    @fact length(xv[:,2]) < 501 --> true
    @fact length(tv) --> 14
    @fact length(xv[:,1]) --> 14
    @fact length(xv[:,2]) --> 14
    @fact tv[end] < 1/3 --> true
    @fact tv[end] --> tmax
    @fact xv[end,1] --> xv[end,2]
    @fact abs(xv[end,1]-exactsol(tv[end], xv[1,1])) < 2e-14 --> true
    @fact abs(xv[end,2]-exactsol(tv[end], xv[1,2])) < 2e-14 --> true

    tmax = 0.33

    tv, xv = taylorinteg(eqs_mov, x0, 0.0, tmax, _order, _abs_tol)
    @fact length(tv) < 501 --> true
    @fact length(xv) < 501 --> true
    @fact length(tv) --> 28
    @fact length(xv) --> 28
    @fact xv[1] --> x0
    @fact tv[end] < 1/3 --> true
    @fact tv[end] --> tmax
    @fact abs(xv[end]-exactsol(tv[end], xv[1])) < 5e-12 --> true

    tv, xv = taylorinteg(eqs_mov, q0, 0.0, tmax, _order, _abs_tol)
    @fact length(tv) < 501 --> true
    @fact length(xv[:,1]) < 501 --> true
    @fact length(xv[:,2]) < 501 --> true
    @fact length(tv) --> 28
    @fact length(xv[:,1]) --> 28
    @fact length(xv[:,2]) --> 28
    @fact tv[end] < 1/3 --> true
    @fact tv[end] --> tmax
    @fact xv[end,1] --> xv[end,2]
    @fact abs(xv[end,1]-exactsol(tv[end], xv[1,1])) < 5e-12 --> true
    @fact abs(xv[end,2]-exactsol(tv[end], xv[1,2])) < 5e-12 --> true
end

facts("Tests: dot{x}=x.^2, x(0) = [3.0,1.0]") do
    eqs_mov(t, x) = x.^2
    exactsol(t, x0) = x0/(1.0-x0*t)
    t0 = 0.0
    q0 = [3.0, 1.0]
    q0T = [TaylorSeries.Taylor1(q0[1], _order), TaylorSeries.Taylor1(q0[2], _order)]
    TaylorIntegration.jetcoeffs!(eqs_mov, t0, q0T)
    @fact q0T[1].coeffs[end] --> 3.0^(_order+1)
    @fact q0T[2].coeffs[end] --> 1.0
    δt = (_abs_tol/q0T[1].coeffs[end-1])^inv(_order-1)
    @fact TaylorIntegration.stepsize(q0T, _abs_tol) --> δt

    tv, xv = taylorinteg(eqs_mov, q0, 0.0, 0.5, _order, _abs_tol)
    @fact length(tv) --> 501
    @fact tv[end] < 1/3 --> true

    trange = 0.0:1/8:1.0
    xv = taylorinteg(eqs_mov, q0, trange, _order, _abs_tol)
    @fact size(xv) --> (9,2)
    @fact q0 --> [3.0, 1.0]
    @fact typeof(xv) --> Array{eltype(q0),2}
    @fact (isnan(xv[4,1]) && isnan(xv[4,2])) --> true
    @fact (isnan(xv[end,1]) && isnan(xv[end,2])) --> true
    @fact abs(xv[3,2] - 4/3) ≤ eps(4/3) --> true
    @fact abs(xv[2,1] - 4.8) ≤ eps(4.8) --> true
end

facts("Test non-autonomous ODE: dot{x}=cos(t)") do
    f(t, x) = [one(x[1]), cos(x[1])]
    t0 = 0//1
    tmax = 10.25*(2pi)
    abstol = 1e-20
    order = 25
    x0 = [t0, 0.0] #initial conditions such that x(t)=sin(t)
    tT, xT = taylorinteg(f, x0, t0, tmax, order, abstol)
    @fact length(tT) < 501 --> true
    @fact length(xT[:,1]) < 501 --> true
    @fact length(xT[:,2]) < 501 --> true
    @fact tT[1] == t0 --> true
    @fact xT[1,1] == x0[1] --> true
    @fact xT[1,2] == x0[2] --> true
    @fact tT[end] == xT[end,1] --> true
    @fact abs(sin(tmax)-xT[end,2]) < 1e-14 --> true

    tmax = 15*(2pi)
    tT, xT = taylorinteg(f, x0, t0, tmax, order, abstol)
    @fact length(tT) < 501 --> true
    @fact length(xT[:,1]) < 501 --> true
    @fact length(xT[:,2]) < 501 --> true
    @fact tT[1] == t0 --> true
    @fact xT[1,1] == x0[1] --> true
    @fact xT[1,2] == x0[2] --> true
    @fact tT[end] == xT[end,1] --> true
    @fact abs(sin(tmax)-xT[end,2]) < 1e-14 --> true
end

facts("Test integration of ODE with complex dependent variables") do
    z0 = complex(0.0, 1.0)
    tr = 0.0:pi/8:2pi

    eqs1(t,z) = -z
    zsol1 = taylorinteg(eqs1, z0, tr, 28, 1.0e-20)
    @fact zsol1[1] == z0 --> true
    @fact isapprox( zsol1[2]  , z0*exp(-tr[2]) ) --> true
    @fact isapprox( zsol1[6], z0*exp(-tr[6]) ) --> true
    @fact isapprox( zsol1[end], z0*exp(-tr[end]) ) --> true
    tt, zsol1 = taylorinteg(eqs1, z0, 0.0, 2pi, 28, 1.0e-20)
    @fact zsol1[1] == z0 --> true
    @fact isapprox( zsol1[2]  , z0*exp(-tt[2]) ) --> true
    @fact isapprox( zsol1[end], z0*exp(-tt[end]) ) --> true

    eqs2(t,z) = im*z
    zsol2 = taylorinteg(eqs2, z0, tr, 28, 1.0e-20)
    @fact zsol2[1] == z0 --> true
    @fact isapprox( zsol2[3], z0*exp( complex(0.0, tr[3])) ) --> true
    @fact isapprox( zsol2[5], z0*exp( complex(0.0, tr[5])) ) --> true
    tt, zsol2 = taylorinteg(eqs2, z0, 0.0, 2pi, 28, 1.0e-20)
    @fact zsol2[1] == z0 --> true
    @fact isapprox( zsol2[2], z0*exp( complex(0.0, tt[2])) ) --> true
    @fact isapprox( zsol2[end], z0*exp( complex(0.0, tt[end])) ) --> true

    eqs3(t,z) = [ eqs1(t,z[1]), eqs2(t,z[2]) ]
    zsol3 = taylorinteg(eqs3, [z0, z0], tr, 28, 1.0e-20)
    @fact isapprox( zsol3[4,1], z0*exp(-tr[4]) ) --> true
    @fact isapprox( zsol3[7,1], z0*exp(-tr[7]) ) --> true
    @fact isapprox( zsol3[4,2], z0*exp( complex(0.0, tr[4])) ) --> true
    @fact isapprox( zsol3[7,2], z0*exp( complex(0.0, tr[7])) ) --> true
    tt, zsol3 = taylorinteg(eqs3, [z0,z0], 0.0, 2pi, 28, 1.0e-20)
    @fact zsol3[1,1] == z0 --> true
    @fact zsol3[1,2] == z0 --> true
    @fact isapprox( zsol3[2,1], z0*exp( -tt[2]) ) --> true
    @fact isapprox( zsol3[end,1], z0*exp( -tt[end]) ) --> true
    @fact isapprox( zsol3[2,2], z0*exp( complex(0.0, tt[2])) ) --> true
    @fact isapprox( zsol3[end,2], z0*exp( complex(0.0, tt[end])) ) --> true

end

exitstatus()
