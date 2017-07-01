# This file is part of the TaylorIntegration.jl package; MIT licensed

include("../src/TaylorIntegration.jl")
using TaylorSeries, TaylorIntegration, FactCheck
# FactCheck.setstyle(:compact)

const _order = 28
const _abstol = 1.0E-20
const vT = zeros(_order+1)
vT[2] = 1.0

facts("Tests: dot{x}=x^2, x(0) = 1") do
    eqs_mov(t, x) = x^2
    t0 = 0.0
    x0 = 1.0
    x0T = Taylor1(x0, _order)
    vT[1] = t0
    TaylorIntegration.jetcoeffs!(eqs_mov, t0, x0T, vT)
    @fact x0T.coeffs[end] --> 1.0
    δt = _abstol^inv(_order-1)
    @fact TaylorIntegration.stepsize(x0T, _abstol) --> δt

    tv, xv = taylorinteg(eqs_mov, x0, 0.0, 1.0, _order, _abstol)
    @fact length(tv) --> 501
    @fact length(xv) --> 501
    @fact xv[1] --> x0
    @fact tv[end] < 1.0 --> true

    trange = 0.0:1/8:1.0
    xv = taylorinteg(eqs_mov, x0, trange, _order, _abstol)
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
    x0T = Taylor1(x0, _order)
    vT[1] = t0
    TaylorIntegration.jetcoeffs!(eqs_mov, t0, x0T, vT)
    @fact x0T.coeffs[end] --> 3.0^(_order+1)
    δt = (_abstol/x0T.coeffs[end-1])^inv(_order-1)
    @fact TaylorIntegration.stepsize(x0T, _abstol) --> δt

    tv, xv = taylorinteg(eqs_mov, x0, 0.0, tmax, _order, _abstol)
    @fact length(tv) < 501 --> true
    @fact length(xv) < 501 --> true
    @fact length(tv) --> 14
    @fact length(xv) --> 14
    @fact xv[1] --> x0
    @fact tv[end] < 1/3 --> true
    @fact tv[end] --> tmax
    @fact abs(xv[end]-exactsol(tv[end], xv[1])) < 2e-14 --> true

    function eqs_mov!(t, x, Dx)
        for i in eachindex(x)
            Dx[i] = x[i]^2
        end
        nothing
    end

    tv, xv = taylorinteg(eqs_mov!, q0, 0.0, tmax, _order, _abstol)
    @fact length(tv) < 501 --> true
    @fact length(xv[:,1]) < 501 --> true
    @fact length(xv[:,2]) < 501 --> true
    @fact length(tv) --> 14
    @fact length(xv[:,1]) --> 14
    @fact length(xv[:,2]) --> 14
    @fact xv[1,1:end] --> q0
    @fact tv[end] < 1/3 --> true
    @fact tv[end] --> tmax
    @fact xv[end,1] --> xv[end,2]
    @fact abs(xv[end,1]-exactsol(tv[end], xv[1,1])) < 2e-14 --> true
    @fact abs(xv[end,2]-exactsol(tv[end], xv[1,2])) < 2e-14 --> true

    tmax = 0.33

    tv, xv = taylorinteg(eqs_mov, x0, 0.0, tmax, _order, _abstol)
    @fact length(tv) < 501 --> true
    @fact length(xv) < 501 --> true
    @fact length(tv) --> 28
    @fact length(xv) --> 28
    @fact xv[1] --> x0
    @fact tv[end] < 1/3 --> true
    @fact tv[end] --> tmax
    @fact abs(xv[end]-exactsol(tv[end], xv[1])) < 5e-12 --> true

    tv, xv = taylorinteg(eqs_mov!, q0, 0.0, tmax, _order, _abstol)
    @fact length(tv) < 501 --> true
    @fact length(xv[:,1]) < 501 --> true
    @fact length(xv[:,2]) < 501 --> true
    @fact length(tv) --> 28
    @fact length(xv[:,1]) --> 28
    @fact length(xv[:,2]) --> 28
    @fact xv[1,1:end] --> q0
    @fact tv[end] < 1/3 --> true
    @fact tv[end] --> tmax
    @fact xv[end,1] --> xv[end,2]
    @fact abs(xv[end,1]-exactsol(tv[end], xv[1,1])) < 5e-12 --> true
    @fact abs(xv[end,2]-exactsol(tv[end], xv[1,2])) < 5e-12 --> true
end

facts("Tests: dot{x}=x.^2, x(0) = [3.0,1.0]") do
    function eqs_mov!(t, x, Dx)
        for i in eachindex(x)
            Dx[i] = x[i]^2
        end
        nothing
    end
    exactsol(t, x0) = x0/(1.0-x0*t)
    t0 = 0.0
    q0 = [3.0, 1.0]
    q0T = [Taylor1(q0[1], _order), Taylor1(q0[2], _order)]
    xdotT = Array{Taylor1{Float64}}(length(q0))
    xaux = Array{Taylor1{Float64}}(length(q0))
    vT[1] = t0
    TaylorIntegration.jetcoeffs!(eqs_mov!, t0, q0T, xdotT, xaux, vT)
    @fact q0T[1].coeffs[end] --> 3.0^(_order+1)
    @fact q0T[2].coeffs[end] --> 1.0
    δt = (_abstol/q0T[1].coeffs[end-1])^inv(_order-1)
    @fact TaylorIntegration.stepsize(q0T, _abstol) --> δt

    tv, xv = taylorinteg(eqs_mov!, q0, 0.0, 0.5, _order, _abstol)
    @fact length(tv) --> 501
    @fact xv[1,:] --> q0
    @fact tv[end] < 1/3 --> true

    trange = 0.0:1/8:1.0
    xv = taylorinteg(eqs_mov!, q0, trange, _order, _abstol)
    @fact size(xv) --> (9,2)
    @fact q0 --> [3.0, 1.0]
    @fact typeof(xv) --> Array{eltype(q0),2}
    @fact xv[1,1:end] --> q0
    @fact (isnan(xv[4,1]) && isnan(xv[4,2])) --> true
    @fact (isnan(xv[end,1]) && isnan(xv[end,2])) --> true
    @fact abs(xv[3,2] - 4/3) ≤ eps(4/3) --> true
    @fact abs(xv[2,1] - 4.8) ≤ eps(4.8) --> true
end

facts("Test non-autonomous ODE (1): dot{x}=cos(t)") do
    function f!(t, x, Dx)
        Dx[1] = one(x[1])
        Dx[2] = cos(x[1])
        nothing
    end
    t0 = 0//1
    tmax = 10.25*(2pi)
    abstol = 1e-20
    order = 25
    x0 = [t0, 0.0] #initial conditions such that x(t)=sin(t)
    tT, xT = taylorinteg(f!, x0, t0, tmax, order, abstol)
    @fact length(tT) < 501 --> true
    @fact length(xT[:,1]) < 501 --> true
    @fact length(xT[:,2]) < 501 --> true
    @fact xT[1,1:end] --> x0
    @fact tT[1] == t0 --> true
    @fact xT[1,1] == x0[1] --> true
    @fact xT[1,2] == x0[2] --> true
    @fact tT[end] == xT[end,1] --> true
    @fact abs(sin(tmax)-xT[end,2]) < 1e-14 --> true

    tmax = 15*(2pi)
    tT, xT = taylorinteg(f!, x0, t0, tmax, order, abstol)
    @fact length(tT) < 501 --> true
    @fact length(xT[:,1]) < 501 --> true
    @fact length(xT[:,2]) < 501 --> true
    @fact xT[1,1:end] --> x0
    @fact tT[1] == t0 --> true
    @fact xT[1,1] == x0[1] --> true
    @fact xT[1,2] == x0[2] --> true
    @fact tT[end] == xT[end,1] --> true
    @fact abs(sin(tmax)-xT[end,2]) < 1e-14 --> true
end

facts("Test non-autonomous ODE (2): dot{x}=cos(t)") do
    f!(t, x) = cos(t)
    t0 = 0//1
    tmax = 10.25*(2pi)
    abstol = 1e-20
    order = 25
    x0 = 0.0 #initial conditions such that x(t)=sin(t)
    tT, xT = taylorinteg(f!, x0, t0, tmax, order, abstol)
    @fact length(tT) < 501 --> true
    @fact length(xT) < 501 --> true
    # @fact length(xT[:,2]) < 501 --> true
    @fact xT[1] --> x0
    @fact tT[1] == t0 --> true
    @fact abs(sin(tmax)-xT[end]) < 1e-14 --> true

    tmax = 15*(2pi)
    tT, xT = taylorinteg(f!, x0, t0, tmax, order, abstol)
    @fact length(tT) < 501 --> true
    @fact length(xT) < 501 --> true
    @fact xT[1] --> x0
    @fact abs(sin(tmax)-xT[end]) < 1e-14 --> true
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

    function eqs3!(t, z, Dz)
        Dz[1] = eqs1(t,z[1])
        Dz[2] = eqs2(t,z[2])
        nothing
    end
    zsol3 = taylorinteg(eqs3!, [z0, z0], tr, 28, 1.0e-20)
    @fact isapprox( zsol3[4,1], z0*exp(-tr[4]) ) --> true
    @fact isapprox( zsol3[7,1], z0*exp(-tr[7]) ) --> true
    @fact isapprox( zsol3[4,2], z0*exp( complex(0.0, tr[4])) ) --> true
    @fact isapprox( zsol3[7,2], z0*exp( complex(0.0, tr[7])) ) --> true
    tt, zsol3 = taylorinteg(eqs3!, [z0,z0], 0.0, 2pi, 28, 1.0e-20)
    @fact zsol3[1,1] == z0 --> true
    @fact zsol3[1,2] == z0 --> true
    @fact isapprox( zsol3[2,1], z0*exp( -tt[2]) ) --> true
    @fact isapprox( zsol3[end,1], z0*exp( -tt[end]) ) --> true
    @fact isapprox( zsol3[2,2], z0*exp( complex(0.0, tt[2])) ) --> true
    @fact isapprox( zsol3[end,2], z0*exp( complex(0.0, tt[end])) ) --> true

end

facts("Test jet transport (t0, tmax): 1-dim case") do
    f(t, x) = x^2
    p = set_variables("ξ", numvars=1, order=5)
    x0 = 3.0 #"nominal" initial condition
    x0TN = x0 + p[1] #jet transport initial condition
    t0=0.0
    tmax=0.3
    tvTN, xvTN = taylorinteg(f, x0TN, t0, tmax, _order, _abstol, maxsteps=500)
    tv, xv = taylorinteg(f, x0, t0, tmax, _order, _abstol, maxsteps=500)
    exactsol(t, x0, t0) = x0./(1.0-x0.*(t-t0)) #the analytical solution
    δsol = exactsol(tvTN[end], x0TN, t0)-xvTN[end]
    δcoeffs = map(y->y[1], map(x->x.coeffs, δsol.coeffs))

    @fact isapprox(δcoeffs, zeros(6), atol=1e-10, rtol=0) --> true
    @fact isapprox(x0, evaluate(xvTN[1])) --> true
    @fact isapprox(xv[1], evaluate(xvTN[1])) --> true
    @fact isapprox(xv[end], evaluate(xvTN[end])) --> true

    g(t, x) = 0.3x
    y0 = 1.0 #"nominal" initial condition
    y0TN = y0 + p[1] #jet transport initial condition
    uvTN, yvTN = taylorinteg(g, y0TN, t0, 10/0.3, _order, _abstol, maxsteps=500);
    uv, yv = taylorinteg(g, y0, t0, 10/0.3, _order, _abstol, maxsteps=500);
    exactsol_g(u, y0, u0) = y0*exp.(0.3(u-u0))
    exactsol_g(uv, y0, t0)
    δsol_g = exactsol_g(uvTN[end], y0TN, t0)-yvTN[end]
    δcoeffs_g = map(y->y[1], map(x->x.coeffs, δsol_g.coeffs))

    @fact isapprox(δcoeffs_g, zeros(6), atol=1e-10, rtol=0) --> true
end

facts("Test jet transport (trange): 1-dim case") do
    f(t, x) = x^2
    p = set_variables("ξ", numvars=1, order=5)
    x0 = 3.0 #"nominal" initial condition
    x0TN = x0 + p[1] #jet transport initial condition
    tv = 0.0:0.05:0.33
    xvTN = taylorinteg(f, x0TN, tv, _order, _abstol, maxsteps=500)
    xv = taylorinteg(f, x0, tv, _order, _abstol, maxsteps=500)
    exactsol(t, x0, t0) = x0./(1.0-x0.*(t-t0)) #the analytical solution
    δsol = exactsol(tv[end], x0TN, tv[1])-xvTN[end]
    δcoeffs = map(y->y[1], map(x->x.coeffs, δsol.coeffs))

    @fact length(tv) == length(xvTN) --> true
    @fact isapprox(δcoeffs, zeros(6), atol=1e-10, rtol=0) --> true

    g(t, x) = 0.3x
    y0 = 1.0 #"nominal" initial condition
    y0TN = y0 + p[1] #jet transport initial condition
    tv = 0.0:1/0.3:10/0.3
    yvTN = taylorinteg(g, y0TN, tv, _order, _abstol, maxsteps=500);
    yv = taylorinteg(g, y0, tv, _order, _abstol, maxsteps=500);
    exactsol_g(u, y0, u0) = y0*exp.(0.3(u-u0))
    δsol_g = exactsol_g(tv[end], y0TN, tv[1])-yvTN[end]
    δcoeffs_g = map(y->y[1], map(x->x.coeffs, δsol_g.coeffs))

    @fact length(tv) == length(yvTN) --> true
    @fact isapprox(δcoeffs_g, zeros(6), atol=1e-10, rtol=0) --> true
end

facts("Test jet transport (t0,tmax): harmonic oscillator") do
    function harmosc!(t, x, dx) #the harmonic oscillator ODE
        dx[1] = x[2]
        dx[2] = -x[1]
        nothing
    end
    p = set_variables("ξ", numvars=2, order=5)
    x0 = [-1.0,0.45]
    x0TN = x0 + p
    tvTN, xvTN = taylorinteg(harmosc!, x0TN, 0.0, 10pi, _order, _abstol, maxsteps=1)
    @fact length(tvTN) <= 2 --> true
    @fact length(xvTN[:,1]) <= 2 --> true
    @fact length(xvTN[:,2]) <= 2 --> true
    tvTN, xvTN = taylorinteg(harmosc!, x0TN, 0.0, 10pi, _order, _abstol, maxsteps=500)
    tv  , xv   = taylorinteg(harmosc!, x0  , 0.0, 10pi, _order, _abstol, maxsteps=500)
    x_analyticsol(t,x0,p0) = p0*sin(t)+x0*cos(t)
    p_analyticsol(t,x0,p0) = p0*cos(t)-x0*sin(t)
    x_δsol = x_analyticsol(tvTN[end], x0TN[1], x0TN[2])-xvTN[end,1]
    x_δcoeffs = map(y->y[1], map(x->x.coeffs, x_δsol.coeffs))
    p_δsol = p_analyticsol(tvTN[end], x0TN[1], x0TN[2])-xvTN[end,2]
    p_δcoeffs = map(y->y[1], map(x->x.coeffs, p_δsol.coeffs))

    @fact (length(tvTN), length(x0)) == size(xvTN) --> true
    @fact isapprox(x_δcoeffs, zeros(6), atol=1e-10, rtol=0) --> true
    @fact isapprox(x0, map( x->evaluate(x), xvTN[1,:])) --> true
    @fact isapprox(xv[1,:], map( x->evaluate(x), xvTN[1,:])) --> true # nominal solution must coincide with jet evaluated at ξ=(0,0) at initial time
    @fact isapprox(xv[end,:], map( x->evaluate(x), xvTN[end,:])) --> true #nominal solution must coincide with jet evaluated at ξ=(0,0) at final time

    xvTN_0 = map( x->evaluate(x), xvTN ) # the jet evaluated at the nominal solution
    @fact isapprox(xv[end,:], xvTN_0[end,:]) --> true # nominal solution must coincide with jet evaluated at ξ=(0,0) at final time
    @fact isapprox(xvTN_0[1,:], xvTN_0[end,:]) --> true # end point must coincide with a full period
end

facts("Test jet transport (trange): simple pendulum") do

    function pendulum!(t, x, dx) #the simple pendulum ODE
        dx[1] = x[2]
        dx[2] = -sin(x[1])
        nothing
    end

    varorder = 2 #the order of the variational expansion
    p = set_variables("ξ", numvars=2, order=varorder) #TaylorN steup
    q0 = [1.3, 0.0] #the initial conditions
    q0TN = q0 + p #parametrization of a small neighbourhood around the initial conditions
    # T is the librational period == 4Elliptic.K(sin(q0[1]/2)^2) # this is an explicit value that will be used until Elliptic.K works with julia 0.6
    T = 7.019250311844546
    t0 = 0.0 #the initial time
    tmax = T #the final time
    integstep = 0.25*T #the time interval between successive evaluations of the solution vector

    #the time range
    tr = t0:integstep:tmax;
    #xv is the solution vector representing the propagation of the initial condition q0 propagated until time T
    xv = taylorinteg(pendulum!, q0, tr, _order, _abstol, maxsteps=100)
    #xvTN is the solution vector representing the propagation of the initial condition q0 plus variations (ξ₁,ξ₂) propagated until time T
    #note that q0 is a Vector{Float64}, but q0TN is a Vector{TaylorN{Float64}}
    #but apart from that difference, we're calling `taylorinteg` essentially with the same parameters!
    #thus, jet transport is reduced to a beautiful application of Julia's multiple dispatch!
    xvTN = taylorinteg(pendulum!, q0TN, tr, _order, _abstol, maxsteps=1)
    @fact size(xvTN) == (length(tr), length(q0)) --> true
    @fact prod(isnan.(xvTN[2:end,:])) --> true
    xvTN = taylorinteg(pendulum!, q0TN, tr, _order, _abstol, maxsteps=100)

    xvTN_0 = map( x->evaluate(x, [0.0, 0.0]), xvTN ) # the jet evaluated at the nominal solution

    @fact isapprox(xvTN_0[1,:], xvTN_0[end,:]) --> true #end point must coincide with a full period
    @fact isapprox(xv, xvTN_0) --> true #nominal solution must coincide with jet evaluated at ξ=(0,0)

    #testing another jet transport method:
    tv, xv = taylorinteg(pendulum!, q0, t0, tmax, _order, _abstol, maxsteps=100)
    @fact isapprox(xv[1,:], xvTN_0[1,:]) --> true #nominal solution must coincide with jet evaluated at ξ=(0,0) at initial time
    @fact isapprox(xv[end,:], xvTN_0[end,:]) --> true #nominal solution must coincide with jet evaluated at ξ=(0,0) at final time

    # a small displacement
    disp = 0.0001 #works even with 0.001, but we're giving it some margin

    #compare the jet solution evaluated at various variation vectors ξ, wrt to full solution, at each time evaluation point
    srand(14908675)
    for i in 1:10
        # generate a random angle
        ϕ = 2pi*rand()
        # generate a random point on a circumference of radius disp
        ξ = disp*[cos(ϕ), sin(ϕ)]
        #propagate in time full solution with initial condition q0+ξ
        xv_disp = taylorinteg(pendulum!, q0+ξ, tr, _order, _abstol, maxsteps=100)
        #evaluate jet at q0+ξ
        xvTN_disp = map( x->evaluate(x, ξ), xvTN )
        #the propagated exact solution at q0+ξ should be approximately equal to the propagated jet solution evaluated at the same point:
        @fact isapprox( xvTN_disp, xv_disp ) --> true
    end

end

facts("Test Lyapunov spectrum integrator (t0, tmax): Lorenz system") do

    x0 = [19.0, 20.0, 50.0] #the initial condition
    t0 = 0.0 #the initial time
    tmax = t0+20.0 #final time of integration

    #Lorenz system ODE:
    function lorenz!(t, x, dx)
        dx[1] = σ*(x[2]-x[1])
        dx[2] = x[1]*(ρ-x[3])-x[2]
        dx[3] = x[1]*x[2]-β*x[3]
        nothing
    end

    #Lorenz system parameters
    σ = 16.0
    β = 4.0
    ρ = 45.92

    #Calculate trace of Lorenz system Jacobian via TaylorSeries.jacobian:
    xi = set_variables("δ", order=1, numvars=length(x0))
    x0TN = [ x0[1]+xi[1], x0[2]+xi[2], x0[3]+xi[3] ]
    dx0TN = similar(x0TN)
    lorenz!(t0, x0TN, dx0TN)
    lorenztr = trace(jacobian(dx0TN)) #trace of Lorenz system Jacobian matrix

    @fact lorenztr == -(σ+one(Float64)+β) --> true

    tv, xv, λv = liap_taylorinteg(lorenz!, x0, t0, tmax, 28, _abstol; maxsteps=2)

    @fact size(tv) == (3,) --> true
    @fact size(xv) == (3,3) --> true
    @fact size(λv) == (3,3) --> true

    tv, xv, λv = liap_taylorinteg(lorenz!, x0, t0, tmax, 28, _abstol; maxsteps=2000)

    @fact xv[1,:] == x0 --> true
    @fact tv[1] == t0 --> true
    @fact length(tv[:,1]) < 2001 --> true
    @fact length(xv[:,1]) < 2001 --> true
    @fact length(xv[:,2]) < 2001 --> true
    @fact length(xv[:,3]) < 2001 --> true
    @fact length(λv[:,1]) < 2001 --> true
    @fact length(λv[:,2]) < 2001 --> true
    @fact length(λv[:,3]) < 2001 --> true
    @fact size(xv) == size(λv) --> true
    @fact isapprox(sum(λv[1,:]), lorenztr) --> false
    @fact isapprox(sum(λv[end,:]), lorenztr) --> true
    mytol = 1e-4
    @fact isapprox(λv[end,1], 1.47167, rtol=mytol, atol=mytol) -->true
    @fact isapprox(λv[end,2], -0.00830, rtol=mytol, atol=mytol) -->true
    @fact isapprox(λv[end,3], -22.46336, rtol=mytol, atol=mytol) -->true

end

facts("Test Lyapunov spectrum integrator (trange): Lorenz system") do

    x0 = [19.0, 20.0, 50.0] #the initial condition
    t0 = 0.0 #the initial time
    tmax = t0+20.0 #final time of integration

    #Lorenz system ODE:
    function lorenz!(t, x, dx)
        dx[1] = σ*(x[2]-x[1])
        dx[2] = x[1]*(ρ-x[3])-x[2]
        dx[3] = x[1]*x[2]-β*x[3]
        nothing
    end

    #Lorenz system parameters
    σ = 16.0
    β = 4.0
    ρ = 45.92

    #Calculate trace of Lorenz system Jacobian via TaylorSeries.jacobian:
    xi = set_variables("δ", order=1, numvars=length(x0))
    x0TN = [ x0[1]+xi[1], x0[2]+xi[2], x0[3]+xi[3] ]
    dx0TN = similar(x0TN)
    lorenz!(t0, x0TN, dx0TN)
    lorenztr = trace(jacobian(dx0TN)) #trace of Lorenz system Jacobian matrix

    @fact lorenztr == -(σ+one(eltype(x0))+β) --> true

    xw, λw = liap_taylorinteg(lorenz!, x0, t0:1.0:tmax, _order, _abstol; maxsteps=2)

    @fact size(xw) == (length(t0:1.0:tmax), length(x0)) --> true
    @fact size(λw) == (length(t0:1.0:tmax), length(x0)) --> true
    @fact prod(isnan.(xw[2:end,:])) --> true

    xw, λw = liap_taylorinteg(lorenz!, x0, t0:1.0:tmax, _order, _abstol; maxsteps=2000)

    @fact xw[1,:] == x0 --> true
    @fact size(xw) == size(λw) --> true
    @fact isapprox(sum(λw[1,:]), lorenztr) --> false
    @fact isapprox(sum(λw[end,:]), lorenztr) --> true
    mytol = 1e-4
    @fact isapprox(λw[end,1], 1.47167, rtol=mytol, atol=mytol) -->true
    @fact isapprox(λw[end,2], -0.00830, rtol=mytol, atol=mytol) -->true
    @fact isapprox(λw[end,3], -22.46336, rtol=mytol, atol=mytol) -->true

end

facts("Test ODE integration with BigFloats: simple pendulum") do

    function pendulum!(t, x, dx) #the simple pendulum ODE
        dx[1] = x[2]
        dx[2] = -sin(x[1])
        nothing
    end

    q0 = [1.3, 0.0] #the initial conditions
    # T is the pendulum's librational period == 4Elliptic.K(sin(q0[1]/2)^2)
    # we will evaluate the elliptic integral K using TaylorIntegration.jl:
    g(t,x) = (1-((sin(q0[1]/2))^2)*(sin(t)^2))^(-0.5) # K elliptic integral kernel
    tvk, xvk = taylorinteg(g, 0.0, 0.0, BigFloat(π)/2, 25, 1e-20)
    println("eltype(tvk) = ", eltype(tvk))
    println("eltype(xvk) = ", eltype(xvk))
    @fact eltype(tvk) == BigFloat --> true
    @fact eltype(xvk) == BigFloat --> true
    T = 4xvk[end]
    @fact typeof(T) == BigFloat --> true
    @fact abs(T-7.019250311844546) < eps(10.0) --> true

    t0 = 0.0 #the initial time
    tmax = T #the final time
    @fact typeof(tmax) == BigFloat --> true

    tv, xv = taylorinteg(pendulum!, q0, t0, tmax, _order, _abstol; maxsteps=1)
    @fact eltype(tv) == BigFloat --> true
    @fact eltype(xv) == BigFloat --> true
    @fact length(tv) == 2 --> true
    @fact length(xv[:,1]) == 2 --> true
    @fact length(xv[:,2]) == 2 --> true

    #note that tmax is a BigFloat
    tv, xv = taylorinteg(pendulum!, q0, t0, tmax, _order, _abstol)
    @fact length(tv) < 501 --> true
    @fact length(xv[:,1]) < 501 --> true
    @fact length(xv[:,2]) < 501 --> true
    @fact norm(xv[end,:]-q0,Inf) < eps(10.0) --> true

end

exitstatus()
