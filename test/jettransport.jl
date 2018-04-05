# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration, Elliptic
using LinearAlgebra: norm
using Test

const _order = 28
const _abstol = 1.0E-20

f(t, x) = x^2
g(t, x) = 0.3x

@testset "Test Taylor1 jet transport (t0, tmax): 1-dim case" begin
    p = Taylor1([0.0,1.0], 5)
    x0 = 3.0 #"nominal" initial condition
    x0T1 = x0 + p #jet transport initial condition
    t0 = 0.0
    tmax = 0.3
    tvT1, xvT1 = taylorinteg(f, x0T1, t0, tmax, _order, _abstol, maxsteps=1)
    @test size(tvT1) == (2,)
    @test size(xvT1) == (2,)
    tvT1, xvT1 = taylorinteg(f, x0T1, t0, tmax, _order, _abstol)
    tv, xv = taylorinteg(f, x0, t0, tmax, _order, _abstol)
    exactsol(t, x0, t0) = x0/(1.0-x0*(t-t0)) #the analytical solution
    δsol = exactsol(tvT1[end], x0T1, t0)-xvT1[end] #analytical vs jet transport diff at end of integration
    δcoeffs = δsol.coeffs
    @test isapprox(δcoeffs, zeros(6), atol=1e-10, rtol=0)
    @test isapprox(x0, evaluate(xvT1[1]))
    @test isapprox(xv[1], evaluate(xvT1[1]))
    @test isapprox(xv[end], evaluate(xvT1[end]))
    xvT1_0 = evaluate.(xvT1)
    @test norm(exactsol.(tvT1, x0, t0)-xvT1_0, Inf) < 1E-12
    for i in 1:5
        disp = 0.001*rand() #a small, random displacement
        x0_disp = x0+disp
        tv_disp, xv_disp = taylorinteg(f, x0_disp, t0, tmax, _order, _abstol)
        xvT1_disp = evaluate.(xvT1, disp)
        @test norm(exactsol.(tvT1, x0_disp, t0)-xvT1_disp, Inf) < 1E-12 #analytical vs jet transport
        @test norm(x0_disp-evaluate(xvT1[1], disp), Inf) < 1E-12
        @test norm(xv_disp[1]-evaluate(xvT1[1], disp), Inf) < 1E-12
        @test norm(xv_disp[end]-evaluate(xvT1[end], disp), Inf) < 1E-12
    end

    y0 = 1.0 #"nominal" initial condition
    u0 = 0.0 #initial time
    y0T1 = y0 + p #jet transport initial condition
    uvT1, yvT1 = taylorinteg(g, y0T1, t0, 10/0.3, _order, _abstol, maxsteps=1) #warmup lap
    # test maxsteps break
    @test size(uvT1) == (2,)
    @test size(yvT1) == (2,)
    uvT1, yvT1 = taylorinteg(g, y0T1, t0, 10/0.3, _order, _abstol) #Taylor1 jet transport integration
    uv, yv = taylorinteg(g, y0, t0, 10/0.3, _order, _abstol) #reference integration
    exactsol_g(u, y0, u0) = y0*exp(0.3(u-u0))  #the analytical solution
    δsol_g = exactsol_g(uvT1[end], y0T1, t0)-yvT1[end] #analytical vs jet transport diff at end of integration
    δcoeffs_g = δsol_g.coeffs
    @test isapprox(δcoeffs_g, zeros(6), atol=1e-10, rtol=0)
    @test isapprox(y0, evaluate(yvT1[1]), atol=1e-10, rtol=0)
    @test isapprox(yv[1], evaluate(yvT1[1]), atol=1e-10, rtol=0)
    @test isapprox(yv[end], evaluate(yvT1[end]), atol=1e-10, rtol=0)
    yvT1_0 = evaluate.(yvT1)
    @test norm(exactsol_g.(uvT1, y0, u0)-yvT1_0, Inf) < 1E-9
    for i in 1:5
        disp = 0.001*rand() #a small, random displacement
        y0_disp = y0+disp
        uv_disp, yv_disp = taylorinteg(g, y0_disp, u0, 10/0.3, _order, _abstol)
        yvT1_disp = evaluate.(yvT1, disp)
        @test norm(exactsol_g.(uvT1, y0_disp, u0)-yvT1_disp, Inf) < 1E-9 #analytical vs jet transport
        @test norm(y0_disp-evaluate(yvT1[1], disp), Inf) < 1E-9
        @test norm(yv_disp[1]-evaluate(yvT1[1], disp), Inf) < 1E-9
        @test norm(yv_disp[end]-evaluate(yvT1[end], disp), Inf) < 1E-9
    end
end

# @testset "Test TaylorN jet transport (t0, tmax): 1-dim case" begin
#     p = set_variables("ξ", numvars=1, order=5)
#     x0 = 3.0 #"nominal" initial condition
#     x0TN = x0 + p[1] #jet transport initial condition
#     t0=0.0
#     tmax=0.3
#     tvTN, xvTN = taylorinteg(f, x0TN, t0, tmax, _order, _abstol, maxsteps=1)
#     @test size(tvTN) == (2,)
#     @test size(xvTN) == (2,)
#     tvTN, xvTN = taylorinteg(f, x0TN, t0, tmax, _order, _abstol)
#     tv, xv = taylorinteg(f, x0, t0, tmax, _order, _abstol)
#     exactsol(t, x0, t0) = x0/(1.0-x0*(t-t0)) #the analytical solution
#     δsol = exactsol(tvTN[end], x0TN, t0)-xvTN[end]
#     δcoeffs = map(y->y[1], map(x->x.coeffs, δsol.coeffs))
#     @test isapprox(δcoeffs, zeros(6), atol=1e-10, rtol=0)
#     @test isapprox(x0, evaluate(xvTN[1]))
#     @test isapprox(xv[1], evaluate(xvTN[1]))
#     @test isapprox(xv[end], evaluate(xvTN[end]))
#     xvTN_0 = evaluate.(xvTN)
#     @test norm(exactsol.(tvTN, x0, t0)-xvTN_0, Inf) < 1E-12
#     for i in 1:5
#         disp = 0.001*rand() #a small, random displacement
#         x0_disp = x0+disp
#         dv = map(x->[disp], tvTN) #a vector of identical displacements
#         tv_disp, xv_disp = taylorinteg(f, x0_disp, t0, tmax, _order, _abstol)
#         xvTN_disp = evaluate.(xvTN, dv)
#         @test norm(exactsol.(tvTN, x0_disp, t0)-xvTN_disp, Inf) < 1E-12 #analytical vs jet transport
#         @test norm(x0_disp-evaluate(xvTN[1], [disp]), Inf) < 1E-12
#         @test norm(xv_disp[1]-evaluate(xvTN[1], [disp]), Inf) < 1E-12
#         @test norm(xv_disp[end]-evaluate(xvTN[end], [disp]), Inf) < 1E-12
#     end
#
#     y0 = 1.0 #"nominal" initial condition
#     u0 = 0.0 #initial time
#     y0TN = y0 + p[1] #jet transport initial condition
#     uvTN, yvTN = taylorinteg(g, y0TN, u0, 10/0.3, _order, _abstol, maxsteps=1)
#     @test size(uvTN) == (2,)
#     @test size(yvTN) == (2,)
#     uvTN, yvTN = taylorinteg(g, y0TN, u0, 10/0.3, _order, _abstol)
#     uv, yv = taylorinteg(g, y0, u0, 10/0.3, _order, _abstol)
#     exactsol_g(u, y0, u0) = y0*exp(0.3(u-u0))
#     δsol_g = exactsol_g(uvTN[end], y0TN, u0)-yvTN[end]
#     δcoeffs_g = map(y->y[1], map(x->x.coeffs, δsol_g.coeffs))
#     @test isapprox(δcoeffs_g, zeros(6), atol=1e-10, rtol=0)
#     @test isapprox(y0, evaluate(yvTN[1]), atol=1e-10, rtol=0)
#     @test isapprox(yv[1], evaluate(yvTN[1]), atol=1e-10, rtol=0)
#     @test isapprox(yv[end], evaluate(yvTN[end]), atol=1e-10, rtol=0)
#     yvTN_0 = evaluate.(yvTN)
#     @test norm(exactsol_g.(uvTN, y0, u0)-yvTN_0, Inf) < 1E-9
#     for i in 1:5
#         disp = 0.001*rand() #a small, random displacement
#         y0_disp = y0+disp
#         dv = map(x->[disp], uvTN) #a vector of identical displacements
#         uv_disp, yv_disp = taylorinteg(g, y0_disp, u0, 10/0.3, _order, _abstol)
#         yvTN_disp = evaluate.(yvTN, dv)
#         @test norm(exactsol_g.(uvTN, y0_disp, u0)-yvTN_disp, Inf) < 1E-9 #analytical vs jet transport
#         @test norm(y0_disp-yvTN[1]([disp]), Inf) < 1E-9
#         @test norm(yv_disp[1]-yvTN[1]([disp]), Inf) < 1E-9
#         @test norm(yv_disp[end]-yvTN[end]([disp]), Inf) < 1E-9
#     end
# end

@testset "Test Taylor1 jet transport (trange): 1-dim case" begin
    p = Taylor1([0.0,1.0], 5)
    x0 = 3.0 #"nominal" initial condition
    x0T1 = x0 + p #jet transport initial condition
    tv = 0.0:0.05:0.33
    xvT1 = taylorinteg(f, x0T1, tv, _order, _abstol, maxsteps=1)
    @test size(xvT1) == (7,)
    ta = collect(tv)
    xvT1 = taylorinteg(f, x0T1, ta, _order, _abstol, maxsteps=1)
    @test size(xvT1) == (7,)
    xvT1 = taylorinteg(f, x0T1, tv, _order, _abstol)
    xvT11 = taylorinteg(f, x0T1, ta, _order, _abstol)
    @test xvT11 == xvT1
    xv = taylorinteg(f, x0, tv, _order, _abstol)
    exactsol(t, x0, t0) = x0/(1.0-x0*(t-t0)) #the analytical solution
    δsol = exactsol(tv[end], x0T1, tv[1])-xvT1[end]
    δcoeffs = δsol.coeffs
    @test length(tv) == length(xvT1)
    @test isapprox(δcoeffs, zeros(6), atol=1e-10, rtol=0)
    xv_analytical = exactsol.(tv, x0, tv[1])
    xvT1_0 = xvT1.()
    @test isapprox(xv_analytical, xvT1_0, atol=1e-10, rtol=0)
    for i in 1:5
        disp = 0.001*rand() #a small, random displacement
        xv_disp = taylorinteg(f, x0+disp, tv, _order, _abstol)
        xv_disp2 = taylorinteg(f, x0+disp, ta, _order, _abstol)
        @test xv_disp == xv_disp2
        xvT1_disp = xvT1.(disp)
        @test norm(exactsol.(tv, x0+disp, tv[1])-xvT1_disp, Inf) < 1E-12 #analytical vs jet transport
        @test norm(xv_disp-xvT1_disp, Inf) < 1E-12 # integration vs jet transport
    end

    y0 = 1.0 #"nominal" initial condition
    y0T1 = y0 + p #jet transport initial condition
    uv = 0.0:1/0.3:10/0.3
    yvT1 = taylorinteg(g, y0T1, uv, _order, _abstol, maxsteps=1)
    @test size(yvT1) == (11,)
    yvT1 = taylorinteg(g, y0T1, uv, _order, _abstol)
    yv = taylorinteg(g, y0, uv, _order, _abstol)
    exactsol_g(u, y0, u0) = y0*exp(0.3(u-u0))
    δsol_g = exactsol_g(uv[end], y0T1, uv[1])-yvT1[end]
    δcoeffs_g = δsol_g.coeffs
    @test length(uv) == length(yvT1)
    @test isapprox(δcoeffs_g, zeros(6), atol=1e-10, rtol=0)
    yv_analytical = exactsol_g.(uv, y0, uv[1])
    yvT1_0 = evaluate.(yvT1)
    @test isapprox(yv_analytical, yvT1_0, atol=1e-10, rtol=0)
    for i in 1:5
        disp = 0.001*rand() #a small, random displacement
        yv_disp = taylorinteg(g, y0+disp, uv, _order, _abstol)
        yvT1_disp = evaluate.(yvT1, disp)
        @test norm(exactsol_g.(uv, y0+disp, uv[1])-yvT1_disp, Inf) < 1E-9 #analytical vs jet transport
        @test norm(yv_disp-yvT1_disp, Inf) < 1E-9 # integration vs jet transport
    end
end

@testset "Test TaylorN jet transport (trange): 1-dim case" begin
    p = set_variables("ξ", numvars=1, order=5)
    x0 = 3.0 #"nominal" initial condition
    x0TN = x0 + p[1] #jet transport initial condition
    tv = 0.0:0.05:0.33
    xvTN = taylorinteg(f, x0TN, tv, _order, _abstol, maxsteps=1)
    @test size(xvTN) == (7,)
    ta = collect(tv)
    xvTN = taylorinteg(f, x0TN, ta, _order, _abstol, maxsteps=1)
    @test size(xvTN) == (7,)
    xvTN = taylorinteg(f, x0TN, tv, _order, _abstol)
    xvTN_ = taylorinteg(f, x0TN, ta, _order, _abstol)
    @test xvTN == xvTN_
    xv = taylorinteg(f, x0, tv, _order, _abstol)
    exactsol(t, x0, t0) = x0/(1.0-x0*(t-t0)) #the analytical solution
    δsol = exactsol(tv[end], x0TN, tv[1])-xvTN[end]
    δcoeffs = map(y->y[1], map(x->x.coeffs, δsol.coeffs))
    @test length(tv) == length(xvTN)
    @test isapprox(δcoeffs, zeros(6), atol=1e-10, rtol=0)
    xv_analytical = exactsol.(tv, x0, tv[1])
    xvTN_0 = evaluate.(xvTN)
    @test isapprox(xv_analytical, xvTN_0, atol=1e-10, rtol=0)
    for i in 1:5
        disp = 0.001*rand() #a small, random displacement
        dv = map(x->[disp], tv) #a vector of identical displacements
        xv_disp = taylorinteg(f, x0+disp, tv, _order, _abstol)
        xv_disp_ = taylorinteg(f, x0+disp, ta, _order, _abstol)
        @test xv_disp == xv_disp_
        xvTN_disp = evaluate.(xvTN, dv)
        @test norm(exactsol.(tv, x0+disp, tv[1])-xvTN_disp, Inf) < 1E-12 #analytical vs jet transport
        @test norm(xv_disp-xvTN_disp, Inf) < 1E-12 # integration vs jet transport
    end

    y0 = 1.0 #"nominal" initial condition
    y0TN = y0 + p[1] #jet transport initial condition
    uv = 0.0:1/0.3:10/0.3
    yvTN = taylorinteg(g, y0TN, uv, _order, _abstol, maxsteps=1)
    @test size(yvTN) == (11,)
    yvTN = taylorinteg(g, y0TN, uv, _order, _abstol)
    yv = taylorinteg(g, y0, uv, _order, _abstol)
    exactsol_g(u, y0, u0) = y0*exp(0.3(u-u0))
    δsol_g = exactsol_g(uv[end], y0TN, uv[1])-yvTN[end]
    δcoeffs_g = map(y->y[1], map(x->x.coeffs, δsol_g.coeffs))
    @test length(uv) == length(yvTN)
    @test isapprox(δcoeffs_g, zeros(6), atol=1e-10, rtol=0)
    yv_analytical = exactsol_g.(uv, y0, uv[1])
    yvTN_0 = evaluate.(yvTN)
    @test isapprox(yv_analytical, yvTN_0, atol=1e-10, rtol=0)
    for i in 1:5
        disp = 0.001*rand() #a small, random displacement
        dv = map(x->[disp], uv) #a vector of identical displacements
        yv_disp = taylorinteg(g, y0+disp, uv, _order, _abstol)
        yvTN_disp = evaluate.(yvTN, dv)
        @test norm(exactsol_g.(uv, y0+disp, uv[1])-yvTN_disp, Inf) < 1E-9 #analytical vs jet transport
        @test norm(yv_disp-yvTN_disp, Inf) < 1E-9 # integration vs jet transport
    end
end

function harmosc!(t, x, dx) #the harmonic oscillator ODE
    dx[1] = x[2]
    dx[2] = -x[1]*x[3]^2
    dx[3] = zero(x[1])
    nothing
end

@testset "Test Taylor1 jet transport (t0,tmax): harmonic oscillator" begin
    t = Taylor1([0.0, 1.0], 10)
    ω0 = 1.0
    x0 = [0.0,ω0,ω0]
    x0T1 = x0+[0t,t,t]
    tv1, xv1 = taylorinteg(harmosc!, x0T1, 0.0, 100pi, _order, _abstol, maxsteps=1)
    @test size(tv1) == (2,)
    @test size(xv1) == (2, 3)
    tv1, xv1 = taylorinteg(harmosc!, x0T1, 0.0, 100pi, _order, _abstol, maxsteps=2000)
    y0 = evaluate.(xv1)
    x1(t,δω) = sin((ω0+δω)*t)
    x2(t,δω) = (ω0+δω)*cos((ω0+δω)*t)
    @test norm(y0[:,1]-x1.(tv1,0.0), Inf) < 1E-11
    @test norm(y0[:,2]-x2.(tv1,0.0), Inf) < 1E-11
    for i in 1:5
        δω=0.001*rand()
        x0_disp = x0+[0.0,δω,δω]
        tv, xv = taylorinteg(harmosc!, x0_disp, 0.0, 100pi, _order, _abstol, maxsteps=2000)
        y1 = evaluate.(xv1, δω)
        @test norm(y1[:,1]-x1.(tv1,δω),Inf) < 1E-11
        @test norm(y1[:,2]-x2.(tv1,δω),Inf) < 1E-11
        @test norm(x0_disp-evaluate.(xv1[1,:], δω), Inf) < 1E-11
        @test norm(xv[1,:]-evaluate.(xv1[1,:], δω), Inf) < 1E-11
        @test norm(xv[end,:]-evaluate.(xv1[end,:], δω), Inf) < 1E-11
    end
end

@testset "Test Taylor1 jet transport (trange): harmonic oscillator" begin
    t = Taylor1([0.0, 1.0], 10)
    ω0 = 1.0
    x0 = [0.0,ω0,ω0]
    x0T1 = x0+[0t,t,t]
    tv = 0.0:0.25*(2pi):100pi
    xv1 = taylorinteg(harmosc!, x0T1, tv, _order, _abstol, maxsteps=1)
    @test length(tv) == 201
    @test size(xv1) == (201, 3)
    ta = collect(tv)
    xv1 = taylorinteg(harmosc!, x0T1, ta, _order, _abstol, maxsteps=1)
    @test length(ta) == 201
    @test size(xv1) == (201, 3)
    xv1 = taylorinteg(harmosc!, x0T1, tv, _order, _abstol, maxsteps=2000)
    xv1_ = taylorinteg(harmosc!, x0T1, ta, _order, _abstol, maxsteps=2000)
    @test xv1 == xv1_
    y0 = evaluate.(xv1)
    x1(t,δω) = sin((ω0+δω)*t)
    x2(t,δω) = (ω0+δω)*cos((ω0+δω)*t)
    @test norm(y0[:,1]-x1.(tv,0.0), Inf) < 1E-11
    @test norm(y0[:,2]-x2.(tv,0.0), Inf) < 1E-11
    for i in 1:5
        δω=0.001*rand()
        x0_disp = x0+[0.0,δω,δω]
        xv = taylorinteg(harmosc!, x0_disp, tv, _order, _abstol, maxsteps=2000)
        xv_ = taylorinteg(harmosc!, x0_disp, ta, _order, _abstol, maxsteps=2000)
        @test xv == xv_
        y1 = evaluate.(xv1, δω)
        @test norm(y1[:,1]-x1.(tv,δω), Inf) < 1E-11
        @test norm(y1[:,2]-x2.(tv,δω), Inf) < 1E-11
        @test norm(y1-xv, Inf) < 1E-11
    end
end

function harmosc!(t, x, dx) #the harmonic oscillator ODE
    dx[1] = x[2]
    dx[2] = -x[1]
    nothing
end

@testset "Test TaylorN jet transport (t0,tmax): harmonic oscillator" begin
    p = set_variables("ξ", numvars=2, order=5)
    x0 = [-1.0,0.45]
    x0TN = x0 + p
    tvTN, xvTN = taylorinteg(harmosc!, x0TN, 0.0, 10pi, _order, _abstol, maxsteps=1)
    @test length(tvTN) == 2
    @test size(xvTN) == (length(tvTN), length(x0TN))
    tvTN, xvTN = taylorinteg(harmosc!, x0TN, 0.0, 10pi, _order, _abstol)
    tv , xv = taylorinteg(harmosc!, x0  , 0.0, 10pi, _order, _abstol)
    x_analyticsol(t,x0,p0) = p0*sin(t)+x0*cos(t)
    p_analyticsol(t,x0,p0) = p0*cos(t)-x0*sin(t)
    x_δsol = x_analyticsol(tvTN[end], x0TN[1], x0TN[2])-xvTN[end,1]
    x_δcoeffs = map(y->y[1], map(x->x.coeffs, x_δsol.coeffs))
    p_δsol = p_analyticsol(tvTN[end], x0TN[1], x0TN[2])-xvTN[end,2]
    p_δcoeffs = map(y->y[1], map(x->x.coeffs, p_δsol.coeffs))

    @test (length(tvTN), length(x0)) == size(xvTN)
    @test isapprox(x_δcoeffs, zeros(6), atol=1e-10, rtol=0)
    @test isapprox(x0, map( x->evaluate(x), xvTN[1,:]))
    @test isapprox(xv[1,:], map( x->evaluate(x), xvTN[1,:])) # nominal solution must coincide with jet evaluated at ξ=(0,0) at initial time
    @test isapprox(xv[end,:], map( x->evaluate(x), xvTN[end,:])) #nominal solution must coincide with jet evaluated at ξ=(0,0) at final time

    xvTN_0 = map( x->evaluate(x), xvTN ) # the jet evaluated at the nominal solution
    @test isapprox(xv[end,:], xvTN_0[end,:]) # nominal solution must coincide with jet evaluated at ξ=(0,0) at final time
    @test isapprox(xvTN_0[1,:], xvTN_0[end,:]) # end point must coincide with a full period
end

function pendulum!(t, x, dx) #the simple pendulum ODE
    dx[1] = x[2]
    dx[2] = -sin(x[1])
    nothing
end

@testset "Test TaylorN jet transport (trange): simple pendulum" begin
    varorder = 2 #the order of the variational expansion
    p = set_variables("ξ", numvars=2, order=varorder) #TaylorN steup
    q0 = [1.3, 0.0] #the initial conditions
    q0TN = q0 + p #parametrization of a small neighbourhood around the initial conditions
    # T is the librational period == 4Elliptic.K(sin(q0[1]/2)^2)
    T = 4Elliptic.K(sin(q0[1]/2)^2) # equals 7.019250311844546
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
    @test size(xvTN) == (5,2)
    xvTN = taylorinteg(pendulum!, q0TN, tr, _order, _abstol, maxsteps=100)

    xvTN_0 = map( x->evaluate(x, [0.0, 0.0]), xvTN ) # the jet evaluated at the nominal solution

    @test isapprox(xvTN_0[1,:], xvTN_0[end,:]) #end point must coincide with a full period
    @test isapprox(xv, xvTN_0) #nominal solution must coincide with jet evaluated at ξ=(0,0)

    #testing another jet transport method:
    tv, xv = taylorinteg(pendulum!, q0, t0, tmax, _order, _abstol, maxsteps=100)
    @test isapprox(xv[1,:], xvTN_0[1,:]) #nominal solution must coincide with jet evaluated at ξ=(0,0) at initial time
    @test isapprox(xv[end,:], xvTN_0[end,:]) #nominal solution must coincide with jet evaluated at ξ=(0,0) at final time

    # a small displacement
    disp = 0.0001 #works even with 0.001, but we're giving it some margin

    #compare the jet solution evaluated at various variation vectors ξ, wrt to full solution, at each time evaluation point
    xv_disp = taylorinteg(pendulum!, q0+[disp,0.0], tr, _order, _abstol, maxsteps=1)
    @test size(xv_disp) == (5,2)
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
        @test isapprox( xvTN_disp, xv_disp )
    end
end
