# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorSeries, TaylorIntegration
using Base.Test

const _order = 28
const _abstol = 1.0E-20

function pendulum!(t, x, dx)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
    nothing
end

g(t, x, dx) = x[2]

@testset "Test intersection of surfaces and root-finding: simple pendulum" begin
    const t0 = 0.0
    const x0 = [1.3, 0.0]
    const T = 7.019250311844546

    p = set_variables("ξ", numvars=length(x0), order=2)
    x0N = x0 + p
    ξ = Taylor1(2)
    x01 = x0 + ξ

    #warm-up lap and preliminary tests
    tv, xv, tvS, xvS, gvS = taylorinteg(pendulum!, g, x0, t0, T, _order, _abstol, maxsteps=1)
    @test size(tv) == (2,)
    @test tv[1] == t0
    @test size(xv) == (2,2)
    @test xv[1,:] == x0
    @test size(tvS) == (0,)
    @test size(xvS) == (0,2)
    @test size(gvS) == (0,)
    tvN, xvN, tvSN, xvSN, gvSN = taylorinteg(pendulum!, g, x0N, t0, 3T, _order, _abstol, maxsteps=1)
    @test eltype(tvN) == Float64
    @test eltype(xvN) == TaylorN{Float64}
    @test eltype(tvSN) == TaylorN{Float64}
    @test eltype(xvSN) == TaylorN{Float64}
    @test eltype(gvSN) == TaylorN{Float64}
    @test size(tvN) == (2,)
    @test tvN[1] == t0
    @test size(xvN) == (2,2)
    @test xvN[1,:] == x0N
    @test size(tvSN) == (0,)
    @test size(xvSN) == (0,2)
    @test size(gvSN) == (0,)
    tv1, xv1, tvS1, xvS1, gvS1 = taylorinteg(pendulum!, g, x01, t0, 3T, _order, _abstol, maxsteps=1)
    @test eltype(tv1) == Float64
    @test eltype(xv1) == Taylor1{Float64}
    @test eltype(tvS1) == Taylor1{Float64}
    @test eltype(xvS1) == Taylor1{Float64}
    @test eltype(gvS1) == Taylor1{Float64}
    @test size(tv1) == (2,)
    @test tv1[1] == t0
    @test size(xv1) == (2,2)
    @test xv1[1,:] == x01
    @test size(tvS1) == (0,)
    @test size(xvS1) == (0,2)
    @test size(gvS1) == (0,)

    #testing 0-th order root-finding
    tv, xv, tvS, xvS, gvS = taylorinteg(pendulum!, g, x0, t0, 3T, _order, _abstol, maxsteps=1000)
    @test tv[1] == t0
    @test xv[1,:] == x0
    @test size(tvS) == (5,)
    @test norm(tvS-[T/2,T,3T/2,2T,5T/2],Inf) < 1E-13
    @test norm(gvS,Inf) < eps()

    #testing 0-th order root-finding + TaylorN jet transport
    tvN, xvN, tvSN, xvSN, gvSN = taylorinteg(pendulum!, g, x0N, t0, 3T, _order, _abstol, maxsteps=1000)
    @test size(tvSN) == size(tvS)
    @test size(xvSN) == size(xvS)
    @test size(gvSN) == size(gvS)
    @test norm( map(z->map(y->y.coeffs, z.coeffs), gvSN) ) < 1E-14
    @test norm(evaluate.(tvSN)-[T/2,T,3T/2,2T,5T/2],Inf) < 1E-13
    @test norm(evaluate.(xvSN)-xvS, Inf) < 1E-15

    #testing 0-th root-finding + Taylor1 jet transport
    tv1, xv1, tvS1, xvS1, gvS1 = taylorinteg(pendulum!, g, x01, t0, 3T, _order, _abstol, maxsteps=1000)
    @test size(tvS1) == size(tvS)
    @test size(xvS1) == size(xvS)
    @test size(gvS1) == size(gvS)
    @test norm( map(z->z.coeffs, gvS1) ) < 1E-14
    @test norm(evaluate.(tvS1)-[T/2,T,3T/2,2T,5T/2],Inf) < 1E-13
    @test norm(evaluate.(xvS1)-xvS, Inf) < 1E-15

    #testing surface higher order crossing detections and root-finding
    tv, xv, tvS, xvS, gvS = taylorinteg(pendulum!, g, x0, t0, 3T, _order, _abstol, maxsteps=1000, eventorder=2, maxnriters=2)
    @test tv[1] == t0
    @test xv[1,:] == x0
    @test size(tvS) == (5,)
    @test norm(tvS-[T/2,T,3T/2,2T,5T/2],Inf) < 1E-13
    @test norm(gvS,Inf) < 1E-15

    #testing higher order root-finding + TaylorN jet transport
    tvN, xvN, tvSN, xvSN, gvSN = taylorinteg(pendulum!, g, x0N, t0, 3T, _order, _abstol, maxsteps=1000, eventorder=2)
    @test size(tvSN) == size(tvS)
    @test size(xvSN) == size(xvS)
    @test size(gvSN) == size(gvS)
    @test norm( map(z->map(y->y.coeffs, z.coeffs), gvSN) ) < 1E-12
    @test norm(evaluate.(tvSN)-[T/2,T,3T/2,2T,5T/2],Inf) < 1E-13
    @test norm(evaluate.(xvSN)-xvS, Inf) < 1E-14

    #testing higher root-finding + Taylor1 jet transport
    tv1, xv1, tvS1, xvS1, gvS1 = taylorinteg(pendulum!, g, x01, t0, 3T, _order, _abstol, maxsteps=1000, eventorder=2)
    @test size(tvS1) == size(tvS)
    @test size(xvS1) == size(xvS)
    @test size(gvS1) == size(gvS)
    @test norm( map(z->z.coeffs, gvS1) ) < 1E-12
    @test norm(evaluate.(tvS1)-[T/2,T,3T/2,2T,5T/2],Inf) < 1E-13
    @test norm(evaluate.(xvS1)-xvS, Inf) < 1E-14
end
