# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorSeries, TaylorIntegration
using Base.Test

const _order = 28
const _abstol = 1.0E-20

@testset "Test intersection of surfaces and root-finding: simple pendulum" begin
    function pendulum!(t, x, dx)
        dx[1] = x[2]
        dx[2] = -sin(x[1])
        nothing
    end

    g(t, x, dx) = x[2]

    const t0 = 0.0
    const x0 = [1.3, 0.0]
    const T = 7.019250311844546

    #warm-up lap and preliminary tests
    tv, xv, tvS, xvS, gvS = taylorinteg(pendulum!, g, x0, t0, T, _order, _abstol, maxsteps=1)
    @test size(tv) == (2,)
    @test tv[1] == t0
    @test size(xv) == (2,2)
    @test xv[1,:] == x0
    @test size(tvS) == (0,)
    @test size(xvS) == (0,2)
    @test size(gvS) == (0,)

    #testing surface crossing detections and root-finding
    tv, xv, tvS, xvS, gvS = taylorinteg(pendulum!, g, x0, t0, 1.25T, _order, _abstol, maxsteps=100000)
    @test tv[1] == t0
    @test xv[1,:] == x0
    @test size(tvS) == (2,)
    @test norm(tvS-[T/2,T],Inf) < 1E-13
    @test norm(gvS,Inf) < eps()
end