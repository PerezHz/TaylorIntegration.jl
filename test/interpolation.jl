# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration
using Test
using LinearAlgebra: norm, transpose

const _order = 28
const _abstol = 1.0E-20
const tT = Taylor1(_order)

@testset "Tests: Taylor interpolation, scalar case" begin
    eqs_mov(t, x) = x^2
    exactsol(t, x0) = x0/(1.0-x0*t) #the analytical solution
    t0 = 0.0
    tmax = 0.3
    x0 = 3.0
    tv, xv = taylorinteg(eqs_mov, x0, t0, tmax, _order, _abstol)
    tv2, xv2 = taylorinteg(eqs_mov, x0, t0, tmax, _order, _abstol, dense=false)
    # taylorinteg should select dense=false by default
    @test tv == tv2
    @test xv == xv2
    tinterp = taylorinteg(eqs_mov, x0, t0, tmax, _order, _abstol, dense=true)
    @test tinterp.t == tv
    # the interpolator evaluated at tv should equal the solution without interpolation
    @test tinterp.(tinterp.t) == xv
    @test tinterp(0//1) == x0
    @test tinterp(t0) == x0
    @test tinterp(tmax) == xv[end]
    @test_throws AssertionError tinterp(tinterp.t[1]-0.1)
    @test_throws AssertionError tinterp(tinterp.t[end]+0.1)
    tv3 = sort(  union(t0, tmax, tmax*rand(20))  )
    @test all(diff(tv3).>0)
    sol_interp_exact = @__dot__ tinterp(tv3) - exactsol(tv3, x0)
    @test norm(sol_interp_exact, Inf) < 1e-13
end

@testset "Tests: Taylor interpolation, vectorial case" begin
    function f!(t, x, Dx)
        Dx[1] = one(t)
        Dx[2] = cos(t)
        nothing
    end
    t0 = 0//1
    tmax = 10.25*(2pi)
    x0 = [t0, 0.0] #initial conditions such that x(t)=sin(t)
    tv, xv = taylorinteg(f!, x0, t0, tmax, _order, _abstol)
    tv2, xv2 = taylorinteg(f!, x0, t0, tmax, _order, _abstol, dense=false)
    # @time tv2, xv2 = taylorinteg(f!, x0, t0, tmax, _order, _abstol, dense=false)
    @test tv == tv2
    @test xv == xv2
    tinterp = taylorinteg(f!, x0, t0, tmax, _order, _abstol, dense=true)
    # @time tinterp = taylorinteg(f!, x0, t0, tmax, _order, _abstol, dense=true)
    @test tinterp.t == tv
    @test transpose(hcat(tinterp.(tv)...)) == xv
    @test tinterp(t0) == x0
    @test tinterp(tmax) == xv[end,:]
end
