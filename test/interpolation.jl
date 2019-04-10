# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration
using Test

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
    @test tv == tv2
    @test xv == xv2
    tinterp = taylorinteg(eqs_mov, x0, t0, tmax, _order, _abstol, dense=true)
    @test tinterp.t == tv
    @test tinterp.(tinterp.t) == xv
    @test_throws AssertionError tinterp(tinterp.t[1]-0.1)
    @test_throws AssertionError tinterp(tinterp.t[end]+0.1)
end