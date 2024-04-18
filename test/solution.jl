# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration
using Test
using Logging
import Logging: Warn

@testset "Testing `solution.jl`" begin
    tv = [1.,2]
    xv = rand(2,2)
    psol = Taylor1.(rand(2,1),2)
    nsteps = 2
    # @show @code_warntype TaylorIntegration.build_solution(tv, xv, psol, nsteps)
    sol2 = TaylorIntegration.build_solution(tv, xv, psol, nsteps)
    @test sol2 isa TaylorSolution{Float64, Float64, 2}
    @test string(sol2) == "tspan: (1.0, 2.0), x: 2 Float64 variables"
    # @show @code_warntype TaylorIntegration.build_solution(tv, Vector(xv[1,:]), Vector(psol[1,:]), nsteps)
    sol1 = TaylorIntegration.build_solution(tv, Vector(xv[1,:]), Vector(psol[1,:]), nsteps)
    @test sol1 isa TaylorSolution{Float64, Float64, 1}
    @test string(sol1) == "tspan: (1.0, 2.0), x: 1 Float64 variable"
    tv = 0.1:0.1:1.1
    xv = rand(2, length(tv))
    sol = TaylorIntegration.build_solution(tv, xv, Taylor1.(xv, 2), 9)
    t1 = 0.35 + Taylor1(get_variables()[1],2)
    ind, δt = TaylorIntegration.timeindex(sol, t1)
    @test ind == 3
    @test δt == t1 - sol.t[ind]
    tv = collect((0:0.25:2)*pi)
    xv = Matrix(hcat(sin.(tv), cos.(tv))')
    psolv = Matrix(hcat(sin.(tv .+ Taylor1(25)), cos.(tv .+ Taylor1(25)))')
    sol = TaylorIntegration.build_solution(tv, xv, psolv, length(tv)-2)
    @test sol(sol.t[1]) == sol.x[1,:]
    @test norm(sol(sol.t[end]) - sol.x[end,:], Inf) < 1e-14
end