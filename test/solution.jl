# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration
using Test
using Logging
import Logging: Warn
using InteractiveUtils

@testset "Testing `solution.jl`" begin
    tv = [1.,2]
    xv = rand(2,2)
    psol = Taylor1.(rand(2,1),2)
    nsteps = 2
    # @show @code_warntype TaylorIntegration.build_solution(tv, xv, psol, nsteps)
    sol2 = TaylorIntegration.build_solution(tv, xv, psol, nsteps)
    @test sol2 isa TaylorSolution{Float64, Float64, 2}
    # @show @code_warntype TaylorIntegration.build_solution(tv, Vector(xv[1,:]), Vector(psol[1,:]), nsteps)
    sol1 = TaylorIntegration.build_solution(tv, Vector(xv[1,:]), Vector(psol[1,:]), nsteps)
    @test sol1 isa TaylorSolution{Float64, Float64, 1}
end