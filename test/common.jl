using TaylorIntegration, Test, DiffEqBase
using LinearAlgebra: norm

@testset "Testing `common.jl`" begin

    f(u,p,t) = u
    @testset "Test integration of ODE with numbers in common interface" begin
        u0 = 0.5
        tspan = (0.0, 1.0)
        prob = ODEProblem(f, u0, tspan)
        sol = solve(prob, TaylorMethod(50), abstol=1e-20)

        @test sol[end] - 0.5*exp(1) < 1e-12
    end

    f!(du, u, p, t) = (du .= u)
    @testset "Test integration of ODE with abstract arrays in common interface" begin
        u0 = rand(4, 2)
        tspan = (0.0, 1.0)
        prob = ODEProblem(f!, u0, tspan)
        sol = solve(prob, TaylorMethod(50), abstol=1e-20)

        @test norm(sol[end] - u0.*exp(1)) < 1e-12
    end

    tspan = (0.0,5.0)
    saveat_inputs = ([], 0:1:(tspan[2]+5), 0:1:tspan[2], 3:1:tspan[2], collect(0:1:tspan[2]))
    @testset "Test saveat behavior with numbers in common interface" begin
        u0 = 1.0
        prob = ODEProblem(f, u0, tspan)
        sol = solve(prob, TaylorMethod(20), abstol=1e-20)
        s = saveat_inputs[1]
        sol_taylor = solve(prob, TaylorMethod(20), abstol=1e-20, saveat = s)
        @test all(sol.t .== sol_taylor.t)
        @test all(sol_taylor.u .== sol.u)
        @test length(sol_taylor.t) == length(sol_taylor.u)
        s = saveat_inputs[2]
        sol_taylor = solve(prob, TaylorMethod(20), abstol=1e-20, saveat = s)
        @test sol_taylor.t == saveat_inputs[3]
        for s ∈ saveat_inputs[3:end]
            sol_taylor = solve(prob, TaylorMethod(20), abstol=1e-20, saveat = s)
            @test all(s .== sol_taylor.t)
            @test length(sol_taylor.t) == length(sol_taylor.u)
        end
    end

    @testset "Test saveat behavior with abstract arrays in common interface" begin
        u0 = rand(2)
        prob = ODEProblem(f!, u0, tspan)
        sol = solve(prob, TaylorMethod(20), abstol=1e-20)
        s = saveat_inputs[1]
        sol_taylor = solve(prob, TaylorMethod(20), abstol=1e-20, saveat = s)
        @test all(sol.t .== sol_taylor.t)
        @test all(sol_taylor.u .== sol.u)
        @test length(sol_taylor.t) == length(sol_taylor.u)
        s = saveat_inputs[2]
        sol_taylor = solve(prob, TaylorMethod(20), abstol=1e-20, saveat = s)
        @test sol_taylor.t == saveat_inputs[3]
        for s ∈ saveat_inputs[3:end]
            sol_taylor = solve(prob, TaylorMethod(20), abstol=1e-20, saveat = s)
            @test all(s .== sol_taylor.t)
            @test length(sol_taylor.t) == length(sol_taylor.u)
        end
    end
end
