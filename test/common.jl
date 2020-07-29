using TaylorIntegration, Test, DiffEqBase
using LinearAlgebra: norm
using StaticArrays

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

    @testset "Test throwing errors in common interface" begin
        u0 = rand(4, 2)
        tspan = (0.0, 1.0)
        prob1 = ODEProblem(f!, u0, tspan)
        sol = solve(prob1, TaylorMethod(50), abstol=1e-20)

        # `order` is not specified
        @test_throws ErrorException solve(prob, TaylorMethod(), abstol=1e-20)

        # Using a `callback`
        prob2 = ODEProblem(f!, u0, tspan, callback=nothing)
        @test_throws ErrorException solve(prob2, TaylorMethod(10), abstol=1e-20)
    end

    @testset "Test integration of DynamicalODEPoblem" begin
        function iip_q̇(dq,p,q,params,t)
            dq[1] = p[1]
            dq[2] = p[2]
        end

        function iip_ṗ(dp,p,q,params,t)
            dp[1] = -q[1] * (1 + 2q[2])
            dp[2] = -q[2] - (q[1]^2 - q[2]^2)
        end

        iip_q0 = [0.1, 0.]
        iip_p0 = [0., 0.5]


        function oop_q̇(p, q, params, t)
            p
        end

        function oop_ṗ(p, q, params, t)
            dp1 = -q[1] * (1 + 2q[2])
            dp2 = -q[2] - (q[1]^2 - q[2]^2)
            @SVector [dp1, dp2]
        end

        oop_q0 = @SVector [0.1, 0.]
        oop_p0 = @SVector [0., 0.5]

        T(p) = 1//2 * norm(p)^2
        V(q) = 1//2 * (q[1]^2 + q[2]^2 + 2q[1]^2 * q[2]- 2//3 * q[2]^3)
        H(p,q, params) = T(p) + V(q)

        E = H(iip_p0, iip_q0, nothing)

        energy_err(sol) = maximum(i->H([sol[1,i], sol[2,i]], [sol[3,i], sol[4,i]], nothing)-E, 1:length(sol.u))

        iip_prob = DynamicalODEProblem(iip_ṗ, iip_q̇, iip_p0, iip_q0, (0., 100.))
        oop_prob = DynamicalODEProblem(oop_ṗ, oop_q̇, oop_p0, oop_q0, (0., 100.))

        sol1 = solve(iip_prob, TaylorMethod(50), abstol=1e-20)
        @test energy_err(sol1) < 1e-10

        sol2 = solve(oop_prob, TaylorMethod(50), abstol=1e-20)
        @test energy_err(sol2) < 1e-10
    end
end
