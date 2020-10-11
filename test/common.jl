using TaylorIntegration, Test, DiffEqBase
using LinearAlgebra: norm

@testset "Testing `common.jl`" begin

    f(u,p,t) = u
    g(u,p,t) = cos(t)
    @testset "Test integration of ODE with numbers in common interface" begin
        u0 = 0.5
        tspan = (0.0, 1.0)
        prob = ODEProblem(f, u0, tspan)
        sol = solve(prob, TaylorMethod(50), abstol=1e-20)
        @test abs(sol[end] - u0*exp(1)) < 1e-12
        u0 = 0.0
        tspan = (0.0, 11pi)
        prob = ODEProblem(g, u0, tspan)
        sol = solve(prob, TaylorMethod(50), abstol=1e-20)
        @test abs(sol[end] - sin(sol.t[end])) < 1e-12
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

    function harmosc!(dx, x, p, t)
        dx[1] = x[2]
        dx[2] = - x[1]
        return nothing
    end
    tspan = (0.0, 10pi)
    abstol=1e-20 # 1e-16
    order = 25 # Taylor expansion order wrt time
    u0 = [1.0; 0.0]
    prob = ODEProblem(harmosc!, u0, tspan)
    @testset "Test consistency with taylorinteg" begin
        sol = solve(prob, TaylorMethod(order), abstol=abstol)
        tv1, xv1 = taylorinteg(harmosc!, u0, tspan[1], tspan[2], order, abstol)
        @test sol.t == tv1
        @test xv1[end,:] == sol[end]
        tT = Taylor1(tspan[1], order)
        xT = Taylor1.(u0, order)
    end

    @testset "Test use of callbacks in common interface" begin
        # discrete callback: example taken from DifferentialEquations.jl docs:
        # https://diffeq.sciml.ai/dev/features/callback_functions/#Using-Callbacks
        t_cb = 1.0pi
        condition(u,t,integrator) = t == t_cb
        affect!(integrator) = integrator.u[1] += 0.1
        cb = DiscreteCallback(condition,affect!)
        sol = solve(prob, TaylorMethod(order), abstol=abstol, tstops=[t_cb], callback=cb)
        @test sol.t[4] == t_cb
        @test sol.t[4] == sol.t[5]
        @test sol[4][1] != sol[5][1]
        @test abs(sol[5][1] - sol[4][1] - 0.1) < 1e-14
    end

    @testset "Test parsed jetcoeffs! method in common interface" begin
        @taylorize function integ_vec(dx, x, p, t)
            local λ = p[1]
            dx[1] = cos(t)
            dx[2] = -λ*sin(t)
            return dx
        end
        @test (@isdefined integ_vec)
        x0 = [0.0, 1.0]
        tspan = (0.0, pi)
        prob = ODEProblem(integ_vec, x0, tspan, [1.0])
        sol1 = solve(prob, TaylorMethod(order), abstol=abstol, parse_eqs=false)
        sol2 = solve(prob, TaylorMethod(order), abstol=abstol) # parse_eqs=true
        @test length(sol1.t) == length(sol2.t)
        @test sol1.t == sol2.t
        @test sol1.u == sol2.u
        tv, xv = taylorinteg(integ_vec, x0, tspan[1], tspan[2], order, abstol, [1.0])
        @test norm(sol1[end][1] - sin(sol1.t[end]), Inf) < 1.0e-15
        @test norm(sol1[end][2] - cos(sol1.t[end]), Inf) < 1.0e-15
    end

    @testset "Test throwing errors in common interface" begin
        u0 = rand(4, 2)
        tspan = (0.0, 1.0)
        prob1 = ODEProblem(f!, u0, tspan)
        sol = solve(prob1, TaylorMethod(50), abstol=1e-20)

        # `order` is not specified
        @test_throws ErrorException solve(prob, TaylorMethod(), abstol=1e-20)
    end
end
