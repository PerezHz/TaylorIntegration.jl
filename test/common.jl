using Test, DiffEqBase, TaylorIntegration
using LinearAlgebra: norm

@testset "Testing `common.jl`" begin

    f(u,p,t) = u
    g(u,p,t) = cos(t)
    @testset "Test integration of ODE with numbers in common interface" begin
        u0 = 0.5
        tspan = (0.0, 1.0)
        prob = ODEProblem(f, u0, tspan)
        sol = solve(prob, TaylorMethod(50), abstol=1e-20)
        @test TaylorIntegration.alg_order(TaylorMethod(50)) == 50
        @test TaylorIntegration.alg_order(sol.alg) == TaylorIntegration.alg_order(TaylorMethod(50))
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
    end

    @testset "Test discrete callback in common interface" begin
        # discrete callback: example taken from DifferentialEquations.jl docs:
        # https://diffeq.sciml.ai/dev/features/callback_functions/#Using-Callbacks
        t_cb = 1.0pi
        condition(u,t,integrator) = t == t_cb
        affect!(integrator) = integrator.u[1] += 0.1
        cb = DiscreteCallback(condition,affect!)
        sol = solve(prob, TaylorMethod(order), abstol=abstol, tstops=[t_cb], callback=cb)
        @test sol.t[4] == t_cb
        @test sol.t[4] == sol.t[5]
        @test sol[4][1] + 0.1 == sol[5][1]
    end

    @testset "Test continuous callback in common interface" begin
        # continuous callback: example taken from DifferentialEquations.jl docs:
        # https://diffeq.sciml.ai/stable/features/callback_functions/#Example-1:-Bouncing-Ball
        @taylorize function f(du,u,p,t)
            local g_acc = p
            du[1] = u[2]
            du[2] = -g_acc + zero(u[2])
        end
        function condition(u,t,integrator)
            u[1]
        end
        #### TODO: fix main loop so that it's not necessary to update cache manually
        function affect!(integrator)
            integrator.u[2] = -integrator.u[2]
            # after affecting current state, Taylor expansions in cache should be updated as well
            TaylorIntegration.update_jetcoeffs_cache!(integrator)
            return nothing
        end
        cb = ContinuousCallback(condition,affect!)
        u0 = [50.0,0.0]
        tspan = (0.0,15.0)
        p = 9.8
        prob = ODEProblem(f,u0,tspan,p)
        sol = solve(prob, TaylorMethod(25), abstol=1e-16, callback=cb)
        tb = sqrt(2*50/9.8) # bounce time
        @test abs(tb - sol.t[9]) < 1e-14
        @test sol.t[9] == sol.t[10]
        @test sol[9][1] == sol[10][1]
        @test sol[9][2] == -sol[10][2] # check that callback was applied correctly (1st bounce)
        @test abs(3tb - sol.t[19]) < 1e-14
        @test sol.t[19] == sol.t[20]
        @test sol[19][1] == sol[20][1]
        @test sol[19][2] == -sol[20][2] # check that callback was applied correctly (2nd bounce)
    end

    @testset "Test vector continuous callback in common interface" begin
        # vector continuous callback: example taken from DifferentialEquations.jl docs:
        # https://diffeq.sciml.ai/dev/features/callback_functions/#VectorContinuousCallback-Example
        @taylorize function f(du,u,p,t)
            local g_acc = p
            du[1] = u[2]
            du[2] = -g_acc + zero(u[2])
            du[3] = u[4]
            du[4] = zero(u[4])
        end

        function condition(out,u,t,integrator) # Event when event_f(u,t) == 0
            out[1] = u[1]
            out[2] = (u[3] - 10.0)u[3]
        end

        #### TODO: fix main loop so that it's not necessary to update cache manually
        function affect!(integrator, idx)
            if idx == 1
                integrator.u[2] = -0.9integrator.u[2]
            elseif idx == 2
                integrator.u[4] = -0.9integrator.u[4]
            end
            # after affecting current state, Taylor expansions in cache should be updated as well
            TaylorIntegration.update_jetcoeffs_cache!(integrator)
        end

        cb = VectorContinuousCallback(condition, affect!, 2)

        u0 = [50.0, 0.0, 0.0, 2.0]
        tspan = (0.0, 15.0)
        p = 9.8
        prob = ODEProblem(f, u0, tspan, p)
        sol = solve(prob, TaylorMethod(25), abstol=1e-16, callback=cb)
        tb = sqrt(2*50/9.8) # bounce time
        @test abs(tb - sol.t[8]) < 1e-14
        @test sol.t[8] == sol.t[9]
        @test sol[9][1] == sol[8][1]
        @test sol[9][2] == -0.9sol[8][2]
        @test sol[9][3] == sol[8][3]
        @test sol[9][4] == sol[8][4]
        @test sol.t[13] == sol.t[14]
        @test sol[14][1] == sol[13][1]
        @test sol[14][2] == sol[13][2]
        @test sol[14][3] == sol[13][3]
        @test sol[14][4] == -0.9sol[13][4]
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
