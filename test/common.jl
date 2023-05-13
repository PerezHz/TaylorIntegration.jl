using TaylorIntegration, Test
using OrdinaryDiffEq
using LinearAlgebra: norm
using StaticArrays
using Logging
import Logging: Warn

@testset "Testing `common.jl`" begin

    local TI = TaylorIntegration
    max_iters_reached() = "Maximum number of integration steps reached; exiting.\n"

    f(u,p,t) = u
    g(u,p,t) = cos(t)
    f!(du, u, p, t) = (du .= u)

    @testset "Test integration of ODE with numbers in common interface" begin
        u0 = 0.5
        tspan = (0.0, 1.0)
        prob = ODEProblem(f, u0, tspan)
        sol = (@test_logs min_level=Logging.Warn solve(prob, TaylorMethod(50), abstol=1e-20))
        @test TI.alg_order(TaylorMethod(50)) == 50
        @test TI.alg_order(sol.alg) == TI.alg_order(TaylorMethod(50))
        @test abs(sol[end] - u0*exp(1)) < 1e-12
        u0 = 0.0
        tspan = (0.0, 11pi)
        prob = ODEProblem(g, u0, tspan)
        sol = (@test_logs min_level=Logging.Warn solve(prob, TaylorMethod(50), abstol=1e-20))
        @test abs(sol[end] - sin(sol.t[end])) < 1e-12
    end

    @testset "Test integration of ODE with abstract arrays in common interface" begin
        u0 = rand(4, 2)
        tspan = (0.0, 1.0)
        prob = ODEProblem(f!, u0, tspan)
        sol = solve(prob, TaylorMethod(50), abstol=1e-20)
        @test norm(sol[end] - u0.*exp(1)) < 1e-12
        @taylorize f_oop(u, p, t) = u
        prob_oop = ODEProblem(f_oop, u0, tspan)
        sol_oop = solve(prob_oop, TaylorMethod(50), abstol=1e-20)
        @test norm(sol_oop[end] - u0.*exp(1)) < 1e-12
        @test sol.t == sol_oop.t
        @test sol.u == sol_oop.u
    end

    local tspan = (0.0,5.0)
    local saveat_inputs = ([], 0:1:(tspan[2]+5), 0:1:tspan[2], 3:1:tspan[2], collect(0:1:tspan[2]))
    @testset "Test saveat behavior with numbers in common interface" begin
        u0 = 1.0
        prob = ODEProblem(f, u0, tspan)
        sol = (@test_logs min_level=Logging.Warn solve(prob, TaylorMethod(20), abstol=1e-20))
        s = saveat_inputs[1]
        sol_taylor = (@test_logs min_level=Logging.Warn solve(prob, TaylorMethod(20), abstol=1e-20, saveat = s))
        @test all(sol.t .== sol_taylor.t)
        @test all(sol_taylor.u .== sol.u)
        @test length(sol_taylor.t) == length(sol_taylor.u)
        s = saveat_inputs[2]
        sol_taylor = (@test_logs min_level=Logging.Warn solve(prob, TaylorMethod(20), abstol=1e-20, saveat = s))
        @test sol_taylor.t == saveat_inputs[3]
        for s ∈ saveat_inputs[3:end]
            sol_taylor = (@test_logs min_level=Logging.Warn solve(prob, TaylorMethod(20), abstol=1e-20, saveat = s))
            @test all(s .== sol_taylor.t)
            @test length(sol_taylor.t) == length(sol_taylor.u)
        end
    end

    @testset "Test saveat behavior with abstract arrays in common interface" begin
        u0 = rand(2)
        prob = ODEProblem(f!, u0, tspan)
        sol = (@test_logs min_level=Logging.Warn solve(prob, TaylorMethod(20), abstol=1e-20))
        s = saveat_inputs[1]
        sol_taylor = (@test_logs min_level=Logging.Warn solve(prob, TaylorMethod(20), abstol=1e-20, saveat = s))
        @test all(sol.t .== sol_taylor.t)
        @test all(sol_taylor.u .== sol.u)
        @test length(sol_taylor.t) == length(sol_taylor.u)
        s = saveat_inputs[2]
        sol_taylor = (@test_logs min_level=Logging.Warn solve(prob, TaylorMethod(20), abstol=1e-20, saveat = s))
        @test sol_taylor.t == saveat_inputs[3]
        for s ∈ saveat_inputs[3:end]
            sol_taylor = (@test_logs min_level=Logging.Warn solve(prob, TaylorMethod(20), abstol=1e-20, saveat = s))
            @test all(s .== sol_taylor.t)
            @test length(sol_taylor.t) == length(sol_taylor.u)
        end
    end

    function harmosc!(dx, x, p, t)
        dx[1] = x[2]
        dx[2] = - x[1]
        return nothing
    end
    @testset "Test consistency with taylorinteg" begin
        tspan = (0.0, 10pi)
        abstol = 1e-20
        order = 25 # Taylor expansion order wrt time
        u0 = [1.0; 0.0]
        prob = ODEProblem(harmosc!, u0, tspan)
        sol = (@test_logs min_level=Logging.Warn solve(prob, TaylorMethod(order), abstol=abstol))
        tv1, xv1 = (@test_logs min_level=Logging.Warn taylorinteg(
            harmosc!, u0, tspan[1], tspan[2], order, abstol))
        @test sol.t == tv1
        @test xv1[end,:] == sol[end]
    end

    @testset "Test discrete callback in common interface" begin
        # discrete callback: example taken from DifferentialEquations.jl docs:
        # https://diffeq.sciml.ai/dev/features/callback_functions/#Using-Callbacks
        tspan = (0.0, 10pi)
        abstol = 1e-20
        order = 25 # Taylor expansion order wrt time
        u0 = [1.0; 0.0]
        prob = ODEProblem(harmosc!, u0, tspan)
        t_cb = 1.0pi
        condition(u,t,integrator) = t == t_cb
        affect!(integrator) = integrator.u[1] += 0.1
        cb = DiscreteCallback(condition,affect!)
        sol = (@test_logs min_level=Logging.Warn solve(prob, TaylorMethod(order), abstol=abstol, tstops=[t_cb], callback=cb))
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
        function affect!(integrator)
            integrator.u[2] = -integrator.u[2]
            return nothing
        end
        cb = ContinuousCallback(condition,affect!)
        u0 = [50.0,0.0]
        tspan = (0.0,15.0)
        p = 9.8
        prob = ODEProblem(f,u0,tspan,p)
        # Avoid log-checks for Julia versions before v1.6
        if VERSION < v"1.6"
            sol = solve(prob, TaylorMethod(25), abstol=1e-16, callback=cb)
        else
            sol = (@test_logs min_level=Logging.Warn solve(prob, TaylorMethod(25), abstol=1e-16, callback=cb))
        end
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
        @taylorize function ff(du,u,p,t)
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

        function affect!(integrator, idx)
            if idx == 1
                integrator.u[2] = -0.9integrator.u[2]
            elseif idx == 2
                integrator.u[4] = -0.9integrator.u[4]
            end
        end

        cb = VectorContinuousCallback(condition, affect!, 2)

        u0 = [50.0, 0.0, 0.0, 2.0]
        tspan = (0.0, 15.0)
        p = 9.8
        prob = ODEProblem(ff, u0, tspan, p)
        # Avoid log-checks for Julia versions before v1.6
        if VERSION < v"1.6"
            sol = solve(prob, TaylorMethod(25), abstol=1e-16, callback=cb)
        else
            sol = (@test_logs min_level=Logging.Warn solve(prob, TaylorMethod(25), abstol=1e-16, callback=cb))
        end
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
        b1 = 3.0
        order = 25 # Taylor expansion order wrt time
        abstol = 1.0e-20
        @taylorize xdot1(x, p, t) = b1-x^2
        @test (@isdefined xdot1)
        x0 = 1.0
        tspan = (0.0, 1000.0)
        prob = ODEProblem(xdot1, x0, tspan)
        sol1 = (@test_logs min_level=Logging.Warn solve(
            prob, TaylorMethod(order), abstol=abstol, parse_eqs=false))
        @test sol1.alg.parse_eqs == false
        sol2 = (@test_logs min_level=Logging.Warn solve(
            prob, TaylorMethod(order), abstol=abstol))
        sol3 = (@test_logs min_level=Logging.Warn solve(
            prob, TaylorMethod(order), abstol=abstol, parse_eqs=false))
        @test sol2.alg.parse_eqs
        @test !sol3.alg.parse_eqs
        @test sol1.t == sol2.t
        @test sol1.u == sol2.u
        @test sol1.t == sol3.t
        @test sol1.u == sol3.u

        @taylorize function integ_vec(dx, x, p, t)
            local λ = p[1]
            dx[1] = cos(t)
            dx[2] = -λ*sin(t)
            return dx
        end
        @test (@isdefined integ_vec)
        x0 = [0.0, 1.0]
        tspan = (0.0, 11pi)
        prob = ODEProblem(integ_vec, x0, tspan, [1.0])
        sol1 = (@test_logs min_level=Logging.Warn solve(
            prob, TaylorMethod(order), abstol=abstol, parse_eqs=false))
        sol2 = (@test_logs min_level=Logging.Warn solve(
            prob, TaylorMethod(order), abstol=abstol)) # parse_eqs=true
        @test sol2.alg.parse_eqs == true
        @test length(sol1.t) == length(sol2.t)
        @test sol1.t == sol2.t
        @test sol1.u == sol2.u
        tv, xv = (@test_logs min_level=Logging.Warn taylorinteg(
            integ_vec, x0, tspan[1], tspan[2], order, abstol, [1.0]))
        @test sol1.t == tv
        @test sol1[1,:] == xv[:,1]
        @test sol1[2,:] == xv[:,2]
        @test norm(sol1[end][1] - sin(sol1.t[end]), Inf) < 1.0e-14
        @test norm(sol1[end][2] - cos(sol1.t[end]), Inf) < 1.0e-14

        V(x, y) = - (1-μ)/sqrt((x-μ)^2+y^2) - μ/sqrt((x+1-μ)^2+y^2)
        H_pcr3bp(x, y, px, py) = (px^2+py^2)/2 - (x*py-y*px) + V(x, y)
        H_pcr3bp(x) = H_pcr3bp(x...)
        J0 = -1.58 # Jacobi constant
        q0 = [-0.8, 0.0, 0.0, -0.6276410653920694] # initial condition
        μ = 0.01 # mass parameter
        tspan = (0.0, 1000.0)
        p = [μ]
        @taylorize function pcr3bp!(dq, q, param, t)
            local μ = param[1]
            local onemμ = 1 - μ
            x1 = q[1]-μ
            x1sq = x1^2
            y = q[2]
            ysq = y^2
            r1_1p5 = (x1sq+ysq)^1.5
            x2 = q[1]+onemμ
            x2sq = x2^2
            r2_1p5 = (x2sq+ysq)^1.5
            dq[1] = q[3] + q[2]
            dq[2] = q[4] - q[1]
            dq[3] = (-((onemμ*x1)/r1_1p5) - ((μ*x2)/r2_1p5)) + q[4]
            dq[4] = (-((onemμ*y )/r1_1p5) - ((μ*y )/r2_1p5)) - q[3]
            return nothing
        end
        prob = ODEProblem(pcr3bp!, q0, tspan, p)
        sol1 = (@test_logs min_level=Logging.Warn solve(
            prob, TaylorMethod(order), abstol=abstol))
        @test sol1.alg.parse_eqs == true
        sol2 = (@test_logs min_level=Logging.Warn solve(
            prob, TaylorMethod(order), abstol=abstol, parse_eqs=false))
        @test sol2.alg.parse_eqs == false
        @test norm( H_pcr3bp(sol1.u[end]) - J0 ) < 1e-10
        @test norm( H_pcr3bp(sol2.u[end]) - J0 ) < 1e-10
        @test sol1.u == sol2.u
        @test sol1.t == sol2.t

        # Same as harmosc1dchain!, but adding a `@threads for`
        @taylorize function harmosc1dchain!(dq, q, params, t)
            local N = Int(length(q)/2)
            local _eltype_q_ = eltype(q)
            local μ = params
            X = Array{_eltype_q_}(undef, N, N)
            accX = Array{_eltype_q_}(undef, N) #acceleration
            for j in 1:N
                accX[j] = zero(q[1])
                dq[j] = q[N+j]
            end
            #compute accelerations
            for j in 1:N
                for i in 1:N
                    if i == j
                    else
                        X[i,j] = q[i]-q[j]
                        temp_001 = accX[j] + (μ[i]*X[i,j])
                        accX[j] = temp_001
                    end #if i != j
                end #for, i
            end #for, j
            for i in 1:N
                dq[N+i] = accX[i]
            end
            nothing
        end

        @taylorize function harmosc1dchain_threads!(dq, q, params, t)
            local N = Int(length(q)/2)
            local _eltype_q_ = eltype(q)
            local μ = params
            X = Array{_eltype_q_}(undef, N, N)
            accX = Array{_eltype_q_}(undef, N) #acceleration
            for j in 1:N
                accX[j] = zero(q[1])
                dq[j] = q[N+j]
            end
            #compute accelerations
            Threads.@threads for j in 1:N
                for i in 1:N
                    if i == j
                    else
                        X[i,j] = q[i]-q[j]
                        temp_001 = accX[j] + (μ[i]*X[i,j])
                        accX[j] = temp_001
                    end #if i != j
                end #for, i
            end #for, j
            for i in 1:N
                dq[N+i] = accX[i]
            end
            nothing
        end

        N = 200
        x0 = 10randn(2N)
        μ = 1e-7rand(N)

        @show Threads.nthreads()

        probHO = ODEProblem(harmosc1dchain!, x0, (0.0, 10000.0), μ)
        solHO = (@test_logs min_level=Logging.Warn solve(
            probHO, TaylorMethod(order), abstol=abstol, parse_eqs=false))
        probHOT = ODEProblem(harmosc1dchain_threads!, x0, (0.0, 10000.0), μ)
        solHOT = (@test_logs min_level=Logging.Warn solve(
            probHOT, TaylorMethod(order), abstol=abstol))
        solHOnT = (@test_logs min_level=Logging.Warn solve(
            probHOT, TaylorMethod(order), abstol=abstol, parse_eqs=false))

        @test solHOT.u == solHOnT.u
        @test solHOT.t == solHOnT.t
        @test solHO.u == solHOT.u
        @test solHO.t == solHOT.t
    end

    @testset "Test throwing errors in common interface" begin
        # NOTE: The original test defines u0 as a matrix (rand(4,2)) but that doesn't work
        u0 = rand(4)#rand(4, 2)
        tspan = (0.0, 1.0)
        prob1 = ODEProblem(f!, u0, tspan)
        sol = (@test_logs min_level=Logging.Warn solve(prob1, TaylorMethod(50), abstol=1e-20))

        # `order` is not specified
        @test_throws ErrorException solve(prob, TaylorMethod(), abstol=1e-20)
    end

    ### DynamicalODEProblem tests (see #108, #109)
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

        T(p) = 1//2 * (p[1]^2 + p[2]^2)
        V(q) = 1//2 * (q[1]^2 + q[2]^2 + 2q[1]^2 * q[2]- 2//3 * q[2]^3)
        H(p,q, params) = T(p) + V(q)

        E = H(iip_p0, iip_q0, nothing)

        energy_err(sol) = maximum(i->H([sol[1,i], sol[2,i]], [sol[3,i], sol[4,i]], nothing)-E, 1:length(sol.u))

        iip_prob = DynamicalODEProblem(iip_ṗ, iip_q̇, iip_p0, iip_q0, (0., 100.))
        oop_prob = DynamicalODEProblem(oop_ṗ, oop_q̇, oop_p0, oop_q0, (0., 100.))

        sol1 = (@test_logs min_level=Logging.Warn solve(iip_prob, TaylorMethod(50), abstol=1e-20))
        @test energy_err(sol1) < 1e-10

        sol2 = (@test_logs min_level=Logging.Warn solve(oop_prob, TaylorMethod(50), abstol=1e-20))
        @test energy_err(sol2) < 1e-10

        @test sol1.t == sol2.t
        @test sol1.u[:] == sol2.u[:]
    end

end
