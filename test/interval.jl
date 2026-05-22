using TaylorIntegration, Test
using IntervalArithmetic
using Logging
import Logging: Warn

@testset "Testing integrations with intervals" begin
    local TI = TaylorIntegration

    local _order = 25
    local _abstol = 1.0E-20
    local _reltol = 1.0E-15

    max_iters_reached() = "Maximum number of integration steps reached; exiting.\n"
    zero_stepsize() = "The step-size is zero; aborting integration."

    @testset "Tests: dot{x}=x^2, x(0) = 1" begin
        eqs_mov(x, p, t) = x^2
        t0 = 0.0
        x0 = interval(1.0)
        sol = (@test_no_logs (Warn, zero_stepsize()) taylorinteg(
            eqs_mov,
            x0,
            0.0,
            1.0,
            _order,
            _abstol,
        ))

        @test isa(sol, TaylorSolution{Float64,Interval{Float64},1})
        tv = sol.t
        xv = sol.x
        @test length(tv) < 501
        @test length(xv) < 501
        @test isequal_interval(xv[1], x0)
        @test tv[end] < 1.0

    end

    @testset "Test non-autonomous ODE (2): dot{x}=cos(t)" begin
        function f!(Dx, x, p, t)
            local cero = zero(x[1])
            Dx[1] = one(cero+t)
            Dx[2] = cos(cero+t)
            nothing
        end
        t0 = 0.0
        tmax = 10.25 * (2pi)
        abstol = 1e-20
        order = 25
        x0 = [interval(t0), interval(0.0)] #initial conditions such that x(t)=sin(t)

        sol = (@test_no_logs min_level = Logging.Warn taylorinteg(
            f!,
            x0,
            t0,
            tmax,
            order,
            abstol,
        ))
        tv = sol.t
        xv = sol.x
        @test length(tv) < 501
        @test length(xv[:, 1]) < 501
        @test length(xv[:, 2]) < 501
        @test isequal_interval(xv[1, 1:end], x0)
        @test tv[1] < tv[end]
        @test in_interval(tv[end], xv[end, 1])
        @test abs(sin(tmax) - xv[end, 2]) < 5e-14

        # Backward integration
        solb = (@test_no_logs min_level = Logging.Warn taylorinteg(
            f!,
            [interval(tmax), sin(interval(tmax))],
            tmax,
            t0,
            order,
            abstol,
            dense = true,
        ))
        tb = solb.t
        xb = solb.x
        @test length(tb) < 501
        @test length(xb[:, 1]) < 501
        @test length(xb[:, 2]) < 501
        @test tb[1] > tb[end]
        @test all(in_interval.([tmax, sin(tmax)], xb[1, 1:end]))
        @test in_interval(tb[end], xb[end, 1])
        @test abs(sin(t0) - xb[end, 2]) < 5e-14
        @test all(in_interval.([tmax, sin(tmax)], solb(tmax)))
        @test all(isequal_interval.(xb[end, :], solb(x0[1])))
    end

end