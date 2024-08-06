using TaylorIntegration
using StaticArrays
using Test

@testset "Automatic Domain Splitting" begin
    @testset "2D Kepler problem" begin
        using TaylorIntegration: countnodes, timeshift!

        # This example is based upon section 3 of
        # https://doi.org/10.1007/s10569-015-9618-3.

        # Dynamical function
        @taylorize function kepler_eqs!(dq, q, params, t)
            dq[1] = q[3]
            dq[2] = q[4]
            rr = ( q[1]^2 + q[2]^2 )^(3/2)
            dq[3] = - q[1] / rr
            dq[4] = - q[2] / rr
        end
        # Initial conditons (plain)
        q00 = [1.0, 0.0, 0.0, sqrt(1.5)]
        # Jet transport variables
        dq = set_variables("dx dy", numvars = 2, order = 5)
        # Initial conditions (jet transport)
        q0 = q00 .+ [0.008, 0.08, 0.0, 0.0] .* vcat(dq, dq)
        # Initial time
        t0 = 0.0
        # Final time
        tmax = 34.85
        # Taylor1 order
        order = 25
        # Splitting tolerance
        stol = 0.008
        # Absolute tolerance
        abstol = 1e-20
        # Dynamical function parameters
        params = nothing
        # Maximum allowed splits
        maxsplits = 25
        # Maximum allowed steps
        maxsteps = 1_000

        # ADS taylorinteg (parse_eqs = false)
        # Warmup
        q1 = ADSTaylorSolution([-1.0, -1.0], [1.0, 1.0], q0)
        taylorinteg(kepler_eqs!, q1, t0, tmax, order, stol, abstol, params;
                    maxsplits = 1, maxsteps = 1, parse_eqs = false);
        # Full integration
        q1 = ADSTaylorSolution([-1.0, -1.0], [1.0, 1.0], q0)
        taylorinteg(kepler_eqs!, q1, t0, tmax, order, stol, abstol, params;
                    maxsplits = maxsplits, maxsteps = maxsteps, parse_eqs = false);

        @test isa(q1, ADSTaylorSolution{Float64, 2, 4})
        @test iszero(q1.depth)
        @test iszero(q1.t)
        @test q1.lo == [-1.0, -1.0]
        @test q1.hi == [1.0, 1.0]
        @test q1.x == q0
        @test constant_term.(q1.p) == q0
        @test isnothing(q1.parent)
        @test isa(q1.left, ADSTaylorSolution{Float64, 2, 4})
        @test isnothing(q1.right)

        #=
        ts1 = timesvector(nv1)
        @test ts1[1] == t0
        @test ts1[end] == tmax
        @test length(ts1) == 85
        =#

        @test isone(countnodes(q1, 0))
        # @test countnodes(q1, length(ts1)-1) == maxsplits
        # @test iszero(countnodes(nv1, length(ts1)))
        @test iszero(countnodes(q1, prevfloat(t0)))
        @test countnodes(q1, t0) == 1
        @test countnodes(q1, tmax) == maxsplits
        @test iszero(countnodes(q1, nextfloat(tmax)))

        # ADS taylorinteg (parse_eqs = true)
        # Warmup
        q2 = ADSTaylorSolution([-1.0, -1.0], [1.0, 1.0], q0)
        taylorinteg(kepler_eqs!, q2, t0, tmax, order, stol, abstol, params;
                    maxsplits = 1, maxsteps = 1, parse_eqs = true);
        # Full integration
        q2 = ADSTaylorSolution([-1.0, -1.0], [1.0, 1.0], q0)
        taylorinteg(kepler_eqs!, q2, t0, tmax, order, stol, abstol, params;
                    maxsplits = maxsplits, maxsteps = maxsteps, parse_eqs = true);

        @test isa(q2, ADSTaylorSolution{Float64, 2, 4})
        @test iszero(q2.depth)
        @test iszero(q2.t)
        @test q2.lo == [-1.0, -1.0]
        @test q2.hi == [1.0, 1.0]
        @test q2.x == q0
        @test constant_term.(q2.p) == q0
        @test isnothing(q2.parent)
        @test isa(q2.left, ADSTaylorSolution{Float64, 2, 4})
        @test isnothing(q2.right)

        #=
        ts2 = timesvector(nv2)
        @test ts2[1] == t0
        @test ts2[end] == tmax
        @test length(ts2) == 85
        =#

        @test isone(countnodes(q2, 0))
        # @test countnodes(q2, length(ts2)-1) == maxsplits
        # @test iszero(countnodes(q2, length(ts2)))
        @test iszero(countnodes(q2, prevfloat(t0)))
        @test countnodes(q2, t0) == 1
        @test countnodes(q2, tmax) == maxsplits
        @test iszero(countnodes(q2, nextfloat(tmax)))

        # Compatibility between parse_eqs = false/true

        #=
        @test ts1 == ts2

        s1, x1 = nv1(t0)
        s2, x2 = nv2(t0)
        @test length(s1) == length(s2) == 1
        @test s1[1] == s2[1] == dom
        @test size(x1) == size(x2) == (4, 1)
        @test x1[:, 1] == x2[:, 1] == q0

        s1, x1 = nv1(tmax)
        s2, x2 = nv2(tmax)
        @test length(s1) == length(s2) == maxsplits
        @test s1 == s2
        @test size(x1) == size(x2) == (4, maxsplits)
        @test x1 == x2

        y1 = nv1(t0, SVector(0.1, 0.1))
        y2 = nv2(t0, SVector(0.1, 0.1))
        @test y1 == y2
        y1 = nv1(tmax, SVector(0.1, 0.1))
        y2 = nv2(tmax, SVector(0.1, 0.1))
        @test y1 == y2
        =#

        # ADS vs Monte Carlo both in cartesian coordinates and keplerian elements

        #=
        # Semimajor axis and eccentricity
        function ae(rv)
            x, y, u, v = rv
            r = sqrt(x^2 + y^2)
            vsq = u^2 + v^2
            a = 1 / ( (2/r)- vsq )
            hsq = (x*v - y*u)^2
            e = sqrt(1 - hsq/a)
            return a, e
        end
        # 16 points in the boundary of the domain
        side = LinRange(-1, 1, 5)
        boundary = vcat(
            map(x -> [x, -1], side), # Bottom
            map(x -> [1, x], side),  # Right
            map(x -> [-x, 1], side),  # Top
            map(x -> [-1, -x], side)  # Left
        )
        unique!(boundary)

        for s in boundary
            tv, xv = taylorinteg(kepler_eqs!, q0_(s), t0, tmax, order, abstol, Val(false), params;
                                 maxsteps, parse_eqs = false)
            rfvmc = xv[end, :]

            rv0ads, rvfads = nv1(t0, s), nv1(tmax, s)

            @test maximum(@. abs((rvfads - rfvmc) / rfvmc)) < 0.03
            @test maximum(@. abs((rvfads - rfvmc) / rfvmc)) < 0.03

            a0, e0 = ae(rv0ads)
            af, ef = ae(rvfads)

            @test abs((af - a0) / a0) < 0.07
            @test abs((ef - e0) / e0) < 0.07

            tv, xv = taylorinteg(kepler_eqs!, q0_(s), t0, tmax, order, abstol, Val(false), params;
                                 maxsteps, parse_eqs = true)
            rfvmc = xv[end, :]

            rv0ads, rvfads = nv2(t0, s), nv2(tmax, s)

            @test maximum(@. abs((rvfads - rfvmc) / rfvmc)) < 0.03
            @test maximum(@. abs((rvfads - rfvmc) / rfvmc)) < 0.03

            a0, e0 = ae(rv0ads)
            af, ef = ae(rvfads)

            @test abs((af - a0) / a0) < 0.07
            @test abs((ef - e0) / e0) < 0.07
        end
        =#

        # timeshift!

        timeshift!(q2, 1.0)
        for (n1, n2) in zip(PreOrderDFS(q1), PreOrderDFS(q2))
            @test n2.t == n1.t + 1.0
        end
    end
end