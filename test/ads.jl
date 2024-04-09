using TaylorIntegration
using StaticArrays
using Test

@testset "Automatic Domain Splitting" begin
    @testset "ADSDomain constructors" begin
        # Test ADSDomain constructors
        a = @SVector rand(Float64, 5)
        b = @SVector rand(Float64, 5)
        dom1 = ADSDomain(min.(a, b), max.(a, b))
        dom2 = ADSDomain(map((x, y) -> minmax(x, y), a, b)...)
        @test dom1 == dom2
        @test isa(dom1, ADSDomain{5, Float64}) && isa(dom2, ADSDomain{5, Float64})
        @test dom1.lo == dom2.lo == min.(a, b)
        @test dom1.hi == dom2.hi == max.(a, b)
        @test all(dom1.hi .> dom1.lo) && all(dom2.hi .> dom2.lo)
        dom3 = ADSDomain()
        @test isa(dom3, ADSDomain{1, Float64})
        @test dom3.lo == @SVector zeros(Float64, 1)
        @test dom3.hi == @SVector ones(Float64, 1)
        @test (dom3 != dom1) && (dom3 != dom2)
    end

    @testset "2D Kepler problem" begin
        using TaylorIntegration: ADSBinaryNode, countnodes, timesvector, timeshift!

        # This example is based upon section 3 of
        # https://doi.org/10.1007/s10569-015-9618-3.

        # Dynamical function
        @taylorize function kepler_eqs!(dq, q, params, t)
            dq[1] = q[3]
            dq[2] = q[4]
            rr = ( q[1]^2 + q[2]^2 )^(3/2)
            dq[3] = - q[1] / rr
            dq[4] = - q[2] / rr
        end;
        # Initial conditons (plain)
        q00 = [1.0, 0.0, 0.0, sqrt(1.5)]
        # Jet transport variables
        dq = set_variables("dx dy", numvars = 2, order = 5)
        # Initial conditions (jet transport)
        q0_ = q00 .+ [0.008, 0.08, 0.0, 0.0] .* vcat(dq, dq)
        q0 = SVector{4}(q0_)
        # Jet transport domain
        dom = ADSDomain((-1.0, 1.0), (-1.0, 1.0))
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
        _ = taylorinteg(kepler_eqs!, q0, dom, t0, tmax, order, stol, abstol, params;
                        maxsplits = 1, maxsteps = 1, parse_eqs = false);
        # Full integration
        nv1 = taylorinteg(kepler_eqs!, q0, dom, t0, tmax, order, stol, abstol, params;
                        maxsplits = maxsplits, maxsteps = maxsteps, parse_eqs = false);

        @test isa(nv1, ADSBinaryNode{2, 4, Float64})
        @test nv1.s == dom
        @test iszero(nv1.t)
        @test nv1.x == q0
        @test nv1.p == Taylor1.(q0, order)
        @test iszero(nv1.depth)
        @test isnothing(nv1.parent)
        @test isa(nv1.left, ADSBinaryNode{2, 4, Float64})
        @test isnothing(nv1.right)

        ts1 = timesvector(nv1)
        @test ts1[1] == t0
        @test ts1[end] == tmax
        @test length(ts1) == 85

        @test isone(countnodes(nv1, 0))
        @test countnodes(nv1, length(ts1)-1) == maxsplits
        @test iszero(countnodes(nv1, length(ts1)))
        @test iszero(countnodes(nv1, prevfloat(t0)))
        @test countnodes(nv1, t0) == 1
        @test countnodes(nv1, tmax) == maxsplits
        @test iszero(countnodes(nv1, nextfloat(tmax)))

        # ADS taylorinteg (parse_eqs = true)
        # Warmup
        _ = taylorinteg(kepler_eqs!, q0, dom, t0, tmax, order, stol, abstol, params;
                        maxsplits = 1, maxsteps = 1, parse_eqs = true);
        # Full integration
        nv2 = taylorinteg(kepler_eqs!, q0, dom, t0, tmax, order, stol, abstol, params;
                        maxsplits = maxsplits, maxsteps = maxsteps, parse_eqs = true);

        @test isa(nv2, ADSBinaryNode{2, 4, Float64})
        @test nv2.s == dom
        @test iszero(nv2.t)
        @test nv2.x == q0
        @test nv2.p == Taylor1.(q0, order)
        @test iszero(nv2.depth)
        @test isnothing(nv2.parent)
        @test isa(nv2.left, ADSBinaryNode{2, 4, Float64})
        @test isnothing(nv2.right)

        ts2 = timesvector(nv2)
        @test ts2[1] == t0
        @test ts2[end] == tmax
        @test length(ts2) == 85

        @test isone(countnodes(nv2, 0))
        @test countnodes(nv2, length(ts2)-1) == maxsplits
        @test iszero(countnodes(nv2, length(ts2)))
        @test iszero(countnodes(nv2, prevfloat(t0)))
        @test countnodes(nv2, t0) == 1
        @test countnodes(nv2, tmax) == maxsplits
        @test iszero(countnodes(nv2, nextfloat(tmax)))

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

        timeshift!(nv1, 1.0)
        timeshift!(nv2, 1.0)
        _ts1_ = timesvector(nv1)
        _ts2_ = timesvector(nv2)
        @test _ts1_ == ts1 .+ 1.0
        @test _ts2_ == ts2 .+ 1.0
    end
end