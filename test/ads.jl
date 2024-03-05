using TaylorIntegration
using Test

@testset "Automatic Domain Splitting" begin
    using TaylorIntegration: ADSBinaryNode, countnodes, timesvector
    using StaticArrays: SVector

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
    local q00 = [1.0, 0.0, 0.0, sqrt(1.5)]
    # Jet transport variables
    local dq = set_variables("dx dy", numvars = 2, order = 5)
    # Initial conditions (jet transport)
    local q0_ = q00 .+ [0.008, 0.08, 0.0, 0.0] .* vcat(dq, dq)
    local q0 = SVector{4}(q0_)
    # Jet transport domain
    local dom = ADSDomain((-1.0, 1.0), (-1.0, 1.0))
    # Initial time
    local t0 = 0.0
    # Final time
    local tmax = 34.85
    # Taylor1 order
    local order = 25
    # Splitting tolerance
    local stol = 1e-5
    # Absolute tolerance
    local abstol = 1e-20
    # Dynamical function parameters
    local params = nothing
    # Maximum allowed splits
    local maxsplits = 25
    # Maximum allowed steps
    local maxsteps = 1_000

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
    @test isa(nv1.right, ADSBinaryNode{2, 4, Float64})

    ts1 = timesvector(nv1)
    @test ts1[1] == t0
    @test ts1[end] == tmax
    @test length(ts1) == 87

    @test isone(countnodes(nv1, 0))
    @test countnodes(nv1, length(ts1)-1) == maxsplits
    @test iszero(countnodes(nv1, length(ts1)))
    @test iszero(countnodes(nv1, prevfloat(t0)))
    @test countnodes(nv1, t0) == 2
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
    @test isa(nv2.right, ADSBinaryNode{2, 4, Float64})

    ts2 = timesvector(nv2)
    @test ts2[1] == t0
    @test ts2[end] == tmax
    @test length(ts2) == 87

    @test isone(countnodes(nv2, 0))
    @test countnodes(nv2, length(ts2)-1) == maxsplits
    @test iszero(countnodes(nv2, length(ts2)))
    @test iszero(countnodes(nv2, prevfloat(t0)))
    @test countnodes(nv2, t0) == 2
    @test countnodes(nv2, tmax) == maxsplits
    @test iszero(countnodes(nv2, nextfloat(tmax)))

    @test ts1 == ts2

    s1, x1 = nv1(t0)
    s2, x2 = nv2(t0)
    @test length(s1) == length(s2) == 2
    @test s1 == s2
    @test size(x1) == size(x2) == (4, 2)
    @test x1 == x2

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
end