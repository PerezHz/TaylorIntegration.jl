using TaylorIntegration
using Test

@testset "Automatic Domain Splitting" begin
    using TaylorIntegration: ADSBinaryNode, countnodes, timesvector, evaltree
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
    local dq = set_variables("dx dy", numvars = 2, order = 14)
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
    local stol = 3e-4
    # Absolute tolerance
    local abstol = 1e-20
    # Dynamical function parameters
    local params = nothing
    # Maximum allowed splits
    local maxsplits = 15
    # Maximum allowed steps
    local maxsteps = 1_000
    # Use taylorized kepler_eqs!
    local parse_eqs = true

    # Warmup
    nv = taylorinteg(kepler_eqs!, q0, dom, t0, tmax, order, stol, abstol, params;
                     maxsplits = 1, maxsteps, parse_eqs);
    # ADS taylorinteg
    nv = taylorinteg(kepler_eqs!, q0, dom, t0, tmax, order, stol, abstol, params;
                     maxsplits, maxsteps, parse_eqs);

    @test isa(nv, ADSBinaryNode{2, 4, Float64})
    @test nv.s == dom
    @test iszero(nv.t)
    @test nv.x == q0
    @test nv.p == Taylor1.(q0, order)
    @test iszero(nv.depth)
    @test isnothing(nv.parent)
    @test isa(nv.left, ADSBinaryNode{2, 4, Float64})
    @test isnothing(nv.right) 

    @test isone(countnodes(nv, 0))
    @test countnodes(nv, 91) == maxsplits
    @test iszero(countnodes(nv, 92))
    @test iszero(countnodes(nv, prevfloat(t0)))
    @test isone(countnodes(nv, t0))
    @test countnodes(nv, tmax) == maxsplits
    @test iszero(countnodes(nv, nextfloat(tmax)))

    ts = timesvector(nv)
    @test ts[1] == t0
    @test ts[end] == tmax
    @test length(ts) == 92

    s, x = evaltree(nv, t0)
    @test length(s) == 1
    @test s[1] == dom
    @test size(x) == (4, 1)
    @test x[:, 1] == q0
    s, x = evaltree(nv, tmax)
    @test length(s) == maxsplits
    @test size(x) == (4, maxsplits)

    y = evaltree(nv, t0, SVector(0.1, 0.1))
    @test y == [1.0008, 0.008, 0.0, 1.224744871391589]
    y = evaltree(nv, tmax, SVector(0.1, 0.1))
    @test all(isapprox.(
        y,
        [0.6683166200240993, -0.9479380052802993, 0.6733099139393155, 0.8790273308917417];
        atol = 1e-4
    ))
end