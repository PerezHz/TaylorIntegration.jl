# This file is part of the TaylorIntegration.jl package; MIT licensed

using TaylorIntegration
using Test
using InteractiveUtils: methodswith

# Functions are defined at top level (not inside a `@testset`), and with
# distinctive names, to avoid any clash with the other test files.
@taylorize function perf_kepler1!(dq, q, p, t)
    local μ = p
    r_p3d2 = (q[1]^2 + q[2]^2)^1.5
    dq[1] = q[3]
    dq[2] = q[4]
    dq[3] = μ * q[1] / r_p3d2
    dq[4] = μ * q[2] / r_p3d2
    return nothing
end

@taylorize function perf_kepler2!(dq, q, p, t)
    nn, μ = p
    r2 = zero(q[1])
    for i = 1:nn
        r2_aux = r2 + q[i]^2
        r2 = r2_aux
    end
    r_p3d2 = r2^(3 / 2)
    for j = 1:nn
        dq[j] = q[nn+j]
        dq[nn+j] = μ * q[j] / r_p3d2
    end
    nothing
end

@taylorize function perf_kepler5!(dq, q, p, t)
    local μ = -1.0
    r = sqrt(q[1]^2 + q[2]^2)
    r_p3d2 = r^3
    dq[1] = q[3]
    dq[2] = q[4]
    dq[3] = μ * q[1] / r_p3d2
    dq[4] = μ * q[2] / r_p3d2
    return nothing
end

# Function barriers: the integrators are called with concrete types, and
# `@allocated` measures only the (already compiled) integration.
function perf_integ!(f!, q0, t0, tf, abstol, cache, params, maxsteps)
    return TaylorIntegration.taylorinteg!(
        Val(false), f!, q0, t0, tf, abstol, cache, params;
        maxsteps, copy_solution = Val(false),
    )
end

function perf_alloc_steps(f!, q0, t0, tf, order, abstol, params, maxsteps)
    cache = TaylorIntegration.init_cache(Val(false), t0, q0, maxsteps, order, f!, params)
    sol = perf_integ!(f!, q0, t0, tf, abstol, cache, params, maxsteps)  # warm-up
    nsteps = length(sol.t) - 1
    a = @allocated perf_integ!(f!, q0, t0, tf, abstol, cache, params, maxsteps)
    return a, nsteps, cache.parse_eqs
end

function perf_integ_trange!(f!, q0, trange, abstol, cache, params, maxsteps)
    return TaylorIntegration.taylorinteg!(
        f!, q0, trange, abstol, cache, params;
        maxsteps, copy_solution = Val(false),
    )
end

function perf_alloc_trange(f!, q0, trange, order, abstol, params, maxsteps)
    cache = TaylorIntegration.init_cache(Val(false), trange, q0, maxsteps, order, f!, params)
    perf_integ_trange!(f!, q0, trange, abstol, cache, params, maxsteps)  # warm-up
    a = @allocated perf_integ_trange!(f!, q0, trange, abstol, cache, params, maxsteps)
    return a, cache.parse_eqs
end

@testset "Testing performance-related properties" begin

    q0 = [0.2, 0.0, 0.0, 3.0]
    t0 = 0.0
    tf = 2π * 5            # ~5 periods; `2tf` takes about twice as many steps
    order = 28
    abstol = 1.0e-20
    maxsteps = 2_000
    cases = (
        (perf_kepler1!, -1.0),         # `params` is a `Float64`
        (perf_kepler2!, (2, -1.0)),    # `params` is a `Tuple`
        (perf_kepler5!, nothing),      # no `params`
    )

    # With the parsed method of `jetcoeffs!`, the integration loop must not
    # allocate: the allocations may not depend on the number of steps. (A runtime
    # dispatch in the loop allocates at each step, since it boxes the arguments.)
    @testset "No allocations per step (parsed methods)" begin
        for (f!, params) in cases
            a1, n1, pe1 = perf_alloc_steps(f!, q0, t0, tf, order, abstol, params, maxsteps)
            a2, n2, pe2 = perf_alloc_steps(f!, q0, t0, 2tf, order, abstol, params, maxsteps)
            @test pe1 === Val(true)
            @test pe2 === Val(true)
            @test n2 > n1 + 100
            @test a2 == a1
        end
    end

    # Same, for the integration at the times of a `trange`
    @testset "No allocations per step (parsed methods, trange)" begin
        for (f!, params) in cases
            trange1 = collect(range(t0, tf, length = 101))
            trange2 = collect(range(t0, 2tf, length = 201))
            a1, pe1 = perf_alloc_trange(f!, q0, trange1, order, abstol, params, maxsteps)
            a2, pe2 = perf_alloc_trange(f!, q0, trange2, order, abstol, params, maxsteps)
            @test pe1 === Val(true)
            @test pe2 === Val(true)
            @test a2 == a1
        end
    end

    # `_has_parsed_method` must agree with the (slower) `methodswith`
    @testset "_has_parsed_method" begin
        for f in (perf_kepler1!, perf_kepler2!, perf_kepler5!, sin, +)
            for g in (TaylorIntegration.jetcoeffs!, TaylorIntegration._allocate_jetcoeffs!)
                @test TaylorIntegration._has_parsed_method(g, f) ==
                    !isempty(methodswith(Val{f}, g))
            end
        end
        @test TaylorIntegration._has_parsed_method(TaylorIntegration.jetcoeffs!, perf_kepler1!)
        @test !TaylorIntegration._has_parsed_method(TaylorIntegration.jetcoeffs!, sin)
    end
end