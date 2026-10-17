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

# Scalar problem: x' = 3 - x^2
@taylorize perf_riccati(x, p, t) = 3.0 - x^2

# Number of allocations (after a warm-up) and of steps of `taylorinteg(f, ...; tf)`
function perf_count_steps(call, tf)
    call(tf)                                    # warm-up
    a = @allocations call(tf)
    return a, length(call(tf).t) - 1
end

# Pendulum with an event function which does not allocate (it returns an element of the state)
@taylorize function perf_pendulum!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = -sin(x[1])
    return nothing
end
perf_noevent(dx, x, p, t) = (true, x[1])

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

    # Scalar integration: the same property
    @testset "No allocations per step (parsed methods, scalar)" begin
        scalar(tf) = taylorinteg(perf_riccati, 1.0, 0.0, tf, 28, 1.0e-20;
            maxsteps = 1_000, parse_eqs = true, dense = false)
        a1, n1 = perf_count_steps(scalar, 3.0)
        a2, n2 = perf_count_steps(scalar, 10.0)
        @test n2 > n1
        @test a2 == a1
    end

    # Root finding without events (a rotating pendulum: `x[1]` never vanishes). The only allocation
    # per step is the copy of the polynomial of the event function (`_stored_event_tuple`), which
    # avoids aliasing; this test guards that nothing else allocates in the bookkeeping of each step.
    @testset "Root finding: at most one allocation per step (no events)" begin
        roots(tf) = taylorinteg(perf_pendulum!, perf_noevent, [1.0, 3.0], 0.0, tf, 28, 1.0e-20;
            maxsteps = 10_000, parse_eqs = true, dense = false)
        a1, n1 = perf_count_steps(roots, 50.0)
        a2, n2 = perf_count_steps(roots, 100.0)
        @test n2 > n1 + 50
        @test a2 - a1 <= n2 - n1
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