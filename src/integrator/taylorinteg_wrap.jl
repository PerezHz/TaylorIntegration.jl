function taylorinteg_wrap(
    f!,
    bc!,
    q0::AbstractVector{U},
    t0::T,
    tmax::T,
    order::Int,
    abstol::T,
    params = nothing;
    maxsteps::Int = 500,
    parse_eqs::Bool = true,
    dense::Bool = true,
) where {T<:Real,U<:Number}

    # Allocation
    cache = init_cache(Val(dense), t0, q0, maxsteps, order, f!, params; parse_eqs)

    return taylorinteg_wrap!(Val(dense), f!, bc!, q0, t0, tmax, abstol, cache, params; maxsteps)
end

function taylorinteg_wrap!(
    dense::Val{D},
    f!,
    bc!,
    q0::AbstractVector{U},
    t0::T,
    tmax::T,
    abstol::T,
    cache::VectorCache,
    params;
    maxsteps::Int = 500,
) where {T<:Real,U<:Number,D}

    @unpack tv, xv, psol, xaux, t, x, dx, rv, parse_eqs = cache

    # Initial conditions
    x0 = deepcopy(q0)
    bc!(x0, params, t0)
    update!(cache, t0, x0)
    @inbounds tv[1] = t0
    @inbounds xv[:, 1] .= q0
    sign_tstep = copysign(1, tmax - t0)

    # Integration
    nsteps = 1
    while sign_tstep * t0 < sign_tstep * tmax
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep * (tmax - t0))
        evaluate!(x, δt, x0) # new initial condition
        set_psol!(dense, psol, nsteps, x) # Store the Taylor polynomial solution
        t0 += δt
        bc!(x0, params, t0)
        update!(cache, t0, x0)
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds xv[:, nsteps] .= deepcopy.(x0)
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return build_solution(tv, xv, psol, nsteps)
end

function taylorinteg_wrap(
    f!,
    bc!,
    q0::AbstractVector{U},
    trange::AbstractVector{T},
    order::Int,
    abstol::T,
    params = nothing;
    maxsteps::Int = 500,
    parse_eqs::Bool = true,
) where {T<:Real,U<:Number}

    # Check if trange is increasingly or decreasingly sorted
    @assert (issorted(trange) || issorted(trange, rev = true)) "`trange` or `reverse(trange)` must be sorted"

    # Allocation
    cache = init_cache(Val(false), trange, q0, maxsteps, order, f!, params; parse_eqs)

    return taylorinteg_wrap!(f!, bc!, q0, trange, abstol, cache, params; maxsteps)
end

function taylorinteg_wrap!(
    f!,
    bc!,
    q0::AbstractVector{U},
    trange::AbstractVector{T},
    abstol::T,
    cache::VectorTRangeCache,
    params;
    maxsteps::Int = 500,
) where {T<:Real,U<:Number}

    @unpack xv, xaux, x0, x1, t, x, dx, rv, parse_eqs = cache

    # Initial conditions
    @inbounds t0, t1, tmax = trange[1], trange[2], trange[end]
    sign_tstep = copysign(1, tmax - t0)
    @inbounds x0 .= deepcopy(q0)
    bc!(x0, params, t0)
    update!(cache, t0, x0)
    @inbounds xv[:, 1] .= q0

    # Integration
    iter = 2
    nsteps = 1
    while sign_tstep * t0 < sign_tstep * tmax
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep * (tmax - t0))
        evaluate!(x, δt, x0) # new initial condition
        tnext = t0 + δt
        # Evaluate solution at times within convergence radius
        while sign_tstep * t1 < sign_tstep * tnext
            evaluate!(x, t1 - t0, x1)
            bc!(x1, params, t1)
            @inbounds xv[:, iter] .= x1
            iter += 1
            @inbounds t1 = trange[iter]
        end
        if δt == tmax - t0
            bc!(x0, params, tmax)
            @inbounds xv[:, iter] .= x0
            break
        end
        t0 = tnext
        bc!(x0, params, t0)
        update!(cache, t0, x0)
        nsteps += 1
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return build_solution(trange, xv)
end
