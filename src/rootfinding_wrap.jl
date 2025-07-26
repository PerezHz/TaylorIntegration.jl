function findroot_wrap!(
    bc!,
    t,
    x,
    dx,
    g_tupl_old,
    g_tupl,
    eventorder,
    tvS,
    xvS,
    gvS,
    t0,
    δt_old,
    x_dx,
    x_dx_val,
    g_dg,
    g_dg_val,
    nrabstol,
    newtoniter,
    nevents,
)

    if surfacecrossing(g_tupl_old, g_tupl, eventorder)
        #auxiliary variables
        g_val = g_tupl[2]
        g_val_old = g_tupl_old[2]
        nriter = 1
        dof = length(x)

        #first guess: linear interpolation
        slope = (g_val[eventorder] - g_val_old[eventorder]) / δt_old
        dt_li = -(g_val[eventorder] / slope)

        x_dx[1:dof] = x
        x_dx[dof+1:2dof] = dx
        g_dg[1] = derivative(g_val, eventorder)
        g_dg[2] = derivative(g_dg[1])

        #Newton-Raphson iterations
        dt_nr = dt_li
        evaluate!(g_dg, dt_nr, view(g_dg_val, :))

        while nrconvergencecriterion(g_dg_val[1], nrabstol, nriter, newtoniter)
            dt_nr = dt_nr - g_dg_val[1] / g_dg_val[2]
            evaluate!(g_dg, dt_nr, view(g_dg_val, :))
            nriter += 1
        end
        nriter == newtoniter + 1 && @warn("""
          Newton-Raphson did not converge for prescribed tolerance and maximum allowed iterations.
          """)
        evaluate!(x_dx, dt_nr, view(x_dx_val, :))

        tvS[nevents] = t0 + dt_nr
        xvS[:, nevents] .= view(x_dx_val, 1:dof)

        bc!(view(xvS[:, nevents]), params, tvS[nevents])

        gvS[nevents] = g_dg_val[1]

        nevents += 1
    end

    return nevents
end




function taylorinteg_wrap(
    f!,
    bc!,
    g,
    q0::AbstractVector{U},
    t0::T,
    tmax::T,
    order::Int,
    abstol::T,
    params = nothing;
    maxsteps::Int = 500,
    parse_eqs::Bool = true,
    dense::Bool = true,
    eventorder::Int = 0,
    newtoniter::Int = 10,
    nrabstol::T = eps(T),
) where {T<:Real,U<:Number}

    @assert order ≥ eventorder "`eventorder` must be less than or equal to `order`"

    # Allocation
    cache = init_cache(Val(dense), t0, q0, maxsteps, order, f!, params; parse_eqs)

    return taylorinteg_wrap!(
        Val(dense),
        f!,
        bc!,
        g,
        q0,
        t0,
        tmax,
        abstol,
        cache,
        params;
        maxsteps,
        eventorder,
        newtoniter,
        nrabstol,
    )
end

function taylorinteg_wrap!(
    dense::Val{D},
    f!,
    bc!,
    g,
    q0::Array{U,1},
    t0::T,
    tmax::T,
    abstol::T,
    cache::VectorCache,
    params;
    maxsteps::Int = 500,
    eventorder::Int = 0,
    newtoniter::Int = 10,
    nrabstol::T = eps(T),
) where {T<:Real,U<:Number,D}

    @unpack tv, xv, psol, xaux, t, x, dx, rv, parse_eqs = cache

    # Initial conditions
    x0 = deepcopy(q0)
    bc!(x0, params, t0)
    update!(cache, t0, x0)
    @inbounds tv[1] = t0
    @inbounds xv[:, 1] .= deepcopy(q0)
    sign_tstep = copysign(1, tmax - t0)

    # Some auxiliary arrays for root-finding/event detection/Poincaré surface of section evaluation
    g_tupl = g(dx, x, params, t)
    g_tupl_old = g(dx, x, params, t)
    δt = zero(x[1])
    δt_old = zero(x[1])

    x_dx = vcat(x, dx)
    g_dg = vcat(g_tupl[2], g_tupl_old[2])
    x_dx_val = Array{U}(undef, length(x_dx))
    g_dg_val = vcat(evaluate(g_tupl[2]), evaluate(g_tupl_old[2]))

    tvS = Array{U}(undef, maxsteps + 1)
    xvS = similar(xv)
    gvS = similar(tvS)

    # Integration
    nsteps = 1
    nevents = 1 #number of detected events
    while sign_tstep * t0 < sign_tstep * tmax
        δt_old = δt
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep * (tmax - t0))
        evaluate!(x, δt, x0) # new initial condition
        set_psol!(dense, psol, nsteps, x) # Store the Taylor polynomial solution
        g_tupl = g(dx, x, params, t)
        nevents = findroot!(
            t,
            x,
            dx,
            g_tupl_old,
            g_tupl,
            eventorder,
            tvS,
            xvS,
            gvS,
            t0,
            δt_old,
            x_dx,
            x_dx_val,
            g_dg,
            g_dg_val,
            nrabstol,
            newtoniter,
            nevents,
        )
        g_tupl_old = deepcopy(g_tupl)
        t0 += δt
        bc!(x0, params, t0)
        update!(cache, t0, x0)
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds xv[:, nsteps] .= deepcopy(x0)
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return build_solution(tv, xv, psol, tvS, xvS, gvS, nsteps, nevents)
end

function taylorinteg_wrap!(
    f!,
    bc!,
    g,
    q0::AbstractVector{U},
    trange::AbstractVector{T},
    order::Int,
    abstol::T,
    params = nothing;
    maxsteps::Int = 500,
    parse_eqs::Bool = true,
    eventorder::Int = 0,
    newtoniter::Int = 10,
    nrabstol::T = eps(T),
) where {T<:Real,U<:Number}

    @assert order ≥ eventorder "`eventorder` must be less than or equal to `order`"

    # Check if trange is increasingly or decreasingly sorted
    @assert (issorted(trange) || issorted(trange, rev = true)) "`trange` or `reverse(trange)` must be sorted"

    # Allocation
    cache = init_cache(Val(false), trange, q0, maxsteps, order, f!, params; parse_eqs)

    return taylorinteg_wrap!(
        f!,
        bc!,
        g,
        q0,
        trange,
        abstol,
        cache,
        params;
        maxsteps,
        eventorder,
        newtoniter,
        nrabstol,
    )
end

function taylorinteg_wrap!(
    f!,
    bc!,
    g,
    q0::AbstractVector{U},
    trange::AbstractVector{T},
    abstol::T,
    cache::VectorTRangeCache,
    params;
    maxsteps::Int = 500,
    eventorder::Int = 0,
    newtoniter::Int = 10,
    nrabstol::T = eps(T),
) where {T<:Real,U<:Number}

    @unpack tv, xv, xaux, x0, x1, t, x, dx, rv, parse_eqs = cache

    # Initial conditions
    @inbounds t0, t1, tmax = trange[1], trange[2], trange[end]
    sign_tstep = copysign(1, tmax - t0)
    @inbounds x0 .= deepcopy(q0)
    bc!(x0, params, t0)
    update!(cache, t0, x0)
    @inbounds xv[:, 1] .= deepcopy(q0)

    # Some auxiliary arrays for root-finding/event detection/Poincaré surface of section evaluation
    g_tupl = g(dx, x, params, t)
    g_tupl_old = g(dx, x, params, t)
    δt = zero(U)
    δt_old = zero(U)

    x_dx = vcat(x, dx)
    g_dg = vcat(g_tupl[2], g_tupl_old[2])
    x_dx_val = Array{U}(undef, length(x_dx))
    g_dg_val = vcat(evaluate(g_tupl[2]), evaluate(g_tupl_old[2]))

    tvS = Array{U}(undef, maxsteps + 1)
    xvS = similar(xv)
    gvS = similar(tvS)

    # Integration
    iter = 2
    nsteps = 1
    nevents = 1 #number of detected events
    while sign_tstep * t0 < sign_tstep * tmax
        δt_old = δt
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep * (tmax - t0))
        evaluate!(x, δt, x0) # new initial condition
        tnext = t0 + δt
        # Evaluate solution at times within convergence radius
        while sign_tstep * t1 < sign_tstep * tnext
            evaluate!(x, t1 - t0, x1)
            bc!(x1, params, t1)
            @inbounds xv[:, iter] .= deepcopy(x1)
            iter += 1
            @inbounds t1 = trange[iter]
        end
        if δt == tmax - t0
            bc!(x0, params, tmax)
            @inbounds xv[:, iter] .= deepcopy(x0)
            break
        end
        g_tupl = g(dx, x, params, t)
        nevents = findroot!(
            t,
            x,
            dx,
            g_tupl_old,
            g_tupl,
            eventorder,
            tvS,
            xvS,
            gvS,
            t0,
            δt_old,
            x_dx,
            x_dx_val,
            g_dg,
            g_dg_val,
            nrabstol,
            newtoniter,
            nevents,
        )
        g_tupl_old = deepcopy(g_tupl)
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

    return build_solution(trange, xv, tvS, xvS, gvS, nevents)
end