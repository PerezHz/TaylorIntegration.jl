

struct VectorCachePS{XV,XAUX,T,X,DX,RV,PARSE_EQS} <: AbstractVectorCache
    xv::XV
    xaux::XAUX
    t::T
    x::X
    dx::DX
    rv::RV
    parse_eqs::PARSE_EQS
end

function init_cache_ps(
    t0::T,
    q0::Vector{U},
    maxevents::Int,
    order::Int,
    f!,
    params = nothing;
    parse_eqs::Bool = true,
) where {U,T}
    # Initialize the vector of Taylor1 expansions
    t, x, dx = init_expansions(t0, q0, order)
    # Determine if specialized jetcoeffs! method exists
    parse_eqs, rv = _determine_parsing!(parse_eqs, f!, t, x, dx, params)
    # Initialize cache
    dof = length(q0)
    return VectorCachePS(
        Array{U}(undef, dof, maxevents + 1),
        Array{Taylor1{U}}(undef, dof),
        t,
        x,
        dx,
        rv,
        parse_eqs,
    )
end

function findroot_ps!(
    bc!,
    params,
    t0,
    x,
    dx,
    g_tupl_old,
    g_tupl,
    eventorder,
    xvS,
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
        
        if nriter <= newtoniter

            evaluate!(x_dx, dt_nr, view(x_dx_val, :))

            bc!(x_dx_val, params, t0 + dt_nr)

            xvS[:, nevents] .= view(x_dx_val, 1:dof)

            nevents += 1

        end
    end

    return nevents
end


function taylorinteg_ps!(
    f!,
    bc!,
    g,
    q0::Array{U,1},
    t0::T,
    tmax::T,
    abstol::T,
    cache::VectorCachePS,
    params;
    maxsteps::Int = 500,
    maxevents::Int = 500,
    eventorder::Int = 0,
    newtoniter::Int = 10,
    nrabstol::T = eps(T),
) where {T<:Real,U<:Number}

    @unpack xv, xaux, t, x, dx, rv, parse_eqs = cache

    x0 = deepcopy(q0)
    update!(cache, t0, x0)
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

    # Integration
    nsteps = 1
    nevents = 1 #number of detected events
    while sign_tstep * t0 < sign_tstep * tmax
        δt_old = δt
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep * (tmax - t0))
        evaluate!(x, δt, x0) # new initial condition
        g_tupl = g(dx, x, params, t)
        nevents = findroot_ps!(
            bc!,
            params,
            t0,
            x,
            dx,
            g_tupl_old,
            g_tupl,
            eventorder,
            xv,
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
        if nsteps > maxsteps || nevents > maxevents
            break
        end
    end

    return nevents

end

function taylorinteg_ps!(
    f!,
    bc!,
    g,
    lims,
    q0::Array{U,1},
    t0::T,
    tmax::T,
    abstol::T,
    cache::VectorCachePS,
    params;
    maxsteps::Int = 500,
    maxevents::Int = 500,
    eventorder::Int = 0,
    newtoniter::Int = 10,
    nrabstol::T = eps(T),
) where {T<:Real,U<:Number}

    @unpack xv, xaux, t, x, dx, rv, parse_eqs = cache

    x0 = deepcopy(q0)
    update!(cache, t0, x0)
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

    # Integration
    nsteps = 1
    nevents = 1 #number of detected events
    while sign_tstep * t0 < sign_tstep * tmax
        δt_old = δt
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep * (tmax - t0))
        evaluate!(x, δt, x0) # new initial condition
        g_tupl = g(dx, x, params, t)
        nevents = findroot_ps!(
            bc!,
            params,
            t0,
            x,
            dx,
            g_tupl_old,
            g_tupl,
            eventorder,
            xv,
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
        if nsteps > maxsteps || nevents > maxevents
            break
        end
    end

    return nevents

end