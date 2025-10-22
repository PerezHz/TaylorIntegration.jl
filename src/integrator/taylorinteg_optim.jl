# This file is part of the TaylorIntegration.jl package; MIT licensed

struct VectorCacheOptim{XAUX,T,X,DX,RV,PARSE_EQS} <: AbstractVectorCache
    xaux::XAUX
    t::T
    x::X
    dx::DX
    rv::RV
    parse_eqs::PARSE_EQS
end

function init_cache_optim(
    t0::T,
    q0::Vector{U},
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
    return VectorCacheOptim(
        Array{Taylor1{U}}(undef, dof),
        t,
        x,
        dx,
        rv,
        parse_eqs,
    )
end

function taylorinteg_optim!(
    f!,
    q0::AbstractVector{U},
    t0::T,
    tmax::T,
    abstol::T,
    cache::VectorCacheOptim,
    params;
    maxsteps::Int = 500,
) where {T<:Real,U<:Number}

    @unpack xaux, t, x, dx, rv, parse_eqs = cache

    # Initial conditions
    update!(cache, t0, q0)
    sign_tstep = copysign(1, tmax - t0)

    # Integration
    nsteps = 1
    while sign_tstep * t0 < sign_tstep * tmax
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep * (tmax - t0))
        evaluate!(x, δt, q0) # new initial condition
        t0 += δt
        update!(cache, t0, q0)
        nsteps += 1
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end
    
end

function taylorinteg_wrap_optim!(
    f!,
    bc!,
    q0::AbstractVector{U},
    t0::T,
    tmax::T,
    abstol::T,
    cache::VectorCacheOptim,
    params;
    maxsteps::Int = 500,
) where {T<:Real,U<:Number}

    @unpack xaux, t, x, dx, rv, parse_eqs = cache

    # Initial conditions
    bc!(q0, params, t0)
    update!(cache, t0, q0)
    sign_tstep = copysign(1, tmax - t0)

    # Integration
    nsteps = 1
    while sign_tstep * t0 < sign_tstep * tmax
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep * (tmax - t0))
        evaluate!(x, δt, q0) # new initial condition
        t0 += δt
        bc!(q0, params, t0)
        update!(cache, t0, q0)
        nsteps += 1
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

end


function taylorinteg_wrap_optim!(
    f!,
    bc!,
    lims,
    q0::AbstractVector{U},
    t0::T,
    tmax::T,
    abstol::T,
    cache::VectorCacheOptim,
    params;
    maxsteps::Int = 500,
) where {T<:Real,U<:Number}

    @unpack xaux, t, x, dx, rv, parse_eqs = cache

    # Initial conditions
    bc!(q0, params, t0)
    if lims(q0, params, t0)
        @warn("""
        Variable limits was exceded; exiting.
        """)
        return nothing
    end
    update!(cache, t0, q0)
    sign_tstep = copysign(1, tmax - t0)

    # Integration
    nsteps = 1
    while sign_tstep * t0 < sign_tstep * tmax
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep * (tmax - t0))
        evaluate!(x, δt, q0) # new initial condition
        t0 += δt
        bc!(q0, params, t0)
        if lims(q0, params, t0)
            @warn("""
            Variable limits was exceded; exiting.
            """)
            return nothing
        end
        update!(cache, t0, q0)
        nsteps += 1
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            return nothing
        end

    end

    return nothing

end


function taylorinteg_floquet!(
    f!,
    g!,
    bc!,
    lims,
    q0::AbstractVector{U},
    v0::AbstractVector{U},
    t0::T,
    tmax::T,
    abstol::T,
    cache::VectorCacheOptim,
    eigcache::VectorCacheOptim,
    params;
    maxsteps::Int = 500,
) where {T<:Real,U<:Number}

    @unpack xaux, t, x, dx, rv, parse_eqs = cache

    # Initial conditions
    bc!(q0, params, t0)
    update!(cache, t0, q0)
    sign_tstep = copysign(1, tmax - t0)

    # Integration
    nsteps = 1
    while sign_tstep * t0 < sign_tstep * tmax
        δt = taylorstep!(Val(parse_eqs), f!, t, x, dx, xaux, abstol, params, rv) # δt is positive!
        # Below, δt has the proper sign according to the direction of the integration
        δt = sign_tstep * min(δt, sign_tstep * (tmax - t0))
        taylorinteg_optim!(
                            g!,
                            v0,
                            zero(T),
                            δt,
                            abstol,
                            eigcache,
                            params;
                            maxsteps = maxsteps,
                          )
        evaluate!(x, δt, q0) # new initial condition
        t0 += δt
        if lims(q0, params, t0)
            @warn("""
            Variable limits was exceded; exiting.
            """)
            break
        end
        bc!(q0, params, t0)
        update!(cache, t0, q0)
        nsteps += 1
        if nsteps > maxsteps
            @warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end

    end

end




