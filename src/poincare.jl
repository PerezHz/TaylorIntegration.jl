function taylorinteg{T<:Number}(f!, g, q0::Array{T,1}, t0::T, tmax::T,
        order::Int, abstol::T; maxsteps::Int=500)

    # Allocation
    const tv = Array{T}(maxsteps+1)
    dof = length(q0)
    const xv = Array{T}(dof, maxsteps+1)

    # Initialize the vector of Taylor1 expansions
    const x = Array{Taylor1{T}}(dof)
    const dx = Array{Taylor1{T}}(dof)
    const xaux = Array{Taylor1{T}}(dof)
    for i in eachindex(q0)
        @inbounds x[i] = Taylor1( q0[i], order )
    end

    # Initial conditions
    @inbounds tv[1] = t0
    @inbounds xv[:,1] .= q0
    x0 = copy(q0)

    const g_val = Taylor1( zero(T), order )
    const g_val_old = Taylor1( zero(T), order )

    # Integration
    nsteps = 1
    while t0 < tmax
        g_val_old = g_val
        δt = taylorstep!(f!, x, dx, xaux, t0, tmax, x0, order, abstol)
        g_val = g(stateT).coeffs[2]
        if g_val_old[1]*g_val[1] < zero(T)

            println("g_val_old= ", g_val_old)
            println("g_val= ", g_val)

        end
        for i in eachindex(x0)
            @inbounds x[i].coeffs[1] = x0[i]
        end
        t0 += δt
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds xv[:,nsteps] .= x0
        if nsteps > maxsteps
            warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return view(tv,1:nsteps), view(transpose(xv),1:nsteps,:)
end

function taylorinteg{T<:Number}(f!, g, q0::Array{T,1}, trange::Range{T},
        order::Int, abstol::T; maxsteps::Int=500)

    # Allocation
    nn = length(trange)
    dof = length(q0)
    const x0 = similar(q0, T, dof)
    fill!(x0, T(NaN))
    const xv = Array{eltype(q0)}(dof, nn)
    for ind in 1:nn
        @inbounds xv[:,ind] .= x0
    end

    # Initialize the vector of Taylor1 expansions
    const x = Array{Taylor1{T}}(dof)
    const dx = Array{Taylor1{T}}(dof)
    const xaux = Array{Taylor1{T}}(dof)
    for i in eachindex(q0)
        @inbounds x[i] = Taylor1( q0[i], order )
    end

    # Initial conditions
    @inbounds x0 .= q0
    @inbounds xv[:,1] .= q0

    # Integration
    iter = 1
    while iter < nn
        t0, t1 = trange[iter], trange[iter+1]
        nsteps = 0
        while nsteps < maxsteps
            δt = taylorstep!(f!, x, dx, xaux, t0, t1, x0, order, abstol)
            for i in eachindex(x0)
                @inbounds x[i].coeffs[1] = x0[i]
            end
            t0 += δt
            t0 ≥ t1 && break
            nsteps += 1
        end
        if nsteps ≥ maxsteps && t0 != t1
            warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
        iter += 1
        @inbounds xv[:,iter] .= x0
    end

    return transpose(xv)
end
