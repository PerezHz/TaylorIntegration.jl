function poincare{T<:Number}(f!, g, q0::Array{T,1}, t0::T, tmax::T,
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

    #Some auxiliary arrays for root-finding/event detection/Poincaré surface of section evaluation
    const g_val = Taylor1(zero(T), order)
    const g_val_old = Taylor1(zero(T), order)
    const slope = zero(T)
    const dt_li = zero(T)
    const dt_nr = zero(T)
    const δt = zero(T)
    const δt_old = zero(T)

    const x_g_Dg_D2g = vcat(x, dx, zero(x[1]), zero(x[1]))
    const x_g_Dg_D2g_val = Array{T}( length(x_g_Dg_D2g) )

    const tvS = similar(tv)
    const xvS = similar(xv)
    const gvS = similar(tv)

    # Integration
    nsteps = 1
    nevents = 1 #number of detected events
    while t0 < tmax
        δt_old = δt
        δt = taylorstep!(f!, x, dx, xaux, t0, tmax, x0, order, abstol)
        g_val = g(t0, x, dx)
        if g_val_old[1]*g_val[1] < zero(T)
            # println("* * * begin poincare")
            #
            # println("g_val_old= ", g_val_old[1])
            # println("g_val= ", g_val[1])
            # println("g_val_old[1]*g_val[1] =", g_val_old[1]*g_val[1])
            # println("g_val_old[1]*g_val[1] < zero(T) --> ", g_val_old[1]*g_val[1] < zero(T))

            #first guess: linear interpolation
            slope = (g_val[1]-g_val_old[1])/δt_old
            dt_li = -(g_val[1]/slope)

            # println("slope = ", slope)
            # println("dt_li = ", dt_li)

            x_g_Dg_D2g[1:dof] = x
            x_g_Dg_D2g[dof+1:2dof] = dx
            x_g_Dg_D2g[2dof+1] = g_val
            x_g_Dg_D2g[2dof+2] = derivative(g_val)

            #Newton-Raphson iterations
            dt_nr = dt_li
            # println(" * dt_nr0 = ", dt_nr)
            evaluate!(x_g_Dg_D2g[2dof+1:2dof+2], dt_nr, view(x_g_Dg_D2g_val,2dof+1:2dof+2))

            for i in 1:4
                dt_nr = dt_nr-x_g_Dg_D2g_val[2dof+1]/x_g_Dg_D2g_val[2dof+2]
                # println("i = ", i, ", dt_nr = ", dt_nr)
                evaluate!(x_g_Dg_D2g[2dof+1:2dof+2], dt_nr, view(x_g_Dg_D2g_val,2dof+1:2dof+2))
            end
            evaluate!(x_g_Dg_D2g[1:2dof], dt_nr, view(x_g_Dg_D2g_val,1:2dof))

            tvS[nevents] = t0+dt_nr
            xvS[:,nevents] .= view(x_g_Dg_D2g_val,1:dof)
            gvS[nevents] = x_g_Dg_D2g_val[2dof+1] # g(t0+dt_nr,view(x_g_Dg_D2g_val,1:dof),view(x_g_Dg_D2g_val,dof+1:2dof))

            nevents += 1

            #println("g(t*,x*,dx*) = ", g(t0+dt_nr,view(x_g_Dg_D2g_val,1:dof),view(x_g_Dg_D2g_val,dof+1:2dof)))
            # println("x* = ", x_g_Dg_D2g_val[1:dof])
            # println("dx* = ", x_g_Dg_D2g_val[dof+1:2dof], "\n")
            #
            # println("δt = ", δt)
            # println("-δt_old = ", -δt_old)
            # println("* * * end poincare")
        end
        g_val_old = deepcopy(g_val)
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

    return view(tv,1:nsteps), view(transpose(xv),1:nsteps,:), view(tvS,1:nevents-1), view(transpose(xvS),1:nevents-1,:), view(gvS,1:nevents-1)
end

#Poincare + jet transport
function poincare{T<:Number}(f!, g, q0::Array{TaylorN{T},1}, t0::T, tmax::T,
        order::Int, abstol::T; maxsteps::Int=500)

    # Allocation
    const tv = Array{T}(maxsteps+1)
    dof = length(q0)
    const xv = Array{TaylorN{T}}(dof, maxsteps+1)

    # Initialize the vector of Taylor1 expansions
    const x = Array{Taylor1{TaylorN{T}}}(dof)
    const dx = Array{Taylor1{TaylorN{T}}}(dof)
    const xaux = Array{Taylor1{TaylorN{T}}}(dof)
    for i in eachindex(q0)
        @inbounds x[i] = Taylor1( q0[i], order )
    end

    # Initial conditions
    @inbounds tv[1] = t0
    @inbounds xv[:,1] .= q0
    x0 = copy(q0)

    #Some auxiliary arrays for root-finding/event detection/Poincaré surface of section evaluation
    zeroTN = zero(q0[1])
    const g_val = Taylor1(zeroTN, order)
    const g_val_old = Taylor1(zeroTN, order)
    const slope = zeroTN
    const dt_li = zeroTN
    const dt_nr = zeroTN
    const δt = zeroTN
    const δt_old = zeroTN

    const x_g_Dg_D2g = vcat(x, dx, zero(x[1]), zero(x[1]))
    const x_g_Dg_D2g_val = Array{TaylorN{T}}( length(x_g_Dg_D2g) )

    const tvS = Array{TaylorN{T}}( length(tv) )
    const xvS = similar(xv)
    const gvS = similar(tvS)

    # Integration
    nsteps = 1
    nevents = 1 #number of detected events
    while t0 < tmax
        δt_old = δt
        δt = taylorstep!(f!, x, dx, xaux, t0, tmax, x0, order, abstol)
        g_val = g(t0, x, dx)
        if g_val_old[1][1][1]*g_val[1][1][1] < zero(T)
            # println("* * * begin poincare")
            #
            # println("g_val_old= ", g_val_old[1])
            # println("g_val= ", g_val[1])
            # println("g_val_old[1]*g_val[1] =", g_val_old[1]*g_val[1])
            # println("g_val_old[1]*g_val[1] < zero(T) --> ", g_val_old[1]*g_val[1] < zero(T))

            #first guess: linear interpolation
            slope = (g_val[1]-g_val_old[1])/δt_old
            dt_li = -(g_val[1]/slope)

            # println("slope = ", slope)
            # println("dt_li = ", dt_li)

            x_g_Dg_D2g[1:dof] = x
            x_g_Dg_D2g[dof+1:2dof] = dx
            x_g_Dg_D2g[2dof+1] = g_val
            x_g_Dg_D2g[2dof+2] = derivative(g_val)

            #Newton-Raphson iterations
            dt_nr = dt_li
            # println(" * dt_nr0 = ", dt_nr)
            evaluate!(x_g_Dg_D2g[2dof+1:2dof+2], dt_nr, view(x_g_Dg_D2g_val,2dof+1:2dof+2))

            for i in 1:4
                dt_nr = dt_nr-x_g_Dg_D2g_val[2dof+1]/x_g_Dg_D2g_val[2dof+2]
                # println("i = ", i, ", dt_nr = ", dt_nr)
                evaluate!(x_g_Dg_D2g[2dof+1:2dof+2], dt_nr, view(x_g_Dg_D2g_val,2dof+1:2dof+2))
            end
            evaluate!(x_g_Dg_D2g[1:2dof], dt_nr, view(x_g_Dg_D2g_val,1:2dof))

            tvS[nevents] = t0+dt_nr
            xvS[:,nevents] .= view(x_g_Dg_D2g_val,1:dof)
            gvS[nevents] = x_g_Dg_D2g_val[2dof+1] # g(t0+dt_nr,view(x_g_Dg_D2g_val,1:dof),view(x_g_Dg_D2g_val,dof+1:2dof))

            nevents += 1

            #println("g(t*,x*,dx*) = ", g(t0+dt_nr,view(x_g_Dg_D2g_val,1:dof),view(x_g_Dg_D2g_val,dof+1:2dof)))
            # println("x* = ", x_g_Dg_D2g_val[1:dof])
            # println("dx* = ", x_g_Dg_D2g_val[dof+1:2dof], "\n")
            #
            # println("δt = ", δt)
            # println("-δt_old = ", -δt_old)
            # println("* * * end poincare")
        end
        g_val_old = deepcopy(g_val)
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

    return view(tv,1:nsteps), transpose(view(xv,:,1:nsteps)), view(tvS,1:nevents-1), transpose(view(xvS,:,1:nevents-1)), view(gvS,1:nevents-1)
end

function poincare2{T<:Number}(f!, g, q0::Array{T,1}, t0::T, tmax::T,
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

    #Some auxiliary arrays for root-finding/event detection/Poincaré surface of section evaluation
    const g_val = Taylor1(zero(T), order)
    const g_val_old = Taylor1(zero(T), order)
    const slope = zero(T)
    const dt_li = zero(T)
    const dt_nr = zero(T)
    const δt = zero(T)
    const δt_old = zero(T)

    const x_g_Dg_D2g = vcat(x, dx, zero(x[1]), zero(x[1]))
    const x_g_Dg_D2g_val = Array{T}( length(x_g_Dg_D2g) )

    const tvS = similar(tv)
    const xvS = similar(xv)
    const gvS = similar(tv)

    # Integration
    nsteps = 1
    nevents = 1 #number of detected events
    while t0 < tmax
        δt_old = δt
        δt = taylorstep!(f!, x, dx, xaux, t0, tmax, x0, order, abstol)
        g_val = g(t0, x, dx)
        if g_val_old[2]*g_val[2] < zero(T)
            # println("* * * begin poincare")
            #
            # println("g_val_old= ", g_val_old[1])
            # println("g_val= ", g_val[1])
            # println("g_val_old[1]*g_val[1] =", g_val_old[1]*g_val[1])
            # println("g_val_old[1]*g_val[1] < zero(T) --> ", g_val_old[1]*g_val[1] < zero(T))

            #first guess: linear interpolation
            slope = (g_val[2]-g_val_old[2])/δt_old
            dt_li = -(g_val[2]/slope)

            # println("slope = ", slope)
            # println("dt_li = ", dt_li)

            x_g_Dg_D2g[1:dof] = x
            x_g_Dg_D2g[dof+1:2dof] = dx
            x_g_Dg_D2g[2dof+1] = derivative(g_val)
            x_g_Dg_D2g[2dof+2] = derivative(x_g_Dg_D2g[2dof+1])

            #Newton-Raphson iterations
            dt_nr = dt_li
            # println(" * dt_nr0 = ", dt_nr)
            evaluate!(x_g_Dg_D2g[2dof+1:2dof+2], dt_nr, view(x_g_Dg_D2g_val,2dof+1:2dof+2))

            for i in 1:4
                dt_nr = dt_nr-x_g_Dg_D2g_val[2dof+1]/x_g_Dg_D2g_val[2dof+2]
                # println("i = ", i, ", dt_nr = ", dt_nr)
                evaluate!(x_g_Dg_D2g[2dof+1:2dof+2], dt_nr, view(x_g_Dg_D2g_val,2dof+1:2dof+2))
            end
            evaluate!(x_g_Dg_D2g[1:2dof], dt_nr, view(x_g_Dg_D2g_val,1:2dof))

            tvS[nevents] = t0+dt_nr
            xvS[:,nevents] .= view(x_g_Dg_D2g_val,1:dof)
            gvS[nevents] = x_g_Dg_D2g_val[2dof+1]

            nevents += 1

            #println("g(t*,x*,dx*) = ", g(t0+dt_nr,view(x_g_Dg_D2g_val,1:dof),view(x_g_Dg_D2g_val,dof+1:2dof)))
            # println("x* = ", x_g_Dg_D2g_val[1:dof])
            # println("dx* = ", x_g_Dg_D2g_val[dof+1:2dof], "\n")
            #
            # println("δt = ", δt)
            # println("-δt_old = ", -δt_old)
            # println("* * * end poincare")
        end
        g_val_old = deepcopy(g_val)
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

    return view(tv,1:nsteps), view(transpose(xv),1:nsteps,:), view(tvS,1:nevents-1), view(transpose(xvS),1:nevents-1,:), view(gvS,1:nevents-1)
end
