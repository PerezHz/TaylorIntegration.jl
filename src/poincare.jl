# import Base.isless
# isless(x::AbstractSeries, y::AbstractSeries) = isless(evaluate(x), evaluate(y))
# isless(x::AbstractSeries, y::Real) = isless(evaluate(x), y)
# isless(x::Real, y::AbstractSeries) = isless(x, evaluate(y))

myisless(x::AbstractSeries, y::AbstractSeries) = isless(evaluate(x), evaluate(y))
myisless(x::AbstractSeries, y::Real) = isless(evaluate(x), y)
myisless(x::Real, y::AbstractSeries) = isless(x, evaluate(y))

function taylorinteg{T<:Real,U<:Number}(f!, g, q0::Array{U,1}, t0::T, tmax::T,
        order::Int, abstol::T; maxsteps::Int=500, nriter::Int=5)

    # Allocation
    const tv = Array{T}(maxsteps+1)
    dof = length(q0)
    const xv = Array{U}(dof, maxsteps+1)
    const vT = zeros(T, order+1)
    vT[2] = one(T)

    # Initialize the vector of Taylor1 expansions
    const x = Array{Taylor1{U}}(dof)
    const dx = Array{Taylor1{U}}(dof)
    const xaux = Array{Taylor1{U}}(dof)
    for i in eachindex(q0)
        @inbounds x[i] = Taylor1( q0[i], order )
    end

    # Initial conditions
    @inbounds tv[1] = t0
    @inbounds xv[:,1] .= q0
    x0 = copy(q0)

    #Some auxiliary arrays for root-finding/event detection/Poincaré surface of section evaluation
    const g_val = Taylor1(zero(U), order)
    const g_val_old = Taylor1(zero(U), order)
    const slope = zero(U)
    const dt_li = zero(U)
    const dt_nr = zero(U)
    const δt = zero(U)
    const δt_old = zero(U)

    const x_g_Dg_D2g = vcat(x, dx, zero(x[1]), zero(x[1]))
    const x_g_Dg_D2g_val = Array{U}( length(x_g_Dg_D2g) )

    const tvS = similar(tv)
    const xvS = similar(xv)
    const gvS = Array{U}(maxsteps+1)

    #auxiliary range object for Newton-Raphson iterations
    const nrinds = 1:nriter

    # Integration
    nsteps = 1
    nevents = 1 #number of detected events
    while t0 < tmax
        δt_old = δt
        δt = taylorstep!(f!, x, dx, xaux, t0, tmax, x0, order, abstol, vT)
        g_val = g(t0, x, dx)
        if myisless(g_val_old*g_val, zero(U))

            #first guess: linear interpolation
            slope = (g_val[1]-g_val_old[1])/δt_old
            dt_li = -(g_val[1]/slope)

            x_g_Dg_D2g[1:dof] = x
            x_g_Dg_D2g[dof+1:2dof] = dx
            x_g_Dg_D2g[2dof+1] = g_val
            x_g_Dg_D2g[2dof+2] = derivative(g_val)

            #Newton-Raphson iterations
            dt_nr = dt_li
            evaluate!(x_g_Dg_D2g[2dof+1:2dof+2], dt_nr, view(x_g_Dg_D2g_val,2dof+1:2dof+2))

            for i in eachindex(nrinds)
                dt_nr = dt_nr-x_g_Dg_D2g_val[2dof+1]/x_g_Dg_D2g_val[2dof+2]
                evaluate!(x_g_Dg_D2g[2dof+1:2dof+2], dt_nr, view(x_g_Dg_D2g_val,2dof+1:2dof+2))
            end
            evaluate!(x_g_Dg_D2g[1:2dof], dt_nr, view(x_g_Dg_D2g_val,1:2dof))

            tvS[nevents] = t0+dt_nr
            xvS[:,nevents] .= view(x_g_Dg_D2g_val,1:dof)
            gvS[nevents] = x_g_Dg_D2g_val[2dof+1]

            nevents += 1
        end
        g_val_old = deepcopy(g_val)
        for i in eachindex(x0)
            @inbounds x[i][1] = x0[i]
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

function poincare2{T<:Number}(f!, g, q0::Array{T,1}, t0::T, tmax::T,
        order::Int, abstol::T; maxsteps::Int=500, nriter::Int=5)

    # Allocation
    const tv = Array{T}(maxsteps+1)
    dof = length(q0)
    const xv = Array{T}(dof, maxsteps+1)
    const vT = zeros(T, order+1)
    vT[2] = one(T)

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

    #auxiliary range object for Newton-Raphson iterations
    const nrinds = 1:nriter

    # Integration
    nsteps = 1
    nevents = 1 #number of detected events
    while t0 < tmax
        δt_old = δt
        δt = taylorstep!(f!, x, dx, xaux, t0, tmax, x0, order, abstol, vT)
        g_val = g(t0, x, dx)
        if g_val_old[2]*g_val[2] < zero(T)

            #first guess: linear interpolation
            slope = (g_val[2]-g_val_old[2])/δt_old
            dt_li = -(g_val[2]/slope)

            x_g_Dg_D2g[1:dof] = x
            x_g_Dg_D2g[dof+1:2dof] = dx
            x_g_Dg_D2g[2dof+1] = derivative(g_val)
            x_g_Dg_D2g[2dof+2] = derivative(x_g_Dg_D2g[2dof+1])

            #Newton-Raphson iterations
            dt_nr = dt_li
            evaluate!(x_g_Dg_D2g[2dof+1:2dof+2], dt_nr, view(x_g_Dg_D2g_val,2dof+1:2dof+2))

            for i in eachindex(nrinds)
                dt_nr = dt_nr-x_g_Dg_D2g_val[2dof+1]/x_g_Dg_D2g_val[2dof+2]
                evaluate!(x_g_Dg_D2g[2dof+1:2dof+2], dt_nr, view(x_g_Dg_D2g_val,2dof+1:2dof+2))
            end
            evaluate!(x_g_Dg_D2g[1:2dof], dt_nr, view(x_g_Dg_D2g_val,1:2dof))

            tvS[nevents] = t0+dt_nr
            xvS[:,nevents] .= view(x_g_Dg_D2g_val,1:dof)
            gvS[nevents] = x_g_Dg_D2g_val[2dof+1]

            nevents += 1
        end
        g_val_old = deepcopy(g_val)
        for i in eachindex(x0)
            @inbounds x[i][1] = x0[i]
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

#poincare2 + multi-variational jet transport
function poincare2{T<:Number}(f!, g, q0::Array{TaylorN{T},1}, t0::T, tmax::T,
        order::Int, abstol::T; maxsteps::Int=500, nriter::Int=5)

    # Allocation
    const tv = Array{T}(maxsteps+1)
    dof = length(q0)
    const xv = Array{TaylorN{T}}(dof, maxsteps+1)
    const vT = zeros(T, order+1)
    vT[2] = one(T)

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

    #auxiliary range object for Newton-Raphson iterations
    const nrinds = 1:nriter

    # Integration
    nsteps = 1
    nevents = 1 #number of detected events
    while t0 < tmax
        δt_old = δt
        δt = taylorstep!(f!, x, dx, xaux, t0, tmax, x0, order, abstol, vT)
        g_val = g(t0, x, dx)
        if g_val_old[2][1][1]*g_val[2][1][1] < zero(T)

            #first guess: linear interpolation
            slope = (g_val[2]-g_val_old[2])/δt_old
            dt_li = -(g_val[2]/slope)

            x_g_Dg_D2g[1:dof] = x
            x_g_Dg_D2g[dof+1:2dof] = dx
            x_g_Dg_D2g[2dof+1] = derivative(g_val)
            x_g_Dg_D2g[2dof+2] = derivative(x_g_Dg_D2g[2dof+1])

            #Newton-Raphson iterations
            dt_nr = dt_li
            evaluate!(x_g_Dg_D2g[2dof+1:2dof+2], dt_nr, view(x_g_Dg_D2g_val,2dof+1:2dof+2))

            for i in eachindex(nrinds)
                dt_nr = dt_nr-x_g_Dg_D2g_val[2dof+1]/x_g_Dg_D2g_val[2dof+2]
                evaluate!(x_g_Dg_D2g[2dof+1:2dof+2], dt_nr, view(x_g_Dg_D2g_val,2dof+1:2dof+2))
            end
            evaluate!(x_g_Dg_D2g[1:2dof], dt_nr, view(x_g_Dg_D2g_val,1:2dof))

            tvS[nevents] = t0+dt_nr
            xvS[:,nevents] .= view(x_g_Dg_D2g_val,1:dof)
            gvS[nevents] = x_g_Dg_D2g_val[2dof+1]

            nevents += 1
        end
        g_val_old = deepcopy(g_val)
        for i in eachindex(x0)
            @inbounds x[i][1] = x0[i]
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

#poincare2 + 1 variation jet transport
function poincare2{T<:Number}(f!, g, q0::Array{Taylor1{T},1}, t0::T, tmax::T,
        order::Int, abstol::T; maxsteps::Int=500, nriter::Int=5)

    # Allocation
    const tv = Array{T}(maxsteps+1)
    dof = length(q0)
    const xv = Array{Taylor1{T}}(dof, maxsteps+1)
    const vT = zeros(T, order+1)
    vT[2] = one(T)

    # Initialize the vector of Taylor1 expansions
    const x = Array{Taylor1{Taylor1{T}}}(dof)
    const dx = Array{Taylor1{Taylor1{T}}}(dof)
    const xaux = Array{Taylor1{Taylor1{T}}}(dof)
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
    const x_g_Dg_D2g_val = Array{Taylor1{T}}( length(x_g_Dg_D2g) )

    const tvS = Array{Taylor1{T}}( length(tv) )
    const xvS = similar(xv)
    const gvS = similar(tvS)

    #auxiliary range object for Newton-Raphson iterations
    const nrinds = 1:nriter

    # Integration
    nsteps = 1
    nevents = 1 #number of detected events
    while t0 < tmax
        δt_old = δt
        δt = taylorstep!(f!, x, dx, xaux, t0, tmax, x0, order, abstol, vT)
        g_val = g(t0, x, dx)
        if g_val_old[2][1][1]*g_val[2][1][1] < zero(T)

            #first guess: linear interpolation
            slope = (g_val[2]-g_val_old[2])/δt_old
            dt_li = -(g_val[2]/slope)

            x_g_Dg_D2g[1:dof] = x
            x_g_Dg_D2g[dof+1:2dof] = dx
            x_g_Dg_D2g[2dof+1] = derivative(g_val)
            x_g_Dg_D2g[2dof+2] = derivative(x_g_Dg_D2g[2dof+1])

            #Newton-Raphson iterations
            dt_nr = dt_li
            evaluate!(x_g_Dg_D2g[2dof+1:2dof+2], dt_nr, view(x_g_Dg_D2g_val,2dof+1:2dof+2))

            for i in eachindex(nrinds)
                dt_nr = dt_nr-x_g_Dg_D2g_val[2dof+1]/x_g_Dg_D2g_val[2dof+2]
                evaluate!(x_g_Dg_D2g[2dof+1:2dof+2], dt_nr, view(x_g_Dg_D2g_val,2dof+1:2dof+2))
            end
            evaluate!(x_g_Dg_D2g[1:2dof], dt_nr, view(x_g_Dg_D2g_val,1:2dof))

            tvS[nevents] = t0+dt_nr
            xvS[:,nevents] .= view(x_g_Dg_D2g_val,1:dof)
            gvS[nevents] = x_g_Dg_D2g_val[2dof+1]

            nevents += 1
        end
        g_val_old = deepcopy(g_val)
        for i in eachindex(x0)
            @inbounds x[i][1] = x0[i]
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
