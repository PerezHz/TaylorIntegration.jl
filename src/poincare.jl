surfacecrossing{T<:Real}(g::Taylor1{T}, g_old::Taylor1{T}, order::Int) = g[order+1]*g_old[order+1] < zero(T)
surfacecrossing{T<:Real}(g::Taylor1{Taylor1{T}}, g_old::Taylor1{Taylor1{T}}, order::Int) = g[order+1][1]*g_old[order+1][1] < zero(T)
surfacecrossing{T<:Real}(g::Taylor1{TaylorN{T}}, g_old::Taylor1{TaylorN{T}}, order::Int) = g[order+1][1][1]*g_old[order+1][1][1] < zero(T)

function deriv{T<:Number}(n::Int, a::Taylor1{T})
    @assert a.order ≥ n ≥ 0
    if n==0
        return a
    elseif n==1
        return derivative(a)
    else
        return deriv(n-1, derivative(a))
    end
end

function taylorinteg{T<:Real, U<:Number}(f!, g, q0::Array{U,1}, t0::T, tmax::T,
        order::Int, abstol::T; maxsteps::Int=500, nriter::Int=5, eventorder::Int=0)

    # Allocation
    const tv = Array{T}(maxsteps+1)
    dof = length(q0)
    const xv = Array{U}(dof, maxsteps+1)

    # Initialize the vector of Taylor1 expansions
    const t = Taylor1(T, order)
    const x = Array{Taylor1{U}}(dof)
    const dx = Array{Taylor1{U}}(dof)
    const xaux = Array{Taylor1{U}}(dof)

    # Initial conditions
    @inbounds t[1] = t0
    x0 = deepcopy(q0)
    x .= Taylor1.(q0, order)
    @inbounds tv[1] = t0
    @inbounds xv[:,1] .= q0

    #Some auxiliary arrays for root-finding/event detection/Poincaré surface of section evaluation
    const g_val = zero(g(t,x,x))
    const g_val_old = zero(g_val)
    const slope = zero(U)
    const dt_li = zero(U)
    const dt_nr = zero(U)
    const δt = zero(U)
    const δt_old = zero(U)

    const x_dx = vcat(x, dx)
    const g_dg = vcat(g_val, g_val_old)
    const x_dx_val = Array{U}( length(x_dx) )
    const g_dg_val = vcat(evaluate(g_val), evaluate(g_val_old))

    const tvS = Array{U}(maxsteps+1)
    const xvS = similar(xv)
    const gvS = similar(tvS)

    #auxiliary range object for Newton-Raphson iterations
    const nrinds = 1:nriter

    # Integration
    nsteps = 1
    nevents = 1 #number of detected events
    nextevord = eventorder+1
    while t0 < tmax
        δt_old = δt
        δt = taylorstep!(f!, t, x, dx, xaux, t0, tmax, x0, order, abstol)
        g_val = g(t, x, dx)
        if surfacecrossing(g_val_old, g_val, eventorder)

            #first guess: linear interpolation
            slope = (g_val[nextevord]-g_val_old[nextevord])/δt_old
            dt_li = -(g_val[nextevord]/slope)

            x_dx[1:dof] = x
            x_dx[dof+1:2dof] = dx
            g_dg[1] = deriv(eventorder, g_val)
            g_dg[2] = derivative(g_dg[1])

            #Newton-Raphson iterations
            dt_nr = dt_li
            evaluate!(g_dg, dt_nr, view(g_dg_val,:))

            for i in eachindex(nrinds)
                dt_nr = dt_nr-g_dg_val[1]/g_dg_val[2]
                evaluate!(g_dg, dt_nr, view(g_dg_val,:))
            end
            evaluate!(x_dx, dt_nr, view(x_dx_val,:))

            tvS[nevents] = t0+dt_nr
            xvS[:,nevents] .= view(x_dx_val,1:dof)
            gvS[nevents] = g_dg_val[1]

            nevents += 1
        end
        g_val_old = deepcopy(g_val)
        for i in eachindex(x0)
            @inbounds x[i][1] = x0[i]
        end
        t0 += δt
        @inbounds t[1] = t0
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

    return view(tv,1:nsteps), view(transpose(view(xv,:,1:nsteps)),1:nsteps,:), view(tvS,1:nevents-1), view(transpose(view(xvS,:,1:nevents-1)),1:nevents-1,:), view(gvS,1:nevents-1)
end
