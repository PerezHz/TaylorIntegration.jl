eventisdetected{T<:Real}(x::Taylor1{T}, y::Taylor1{T}, r::Real, order::Int) = x[order+1]*y[order+1] < r
eventisdetected{T<:Real}(x::Taylor1{Taylor1{T}}, y::Taylor1{Taylor1{T}}, r::Real, order::Int) = x[order+1][1]*y[order+1][1] < r
eventisdetected{T<:Real}(x::Taylor1{TaylorN{T}}, y::Taylor1{TaylorN{T}}, r::Real, order::Int) = x[order+1][1][1]*y[order+1][1][1] < r

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
    x .= Taylor1.(q0, order)
    x0 = deepcopy(q0)
    @inbounds tv[1] = t0
    @inbounds xv[:,1] .= q0

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
        if eventisdetected(g_val_old, g_val, zero(T), eventorder)

            #first guess: linear interpolation
            slope = (g_val[nextevord]-g_val_old[nextevord])/δt_old
            dt_li = -(g_val[nextevord]/slope)

            x_g_Dg_D2g[1:dof] = x
            x_g_Dg_D2g[dof+1:2dof] = dx
            x_g_Dg_D2g[2dof+1] = deriv(eventorder, g_val)
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
