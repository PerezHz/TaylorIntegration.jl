# This file is part of the TaylorIntegration.jl package; MIT licensed

# jetcoeffs!
function jetcoeffs!{T<:Number}(eqsdiff, t0::T, x::Taylor1{Interval{T}},
        vT::Vector{Interval{T}})
    order = x.order
    vT[1] = @interval t0
    for ord in 1:order
        ordnext = ord+1

        # Set `xaux`, auxiliary Taylor1 variable to order `ord`
        @inbounds xaux = Taylor1( x.coeffs[1:ord] )

        # Equations of motion
        # TODO! define a macro to optimize the eqsdiff
        tT = Taylor1(vT[1:ord])
        dx = eqsdiff(tT, xaux)

        # Recursion relation
        @inbounds x[ordnext] = dx[ord]/ord
    end
    nothing
end

function jetcoeffs!{T<:Number}(eqsdiff!, t0::T, x::Vector{Taylor1{Interval{T}}},
        dx::Vector{Taylor1{Interval{T}}}, xaux::Vector{Taylor1{Interval{T}}},
        vT::Vector{Interval{T}})
    order = x[1].order
    vT[1] = @interval t0
    for ord in 1:order
        ordnext = ord+1

        # Set `xaux`, auxiliary vector of Taylor1 to order `ord`
        for j in eachindex(x)
            @inbounds xaux[j] = Taylor1( x[j].coeffs[1:ord] )
        end

        # Equations of motion
        # TODO! define a macro to optimize the eqsdiff
        tT = Taylor1(vT[1:ord])
        eqsdiff!(tT, xaux, dx)

        # Recursion relations
        for j in eachindex(x)
            @inbounds x[j][ordnext] = dx[j][ord]/ord
        end
    end
    nothing
end

# stepsize
#TODO: Implement Moore's (1966) stepsize subroutine
function stepsize{T<:Real}(x::Taylor1{Interval{T}}, epsilon::T)
    ord = x.order
    h = convert(T, Inf)
    for k in (ord-1, ord)
        @inbounds aux = mag( x[k+1] )
        aux == zero(T) && continue
        @inbounds aux = epsilon*mag(x[1]) / aux
        kinv = one(T)/(k+1)
        aux = aux^kinv
        h = min(h, aux)
    end
    @inbounds disc = mag( x[1] )/mag( x[2] )
    if h < disc
        return h
    else
        h = convert(T, Inf)
        for k in (ord-1, ord)
            @inbounds aux = mag( x[k+1] )
            aux == zero(T) && continue
            @inbounds aux = epsilon*mag(x[2]) / aux
            kinv = one(T)/k
            aux = aux^kinv
            h = min(h, aux)
        end
        return h
    end
end

function stepsize{T<:Real}(q::Array{Taylor1{Interval{T}},1}, epsilon::T)
    h = convert(T, Inf)
    for i in eachindex(q)
        @inbounds hi = stepsize( q[i], epsilon )
        h = min( h, hi )
    end
    return h
end

# taylorstep!
function taylorstep!{T<:Number}(f, x::Taylor1{Interval{T}}, t0::T, t1::T,
        x0::Interval{T}, order::Int, abstol::T, vT::Vector{Interval{T}})
    @assert t1 > t0
    # Compute the Taylor coefficients
    jetcoeffs!(f, t0, x, vT)
    # Compute the step-size of the integration using `abstol`
    δt = stepsize(x, abstol)
    δt = min(δt, t1-t0)
    x0 = evaluate(x, δt)
    return δt, x0
end

function taylorstep!{T<:Number}(f, x::Vector{Taylor1{Interval{T}}},
        dx::Vector{Taylor1{Interval{T}}}, xaux::Vector{Taylor1{Interval{T}}},
        t0::T, t1::T, x0::Array{Interval{T},1}, order::Int, abstol::T,
        vT::Vector{Interval{T}})
    @assert t1 > t0
    # Compute the Taylor coefficients
    jetcoeffs!(f, t0, x, dx, xaux, vT)
    # Compute the step-size of the integration using `abstol`
    δt = stepsize(x, abstol)
    δt = min(δt, t1-t0)
    evaluate!(x, δt, x0)
    return δt
end

# taylorinteg
function taylorinteg{T<:Number}(f, x0::Interval{T}, t0::T, tmax::T, order::Int,
        abstol::T; maxsteps::Int=500)

    # Allocation
    const tv = Array{T}(maxsteps+1)
    const xv = Array{Interval{T}}(maxsteps+1)
    const vT = zeros(Interval{T}, order+1)
    vT[2] = one(Interval{T})

    # Initialize the Taylor1 expansions
    x = Taylor1( x0, order )

    # Initial conditions
    nsteps = 1
    @inbounds tv[1] = t0
    @inbounds xv[1] = x0

    # Integration
    while t0 < tmax
        δt, x0 = taylorstep!(f, x, t0, tmax, x0, order, abstol, vT)
        x[1] = x0
        t0 += δt
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds xv[nsteps] = x0
        if nsteps > maxsteps
            warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    #return tv, xv
    return view(tv,1:nsteps), view(xv,1:nsteps)
end

function taylorinteg{T<:Number}(f, q0::Array{Interval{T},1}, t0::T, tmax::T,
        order::Int, abstol::T; maxsteps::Int=500)

    # Allocation
    const tv = Array{T}(maxsteps+1)
    dof = length(q0)
    const xv = Array{Interval{T}}(dof, maxsteps+1)
    const vT = zeros(Interval{T}, order+1)
    vT[2] = one(Interval{T})

    # Initialize the vector of Taylor1 expansions
    const x = Array{Taylor1{Interval{T}}}(dof)
    const dx = Array{Taylor1{Interval{T}}}(dof)
    const xaux = Array{Taylor1{Interval{T}}}(dof)
    for i in eachindex(q0)
        @inbounds x[i] = Taylor1( q0[i], order )
    end

    # Initial conditions
    @inbounds tv[1] = t0
    @inbounds xv[:,1] .= q0
    x0 = copy(q0)

    # Integration
    nsteps = 1
    while t0 < tmax
        δt = taylorstep!(f, x, dx, xaux, t0, tmax, x0, order, abstol, vT)
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

    # return view(tv,1:nsteps), view(transpose(xv),1:nsteps,:)
    return view(tv,1:nsteps), transpose(view(xv,:,1:nsteps))
end

function taylorinteg{T<:Number}(f, x0::Interval{T}, trange::Range{T},
        order::Int, abstol::T; maxsteps::Int=500)

    # Allocation
    nn = length(trange)
    const xv = Array{Interval{T}}(nn)
    fill!(xv, Interval{T}(NaN))
    const vT = zeros(Interval{T}, order+1)
    vT[2] = one(Interval{T})

    # Initialize the Taylor1 expansions
    x = Taylor1( x0, order )

    # Initial conditions
    @inbounds xv[1] = x0

    # Integration
    iter = 1
    while iter < nn
        t0, t1 = trange[iter], trange[iter+1]
        nsteps = 0
        while nsteps < maxsteps
            δt, x0 = taylorstep!(f, x, t0, t1, x0, order, abstol, vT)
            @inbounds x[1] = x0
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
        @inbounds xv[iter] = x0
    end

    return xv
end

function taylorinteg{T<:Number}(f, q0::Array{Interval{T},1}, trange::Range{T},
        order::Int, abstol::T; maxsteps::Int=500)

    # Allocation
    nn = length(trange)
    dof = length(q0)
    const x0 = similar(q0, Interval{T}, dof)
    fill!(x0, Interval{T}(NaN))
    const xv = Array{eltype(q0)}(dof, nn)
    for ind in 1:nn
        @inbounds xv[:,ind] .= x0
    end
    const vT = zeros(Interval{T}, order+1)
    vT[2] = one(Interval{T})

    # Initialize the vector of Taylor1 expansions
    const x = Array{Taylor1{Interval{T}}}(dof)
    const dx = Array{Taylor1{Interval{T}}}(dof)
    const xaux = Array{Taylor1{Interval{T}}}(dof)
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
            δt = taylorstep!(f, x, dx, xaux, t0, t1, x0, order, abstol, vT)
            for i in eachindex(x0)
                @inbounds x[i][1] = x0[i]
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
