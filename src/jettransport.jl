# This file is part of the TaylorIntegration.jl package; MIT licensed

function jetcoeffs!{T<:Number}(eqsdiff, t0::T, x::Taylor1{TaylorN{T}})
    order = x.order
    for ord in 1:order
        ordnext = ord+1

        # Set `xaux`, auxiliary Taylor1 variable to order `ord`
        @inbounds xaux = Taylor1( x.coeffs[1:ord] )

        # Equations of motion
        # TODO! define a macro to optimize the eqsdiff
        xdot = eqsdiff(t0, xaux)

        # Recursion relation
        @inbounds x.coeffs[ordnext] = xdot.coeffs[ord]/ord
    end
    nothing
end

function jetcoeffs!{T<:Number}(eqsdiff, t0::T, x::Vector{Taylor1{TaylorN{T}}})
    order = x[1].order
    xaux = similar(x)
    for ord in 1:order
        ordnext = ord+1

        # Set `xaux`, auxiliary vector of Taylor1 to order `ord`
        @inbounds for j in eachindex(x)
            xaux[j] = Taylor1( x[j].coeffs[1:ord] )
        end

        # Equations of motion
        # TODO! define a macro to optimize the eqsdiff
        xdot = eqsdiff(t0, xaux)

        # Recursion relations
        @inbounds for j in eachindex(x)
            x[j].coeffs[ordnext] = xdot[j].coeffs[ord]/ord
        end
    end
    nothing
end

function stepsize{T<:Number}(x::Taylor1{TaylorN{T}}, epsilon::T)
    ord = x.order
    h = T(Inf)
    for k in (ord-1, ord)
        @inbounds aux = Array{T}(x.coeffs[k+1].order)
        for i in 1:x.coeffs[k+1].order
            @inbounds aux[i] = norm(x.coeffs[k+1].coeffs[i].coeffs,Inf)
        end
        aux == zeros(T, length(aux)) && continue
        aux = epsilon ./ aux
        kinv = one(T)/k
        aux = aux.^kinv
        h = min(h, minimum(aux))
    end
    return h
end

function stepsize{T<:Number}(q::Array{Taylor1{TaylorN{T}},1}, epsilon::T)
    h = T(Inf)
    for i in eachindex(q)
        @inbounds hi = stepsize( q[i], epsilon )
        h = min( h, hi )
    end
    return h
end

function taylorstep{T<:Number}(f, t0::T, x0::TaylorN{T}, order::Int, abstol::T)
    # Initialize the Taylor1 expansions
    xT = Taylor1( x0, order )
    # Compute the Taylor coefficients
    jetcoeffs!(f, t0, xT)
    # Compute the step-size of the integration using `abstol`
    δt = stepsize(xT, abstol)
    x0 = evaluate(xT, δt)
    return δt, x0
end

function taylorstep!{T<:Number}(f, t0::T, x0::Array{TaylorN{T},1}, order::Int, abstol::T)
    # Initialize the vector of Taylor1 expansions
    xT = Array{Taylor1{TaylorN{T}}}(length(x0))
    for i in eachindex(x0)
        @inbounds xT[i] = Taylor1( x0[i], order )
    end
    # Compute the Taylor coefficients
    jetcoeffs!(f, t0, xT)
    # Compute the step-size of the integration using `abstol`
    δt = stepsize(xT, abstol)
    evaluate!(xT, δt, x0)
    return δt
end

function taylorstep{T<:Number}(f, t0::T, t1::T, x0::TaylorN{T}, order::Int, abstol::T)
    @assert t1 > t0
    # Initialize the Taylor1 expansions
    xT = Taylor1( x0, order )
    # Compute the Taylor coefficients
    jetcoeffs!(f, t0, xT)
    # Compute the step-size of the integration using `abstol`
    δt = stepsize(xT, abstol)
    if δt ≥ t1-t0
        δt = t1-t0
    end
    x0 = evaluate(xT, δt)
    return δt, x0
end

function taylorstep!{T<:Number}(f, t0::T, t1::T, x0::Array{TaylorN{T},1},
        order::Int, abstol::T)
    @assert t1 > t0
    # Initialize the vector of Taylor1 expansions
    xT = Array{Taylor1{TaylorN{T}}}(length(x0))
    for i in eachindex(x0)
        @inbounds xT[i] = Taylor1( x0[i], order )
    end
    # Compute the Taylor coefficients
    jetcoeffs!(f, t0, xT)
    # Compute the step-size of the integration using `abstol`
    δt = stepsize(xT, abstol)
    if δt ≥ t1-t0
        δt = t1-t0
    end
    evaluate!(xT, δt, x0)
    return δt
end

function taylorinteg{T<:Number}(f, x0::TaylorN{T}, t0::T, tmax::T, order::Int,
        abstol::T; maxsteps::Int=500)

    # Allocation
    tv = Array{T}(maxsteps+1)
    xv = Array{TaylorN{T}}(maxsteps+1)

    # Initial conditions
    nsteps = 1
    @inbounds tv[1] = t0
    @inbounds xv[1] = x0

    # Integration
    while t0 < tmax
        xold = x0
        δt, x0 = taylorstep(f, t0, x0, order, abstol)
        if t0+δt ≥ tmax
            x0 = xold
            δt, x0 = taylorstep(f, t0, tmax, x0, order, abstol)
            t0 = tmax
            nsteps += 1
            @inbounds tv[nsteps] = t0
            @inbounds xv[nsteps] = x0
            break
        end
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

function taylorinteg{T<:Number}(f, q0::Array{TaylorN{T},1}, t0::T, tmax::T,
        order::Int, abstol::T; maxsteps::Int=500)

    # Allocation
    tv = Array{T}(maxsteps+1)
    dof = length(q0)
    xv = Array{TaylorN{T}}(dof, maxsteps+1)

    # Initial conditions
    @inbounds tv[1] = t0
    @inbounds xv[:,1] = q0[:]
    x0 = copy(q0)

    # Integration
    nsteps = 1
    while t0 < tmax
        xold = copy(x0)
        δt = taylorstep!(f, t0, x0, order, abstol)
        if t0+δt ≥ tmax
            x0 = xold
            δt = taylorstep!(f, t0, tmax, x0, order, abstol)
            t0 = tmax
            nsteps += 1
            @inbounds tv[nsteps] = t0
            @inbounds xv[:,nsteps] = x0[:]
            break
        end
        t0 += δt
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds xv[:,nsteps] = x0[:]
        if nsteps > maxsteps
            warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return view(tv,1:nsteps), view(xv,:,1:nsteps)' #for xv, first do view, then transpose (otherwise it crashes)
end

function taylorinteg{T<:Number}(f, q0::Array{TaylorN{T},1}, trange::Range{T},
        order::Int, abstol::T; maxsteps::Int=500)

    # Allocation
    nn = length(trange)
    dof = length(q0)
    x0 = similar(q0, TaylorN{T}, dof)
    fill!(x0, TaylorN{T}(NaN))
    xv = Array{eltype(q0)}(dof, nn)
    @inbounds for ind in 1:nn
        xv[:,ind] = x0[:]
    end

    # Initial conditions
    @inbounds x0[:] = q0[:]
    @inbounds xv[:,1] = q0[:]

    # Integration
    iter = 1
    while iter < nn
        t0, t1 = trange[iter], trange[iter+1]
        nsteps = 0
        while nsteps < maxsteps
            xold = copy(x0)
            δt = taylorstep!(f, t0, x0, order, abstol)
            if t0+δt ≥ t1
                x0 = xold
                δt = taylorstep!(f, t0, t1, x0, order, abstol)
                t0 = t1
                break
            end
            t0 += δt
            nsteps += 1
        end
        if nsteps ≥ maxsteps && t0 != t1
            warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
        iter += 1
        @inbounds xv[:,iter] = x0[:]
    end

    return xv'
end

import TaylorSeries.evaluate!

# The code for the function below is another method for the function
# evaluate! which is included in the TaylorSeries.jl package,
# created by Luis Benet and David Sanders, under MIT license.
function evaluate!{T<:Number}(x::Array{Taylor1{TaylorN{T}},1}, δt::T, x0::Array{TaylorN{T},1})
    @assert length(x) == length(x0)
    @inbounds for i in eachindex(x)
        x0[i] = evaluate( x[i], δt )
    end
    nothing
end


## Auxiliary function ##
#
# function lastnonzero{T<:Number}(ac::Vector{T})
#     nonzero::Int = length(ac)
#     for i in length(ac):-1:1
#         if ac[i] != zero(T)
#             nonzero = i-1
#             break
#         end
#     end
#     nonzero
# end
# lastnonzero{T<:Number}(a::Taylor1{T}) = lastnonzero(a.coeffs)
