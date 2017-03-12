# This file is part of the TaylorIntegration.jl package; MIT licensed

"""
    stabilitymatrix!(eqsdiff!, t0, x, δx, dδx, jac)

Updates the matrix `jac::Array{T,2}` (linearized equations of motion)
computed from the equations of motion (`eqsdiff!`), at time `t0`
at `x`; `x` is of type `Vector{T<:Number}`. `δx` and `dδx` are two
auxiliary arrays of type `Vector{TaylorN{T}}` to avoid allocations.

"""
function stabilitymatrix!{T<:Number}(eqsdiff!, t0::T, x::AbstractArray{T,1},
        δx::Array{TaylorN{T},1}, dδx::Array{TaylorN{T},1}, jac::Array{T,2})

    for ind in eachindex(x)
        @inbounds δx[ind] = x[ind] + TaylorN(T,ind,order=1)
    end
    eqsdiff!(t0, δx, dδx)
    jacobian!( jac, dδx )
    nothing
end

"""
    stabilitymatrix!(eqsdiff!, t0, x, δx, dδx, jac)

Updates the matrix `jac::Array{Taylor1{T},2}` (linearized equations of motion)
computed from the equations of motion (`eqsdiff!`), at time `t0`
at `x`; `x` is of type `Vector{Taylor1{T<:Number}}`. `δx` and `dδx` are two
auxiliary arrays of type `Vector{TaylorN{Taylor1{T}}}` to avoid allocations.

"""
function stabilitymatrix!{T<:Number}(eqsdiff!, t0::T, x::AbstractArray{Taylor1{T},1},
        δx::Array{TaylorN{Taylor1{T}},1}, dδx::Array{TaylorN{Taylor1{T}},1},
        jac::Array{Taylor1{T},2})

    for ind in eachindex(x)
        @inbounds δx[ind] = x[ind] + TaylorN(Taylor1{T},ind,order=1)
    end
    eqsdiff!(t0, δx, dδx)
    jacobian!( jac, dδx )
    nothing
end


# Modified from `cgs` and `mgs`, obtained from:
# http://nbviewer.jupyter.org/url/math.mit.edu/~stevenj/18.335/Gram-Schmidt.ipynb
# Classical Gram–Schmidt (Trefethen algorithm 7.1), implemented in the simplest way
# (We could make it faster by unrolling loops to avoid temporaries arrays etc.)
function classicalGS!(A, Q, R, aⱼ, qᵢ, vⱼ)
    m,n = size(A)
    fill!(R, zero(eltype(A)))
    for j = 1:n
        # aⱼ = A[:,j]
        for ind = 1:m
            @inbounds aⱼ[ind] = A[ind,j]
            @inbounds vⱼ[ind] = aⱼ[ind]
        end
        # vⱼ = copy(aⱼ) # use copy so that modifying vⱼ doesn't change aⱼ
        for i = 1:j-1
            # qᵢ = Q[:,i]
            for ind = 1:m
                @inbounds qᵢ[ind] = Q[ind,i]
            end
            @inbounds R[i,j] = dot(qᵢ, aⱼ)
            # vⱼ -= R[i,j] * qᵢ
            @inbounds for ind = 1:m
                vⱼ[ind] -= R[i,j] * qᵢ[ind]
            end
        end
        @inbounds R[j,j] = norm(vⱼ)
        # Q[:,j] = vⱼ / R[j,j]
        for ind = 1:m
            @inbounds Q[ind,j] = vⱼ[ind] / R[j,j]
        end
    end
    return nothing
end
# Modified Gram–Schmidt (Trefethen algorithm 8.1); see also
# http://nbviewer.jupyter.org/url/math.mit.edu/~stevenj/18.335/Gram-Schmidt.ipynb
function modifiedGS!(A, Q, R, aⱼ, qᵢ, vⱼ)
    m,n = size(A)
    fill!(R, zero(eltype(A)))
    for j = 1:n
        # aⱼ = A[:,j]
        for ind = 1:m
            @inbounds aⱼ[ind] = A[ind,j]
            @inbounds vⱼ[ind] = aⱼ[ind]
        end
        # vⱼ = copy(aⱼ) # use copy so that modifying vⱼ doesn't change aⱼ
        for i = 1:j-1
            # qᵢ = Q[:,i]
            for ind = 1:m
                @inbounds qᵢ[ind] = Q[ind,i]
            end
            @inbounds R[i,j] = dot(qᵢ, vⱼ) # ⟵ NOTICE: mgs has vⱼ, clgs has aⱼ
            # vⱼ -= R[i,j] * qᵢ
            @inbounds for ind = 1:m
                vⱼ[ind] -= R[i,j] * qᵢ[ind]
            end
        end
        @inbounds R[j,j] = norm(vⱼ)
        # Q[:,j] = vⱼ / R[j,j]
        for ind = 1:m
            @inbounds Q[ind,j] = vⱼ[ind] / R[j,j]
        end
    end
    return nothing
end

"""
    liap_jetcoeffs!(eqsdiff!, t0, x, dx, xaux, δx, dδx, jac)

Similar to [`jetcoeffs!`](@ref) for the calculation of the Liapunov
spectrum. `jac` is the linearization of the equations of motion,
and `xaux`, `δx` and `dδx` are auxiliary vectors.

"""
function liap_jetcoeffs!{T<:Number}(eqsdiff!, t0::T, x::Vector{Taylor1{T}},
        dx::Vector{Taylor1{T}}, xaux::Vector{Taylor1{T}},
        δx::Array{TaylorN{Taylor1{T}},1}, dδx::Array{TaylorN{Taylor1{T}},1},
        jac::Array{Taylor1{T},2})

    order = x[1].order

    # Dimensions of phase-space: dof
    nx = length(x)
    dof = round(Int, (-1+sqrt(1+4*nx))/2)

    for ord in 1:order
        ordnext = ord+1

        # Set `xaux`, auxiliary vector of Taylor1 to order `ord`
        for j in eachindex(x)
            @inbounds xaux[j] = Taylor1( x[j].coeffs[1:ord] )
        end

        # Equations of motion
        eqsdiff!(t0, xaux, dx)
        # stabilitymatrix!( eqsdiff!, t0, xaux[1:dof], δx, dδx, jac )
        stabilitymatrix!( eqsdiff!, t0, view(xaux,1:dof), δx, dδx, jac )
        @inbounds dx[dof+1:nx] = jac * reshape( xaux[dof+1:nx], (dof,dof) )

        # Recursion relations
        for j in eachindex(x)
            @inbounds x[j].coeffs[ordnext] = dx[j].coeffs[ord]/ord
        end
    end
    nothing
end


"""
    liap_taylorstep!(f, x, dx, xaux, δx, dδx, jac, t0, t1, x0, order, abstol)

Similar to [`taylorstep!`](@ref) for the calculation of the Liapunov
spectrum. `jac` is the linearization of the equations of motion,
and `xaux`, `δx` and `dδx` are auxiliary vectors.

"""
function liap_taylorstep!{T<:Number}(f, x::Vector{Taylor1{T}}, dx::Vector{Taylor1{T}},
        xaux::Vector{Taylor1{T}}, δx::Array{TaylorN{Taylor1{T}},1},
        dδx::Array{TaylorN{Taylor1{T}},1}, jac::Array{Taylor1{T},2}, t0::T, t1::T, x0::Array{T,1},
        order::Int, abstol::T)

    # Compute the Taylor coefficients
    liap_jetcoeffs!(f, t0, x, dx, xaux, δx, dδx, jac)

    # Compute the step-size of the integration using `abstol`
    δt = stepsize(x, abstol)
    δt = min(δt, t1-t0)

    # Update x0
    evaluate!(x, δt, x0)
    return δt
end


"""
    liap_taylorinteg(f, q0, t0, tmax, order, abstol; maxsteps::Int=500)

Similar to [`taylorinteg!`](@ref) for the calculation of the Liapunov
spectrum.

"""
function liap_taylorinteg{T<:Number}(f, q0::Array{T,1}, t0::T, tmax::T,
        order::Int, abstol::T; maxsteps::Int=500)
    # Allocation
    const tv = Array{T}(maxsteps+1)
    dof = length(q0)
    const xv = Array{T}(dof, maxsteps+1)
    const λ = similar(xv)
    const λtsum = similar(q0)
    const jt = eye(T, dof)

    # NOTE: This changes GLOBALLY internal parameters of TaylorN
    global _δv = set_variables("δ", order=1, numvars=dof)

    # Initial conditions
    @inbounds tv[1] = t0
    for ind in 1:dof
        @inbounds xv[ind,1] = q0[ind]
        @inbounds λ[ind,1] = zero(T)
        @inbounds λtsum[ind] = zero(T)
    end
    const x0 = vcat(q0, reshape(jt, dof*dof))
    nx0 = dof*(dof+1)
    t00 = t0

    # Initialize the vector of Taylor1 expansions
    const x = Array{Taylor1{T}}(length(x0))
    for i in eachindex(x0)
        @inbounds x[i] = Taylor1( x0[i], order )
    end

    #Allocate auxiliary arrays
    const dx = Array{Taylor1{T}}(length(x0))
    const xaux = Array{Taylor1{T}}(length(x0))
    const δx = Array{TaylorN{Taylor1{T}}}(dof)
    const dδx = Array{TaylorN{Taylor1{T}}}(dof)
    const jac = Array{Taylor1{T}}(dof,dof)
    for i in eachindex(jac)
        @inbounds jac[i] = zero(x[1])
    end
    const QH = Array{T}(dof,dof)
    const RH = Array{T}(dof,dof)
    const aⱼ = Array{eltype(jt)}( dof )
    const qᵢ = similar(aⱼ)
    const vⱼ = similar(aⱼ)

    # Integration
    nsteps = 1
    while t0 < tmax
        δt = liap_taylorstep!(f, x, dx, xaux, δx, dδx, jac, t0, tmax, x0, order, abstol)
        for ind in eachindex(jt)
            @inbounds jt[ind] = x0[dof+ind]
        end
        modifiedGS!( jt, QH, RH, aⱼ, qᵢ, vⱼ )
        t0 += δt
        tspan = t0-t00
        nsteps += 1
        @inbounds tv[nsteps] = t0
        @inbounds for ind in 1:dof
            xv[ind,nsteps] = x0[ind]
            λtsum[ind] += log(RH[ind,ind])
            λ[ind,nsteps] = λtsum[ind]/tspan
        end
        for ind in eachindex(QH)
            @inbounds x0[dof+ind] = QH[ind]
        end
        for i in eachindex(x0)
            @inbounds x[i] = Taylor1( x0[i], order )
        end
        if nsteps > maxsteps
            warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return view(tv,1:nsteps),  view(transpose(xv),1:nsteps,:),  view(transpose(λ),1:nsteps,:)
end

function liap_taylorinteg{T<:Number}(f, q0::Array{T,1}, trange::Range{T},
        order::Int, abstol::T; maxsteps::Int=500)
    # Allocation
    nn = length(trange)
    dof = length(q0)
    const xv = Array{T}(dof, nn)
    fill!(xv, T(NaN))
    const λ = similar(xv)
    const λtsum = similar(q0)
    const jt = eye(T, dof)

    # NOTE: This changes GLOBALLY internal parameters of TaylorN
    global _δv = set_variables("δ", order=1, numvars=dof)

    # Initial conditions
    @inbounds for ind in 1:dof
        xv[ind,1] = q0[ind]
        λ[ind,1] = zero(T)
        λtsum[ind] = zero(T)
    end

    # Initialize the vector of Taylor1 expansions
    const x0 = vcat(q0, reshape(jt, dof*dof))
    nx0 = length(x0)
    const x = Array{Taylor1{T}}(nx0)
    for i in eachindex(x0)
        @inbounds x[i] = Taylor1( x0[i], order )
    end
    t00 = trange[1]
    tspan = zero(T)

    #Allocate auxiliary arrays
    const dx = Array{Taylor1{T}}(nx0)
    const xaux = Array{Taylor1{T}}(nx0)
    const δx = Array{TaylorN{Taylor1{T}}}(dof)
    const dδx = Array{TaylorN{Taylor1{T}}}(dof)
    const jac = Array{Taylor1{T}}(dof,dof)
    for i in eachindex(jac)
        jac[i] = zero(x[1])
    end
    const QH = Array{T}(dof,dof)
    const RH = Array{T}(dof,dof)
    const aⱼ = Array{eltype(jt)}( dof )
    const qᵢ = similar(aⱼ)
    const vⱼ = similar(aⱼ)

    # Integration
    iter = 1
    while iter < nn
        t0, t1 = trange[iter], trange[iter+1]
        nsteps = 0
        while nsteps < maxsteps
            δt = liap_taylorstep!(f, x, dx, xaux, δx, dδx, jac, t0, t1, x0, order, abstol)
            for ind in eachindex(jt)
                @inbounds jt[ind] = x0[dof+ind]
            end
            modifiedGS!( jt, QH, RH, aⱼ, qᵢ, vⱼ )
            t0 += δt
            nsteps += 1
            @inbounds for ind in 1:dof
                λtsum[ind] += log(RH[ind,ind])
            end
            for ind in eachindex(QH)
                @inbounds x0[dof+ind] = QH[ind]
            end
            for i in eachindex(x0)
                @inbounds x[i] = Taylor1( x0[i], order )
            end
            t0 ≥ t1 && break
        end
        if nsteps ≥ maxsteps && t0 != t1
            warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
        iter += 1
        tspan = t0-t00
        @inbounds for ind in 1:dof
            xv[ind,iter] = x0[ind]
            λ[ind,iter] = λtsum[ind]/tspan
        end
    end

    return transpose(xv),  transpose(λ)
end
