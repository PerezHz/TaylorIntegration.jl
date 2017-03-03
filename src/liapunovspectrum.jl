# This file is part of the TaylorIntegration.jl package; MIT licensed

"""
    stabilitymatrix!{T<:Number, S<:Number}(eqsdiff, t0::T, x::Array{T,1},
        jac::Array{T,2})
    stabilitymatrix!{T<:Number, S<:Number}(eqsdiff, t0::T, x::Array{Taylor1{T},1},
        jac::Array{Taylor1{T},2})

Updates the matrix `jac` (linearized equations of motion)
computed from the equations of motion (`eqsdiff`), at time `t0`
at `x0`.
"""
function stabilitymatrix!{T<:Number}(eqsdiff!, t0::T, x::Array{T,1},
        δx::Array{TaylorN{T},1}, δdx::Array{TaylorN{T},1}, jac::Array{T,2})
    @inbounds for ind in eachindex(x)
        δx[ind] = x[ind] + TaylorN(T,ind,order=1)
    end
    eqsdiff!(t0, δx, δdx)
    jac[:] = jacobian( δdx )
    nothing
end

function stabilitymatrix!{T<:Number}(eqsdiff!, t0::T, x::Array{Taylor1{T},1},
        δx::Array{TaylorN{Taylor1{T}},1}, δdx::Array{TaylorN{Taylor1{T}},1}, jac::Array{Taylor1{T},2})
    @inbounds for ind in eachindex(x)
        δx[ind] = convert(TaylorN{Taylor1{T}}, x[ind]) +
            TaylorN(Taylor1{T},ind,order=1)
    end
    eqsdiff!(t0, δx, δdx)
    jac[:] = jacobian( δdx )
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
        @inbounds for ind = 1:m
            aⱼ[ind] = A[ind,j]
            vⱼ[ind] = aⱼ[ind]
        end
        # vⱼ = copy(aⱼ) # use copy so that modifying vⱼ doesn't change aⱼ
        for i = 1:j-1
            # qᵢ = Q[:,i]
            @inbounds for ind = 1:m
                qᵢ[ind] = Q[ind,i]
            end
            @inbounds R[i,j] = dot(qᵢ, aⱼ)
            # vⱼ -= R[i,j] * qᵢ
            @inbounds for ind = 1:m
                vⱼ[ind] -= R[i,j] * qᵢ[ind]
            end
        end
        @inbounds R[j,j] = norm(vⱼ)
        # Q[:,j] = vⱼ / R[j,j]
        @inbounds for ind = 1:m
            Q[ind,j] = vⱼ[ind] / R[j,j]
        end
    end
    return nothing
end
# Modified Gram–Schmidt (Trefethen algorithm 8.1)
function modifiedGS!(A, Q, R, aⱼ, qᵢ, vⱼ)
    m,n = size(A)
    fill!(R, zero(eltype(A)))
    for j = 1:n
        # aⱼ = A[:,j]
        @inbounds for ind = 1:m
            aⱼ[ind] = A[ind,j]
            # vⱼ[ind] = aⱼ[ind]
        end
        vⱼ = copy(aⱼ) # use copy so that modifying vⱼ doesn't change aⱼ
        for i = 1:j-1
            # qᵢ = Q[:,i]
            @inbounds for ind = 1:m
                qᵢ[ind] = Q[ind,i]
            end
            @inbounds R[i,j] = dot(qᵢ, vⱼ) # ⟵ NOTICE: mgs has vⱼ, clgs has aⱼ
            # vⱼ -= R[i,j] * qᵢ
            @inbounds for ind = 1:m
                vⱼ[ind] -= R[i,j] * qᵢ[ind]
            end
        end
        @inbounds R[j,j] = norm(vⱼ)
        # Q[:,j] = vⱼ / R[j,j]
        @inbounds for ind = 1:m
            Q[ind,j] = vⱼ[ind] / R[j,j]
        end
    end
    return nothing
end


function liap_jetcoeffs!{T<:Number}(eqsdiff!, t0::T, x::Vector{Taylor1{T}},
        dx::Vector{Taylor1{T}}, xaux::Vector{Taylor1{T}},
        δx::Array{TaylorN{Taylor1{T}},1}, δdx::Array{TaylorN{Taylor1{T}},1}, jac::Array{Taylor1{T},2})
    order = x[1].order

    # Dimensions of phase-space: dof
    nx = length(x)
    dof = round(Int, (-1+sqrt(1+4*nx))/2)

    for ord in 1:order
        ordnext = ord+1

        # Set `xaux`, auxiliary vector of Taylor1 to order `ord`
        @inbounds for j in eachindex(x)
            xaux[j] = Taylor1( x[j].coeffs[1:ord] )
        end

        # Equations of motion
        eqsdiff!(t0, xaux, dx)
        stabilitymatrix!( eqsdiff!, t0, xaux[1:dof], δx, δdx, jac )
        @inbounds dx[dof+1:nx] = jac * reshape( xaux[dof+1:nx], (dof,dof) )

        # Recursion relations
        @inbounds for j in eachindex(x)
            x[j].coeffs[ordnext] = dx[j].coeffs[ord]/ord
        end
    end
    nothing
end


function liap_taylorstep!{T<:Number}(f, x::Vector{Taylor1{T}}, dx::Vector{Taylor1{T}},
        xaux::Vector{Taylor1{T}}, δx::Array{TaylorN{Taylor1{T}},1},
        δdx::Array{TaylorN{Taylor1{T}},1}, jac::Array{Taylor1{T},2}, t0::T, t1::T, x0::Array{T,1},
        order::Int, abstol::T)

    # Compute the Taylor coefficients
    liap_jetcoeffs!(f, t0, x, dx, xaux, δx, δdx, jac)

    # Compute the step-size of the integration using `abstol`
    δt = stepsize(x, abstol)
    δt = min(δt, t1-t0)

    # Update x0
    evaluate!(x, δt, x0)
    return δt
end


function liap_taylorinteg{T<:Number}(f, q0::Array{T,1}, t0::T, tmax::T,
        order::Int, abstol::T; maxsteps::Int=500)
    # Allocation
    tv = Array{T}(maxsteps+1)
    dof = length(q0)
    xv = Array{T}(dof, maxsteps+1)
    λ = similar(xv)
    λtsum = similar(q0)
    jt = eye(T, dof)

    # NOTE: This changes GLOBALLY internal parameters of TaylorN
    global _δv = set_variables("δ", order=1, numvars=dof)

    # Initial conditions
    @inbounds tv[1] = t0
    @inbounds for ind in 1:dof
        xv[ind,1] = q0[ind]
        λ[ind,1] = zero(T)
        λtsum[ind] = zero(T)
    end
    x0 = vcat(q0, reshape(jt, dof*dof))
    nx0 = dof*(dof+1)
    t00 = t0

    # Initialize the vector of Taylor1 expansions
    x = Array{Taylor1{T}}(length(x0))
    for i in eachindex(x0)
        @inbounds x[i] = Taylor1( x0[i], order )
    end

    #Allocate auxiliary arrays
    dx = Array{Taylor1{T}}(length(x0))
    xaux = Array{Taylor1{T}}(length(x0))
    δx = Array{TaylorN{Taylor1{T}}}(dof)
    δdx = Array{TaylorN{Taylor1{T}}}(dof)
    jac = Array{Taylor1{T}}(dof,dof)
    for i in eachindex(jac)
        jac[i] = zero(x[1])
    end
    QH = Array{T}(dof,dof)
    RH = Array{T}(dof,dof)
    aⱼ = Array{eltype(jt)}( dof )
    qᵢ = similar(aⱼ)
    vⱼ = similar(aⱼ)

    # Integration
    nsteps = 1
    while t0 < tmax
        δt = liap_taylorstep!(f, x, dx, xaux, δx, δdx, jac, t0, tmax, x0, order, abstol)
        @inbounds for ind in eachindex(jt)
            jt[ind] = x0[dof+ind]
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
        @inbounds for ind in eachindex(QH)
            x0[dof+ind] = QH[ind]
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
