# This file is part of the TaylorIntegration.jl package; MIT licensed

"""
    stabilitymatrix!{T<:Number, S<:Number}(eqsdiff, t0::T, x::Array{T,1},
        jjac::Array{T,2})
    stabilitymatrix!{T<:Number, S<:Number}(eqsdiff, t0::T, x::Array{Taylor1{T},1},
        jjac::Array{Taylor1{T},2})

Updates the matrix `jjac` (linearized equations of motion)
computed from the equations of motion (`eqsdiff`), at time `t0`
at `x0`.
"""
function stabilitymatrix!{T<:Number}(eqsdiff, t0::T, x::Array{T,1},
        jjac::Array{T,2})
    δx = Array{TaylorN{T}}( length(x) )
    @inbounds for ind in eachindex(x)
        δx[ind] = x[ind] + TaylorN(T,ind,order=1)
    end
    jjac[:] = jacobian( eqsdiff(t0, δx) )
    nothing
end
function stabilitymatrix!{T<:Number}(eqsdiff, t0::T, x::Array{Taylor1{T},1},
        jjac::Array{Taylor1{T},2})
    δx = Array{TaylorN{Taylor1{T}}}( length(x) )
    @inbounds for ind in eachindex(x)
        δx[ind] = convert(TaylorN{Taylor1{T}}, x[ind]) +
            TaylorN(Taylor1{T},ind,order=1)
    end
    jjac[:] = jacobian( eqsdiff(t0, δx) )
    nothing
end


# Modified from `cgs` and `mgs`, obtained from:
# http://nbviewer.jupyter.org/url/math.mit.edu/~stevenj/18.335/Gram-Schmidt.ipynb
# Classical Gram–Schmidt (Trefethen algorithm 7.1), implemented in the simplest way
# (We could make it faster by unrolling loops to avoid temporaries arrays etc.)
function classicalGS(A)
    m,n = size(A)
    Q = similar(A)
    R = zeros(eltype(A),n,n)
    aⱼ = Array{eltype(A)}(m)
    qᵢ = similar(aⱼ)
    vⱼ = similar(aⱼ)
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
    return Q, R
end
# Modified Gram–Schmidt (Trefethen algorithm 8.1)
function modifiedGS(A)
    m,n = size(A)
    Q = similar(A)
    R = zeros(eltype(A),n,n)
    aⱼ = Array{eltype(A)}(m)
    qᵢ = similar(aⱼ)
    # vⱼ = similar(aⱼ)
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
    return Q, R
end


function liap_jetcoeffs!{T<:Number}(eqsdiff, t0::T, x::Vector{Taylor1{T}})
    order = x[1].order
    xaux = similar(x)
    xdot = similar(x)

    # Dimensions of phase-space: dof
    nx = length(x)
    dof = round(Int, (-1+sqrt(1+4*nx))/2)
    jjac = Array{Taylor1{T}}(dof,dof)

    for ord in 1:order
        ordnext = ord+1

        # Set `xaux`, auxiliary vector of Taylor1 to order `ord`
        @inbounds for j in eachindex(x)
            xaux[j] = Taylor1( x[j].coeffs[1:ord] )
        end

        # Equations of motion
        @inbounds xdot[1:dof] = eqsdiff(t0, xaux[1:dof])
        stabilitymatrix!( eqsdiff, t0, xaux[1:dof], jjac )
        @inbounds xdot[dof+1:nx] = jjac * reshape( xaux[dof+1:nx], (dof,dof) )

        # Recursion relations
        @inbounds for j in eachindex(x)
            x[j].coeffs[ordnext] = xdot[j].coeffs[ord]/ord
        end
    end
    nothing
end


function liap_taylorstep!{T<:Number}(f, t0::T, tmax::T, x0::Array{T,1},
        order::Int, abstol::T)
    # Initialize the vector of Taylor1 expansions
    xT = Array{Taylor1{T}}(length(x0))
    for i in eachindex(x0)
        @inbounds xT[i] = Taylor1( x0[i], order )
    end

    # Compute the Taylor coefficients
    liap_jetcoeffs!(f, t0, xT)

    # Compute the step-size of the integration using `abstol`
    δt = stepsize(xT, abstol)
    δt = min(δt, tmax-t0)

    # Update x0
    evaluate!(xT, δt, x0)
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
    jt = Array{T}(dof,dof)

    # NOTE: This changes GLOBALLY internal parameters of TaylorN
    global _δv = set_variables("δ", order=1, numvars=dof)

    # Initial conditions
    @inbounds tv[1] = t0
    @inbounds for ind in 1:dof
        xv[ind,1] = q0[ind]
        λ[ind,1] = zero(T)
        λtsum[ind] = zero(T)
    end
    x0 = vcat(q0, reshape(eye(T, dof), dof*dof))
    nx0 = dof*(dof+1)
    t00 = t0

    # Integration
    nsteps = 1
    while t0 < tmax
        δt = liap_taylorstep!(f, t0, tmax, x0, order, abstol)
        @inbounds for ind in eachindex(jt)
            jt[ind] = x0[dof+ind]
        end
        QH, RH = modifiedGS( jt )
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
        if nsteps > maxsteps
            warn("""
            Maximum number of integration steps reached; exiting.
            """)
            break
        end
    end

    return view(tv,1:nsteps), view(xv,:,1:nsteps)', view(λ,:,1:nsteps)'
end

# function liap_taylorinteg{T<:Number}(f, q0::Array{T,1}, trange::Range{T},
#         order::Int, abstol::T; maxsteps::Int=500)
#
#     # Allocation
#     nn = length(trange)
#     dof = length(q0)
#     xv = Array{T}(dof, nn)
#     λ = Array{T}(dof, maxsteps+1)
#     DH = Array{T}(dof)
#     fill!(xv, T(NaN))
#     fill!(λ, T(NaN))
#
#     # Initial conditions
#     @inbounds xv[:,1] = q0[:]
#     @inbounds λ[:,1] = reshape(eye(T), dof*dof)
#     x0 = vcat(q0, λ[:,1])
#     t00 = copy(t0)
#
#     # Integration
#     iter = 1
#     while iter < nn
#         t0, t1 = trange[iter], trange[iter+1]
#         nsteps = 0
#         while nsteps < maxsteps
#             xold = copy(x0)
#             δt = taylorstep!(f, t0, x0, order, abstol)
#             jt = reshape(x0[dof+1:nx0], (dof,dof))
#             VH, DH, WHt = svd(jt)
#             x0[dof+1:nx0] = VH[:]
#             if t0+δt ≥ t1
#                 x0 = xold
#                 δt = taylorstep!(f, t0, t1, x0, order, abstol)
#                 jt = reshape(x0[dof+1:nx0], (dof,dof))
#                 VH, DH, WHt = svd(jt)
#                 x0[dof+1:nx0] = VH[:]
#                 t0 = t1
#                 break
#             end
#             t0 += δt
#             nsteps += 1
#         end
#         if nsteps ≥ maxsteps && t0 != t1
#             warn("""
#             Maximum number of integration steps reached; exiting.
#             """)
#             break
#         end
#         iter += 1
#         @inbounds for ind in 1:dof
#             xv[ind,nsteps] = x0[ind]
#             λ[ind,nsteps] = ((t0-δt-t00)*λ[ind,nsteps-1] + log(DH[ind]))/(t0-t00)
#         end
#     end
#
#     return view(xv,:,1:nsteps)', view(λ,:,1:nsteps)'
# end
